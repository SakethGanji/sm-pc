"""Honest held-out evaluation: apply the dual-critic test verdicts to build a
clean test ground truth, then score the given artifact's policy-certified
thresholds on the cleaned test split. Reports per-intent precision/recall and
overall coverage at the maintained precision floor. This is the generalization
number (thresholds certified on policy, evaluated once on held-out test).

Test verdicts live at data/gold/test_verdicts.json (persistent, in-repo), format:
  {"verdicts": [{"id": "<conversation_id>#<turn_index>", "label": "<intent>|none|ambiguous"}, ...]}
If that file is absent, the eval still runs but against RAW (uncleaned) gold
labels — clearly flagged — so the harness is never blocked.

Importable: build_clean_test() / embed_clean() / evaluate_artifact() are reused
by scripts/stability_report.py. Running as a script keeps the original CLI.

Usage (from trainer/, venv activated):
  python ../scripts/held_out_eval.py [<artifact_name> ...]   # defaults to newest artifact
Override verdicts path with POC_TEST_VERDICTS=/path/to/verdicts.json
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "trainer"))

from semantic_poc.artifact import NPY_FILES
from semantic_poc.cli import load_gold
from semantic_poc.config import load_config
from semantic_poc.context import build_c1
from semantic_poc.embeddings import GeminiEmbedder
from semantic_poc.features import TfIdf
from semantic_poc.paths import ARTIFACTS_DIR, GOLD_DIR
from semantic_poc.runtime import Scorer, sigmoid
from semantic_poc.policy import decide

FLOOR = 0.90
VERDICTS_PATH = Path(os.environ.get("POC_TEST_VERDICTS", GOLD_DIR / "test_verdicts.json"))


def load_verdicts() -> dict:
    if VERDICTS_PATH.exists():
        v = {x["id"]: x["label"] for x in json.loads(VERDICTS_PATH.read_text())["verdicts"]}
        print(f"loaded {len(v)} test verdicts from {VERDICTS_PATH}")
        return v
    print(f"!! WARNING: no verdicts at {VERDICTS_PATH} — scoring against RAW gold "
          f"labels (UNCLEANED). Understates coverage due to known test-label noise; "
          f"NOT the certified generalization number.")
    return {}


def build_clean_test():
    """Return [(row, labelset)] for the test split: verdict wins, ambiguous
    dropped, 'none' -> negative, no-verdict -> keep gold label."""
    verdicts = load_verdicts()
    gold = {f"{r.conversation_id}#{r.turn_index}": r
            for r in load_gold() if r.split == "test"}
    clean, n_amb = [], 0
    for rid, r in gold.items():
        v = verdicts.get(rid)
        if v == "ambiguous":
            n_amb += 1
            continue
        labels = set(r.labels) if v is None else (set() if v == "none" else {v})
        clean.append((r, labels))
    print(f"clean test rows: {len(clean)} ({n_amb} ambiguous dropped); "
          f"positives: {sum(1 for _, l in clean if l)}")
    return clean


def embed_clean(clean) -> np.ndarray:
    embedder = GeminiEmbedder(load_config("embedding"))
    texts = [build_c1(r.previous_agent_utterance, r.raw_transcript) for r, _ in clean]
    return embedder.embed(texts, progress=True).astype(np.float64)


def _load_scorer(name):
    art = ARTIFACTS_DIR / name
    model = json.loads((art / "model.json").read_text())
    mats = {a: np.load(art / f) for a, f in NPY_FILES.items()}
    scorer = Scorer(
        intents=model["intents"], families=model["families"],
        tfidf=TfIdf(json.loads((art / "tfidf_vocab.json").read_text()),
                    np.load(art / "tfidf_idf.npy")),
        sem_coef=mats["sem_coef"], sem_int=mats["sem_int"],
        lex_coef=mats["lex_coef"], lex_int=mats["lex_int"],
        fus_coef=mats["fus_coef"], fus_int=mats["fus_int"],
        platt_a=np.array(model["platt_a"]), platt_b=np.array(model["platt_b"]),
        thresholds=model["thresholds"], tau_low=model["tau_low"],
        max_accepted=model["max_accepted"], exclusive_groups=model["exclusive_groups"])
    return model, scorer


def evaluate_artifact(name, clean, E) -> dict:
    """Score one artifact over the clean test set; return structured per-intent
    precision/recall/coverage (used by the CLI report and stability tracking)."""
    model, sc = _load_scorer(name)
    intents = model["intents"]
    A, B, th = np.array(model["platt_a"]), np.array(model["platt_b"]), model["thresholds"]

    per = {i: {"tp": 0, "fp": 0, "pos": 0} for i in intents}
    caught = pos_tot = 0
    for (r, labels), e in zip(clean, E):
        sem = sc.sem_coef @ e + sc.sem_int
        lex = np.asarray(sc.tfidf.transform([r.raw_transcript]) @ sc.lex_coef.T).ravel() + sc.lex_int
        meta = sc.meta_features(r.raw_transcript, r.previous_agent_utterance)
        fused = sc.fus_coef @ np.concatenate([sem, lex, meta]) + sc.fus_int
        probs = {i: float(p) for i, p in zip(intents, sigmoid(A * fused + B))}
        out = decide(probs, th, model["families"], model["tau_low"],
                     model["max_accepted"], model["exclusive_groups"])
        accepted = {x["name"] for x in out["intents"]}
        for i in intents:
            if i in accepted:
                per[i]["tp" if i in labels else "fp"] += 1
        for lab in labels:
            if lab in per:
                per[lab]["pos"] += 1
            pos_tot += 1
            if lab in accepted:
                caught += 1

    per_intent = {}
    below = []
    for i in intents:
        d = per[i]
        acc = d["tp"] + d["fp"]
        prec = d["tp"] / acc if acc else None
        rec = d["tp"] / d["pos"] if d["pos"] else None
        if prec is not None and prec < FLOOR:
            below.append(i)
        per_intent[i] = {"threshold": th[i], "precision": prec, "recall": rec,
                         "pos": d["pos"], "accepted": acc, "tp": d["tp"], "fp": d["fp"]}
    pos_h = sum(per[i]["pos"] for i in intents if i not in below)
    caught_h = sum(per[i]["tp"] for i in intents if i not in below)
    return {"name": name, "coverage": caught / pos_tot if pos_tot else None,
            "positives": pos_tot, "below_floor": below,
            "coverage_floor_holding": caught_h / pos_h if pos_h else None,
            "per_intent": per_intent}


def print_report(res):
    print(f"\n=== {res['name']}: held-out test coverage @ certified thresholds ===")
    cov = res["coverage"]
    print(f"  overall coverage: {'n/a' if cov is None else round(cov, 3)} "
          f"(positives={res['positives']})")
    for i, d in res["per_intent"].items():
        flag = " <FLOOR" if i in res["below_floor"] else ""
        p = None if d["precision"] is None else round(d["precision"], 3)
        r = None if d["recall"] is None else round(d["recall"], 3)
        print(f"  {i:22s} t={d['threshold']:.3f} prec={p} rec={r} "
              f"pos={d['pos']} accepted={d['accepted']}{flag}")
    cfh = res["coverage_floor_holding"]
    frac = "n/a (no intent holds the floor)" if cfh is None else f"{cfh:.3f}"
    print(f"  coverage on floor-holding intents only: {frac} (dropped: {res['below_floor']})")


def main(argv):
    names = argv or None
    if not names:
        arts = sorted(ARTIFACTS_DIR.glob("artifact-*"), key=lambda p: p.stat().st_mtime)
        if not arts:
            sys.exit("no artifacts found — run `poc train` first")
        names = [arts[-1].name]
        print(f"no artifact given — evaluating newest: {names[0]}")
    clean = build_clean_test()
    E = embed_clean(clean)
    results = []
    for n in names:
        res = evaluate_artifact(n, clean, E)
        print_report(res)
        results.append(res)
    (ARTIFACTS_DIR / "held_out_results.json").write_text(json.dumps(results, indent=1))
    print(f"\nresults -> {ARTIFACTS_DIR / 'held_out_results.json'}")


if __name__ == "__main__":
    main(sys.argv[1:])
