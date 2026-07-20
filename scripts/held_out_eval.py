"""Honest held-out evaluation: apply the dual-critic test verdicts to build a
clean test ground truth, then score the given artifact's policy-certified
thresholds on the cleaned test split. Reports per-intent precision/recall and
overall coverage at the maintained precision floor. This is the generalization
number (thresholds certified on policy, evaluated once on held-out test).

Test verdicts live at data/gold/test_verdicts.json (persistent, in-repo), format:
  {"verdicts": [{"id": "<conversation_id>#<turn_index>", "label": "<intent>|none|ambiguous"}, ...]}
If that file is absent, the eval still runs but against RAW (uncleaned) gold
labels — clearly flagged — so the harness is never blocked. The cleaned number
(the honest generalization figure) requires regenerating those verdicts via the
dual-critic adjudication; see scripts/adj_prep.py + adj_apply.py for the method.

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

if VERDICTS_PATH.exists():
    verdicts = {v["id"]: v["label"] for v in json.loads(VERDICTS_PATH.read_text())["verdicts"]}
    print(f"loaded {len(verdicts)} test verdicts from {VERDICTS_PATH}")
else:
    verdicts = {}
    print(f"!! WARNING: no verdicts at {VERDICTS_PATH} — scoring against RAW gold "
          f"labels (UNCLEANED). This understates coverage due to known test-label "
          f"noise; it is NOT the certified generalization number.")

# Clean test ground truth: adjudicated label wins; rows marked ambiguous are
# dropped; 'none' -> negative. Rows without a verdict keep their gold label.
gold = {r.conversation_id + "#" + str(r.turn_index): r for r in load_gold() if r.split == "test"}
clean = []  # (row, labelset) with ambiguous dropped
n_amb = 0
for rid, r in gold.items():
    v = verdicts.get(rid)
    if v == "ambiguous":
        n_amb += 1
        continue
    if v in (None,):
        labels = set(r.labels)
    elif v == "none":
        labels = set()
    else:
        labels = {v}
    clean.append((r, labels))
print(f"clean test rows: {len(clean)} ({n_amb} ambiguous dropped); "
      f"positives: {sum(1 for _, l in clean if l)}")

embedder = GeminiEmbedder(load_config("embedding"))
texts = [build_c1(r.previous_agent_utterance, r.raw_transcript) for r, _ in clean]
E = embedder.embed(texts, progress=True).astype(np.float64)


def eval_artifact(name):
    art = ARTIFACTS_DIR / name
    model = json.loads((art / "model.json").read_text())
    mats = {a: np.load(art / f) for a, f in NPY_FILES.items()}
    intents = model["intents"]
    scorer = Scorer(
        intents=intents, families=model["families"],
        tfidf=TfIdf(json.loads((art / "tfidf_vocab.json").read_text()), np.load(art / "tfidf_idf.npy")),
        sem_coef=mats["sem_coef"], sem_int=mats["sem_int"],
        lex_coef=mats["lex_coef"], lex_int=mats["lex_int"],
        fus_coef=mats["fus_coef"], fus_int=mats["fus_int"],
        platt_a=np.array(model["platt_a"]), platt_b=np.array(model["platt_b"]),
        thresholds=model["thresholds"], tau_low=model["tau_low"],
        max_accepted=model["max_accepted"], exclusive_groups=model["exclusive_groups"],
    )
    A, B = np.array(model["platt_a"]), np.array(model["platt_b"])
    th = model["thresholds"]

    per = {i: {"tp": 0, "fp": 0, "pos": 0} for i in intents}
    caught = pos_tot = 0
    for (r, labels), e in zip(clean, E):
        sem = scorer.sem_coef @ e + scorer.sem_int
        lex = np.asarray(scorer.tfidf.transform([r.raw_transcript]) @ scorer.lex_coef.T).ravel() + scorer.lex_int
        meta = scorer.meta_features(r.raw_transcript, r.previous_agent_utterance)
        fused = scorer.fus_coef @ np.concatenate([sem, lex, meta]) + scorer.fus_int
        probs = sigmoid(A * fused + B)
        probs_d = {i: float(p) for i, p in zip(intents, probs)}
        out = decide(probs_d, th, model["families"], model["tau_low"],
                     model["max_accepted"], model["exclusive_groups"])
        accepted = {x["name"] for x in out["intents"]}
        for i in intents:
            if i in accepted:
                per[i]["tp" if i in labels else "fp"] += 1
        for lab in labels:
            per[lab]["pos"] += 1
            pos_tot += 1
            if lab in accepted:
                caught += 1

    print(f"\n=== {name}: held-out test coverage @ certified thresholds ===")
    print(f"  overall coverage: {caught}/{pos_tot} = {caught/pos_tot:.3f}")
    below = []
    for i in intents:
        d = per[i]
        acc = d["tp"] + d["fp"]
        prec = d["tp"] / acc if acc else None
        rec = d["tp"] / d["pos"] if d["pos"] else None
        flag = " <FLOOR" if prec is not None and prec < FLOOR else ""
        if prec is not None and prec < FLOOR:
            below.append(i)
        print(f"  {i:22s} t={th[i]:.3f} prec={None if prec is None else round(prec,3)}"
              f" rec={None if rec is None else round(rec,3)} pos={d['pos']} accepted={acc}{flag}")
    # Coverage restricted to intents that actually hold the floor on held-out test:
    held = caught_h = pos_h = 0
    for i in intents:
        if i in below:
            continue
        pos_h += per[i]["pos"]
        caught_h += per[i]["tp"]
    frac = f"{caught_h/pos_h:.3f}" if pos_h else "n/a (no intent holds the floor)"
    print(f"  coverage on floor-holding intents only: {caught_h}/{pos_h} = "
          f"{frac} (dropped: {below})")
    return {"name": name, "coverage": caught / pos_tot, "below_floor": below,
            "coverage_floor_holding": caught_h / pos_h if pos_h else None}


names = sys.argv[1:]
if not names:
    latest = sorted(ARTIFACTS_DIR.glob("artifact-*"), key=lambda p: p.stat().st_mtime)
    if not latest:
        sys.exit("no artifacts found — run `poc train` first")
    names = [latest[-1].name]
    print(f"no artifact given — evaluating newest: {names[0]}")

results = [eval_artifact(n) for n in names]
out_path = ARTIFACTS_DIR / "held_out_results.json"
out_path.write_text(json.dumps(results, indent=1))
print(f"\nresults -> {out_path}")
