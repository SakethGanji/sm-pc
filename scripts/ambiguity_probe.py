"""Run hand-picked ambiguous / edge-case / control utterances through a
trained artifact and report the decision + top probabilities. Not a
certified metric (these aren't labeled, adjudicated eval rows) — a
robustness check: does the model abstain safely on genuinely ambiguous
input, and how close is any abstain to tipping into a false accept?

Add your own cases by pointing --cases at a JSONL file of
{"text": ..., "prev_agent": ... (optional), "note": ...} objects; otherwise
the small built-in default list runs.

Usage (from trainer/, venv activated):
  python ../scripts/ambiguity_probe.py <artifact_name> [--cases cases.jsonl] [--near-miss 0.05]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "trainer"))

from semantic_poc.artifact import NPY_FILES
from semantic_poc.config import load_config
from semantic_poc.context import build_c1
from semantic_poc.embeddings import GeminiEmbedder
from semantic_poc.features import TfIdf
from semantic_poc.paths import ARTIFACTS_DIR
from semantic_poc.runtime import Scorer

DEFAULT_PREV_AGENT = "how can i help you today"

DEFAULT_CASES = [
    {"text": "when are you open", "note": "control -> get_branch_hours"},
    {"text": "i lost my card can you send me a new one", "note": "control -> replace_card"},
    {"text": "i need new checks mailed to me", "note": "control -> order_checks"},
    {"text": "what is my card", "note": "ambiguous -- no clean match"},
    {"text": "i need help with my card", "note": "ambiguous -- replace_card vs order_checks?"},
    {"text": "is my card still active", "note": "ambiguous -- card status, not in taxonomy"},
    {"text": "i want to move some money", "note": "known confusable -- transfer_money vs pay_bill"},
    {"text": "i want to pay something", "note": "known confusable -- pay_bill vs transfer_money"},
    {"text": "something's wrong with my account", "note": "very vague"},
    {"text": "can you check something for me", "note": "very vague"},
]


def load_scorer(artifact_name: str) -> tuple[Scorer, Path]:
    art = ARTIFACTS_DIR / artifact_name
    model = json.loads((art / "model.json").read_text())
    mats = {a: np.load(art / f) for a, f in NPY_FILES.items()}
    scorer = Scorer(
        intents=model["intents"], families=model["families"],
        tfidf=TfIdf(json.loads((art / "tfidf_vocab.json").read_text()), np.load(art / "tfidf_idf.npy")),
        sem_coef=mats["sem_coef"], sem_int=mats["sem_int"],
        lex_coef=mats["lex_coef"], lex_int=mats["lex_int"],
        fus_coef=mats["fus_coef"], fus_int=mats["fus_int"],
        platt_a=np.array(model["platt_a"]), platt_b=np.array(model["platt_b"]),
        thresholds=model["thresholds"], tau_low=model["tau_low"],
        max_accepted=model["max_accepted"], exclusive_groups=model["exclusive_groups"],
    )
    return scorer, art


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("artifact", nargs="?", default=None,
                    help="artifact dir name; default = newest by mtime")
    ap.add_argument("--cases", type=Path, default=None,
                    help="JSONL of {text, prev_agent?, note?}; default = built-in list")
    ap.add_argument("--near-miss", type=float, default=0.05,
                    help="flag an abstain/no_supported_intent as a near-miss if the "
                         "top intent's probability is within this margin of its threshold")
    args = ap.parse_args()

    artifact_name = args.artifact or max(
        ARTIFACTS_DIR.glob("artifact-*"), key=lambda p: p.stat().st_mtime).name
    scorer, art = load_scorer(artifact_name)
    embedder = GeminiEmbedder(load_config("embedding"))

    cases = DEFAULT_CASES
    if args.cases:
        cases = [json.loads(l) for l in open(args.cases) if l.strip()]

    print(f"probing {artifact_name} with {len(cases)} cases\n")
    print(f"{'utterance':45s} {'decision':20s} {'top intent':18s} {'p':>6s} {'thr':>6s}  note")
    print("-" * 120)
    near_misses = []
    for c in cases:
        text, prev_agent = c["text"], c.get("prev_agent", DEFAULT_PREV_AGENT)
        emb = embedder.embed([build_c1(prev_agent, text)])[0].astype(np.float64)
        fwd = scorer.forward(text, prev_agent, emb)
        probs = fwd["probs"]
        top_name, top_p = sorted(probs.items(), key=lambda kv: -kv[1])[0]
        thr = scorer.thresholds[top_name]
        print(f"{text:45s} {fwd['decision']:20s} {top_name:18s} {top_p:.3f} {thr:.3f}  {c.get('note', '')}")
        if fwd["decision"] in ("no_supported_intent", "abstained") and thr - top_p <= args.near_miss:
            near_misses.append((text, top_name, top_p, thr))

    if near_misses:
        print(f"\n{len(near_misses)} NEAR-MISS abstain(s) — within {args.near_miss} of accepting "
              f"(safe today, thin margin; watch these under phrasing/ASR noise):")
        for text, name, p, thr in near_misses:
            print(f"  {text!r} -> {name} p={p:.3f} vs threshold={thr:.3f} (margin {thr - p:.3f})")


if __name__ == "__main__":
    main()
