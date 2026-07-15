"""Quantify how the model does on ASR-damaged turns that DO carry a real intent.
Held-out test only, adjudicated (clean) truth. For each true positive, measure
how far the raw ASR text is from the clean human transcript (word edit distance,
restricted to the words that differ), and whether the model still caught it.

Splits positives into: ASR-clean, lightly-damaged, heavily-damaged, and the
subset where the *core intent keyword* itself was corrupted."""

import difflib
import json
import re
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
from semantic_poc.paths import ARTIFACTS_DIR
from semantic_poc.runtime import Scorer, sigmoid
from semantic_poc.policy import decide

art = max(ARTIFACTS_DIR.glob("artifact-*"), key=lambda p: p.stat().st_mtime)
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
embedder = GeminiEmbedder(load_config("embedding"))
A, B = np.array(model["platt_a"]), np.array(model["platt_b"])

tv = {v["id"]: v["label"] for v in json.load(open("/tmp/adj-test/verdicts.json"))["verdicts"]}
rows = [r for r in load_gold() if r.split == "test"]
def truth(r):
    v = tv.get(f"{r.conversation_id}#{r.turn_index}")
    if v is None:
        return set(r.labels)
    if v in ("none", "ambiguous"):
        return set()
    return {v}

# intent-defining keywords: if any survive verbatim, the core request is intact
KW = {
    "replace_card": ["card"], "order_checks": ["check", "checks", "checkbook", "checkbooks"],
    "transfer_money": ["transfer", "move"], "pay_bill": ["bill", "pay"],
    "check_balance": ["balance"], "reset_password": ["password", "reset"],
    "schedule_appointment": ["appointment", "schedule"], "get_branch_hours": ["hours", "branch"],
}
tok = lambda s: re.findall(r"[a-z0-9']+", s.lower())

pos = [r for r in rows if truth(r)]
texts = [build_c1(r.previous_agent_utterance, r.raw_transcript) for r in pos]
E = embedder.embed(texts).astype(np.float64)

def caught(r, e):
    sem = scorer.sem_coef @ e + scorer.sem_int
    lex = np.asarray(scorer.tfidf.transform([r.raw_transcript]) @ scorer.lex_coef.T).ravel() + scorer.lex_int
    meta = scorer.meta_features(r.raw_transcript, r.previous_agent_utterance)
    fused = scorer.fus_coef @ np.concatenate([sem, lex, meta]) + scorer.fus_int
    probs = {i: float(p) for i, p in zip(intents, sigmoid(A * fused + B))}
    out = decide(probs, model["thresholds"], model["families"], model["tau_low"],
                 model["max_accepted"], model["exclusive_groups"])
    acc = {i["name"] for i in out["intents"]}
    lab = list(truth(r))[0]
    return lab in acc, probs[lab]

buckets = {"clean": [0, 0], "light": [0, 0], "heavy": [0, 0], "kw_lost": [0, 0]}
kw_lost_examples = []
for r, e in zip(pos, E):
    lab = list(truth(r))[0]
    raw_t, hum_t = tok(r.raw_transcript), tok(r.human_transcript)
    sm = difflib.SequenceMatcher(a=hum_t, b=raw_t)
    changed = sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal")
    ratio = changed / max(len(hum_t), 1)
    kw_intact = any(k in raw_t for k in KW[lab])
    hit, p = caught(r, e)
    if not kw_intact:
        b = "kw_lost"
        kw_lost_examples.append((hit, p, lab, r))
    elif changed == 0:
        b = "clean"
    elif ratio < 0.25:
        b = "light"
    else:
        b = "heavy"
    buckets[b][0] += hit
    buckets[b][1] += 1

print(f"Held-out test positives: {len(pos)}\n")
print("Bucket (by how much ASR corrupted the turn vs the clean transcript):")
labels = {"clean": "ASR clean (raw == human)",
          "light": "lightly damaged (<25% words changed, core keyword survived)",
          "heavy": "heavily damaged (>=25% words changed, core keyword survived)",
          "kw_lost": "CORE KEYWORD itself corrupted/dropped by ASR"}
for b in ["clean", "light", "heavy", "kw_lost"]:
    c, n = buckets[b]
    if n:
        print(f"  {labels[b]:60s} caught {c}/{n} = {c/n:.0%}")

print("\nThe hardest slice — ASR destroyed the core request word itself:")
for hit, p, lab, r in sorted(kw_lost_examples, key=lambda x: x[0]):
    mark = "CAUGHT " if hit else "missed "
    print(f"  {mark} p={p:.2f} truth={lab}")
    print(f"      raw ASR: {r.raw_transcript[:95]}")
    print(f"      clean  : {r.human_transcript[:95]}")
