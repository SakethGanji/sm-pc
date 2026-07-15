"""Show every labeled policy-split positive the current artifact misses:
its probability for the true intent, the threshold it failed, and what the
model preferred instead."""

import json
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
from semantic_poc.runtime import Scorer

art = max(ARTIFACTS_DIR.glob("artifact-*"), key=lambda p: p.stat().st_mtime)
model = json.loads((art / "model.json").read_text())
vocab = json.loads((art / "tfidf_vocab.json").read_text())
mats = {a: np.load(art / f) for a, f in NPY_FILES.items()}
scorer = Scorer(
    intents=model["intents"], families=model["families"],
    tfidf=TfIdf(vocab, np.load(art / "tfidf_idf.npy")),
    sem_coef=mats["sem_coef"], sem_int=mats["sem_int"],
    lex_coef=mats["lex_coef"], lex_int=mats["lex_int"],
    fus_coef=mats["fus_coef"], fus_int=mats["fus_int"],
    platt_a=np.array(model["platt_a"]), platt_b=np.array(model["platt_b"]),
    thresholds=model["thresholds"], tau_low=model["tau_low"],
    max_accepted=model["max_accepted"], exclusive_groups=model["exclusive_groups"],
)
embedder = GeminiEmbedder(load_config("embedding"))

rows = [r for r in load_gold() if r.split == "policy" and r.labels and not r.ambiguous]
embs = embedder.embed([build_c1(r.previous_agent_utterance, r.raw_transcript) for r in rows])

print(f"artifact: {art.name} — {len(rows)} labeled policy positives\n")
misses = []
for r, e in zip(rows, embs):
    fwd = scorer.forward(r.raw_transcript, r.previous_agent_utterance, e.astype(np.float64))
    true = r.labels[0]
    accepted = {i["name"] for i in fwd["intents"]}
    if true not in accepted:
        p = fwd["probs"][true]
        t = model["thresholds"][true]
        top = max(fwd["probs"], key=fwd["probs"].get)
        misses.append((true, p, t, top, fwd["probs"][top], r))

misses.sort(key=lambda m: (m[0], -m[1]))
cur = None
for true, p, t, top, top_p, r in misses:
    if true != cur:
        cur = true
        print(f"--- {true} (threshold {'unattainable' if t > 1 else f'{t:.2f}'}) ---")
    gap = "NEAR " if t <= 1 and p >= t - 0.15 else ""
    rival = f" [model prefers {top} p={top_p:.2f}]" if top != true and top_p > p else ""
    print(f"  {gap}p={p:.2f}{rival}")
    if r.previous_agent_utterance:
        print(f"      agent:  {r.previous_agent_utterance[:70]}")
    print(f"      caller: {r.raw_transcript[:90]}")
print(f"\n{len(misses)} missed / {len(rows)} positives "
      f"({1 - len(misses) / len(rows):.0%} caught)")
