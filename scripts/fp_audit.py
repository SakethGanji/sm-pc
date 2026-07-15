"""List the high-scoring 'false positives' per intent on the policy split —
if these read as true requests, measured precision understates true precision
(weak-label noise at the top of the ranking)."""

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
from semantic_poc.runtime import Scorer, sigmoid

art = ARTIFACTS_DIR / "artifact-poc-hvb-001-7ec9fd8a"
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

rows = [r for r in load_gold() if r.split in ("policy", "test") and not r.ambiguous]
texts = [build_c1(r.previous_agent_utterance, r.raw_transcript) for r in rows]
E = embedder.embed(texts).astype(np.float64)
A, B = np.array(model["platt_a"]), np.array(model["platt_b"])

P = np.zeros((len(rows), len(intents)))
for i, (r, e) in enumerate(zip(rows, E)):
    sem = scorer.sem_coef @ e + scorer.sem_int
    lex = np.asarray(scorer.tfidf.transform([r.raw_transcript]) @ scorer.lex_coef.T).ravel() + scorer.lex_int
    meta = scorer.meta_features(r.raw_transcript, r.previous_agent_utterance)
    fused = scorer.fus_coef @ np.concatenate([sem, lex, meta]) + scorer.fus_int
    P[i] = sigmoid(A * fused + B)

for k, intent in enumerate(intents):
    fps = [(p, r) for p, r in zip(P[:, k], rows) if p >= 0.5 and intent not in r.labels]
    fps.sort(key=lambda x: -x[0])
    print(f"\n=== {intent}: {len(fps)} non-positives with p>=0.5 ===")
    for p, r in fps[:10]:
        print(f"  p={p:.3f} split={r.split} task={r.session_task_intent} "
              f"labels={r.labels} acts={r.dialog_acts}")
        print(f"    caller: {r.raw_transcript[:100]}")
