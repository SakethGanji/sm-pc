"""Build adjudication batches for the label-noise cleanup: all unlabeled
non-ambiguous policy+calibration rows, plus train rows flagged by request
regex OR model score >= 0.10 (train selection may be model-guided; the
measurement splits are exhaustive so coverage denominators stay unbiased)."""

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

OUT = Path("/tmp/adj")
OUT.mkdir(exist_ok=True)
BATCH = 40

PAT = re.compile(
    r"(lost my (debit |credit )?card|new (check ?book|checks|card)|order (some |new )?checks"
    r"|transfer (some )?money|pay (my |a )?bill|pay my \w+ bill|check my( account)? balance"
    r"|reset my password|schedule an appointment|branch hours|replace my card|new one|move money)"
)

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
A, B = np.array(model["platt_a"]), np.array(model["platt_b"])

rows = [r for r in load_gold() if not r.conversation_id.startswith("synth-")]

cands = []
for split in ["policy", "calibration", "train"]:
    unl = [r for r in rows if r.split == split and not r.labels and not r.ambiguous]
    if split == "train":
        texts = [build_c1(r.previous_agent_utterance, r.raw_transcript) for r in unl]
        E = embedder.embed(texts, progress=True).astype(np.float64)
        keep = []
        for r, e in zip(unl, E):
            if PAT.search(r.raw_transcript):
                keep.append(r)
                continue
            sem = scorer.sem_coef @ e + scorer.sem_int
            lex = np.asarray(scorer.tfidf.transform([r.raw_transcript]) @ scorer.lex_coef.T).ravel() + scorer.lex_int
            meta = scorer.meta_features(r.raw_transcript, r.previous_agent_utterance)
            fused = scorer.fus_coef @ np.concatenate([sem, lex, meta]) + scorer.fus_int
            if sigmoid(A * fused + B).max() >= 0.10:
                keep.append(r)
        unl = keep
    for r in unl:
        cands.append({
            "id": f"{r.conversation_id}#{r.turn_index}",
            "split": split,
            "agent": r.previous_agent_utterance,
            "caller": r.raw_transcript,
        })
    print(f"{split}: {sum(1 for c in cands if c['split'] == split)} candidates")

for old in OUT.glob("batch_*.jsonl"):
    old.unlink()
n_batches = 0
for i in range(0, len(cands), BATCH):
    with open(OUT / f"batch_{i // BATCH:03d}.jsonl", "w") as f:
        for c in cands[i : i + BATCH]:
            f.write(json.dumps(c) + "\n")
    n_batches += 1
print(f"total {len(cands)} candidates in {n_batches} batches under {OUT}")
