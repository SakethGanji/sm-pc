"""Learning curve: coverage at the precision floor vs training-data volume.
Trains on 25/50/75/100% of train conversations (embeddings cache-only)."""

import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "trainer"))

from semantic_poc.cli import load_gold
from semantic_poc.config import load_config
from semantic_poc.embeddings import GeminiEmbedder
from semantic_poc.features import TfIdf
from semantic_poc.runtime import sigmoid
from semantic_poc.train import (Trained, build_split, fit_branch_full,
                                fit_fusion, fit_platt, fused_logits,
                                pick_thresholds, sweep_and_oof)

taxonomy = load_config("taxonomy")
policy_cfg = load_config("policy")
feat_cfg = load_config("features")
intents = list(taxonomy["intents"].keys())
floor = policy_cfg["precision_floor"]

rows = [r for r in load_gold() if not r.ambiguous]
parts = {s: [r for r in rows if r.split == s] for s in ["train", "calibration", "policy"]}
embedder = GeminiEmbedder(load_config("embedding"))

convs = sorted({r.conversation_id for r in parts["train"]})
rng = random.Random(0)
rng.shuffle(convs)

print(f"{'frac':>5} {'convs':>6} {'pos':>5} {'macro_recall':>12} {'accepted':>9} {'floors_met':>10}")
for frac in [0.25, 0.5, 0.75, 1.0]:
    keep = set(convs[: int(len(convs) * frac)])
    tr_rows = [r for r in parts["train"] if r.conversation_id in keep]
    tfidf = TfIdf.fit([r.raw_transcript for r in tr_rows],
                      min_df=feat_cfg["tfidf"]["min_df"],
                      max_vocab=feat_cfg["tfidf"]["max_vocab"])
    tr = build_split(tr_rows, intents, tfidf, embedder)
    cal = build_split(parts["calibration"], intents, tfidf, embedder)
    pol = build_split(parts["policy"], intents, tfidf, embedder)

    m = Trained(intents=intents, tfidf=tfidf)
    m.sem_cs, oof_sem = sweep_and_oof(tr.X_sem, tr.y, tr.groups, intents)
    m.lex_cs, oof_lex = sweep_and_oof(tr.X_lex, tr.y, tr.groups, intents)
    m.fus_coef, m.fus_int = fit_fusion(oof_sem, oof_lex, tr.meta, tr.y, intents)
    m.sem_coef, m.sem_int = fit_branch_full(tr.X_sem, tr.y, intents, m.sem_cs)
    m.lex_coef, m.lex_int = fit_branch_full(tr.X_lex, tr.y, intents, m.lex_cs)
    m.platt_a, m.platt_b = fit_platt(fused_logits(m, cal), cal.y, intents)
    probs = sigmoid(m.platt_a * fused_logits(m, pol) + m.platt_b)
    thresholds = pick_thresholds(probs, pol.y, intents, floor)

    recalls, accepted, floors_met = [], 0, 0
    for k, intent in enumerate(intents):
        t = thresholds[intent]
        sel = probs[:, k] >= t
        y = pol.y[:, k]
        if y.sum():
            recalls.append(float(sel[y == 1].mean()))
        accepted += int(sel.sum())
        floors_met += int(t <= 1.0)
    print(f"{frac:5.2f} {len(keep):6d} {int(tr.y.sum()):5d} "
          f"{np.mean(recalls):12.3f} {accepted:9d} {floors_met:10d}/8", flush=True)
