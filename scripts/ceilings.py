"""Ceiling analyses for the feasibility session (docs/next-session.md).

1. Certification ceiling — oracle thresholds set with point-estimate precision
   on the pooled policy+test labeled rows (read-only diagnostic peek at test,
   per plan; no model/threshold decision flows back from it).
2. Representation ceiling — per-intent AP/AUC of the fused calibrated scores.
3. ASR ceiling — same eval scoring human_transcript instead of raw_transcript.

Coverage metric everywhere: fraction of labeled positives whose true intent is
in the accepted set under the full decision policy (matches miss_analysis.py).
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

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

ART_NAME = sys.argv[1] if len(sys.argv) > 1 else "artifact-poc-hvb-001-7ec9fd8a"
FLOOR = 0.90

art = ARTIFACTS_DIR / ART_NAME
model = json.loads((art / "model.json").read_text())
vocab = json.loads((art / "tfidf_vocab.json").read_text())
mats = {a: np.load(art / f) for a, f in NPY_FILES.items()}
intents = model["intents"]
scorer = Scorer(
    intents=intents, families=model["families"],
    tfidf=TfIdf(vocab, np.load(art / "tfidf_idf.npy")),
    sem_coef=mats["sem_coef"], sem_int=mats["sem_int"],
    lex_coef=mats["lex_coef"], lex_int=mats["lex_int"],
    fus_coef=mats["fus_coef"], fus_int=mats["fus_int"],
    platt_a=np.array(model["platt_a"]), platt_b=np.array(model["platt_b"]),
    thresholds=model["thresholds"], tau_low=model["tau_low"],
    max_accepted=model["max_accepted"], exclusive_groups=model["exclusive_groups"],
)
embedder = GeminiEmbedder(load_config("embedding"))


def probs_matrix(rows, text_attr="raw_transcript"):
    texts = [build_c1(r.previous_agent_utterance, getattr(r, text_attr)) for r in rows]
    E = embedder.embed(texts, progress=True).astype(np.float64)
    P = np.zeros((len(rows), len(intents)))
    for i, (r, e) in enumerate(zip(rows, E)):
        sem = scorer.sem_coef @ e + scorer.sem_int
        lex = np.asarray(scorer.tfidf.transform([getattr(r, text_attr)]) @ scorer.lex_coef.T).ravel() + scorer.lex_int
        meta = scorer.meta_features(getattr(r, text_attr), r.previous_agent_utterance)
        fused = scorer.fus_coef @ np.concatenate([sem, lex, meta]) + scorer.fus_int
        P[i] = sigmoid(np.array(model["platt_a"]) * fused + np.array(model["platt_b"]))
    return P


def coverage(rows, P, thresholds):
    """(caught, positives, per-intent dict) under the full decision policy."""
    caught, per = 0, {i: [0, 0] for i in intents}
    for r, p in zip(rows, P):
        if not r.labels:
            continue
        probs_d = {i: float(v) for i, v in zip(intents, p)}
        out = decide(probs_d, thresholds, model["families"], model["tau_low"],
                     model["max_accepted"], model["exclusive_groups"])
        accepted = {x["name"] for x in out["intents"]}
        for lab in r.labels:
            per[lab][1] += 1
            if lab in accepted:
                per[lab][0] += 1
                caught += 1
    pos = sum(v[1] for v in per.values())
    return caught, pos, per


def accept_precision(rows, P, thresholds):
    """Per-intent precision over accepted turns (all rows, incl. unlabeled)."""
    out = {}
    for k, intent in enumerate(intents):
        t = thresholds[intent]
        sel = P[:, k] >= t
        if not sel.sum():
            out[intent] = (None, 0)
            continue
        tp = sum(1 for r, s in zip(rows, sel) if s and intent in r.labels)
        out[intent] = (tp / sel.sum(), int(sel.sum()))
    return out


def oracle_thresholds(rows, P, floor):
    y = np.array([[1 if i in r.labels else 0 for i in intents] for r in rows])
    th = {}
    for k, intent in enumerate(intents):
        p, yk = P[:, k], y[:, k]
        best = 2.0
        for t in sorted(set(p[yk == 1])):
            sel = p >= t
            if sel.sum() and yk[sel].mean() >= floor:
                best = float(t)
                break
        th[intent] = best
    return th


rows = [r for r in load_gold() if not r.ambiguous]
pol = [r for r in rows if r.split == "policy"]
tst = [r for r in rows if r.split == "test"]

print("=== embedding / scoring policy split (raw) ===")
P_pol = probs_matrix(pol)

# ---------- baseline ----------
caught, pos, per = coverage(pol, P_pol, model["thresholds"])
print(f"\nBASELINE coverage on policy (certified thresholds): {caught}/{pos} = {caught/pos:.3f}")

# ---------- ceiling 2: representation (policy only, no test needed) ----------
print("\n=== CEILING 2: per-intent AP / AUC of fused scores (policy split) ===")
y_pol = np.array([[1 if i in r.labels else 0 for i in intents] for r in pol])
for k, intent in enumerate(intents):
    yk = y_pol[:, k]
    ap = average_precision_score(yk, P_pol[:, k])
    auc = roc_auc_score(yk, P_pol[:, k])
    # recall among positives at the score that would give perfect ranking cut:
    print(f"  {intent:22s} AP={ap:.3f} AUC={auc:.3f} positives={yk.sum()}")

# ---------- ceiling 1: certification (oracle thresholds on policy+test pool) ----------
print("\n=== CEILING 1: oracle thresholds on pooled policy+test (diagnostic peek) ===")
P_tst = probs_matrix(tst)
pool = pol + tst
P_pool = np.vstack([P_pol, P_tst])
oth = oracle_thresholds(pool, P_pool, FLOOR)

for name, th in [("certified", model["thresholds"]), ("oracle", oth)]:
    c, p, per = coverage(pol, P_pool[: len(pol)], th)
    cp, pp, _ = coverage(pool, P_pool, th)
    prec = accept_precision(pool, P_pool, th)
    print(f"\n{name} thresholds -> policy coverage {c}/{p} = {c/p:.3f}; "
          f"pool coverage {cp}/{pp} = {cp/pp:.3f}")
    for k, intent in enumerate(intents):
        pr, n = prec[intent]
        print(f"  {intent:22s} t={th[intent]:.3f} pool_precision={pr if pr is None else round(pr,3)} "
              f"accepted={n} policy_recall={per[intent][0]}/{per[intent][1]}")

# ---------- ceiling 3: ASR (human transcript, policy split) ----------
print("\n=== CEILING 3: human-transcript eval (policy split) ===")
P_hum = probs_matrix(pol, "human_transcript")
c_h, p_h, per_h = coverage(pol, P_hum, model["thresholds"])
print(f"human-transcript coverage (certified thresholds): {c_h}/{p_h} = {c_h/p_h:.3f}")
for k, intent in enumerate(intents):
    yk = y_pol[:, k]
    ap_r = average_precision_score(yk, P_pol[:, k])
    ap_h = average_precision_score(yk, P_hum[:, k])
    print(f"  {intent:22s} AP raw={ap_r:.3f} human={ap_h:.3f} delta={ap_h-ap_r:+.3f}")

# oracle-on-human upper bound (pool oracle thresholds applied to human-text scores)
c_ho, _, _ = coverage(pol, P_hum, oth)
print(f"human-transcript coverage @ oracle thresholds: {c_ho}/{p_h} = {c_ho/p_h:.3f}")

json.dump({
    "baseline": [caught, pos],
    "oracle_thresholds": oth,
    "certified_thresholds": model["thresholds"],
}, open("/tmp/ceilings_out.json", "w"), indent=1)
print("\nwrote /tmp/ceilings_out.json")
