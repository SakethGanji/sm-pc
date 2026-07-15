"""Training: semantic OVR LR + lexical OVR LR (per-intent C swept on
conversation-grouped folds), fusion LR on out-of-fold logits (§6.6), Platt
calibration on the calibration split, precision-floor thresholds on the
policy split."""

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold

from .context import build_c1
from .features import TfIdf, tokenize
from .schema import GoldRow

C_GRID = [0.25, 1.0, 4.0]
N_FOLDS = 5


@dataclass
class Split:
    rows: list[GoldRow]
    X_sem: np.ndarray
    X_lex: sparse.csr_matrix
    meta: np.ndarray
    y: np.ndarray  # (n, K) binary
    groups: np.ndarray


def build_split(rows: list[GoldRow], intents: list[str], tfidf: TfIdf, embedder) -> Split:
    texts_c1 = [build_c1(r.previous_agent_utterance, r.raw_transcript) for r in rows]
    X_sem = embedder.embed(texts_c1, progress=True).astype(np.float64)
    X_lex = tfidf.transform([r.raw_transcript for r in rows]).astype(np.float64)
    meta = np.array(
        [[len(tokenize(r.raw_transcript)) / 100.0,
          1.0 if r.previous_agent_utterance else 0.0] for r in rows]
    )
    y = np.array([[1 if i in r.labels else 0 for i in intents] for r in rows])
    groups = np.array([r.conversation_id for r in rows])
    return Split(rows, X_sem, X_lex, meta, y, groups)


def _fit_lr(X, y, C: float) -> LogisticRegression:
    return LogisticRegression(C=C, class_weight="balanced", max_iter=3000).fit(X, y)


def sweep_and_oof(X, y_all: np.ndarray, groups: np.ndarray, intents: list[str]):
    """Per-intent C selection by OOF average precision; returns (best_C per
    intent, OOF logits matrix) computed with the winning C."""
    gkf = GroupKFold(n_splits=N_FOLDS)
    folds = list(gkf.split(np.zeros(len(y_all)), groups=groups))
    oof = np.zeros((len(y_all), len(intents)))
    best_cs: list[float] = []
    for k, intent in enumerate(intents):
        y = y_all[:, k]
        best_c, best_ap, best_oof = C_GRID[0], -1.0, None
        for C in C_GRID:
            oof_k = np.zeros(len(y))
            for tr, te in folds:
                if y[tr].sum() == 0:
                    continue
                m = _fit_lr(X[tr], y[tr], C)
                oof_k[te] = m.decision_function(X[te])
            ap = average_precision_score(y, oof_k)
            if ap > best_ap:
                best_c, best_ap, best_oof = C, ap, oof_k
        best_cs.append(best_c)
        oof[:, k] = best_oof
        print(f"  {intent:22s} C={best_c:<4} oof_ap={best_ap:.3f}")
    return best_cs, oof


def fit_branch_full(X, y_all, intents, best_cs):
    coefs, ints = [], []
    for k, _ in enumerate(intents):
        m = _fit_lr(X, y_all[:, k], best_cs[k])
        coefs.append(m.coef_.ravel())
        ints.append(m.intercept_[0])
    return np.vstack(coefs), np.array(ints)


def fit_fusion(oof_sem, oof_lex, meta, y_all, intents):
    F = np.hstack([oof_sem, oof_lex, meta])
    return fit_branch_full(F, y_all, intents, [1.0] * len(intents))


def branch_logits(coef, intercept, X) -> np.ndarray:
    out = X @ coef.T
    if sparse.issparse(out):
        out = np.asarray(out)
    return np.asarray(out) + intercept


def fused_logits(models: "Trained", split: Split) -> np.ndarray:
    sem = branch_logits(models.sem_coef, models.sem_int, split.X_sem)
    lex = branch_logits(models.lex_coef, models.lex_int, split.X_lex)
    F = np.hstack([sem, lex, split.meta])
    return branch_logits(models.fus_coef, models.fus_int, F)


@dataclass
class Trained:
    intents: list[str]
    tfidf: TfIdf
    sem_coef: np.ndarray = field(default=None)
    sem_int: np.ndarray = field(default=None)
    lex_coef: np.ndarray = field(default=None)
    lex_int: np.ndarray = field(default=None)
    fus_coef: np.ndarray = field(default=None)
    fus_int: np.ndarray = field(default=None)
    platt_a: np.ndarray = field(default=None)
    platt_b: np.ndarray = field(default=None)
    thresholds: dict[str, float] = field(default_factory=dict)
    sem_cs: list[float] = field(default_factory=list)
    lex_cs: list[float] = field(default_factory=list)


def fit_platt(fused: np.ndarray, y_all: np.ndarray, intents: list[str]):
    a, b = [], []
    for k, intent in enumerate(intents):
        y = y_all[:, k]
        if y.sum() < 3:
            print(f"  {intent}: <3 positives in calibration — identity calibration")
            a.append(1.0)
            b.append(0.0)
            continue
        m = LogisticRegression(C=1e6, max_iter=3000).fit(fused[:, [k]], y)
        a.append(m.coef_[0, 0])
        b.append(m.intercept_[0])
    return np.array(a), np.array(b)


def pick_thresholds(probs: np.ndarray, y_all: np.ndarray, intents: list[str],
                    precision_floor: float) -> dict[str, float]:
    """Smallest threshold whose point-estimate precision on the policy split
    meets the floor (maximizes coverage subject to the floor)."""
    thresholds = {}
    for k, intent in enumerate(intents):
        p, y = probs[:, k], y_all[:, k]
        best = 2.0  # unattainable -> never accept
        for t in sorted(set(p[y == 1])):
            sel = p >= t
            if sel.sum() and y[sel].mean() >= precision_floor:
                best = float(t)
                break
        if best == 2.0:
            print(f"  {intent}: floor {precision_floor} unattainable on policy split")
        thresholds[intent] = round(best, 6)
    return thresholds
