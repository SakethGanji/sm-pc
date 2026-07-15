"""Offline forward pass over trained parameters — the exact computation the Go
server re-implements. Used for eval, parity fixtures, and replay checks."""

from dataclasses import dataclass

import numpy as np

from .context import build_c1
from .features import TfIdf, tokenize
from .policy import decide


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class Scorer:
    intents: list[str]
    families: dict[str, str]
    tfidf: TfIdf
    sem_coef: np.ndarray  # (K, dim)
    sem_int: np.ndarray  # (K,)
    lex_coef: np.ndarray  # (K, V)
    lex_int: np.ndarray
    fus_coef: np.ndarray  # (K, 2K + n_meta)
    fus_int: np.ndarray
    platt_a: np.ndarray  # (K,)
    platt_b: np.ndarray
    thresholds: dict[str, float]
    tau_low: float
    max_accepted: int
    exclusive_groups: list[list[str]]

    def meta_features(self, raw_text: str, prev_agent: str) -> np.ndarray:
        return np.array(
            [len(tokenize(raw_text)) / 100.0, 1.0 if prev_agent else 0.0],
            dtype=np.float64,
        )

    def forward(self, raw_text: str, prev_agent: str, embedding: np.ndarray) -> dict:
        sem = self.sem_coef @ embedding + self.sem_int
        lex_vec = self.tfidf.transform([raw_text])
        lex = np.asarray(lex_vec @ self.lex_coef.T).ravel() + self.lex_int
        meta = self.meta_features(raw_text, prev_agent)
        fus_in = np.concatenate([sem, lex, meta])
        fused = self.fus_coef @ fus_in + self.fus_int
        probs = sigmoid(self.platt_a * fused + self.platt_b)
        probs_d = {i: float(p) for i, p in zip(self.intents, probs)}
        result = decide(probs_d, self.thresholds, self.families,
                        self.tau_low, self.max_accepted, self.exclusive_groups)
        return {
            "c1_text": build_c1(prev_agent, raw_text),
            "sem_logits": sem.tolist(),
            "lex_logits": lex.tolist(),
            "meta": meta.tolist(),
            "fused_logits": fused.tolist(),
            "probs": probs_d,
            **result,
        }
