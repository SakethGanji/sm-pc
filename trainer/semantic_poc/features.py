"""Lexical TF-IDF (§6.5), implemented explicitly so the Go server can mirror
the math exactly (parity-tested):

  tokens  = regex findall on lowercased text
  terms   = unigrams + bigrams ("w1 w2")
  tf(t)   = 1 + ln(count)              (sublinear)
  idf(t)  = ln((1+N)/(1+df)) + 1       (N = training docs)
  x       = tf*idf, then L2-normalized
"""

import re

import numpy as np
from scipy import sparse

TOKEN_RE = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def terms_of(text: str, ngram_max: int = 2) -> list[str]:
    toks = tokenize(text)
    terms = list(toks)
    if ngram_max >= 2:
        terms += [f"{a} {b}" for a, b in zip(toks, toks[1:])]
    return terms


class TfIdf:
    def __init__(self, vocab: dict[str, int], idf: np.ndarray):
        self.vocab = vocab
        self.idf = idf.astype(np.float32)

    @classmethod
    def fit(cls, docs: list[str], min_df: int = 2, max_vocab: int = 200000) -> "TfIdf":
        df: dict[str, int] = {}
        for d in docs:
            for t in set(terms_of(d)):
                df[t] = df.get(t, 0) + 1
        kept = sorted((t for t, c in df.items() if c >= min_df),
                      key=lambda t: (-df[t], t))[:max_vocab]
        kept.sort()  # deterministic, order-independent vocab indices
        vocab = {t: i for i, t in enumerate(kept)}
        n = len(docs)
        idf = np.array([np.log((1 + n) / (1 + df[t])) + 1 for t in kept], dtype=np.float32)
        return cls(vocab, idf)

    def transform(self, docs: list[str]) -> sparse.csr_matrix:
        indptr, indices, data = [0], [], []
        for d in docs:
            counts: dict[int, int] = {}
            for t in terms_of(d):
                j = self.vocab.get(t)
                if j is not None:
                    counts[j] = counts.get(j, 0) + 1
            row = {j: (1 + np.log(c)) * self.idf[j] for j, c in counts.items()}
            norm = np.sqrt(sum(v * v for v in row.values())) or 1.0
            for j in sorted(row):
                indices.append(j)
                data.append(row[j] / norm)
            indptr.append(len(indices))
        return sparse.csr_matrix(
            (np.array(data, dtype=np.float32), indices, indptr),
            shape=(len(docs), len(self.vocab)),
        )
