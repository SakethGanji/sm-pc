"""§6.3 embedding provider — real Gemini embeddings, pinned config, disk cache.

Cache key = sha256(model|task_type|dim|text); vectors are L2-normalized before
caching so nothing downstream depends on provider normalization behavior.
"""

import hashlib
import sqlite3
import time

import numpy as np

from .paths import CACHE_DIR

BATCH_SIZE = 100


class GeminiEmbedder:
    def __init__(self, cfg: dict):
        self.model = cfg["model"]
        self.task_type = cfg["task_type"]
        self.dim = int(cfg["output_dim"])
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(CACHE_DIR / "embeddings.sqlite")
        self.db.execute("CREATE TABLE IF NOT EXISTS emb (key TEXT PRIMARY KEY, vec BLOB)")
        self._client = None

    def _key(self, text: str) -> str:
        raw = f"{self.model}|{self.task_type}|{self.dim}|{text}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def _fetch_batch(self, texts: list[str]) -> list[np.ndarray]:
        if self._client is None:
            from google import genai

            self._client = genai.Client()  # needs GEMINI_API_KEY
        from google.genai import types

        config = types.EmbedContentConfig(
            task_type=self.task_type, output_dimensionality=self.dim
        )
        for attempt in range(6):
            try:
                resp = self._client.models.embed_content(
                    model=self.model, contents=texts, config=config
                )
                out = []
                for e in resp.embeddings:
                    v = np.asarray(e.values, dtype=np.float32)
                    v /= np.linalg.norm(v) or 1.0
                    out.append(v)
                return out
            except Exception as exc:  # noqa: BLE001 — rate limits / transient 5xx
                if attempt == 5:
                    raise
                wait = min(60, 2**attempt * 2)
                print(f"embed retry in {wait}s: {exc}")
                time.sleep(wait)
        raise RuntimeError("unreachable")

    def embed(self, texts: list[str], progress: bool = False) -> np.ndarray:
        """Return (len(texts), dim) float32, cache-first, order preserved."""
        keys = [self._key(t) for t in texts]
        vecs: dict[str, np.ndarray] = {}
        for chunk in range(0, len(keys), 900):
            ks = list(set(keys[chunk : chunk + 900]))
            q = ",".join("?" * len(ks))
            for k, blob in self.db.execute(f"SELECT key, vec FROM emb WHERE key IN ({q})", ks):
                vecs[k] = np.frombuffer(blob, dtype=np.float32)

        missing: dict[str, str] = {}  # key -> text (dedup)
        for k, t in zip(keys, texts):
            if k not in vecs:
                missing[k] = t
        items = list(missing.items())
        for i in range(0, len(items), BATCH_SIZE):
            batch = items[i : i + BATCH_SIZE]
            fetched = self._fetch_batch([t for _, t in batch])
            for (k, _), v in zip(batch, fetched):
                vecs[k] = v
                self.db.execute("INSERT OR REPLACE INTO emb VALUES (?, ?)", (k, v.tobytes()))
            self.db.commit()
            if progress:
                print(f"embedded {min(i + BATCH_SIZE, len(items))}/{len(items)} new texts")

        return np.stack([vecs[k] for k in keys])
