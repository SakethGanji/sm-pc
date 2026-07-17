"""Minimal Python server — the POC alternative to the Go server.

Loads the newest artifact and exposes the same /classify contract, using the
exact forward pass in runtime.py (so results match the Go server). Standard
library only; no web framework needed. Bring in the Go server later for
production speed — the artifact and the math are identical, so nothing retrains.

Run:  cd trainer && uv run python ../scripts/serve_py.py [--port 8081]
Test: curl -s localhost:8081/classify -H 'Content-Type: application/json' \
        -d '{"current_customer_transcript":"i lost my card","previous_agent_utterance":"how can i help you"}'
"""

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "trainer"))

from semantic_poc.artifact import NPY_FILES
from semantic_poc.config import load_config
from semantic_poc.context import build_c1
from semantic_poc.embeddings import GeminiEmbedder
from semantic_poc.features import TfIdf
from semantic_poc.paths import ARTIFACTS_DIR
from semantic_poc.runtime import Scorer

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8081)
ap.add_argument("--artifact", default=None, help="artifact dir name; default = newest by mtime")
args = ap.parse_args()

art = (ARTIFACTS_DIR / args.artifact) if args.artifact else \
    max(ARTIFACTS_DIR.glob("artifact-*"), key=lambda p: p.stat().st_mtime)
model = json.loads((art / "model.json").read_text())
mats = {a: np.load(art / f) for a, f in NPY_FILES.items()}
scorer = Scorer(
    intents=model["intents"], families=model["families"],
    tfidf=TfIdf(json.loads((art / "tfidf_vocab.json").read_text()), np.load(art / "tfidf_idf.npy")),
    sem_coef=mats["sem_coef"], sem_int=mats["sem_int"],
    lex_coef=mats["lex_coef"], lex_int=mats["lex_int"],
    fus_coef=mats["fus_coef"], fus_int=mats["fus_int"],
    platt_a=np.array(model["platt_a"]), platt_b=np.array(model["platt_b"]),
    thresholds=model["thresholds"], tau_low=model["tau_low"],
    max_accepted=model["max_accepted"], exclusive_groups=model["exclusive_groups"],
)
embedder = GeminiEmbedder(load_config("embedding"))
print(f"serving {art.name} on :{args.port} (Python) — embedding={load_config('embedding')['model']}")


def classify(req: dict) -> dict:
    raw = req.get("current_customer_transcript", "")
    prev_agent = req.get("previous_agent_utterance", "") or ""
    emb = embedder.embed([build_c1(prev_agent, raw)])[0].astype(np.float64)
    fwd = scorer.forward(raw, prev_agent, emb)
    return {
        "decision": fwd["decision"],
        "intents": fwd["intents"],
        "overflow": fwd["overflow"],
        "model_version": art.name,
        "embedding_model_version": load_config("embedding")["model"],
    }


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: dict) -> None:
        payload = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path == "/healthz":
            self._send(200, {"status": "ok", "model_version": art.name})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/classify":
            self._send(404, {"error": "not found"})
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            self._send(200, classify(req))
        except Exception as exc:  # noqa: BLE001
            self._send(500, {"decision": "degraded", "error": str(exc)})

    def log_message(self, *a):  # quiet
        pass


# Single-threaded on purpose: the embedding cache (SQLite) is bound to one
# thread, and for a POC serialized requests are fine (latency is embedding-bound).
HTTPServer(("0.0.0.0", args.port), Handler).serve_forever()
