"""The POC server. Loads the newest artifact and implements the §5 request/
response contract: stitch -> embed -> lexical + semantic -> fusion -> decision
policy.

Run:  cd trainer && source .venv/bin/activate && python ../scripts/serve_py.py [--port 8081]
Test: curl -s localhost:8081/classify -H 'Content-Type: application/json' \
        -d '{"current_customer_transcript":"i lost my card","previous_agent_utterance":"how can i help you"}'
"""

import argparse
import json
import sys
import time
from datetime import datetime
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
from semantic_poc.stitch import PrevSegment, StitcherConfig, stitch

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8081)
ap.add_argument("--artifact", default=None, help="artifact dir name; default = newest by mtime")
args = ap.parse_args()

art = (ARTIFACTS_DIR / args.artifact) if args.artifact else \
    max(ARTIFACTS_DIR.glob("artifact-*"), key=lambda p: p.stat().st_mtime)
model = json.loads((art / "model.json").read_text())
runtime_cfg = json.loads((art / "runtime_config.json").read_text())
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
stitcher_cfg = StitcherConfig.from_config(runtime_cfg["stitcher"])
context_cfg = runtime_cfg["context"]
embedder = GeminiEmbedder(load_config("embedding"))
print(f"serving {art.name} on :{args.port} (Python) — embedding={load_config('embedding')['model']}")


def _parse_ts_ms(timestamp: str) -> int:
    if not timestamp:
        return 0
    try:
        ts = timestamp.replace("Z", "+00:00")
        return int(datetime.fromisoformat(ts).timestamp() * 1000)
    except ValueError:
        return 0  # unparsable timestamp -> stitch gap math sees 0, same as "no timestamp"


def _ms(a: float, b: float) -> int:
    return round((b - a) * 1000)


def classify(req: dict) -> dict:
    t0 = time.monotonic()
    raw = req.get("current_customer_transcript", "")
    prev_agent = req.get("previous_agent_utterance", "") or ""
    resp = {
        "decision": "",
        "intents": [],  # §5.2: a list, never null
        "supersedes_turn_index": None,
        "stitched": False,
        "long_path_used": False,  # LLM condense path — not implemented in the POC
        "model_version": art.name,
        "embedding_model_version": load_config("embedding")["model"],
    }

    prev_seg = req.get("previous_customer_segment")
    prev = None
    if prev_seg:
        prev = PrevSegment(text=prev_seg["text"], turn_index=prev_seg["turn_index"],
                           end_ts_ms=prev_seg["end_ts_ms"], decision=prev_seg["decision"])
    cur_start_ms = _parse_ts_ms(req.get("timestamp", ""))
    st = stitch(prev, raw, cur_start_ms, stitcher_cfg)
    t_stitch = time.monotonic()
    resp["stitched"] = st.stitched
    if st.stitched:
        resp["supersedes_turn_index"] = st.supersedes_index
        print(f"stitched conv={req.get('conversation_id')} turn={req.get('turn_index')} "
              f"rule={st.rule_fired} gap_ms={st.gap_ms} "
              f"prev={prev.text!r} cur={raw!r} -> {st.text!r}")

    c1 = build_c1(prev_agent, st.text, context_cfg["agent_tag"], context_cfg["customer_tag"])
    try:
        emb = embedder.embed([c1])[0].astype(np.float64)
    except Exception as exc:  # noqa: BLE001 — infra failure -> degraded, never a semantic decision
        print(f"embedding degraded conv={req.get('conversation_id')} turn={req.get('turn_index')}: {exc}")
        resp["decision"] = "degraded"
        resp["error"] = "embedding_unavailable"
        resp["latency_ms"] = {"stitch": _ms(t0, t_stitch), "embedding": _ms(t_stitch, time.monotonic()),
                              "scoring": 0, "total": _ms(t0, time.monotonic())}
        return resp
    t_embed = time.monotonic()

    fwd = scorer.forward(st.text, prev_agent, emb)
    t_scored = time.monotonic()

    resp["decision"] = fwd["decision"]
    resp["intents"] = fwd["intents"]
    if fwd.get("overflow"):
        print(f"accepted-cap overflow conv={req.get('conversation_id')} turn={req.get('turn_index')}")
    resp["latency_ms"] = {
        "stitch": _ms(t0, t_stitch), "embedding": _ms(t_stitch, t_embed),
        # Python's forward pass isn't split into lexical/semantic/fusion stages
        # the way the (removed) Go server's was — reported together.
        "scoring": _ms(t_embed, t_scored), "total": _ms(t0, time.monotonic()),
    }
    return resp


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
        n = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError as exc:
            self._send(400, {"error": str(exc)})
            return
        self._send(200, classify(req))

    def log_message(self, *a):  # quiet
        pass


# Single-threaded on purpose: the embedding cache (SQLite) is bound to one
# thread, and for a POC serialized requests are fine (latency is embedding-bound).
HTTPServer(("0.0.0.0", args.port), Handler).serve_forever()
