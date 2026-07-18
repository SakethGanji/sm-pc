#!/usr/bin/env bash
# Full end-to-end: train -> unit tests -> serve -> replay.
# Assumes GEMINI_API_KEY has working billing.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; source .env; set +a

echo "=== [1/4] train (embeds via Gemini, cache-first) ==="
(cd trainer && uv run poc train)

echo "=== [2/4] trainer unit tests (incl. stitcher) ==="
(cd trainer && uv run pytest -q)

echo "=== [3/4] serve ==="
ARTIFACT_DIR=$(ls -dt artifacts/artifact-* | head -1)
(cd trainer && uv run python ../scripts/serve_py.py --port 8081 --artifact "$(basename "$ARTIFACT_DIR")") &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT
for _ in $(seq 1 40); do
  curl -sf http://localhost:8081/healthz >/dev/null && break
  sleep 0.5
done
curl -sf http://localhost:8081/healthz

echo "=== [4/4] replay test conversations through the live server ==="
(cd trainer && uv run poc replay --n-conversations 8)

echo "=== E2E COMPLETE ==="
