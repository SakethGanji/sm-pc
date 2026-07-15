#!/usr/bin/env bash
# Full end-to-end: train -> unit tests -> Go parity -> serve -> replay.
# Assumes GEMINI_API_KEY has working billing.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; source .env; set +a
export PATH="$HOME/.local/go/bin:$PATH"

echo "=== [1/5] train (embeds via Gemini, cache-first) ==="
(cd trainer && uv run poc train)

echo "=== [2/5] trainer unit tests ==="
(cd trainer && uv run pytest -q)

echo "=== [3/5] Go tests incl. Python<->Go parity ==="
(cd server && go test ./...)

echo "=== [4/5] serve ==="
(cd server && go build -o bin/serve ./cmd/serve)
server/bin/serve -addr :8080 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT
for _ in $(seq 1 40); do
  curl -sf http://localhost:8080/healthz >/dev/null && break
  sleep 0.5
done
curl -sf http://localhost:8080/healthz

echo "=== [5/5] replay test conversations through the live server ==="
(cd trainer && uv run poc replay --n-conversations 8)

echo "=== E2E COMPLETE ==="
