#!/usr/bin/env bash
# Poll the Gemini API until billing works, then run the full e2e once.
# One tiny embedContent probe per 30 min; ~16 h max.
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; source .env; set +a
mkdir -p logs

probe() {
  curl -s -o /dev/null -w '%{http_code}' -m 20 -X POST \
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent" \
    -H "Content-Type: application/json" -H "x-goog-api-key: $GEMINI_API_KEY" \
    -d '{"content":{"parts":[{"text":"ping"}]},"taskType":"CLASSIFICATION","outputDimensionality":768}'
}

for i in $(seq 1 32); do
  code=$(probe)
  echo "$(date -Is) probe $i: HTTP $code"
  if [ "$code" = "200" ]; then
    echo "$(date -Is) billing live — starting e2e"
    bash scripts/e2e_once.sh
    exit $?
  fi
  sleep 1800
done
echo "$(date -Is) gave up after 16h — credits still depleted"
exit 3
