#!/usr/bin/env bash
# Poll the Gemini API (both keys in .env) until one has working quota, then
# promote it to GEMINI_API_KEY and run the full e2e once.
# INTERVAL seconds between rounds (default 30 min), ATTEMPTS max rounds.
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; source .env; set +a
mkdir -p logs
INTERVAL="${INTERVAL:-1800}"
ATTEMPTS="${ATTEMPTS:-32}"

probe() { # $1 = api key
  curl -s -o /dev/null -w '%{http_code}' -m 20 -X POST \
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent" \
    -H "Content-Type: application/json" -H "x-goog-api-key: $1" \
    -d '{"content":{"parts":[{"text":"ping"}]},"taskType":"CLASSIFICATION","outputDimensionality":768}'
}

for i in $(seq 1 "$ATTEMPTS"); do
  for key in "${GEMINI_API_KEY:-}" "${GEMINI_API_KEY_ALT:-}"; do
    [ -z "$key" ] && continue
    code=$(probe "$key")
    echo "$(date -Is) probe $i key=...${key: -6}: HTTP $code"
    if [ "$code" = "200" ]; then
      echo "$(date -Is) quota live on key ...${key: -6} — starting e2e"
      printf 'GEMINI_API_KEY=%s\nGEMINI_API_KEY_ALT=%s\n' "$key" "$GEMINI_API_KEY_ALT" > .env
      bash scripts/e2e_once.sh
      exit $?
    fi
  done
  sleep "$INTERVAL"
done
echo "$(date -Is) gave up — no key has working quota"
exit 3
