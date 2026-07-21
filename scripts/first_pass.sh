#!/usr/bin/env bash
# First pass — let the LLM label EVERYTHING, then train. Directional only: the
# eval set is LLM-labeled, so the number is optimistic (shared bias). Afterward,
# CORRECT the labels — especially the test split — and retrain for an honest number.
#
# Run it in the background and watch the log:
#   nohup bash scripts/first_pass.sh > first_pass.log 2>&1 &
#   tail -f first_pass.log
#
# Pass through autolabel flags, e.g. a cheap trial run first:
#   bash scripts/first_pass.sh --limit 200
#   bash scripts/first_pass.sh --model gemini-2.5-flash
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"
set -a; source .env; set +a                        # GEMINI_API_KEY
export POC_GOLD_PATH="${POC_GOLD_PATH:-$REPO/data/gold/gold.duckdb}"
cd trainer && source .venv/bin/activate

echo "=== [1/5] ingest → gold.duckdb ==="
poc ingest
poc counts
echo "=== [2/5] partition (freeze the test set) ==="
poc partition
poc snapshot --label pre-autolabel
echo "=== [3/5] auto-label EVERYTHING (LLM; eval included → directional) ==="
python ../scripts/autolabel.py "$@"
poc counts
echo "=== [4/5] train ==="
poc train
echo "=== [5/5] held-out number (DIRECTIONAL — eval is LLM-labeled) ==="
python ../scripts/held_out_eval.py
poc snapshot --label first-pass
echo "=== FIRST PASS COMPLETE ==="
echo "Next: correct labels (esp. the test split → honest number), then re-run 'poc train'."
