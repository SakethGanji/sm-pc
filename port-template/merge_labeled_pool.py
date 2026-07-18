"""
Merge label_helper.py's output into data/gold/gold.jsonl.

Rows from label_helper.py are already GoldRow-shaped, so this is mostly a
dedup + append + honest-summary step — not a format conversion.

Usage:
    cd trainer && uv run python ../port-template/merge_labeled_pool.py \\
        ../port-template/labeled_pool.jsonl

Prints a verified vs. chatbot-only breakdown so you know, before you train,
how much of what you just added was double-checked by a human. Since you can't
choose which conversation lands in your eval split later (time+conversation
assigns it automatically), unverified rows are a real risk if they happen to
land in calibration/policy/test — the printed summary is your prompt to go
back and verify any batch that's mostly chatbot-only before you trust results.
"""

import json
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "trainer"))
from semantic_poc.paths import GOLD_DIR  # noqa: E402
from semantic_poc.schema import GoldRow  # noqa: E402

pool_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("labeled_pool.jsonl")
gold_path = GOLD_DIR / "gold.jsonl"

existing_ids = set()
if gold_path.exists():
    for line in open(gold_path):
        r = json.loads(line)
        existing_ids.add(f"{r['conversation_id']}#{r['turn_index']}")

pool_rows = [json.loads(l) for l in open(pool_path) if l.strip()]

# Validate every row against the real schema BEFORE writing anything — a bad
# row from label_helper.py should fail loudly here, not silently corrupt
# gold.jsonl and surface as a cryptic error the next time it's loaded.
for r in pool_rows:
    try:
        GoldRow.model_validate(r)
    except Exception as exc:  # noqa: BLE001
        rid = f"{r.get('conversation_id')}#{r.get('turn_index')}"
        sys.exit(f"invalid row {rid} in {pool_path}: {exc}")

stats = Counter()
new_rows = []
for r in pool_rows:
    rid = f"{r['conversation_id']}#{r['turn_index']}"
    if rid in existing_ids:
        stats["skipped_dup"] += 1
        continue
    verified = "manual:verified" in r.get("dialog_acts", [])
    stats["verified" if verified else "chatbot_only"] += 1
    stats["positive" if r["labels"] else "negative"] += 1
    new_rows.append(r)
    existing_ids.add(rid)

if not new_rows:
    print("nothing new to merge.")
    sys.exit(0)

if gold_path.exists():
    shutil.copy(gold_path, GOLD_DIR / "gold.pre-manual-label.jsonl")
GOLD_DIR.mkdir(parents=True, exist_ok=True)
with open(gold_path, "a") as f:
    for r in new_rows:
        f.write(json.dumps(r) + "\n")

print(json.dumps(dict(stats), indent=1))
print(f"\nmerged {len(new_rows)} rows into {gold_path}")

unverified_frac = stats["chatbot_only"] / max(1, stats["verified"] + stats["chatbot_only"])
if unverified_frac > 0.3:
    print(f"\nWARNING: {unverified_frac:.0%} of merged rows are chatbot-only "
          f"(not personally confirmed). Since any conversation could land in "
          f"your eval split, consider going back and verifying more before "
          f"trusting the certified precision numbers.")

print("\nNext: uv run poc partition && uv run poc train")
