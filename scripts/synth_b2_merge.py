"""Merge verified bucket-2 synthetic rows into gold.jsonl as train-only rows.
Keeps a wrapped row only if the blind dual-critic verdict matches its intended
label (positives) or both agree 'none' (hard negatives / preamble-only).
Near-duplicate filter against the real corpus and within the batch.

Usage: python synth_b2_merge.py /tmp/synth-b2/verdicts.json
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "trainer"))

from semantic_poc.cli import load_gold, save_gold
from semantic_poc.paths import GOLD_DIR

verdicts = {v["id"]: v["label"] for v in json.load(open(sys.argv[1]))["verdicts"]}
intended = json.load(open("/tmp/synth-b2/intended.json"))
batches = sorted(Path("/tmp/synth-b2").glob("batch_*.jsonl"))
cand = {}
for b in batches:
    for line in open(b):
        r = json.loads(line)
        cand[r["id"]] = r


def norm(text):
    return frozenset(re.findall(r"[a-z0-9']+", text.lower()))


rows = load_gold()
if any(r.conversation_id.startswith("synth-b2-") for r in rows):
    print("gold already has synth-b2 rows — aborting")
    sys.exit(1)

real_texts = {r.raw_transcript for r in rows if not r.conversation_id.startswith("synth-")}
seen = []
kept, stats = [], Counter()
for rid, meta in intended.items():
    want = meta["labels"]  # [] for negatives, [intent] for positives
    v = verdicts.get(rid)
    if v is None:
        stats["no_verdict"] += 1
        continue
    want_label = want[0] if want else "none"
    if v != want_label:
        stats[f"reject_{meta['kind']}"] += 1
        continue
    text = cand[rid]["caller"]
    key = norm(text)
    if text in real_texts or any(len(key & s) / max(1, len(key | s)) > 0.85 for s in seen):
        stats["dup"] += 1
        continue
    seen.append(key)
    kept.append((rid, meta, cand[rid]))
    stats[f"keep_{meta['kind']}"] += 1

from semantic_poc.schema import GoldRow

new_rows = []
for n, (rid, meta, c) in enumerate(kept):
    new_rows.append(GoldRow(
        conversation_id=f"synth-b2-{n}",
        turn_index=1,
        timestamp_ms=0,
        raw_transcript=c["caller"],
        human_transcript=c["caller"],
        previous_agent_utterance=c.get("agent", "") or "",
        dialog_acts=[f"synthetic:b2:{meta['kind']}"],
        session_task_intent=(meta["labels"][0] if meta["labels"] else "no_intent"),
        labels=meta["labels"],
        split="train",
    ))

rows.extend(new_rows)
save_gold(rows)
print(json.dumps(dict(stats), indent=1))
n_pos = sum(1 for r in new_rows if r.labels)
print(f"\nappended {len(new_rows)} synth-b2 train rows ({n_pos} positive, "
      f"{len(new_rows) - n_pos} negative)")
print(f"survival: {len(kept)}/{len(intended)} = {len(kept)/len(intended):.0%}")
