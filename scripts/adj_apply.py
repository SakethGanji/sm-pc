"""Apply dual-critic adjudication verdicts (/tmp/adj/verdicts.json) to
gold.jsonl. Agreement on an intent -> labels=[intent]; agreement on none ->
labels=[]; disagreement/unsure -> ambiguous=True (excluded from train+eval).
Test split is never touched. Backs up gold.jsonl first."""

import json
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "trainer"))

from semantic_poc.cli import load_gold, save_gold
from semantic_poc.paths import GOLD_DIR

verdicts = {v["id"]: v["label"] for v in json.load(open("/tmp/adj/verdicts.json"))["verdicts"]}

prior = {}
for line in open(GOLD_DIR / "adjudications.tsv"):
    line = line.strip()
    if line:
        rid, v = line.split("\t")
        prior[rid] = v

shutil.copy(GOLD_DIR / "gold.jsonl", GOLD_DIR / "gold.pre-adj-b2.jsonl")
rows = load_gold()

stats = Counter()
flips_of_prior_no = []
per_split_promoted = Counter()
for r in rows:
    rid = f"{r.conversation_id}#{r.turn_index}"
    v = verdicts.get(rid)
    if v is None:
        continue
    if r.split == "test":  # safety: never relabel frozen test
        stats["skipped_test"] += 1
        continue
    if r.labels or r.ambiguous:
        stats["skipped_already_labeled"] += 1
        continue
    stats["seen"] += 1
    if v == "ambiguous":
        r.ambiguous = True
        stats["marked_ambiguous"] += 1
    elif v == "none":
        stats["confirmed_negative"] += 1
    else:
        r.labels = [v]
        stats["promoted"] += 1
        per_split_promoted[r.split] += 1
        if prior.get(rid) == "no":
            flips_of_prior_no.append((rid, v, r.raw_transcript[:70]))

save_gold(rows)
with open(GOLD_DIR / "adjudications-b2.tsv", "w") as f:
    for rid, v in sorted(verdicts.items()):
        f.write(f"{rid}\t{v}\n")

print(json.dumps(stats, indent=1))
print("promoted per split:", dict(per_split_promoted))
print(f"\nflips of prior 'no' adjudications: {len(flips_of_prior_no)}")
for rid, v, txt in flips_of_prior_no[:15]:
    print(f"  {rid} -> {v}: {txt}")

lab = Counter()
for r in rows:
    if r.labels and not r.ambiguous and not r.conversation_id.startswith("synth-"):
        lab[r.split] += 1
print("\nlabeled real rows per split now:", dict(lab))
