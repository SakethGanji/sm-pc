"""Conversation-grouped, time-ordered partitions (§7.4).

Whole conversations never cross partitions; splits follow real session start
time (train -> calibration -> policy -> test, newest last). The test slice is
hashed at partition time and evaluated once, after freeze.
"""

import hashlib
import json
from collections import Counter

from .schema import GoldRow

FRACTIONS = {"train": 0.60, "calibration": 0.15, "policy": 0.15, "test": 0.10}
ORDER = ["train", "calibration", "policy", "test"]


def assign_splits(rows: list[GoldRow]) -> dict:
    conv_ts: dict[str, int] = {}
    for r in rows:
        conv_ts[r.conversation_id] = min(r.timestamp_ms, conv_ts.get(r.conversation_id, 1 << 62))
    convs = sorted(conv_ts, key=lambda c: (conv_ts[c], c))

    n = len(convs)
    split_of: dict[str, str] = {}
    start = 0
    for name in ORDER:
        end = n if name == ORDER[-1] else start + round(FRACTIONS[name] * n)
        for c in convs[start:end]:
            split_of[c] = name
        start = end

    for r in rows:
        r.split = split_of[r.conversation_id]

    test_payload = json.dumps(
        [r.model_dump() for r in rows if r.split == "test"], sort_keys=True
    ).encode()
    manifest = {
        "conversations": {name: sum(1 for c in convs if split_of[c] == name) for name in ORDER},
        "rows": dict(Counter(r.split for r in rows)),
        "labeled_rows": dict(Counter(r.split for r in rows if r.labels)),
        "test_rows_sha256": hashlib.sha256(test_payload).hexdigest(),
    }
    return manifest
