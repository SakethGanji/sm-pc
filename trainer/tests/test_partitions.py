from semantic_poc.partitions import assign_splits
from semantic_poc.schema import GoldRow


def row(conv: str, idx: int, ts: int) -> GoldRow:
    return GoldRow(conversation_id=conv, turn_index=idx, timestamp_ms=ts,
                   raw_transcript="x", human_transcript="x",
                   previous_agent_utterance="", dialog_acts=[],
                   session_task_intent="pay_bill", labels=[])


def test_conversations_never_split_and_time_ordered():
    rows = [row(f"c{i}", j, 1000 * i + j) for i in range(20) for j in range(3)]
    manifest = assign_splits(rows)
    split_of = {}
    for r in rows:
        assert split_of.setdefault(r.conversation_id, r.split) == r.split
    order = ["train", "calibration", "policy", "test"]
    max_ts = {s: max(r.timestamp_ms for r in rows if r.split == s) for s in order}
    min_ts = {s: min(r.timestamp_ms for r in rows if r.split == s) for s in order}
    for earlier, later in zip(order, order[1:]):
        assert max_ts[earlier] < min_ts[later]
    assert manifest["conversations"]["train"] == 12
    assert "test_rows_sha256" in manifest
