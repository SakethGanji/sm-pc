"""HVB corpus -> gold rows (one per caller ASR segment)."""

import json
from pathlib import Path

from .schema import GoldRow, PrevCallerSegment
from .weak_labels import label_conversation


def ingest_corpus(raw_dir: Path, task_type_map: dict[str, str]) -> list[GoldRow]:
    rows: list[GoldRow] = []
    skipped = 0
    for meta_path in sorted((raw_dir / "metadata").glob("*.json")):
        meta = json.loads(meta_path.read_text())
        sid = meta["sid"]
        tasks = meta.get("tasks") or []
        task_type = tasks[0].get("task_type") if tasks else None
        if task_type not in task_type_map:
            skipped += 1
            continue
        intent = task_type_map[task_type]

        transcript_path = raw_dir / "transcript" / f"{sid}.json"
        if not transcript_path.exists():
            skipped += 1
            continue
        segments = json.loads(transcript_path.read_text())
        segments.sort(key=lambda s: s["index"])

        conv_rows: list[GoldRow] = []
        last_agent_text = ""
        prev_seg = None  # previous segment regardless of speaker
        for seg in segments:
            if seg["speaker_role"] == "caller":
                prev_caller = None
                if prev_seg is not None and prev_seg["speaker_role"] == "caller":
                    gap = seg["start_timestamp_ms"] - (
                        prev_seg["start_timestamp_ms"] + prev_seg["duration_ms"]
                    )
                    prev_caller = PrevCallerSegment(
                        turn_index=prev_seg["index"],
                        text=prev_seg["transcript"],
                        gap_ms=max(0, gap),
                    )
                conv_rows.append(
                    GoldRow(
                        conversation_id=sid,
                        turn_index=seg["index"],
                        timestamp_ms=seg["start_timestamp_ms"],
                        raw_transcript=seg["transcript"],
                        human_transcript=seg["human_transcript"],
                        previous_agent_utterance=last_agent_text,
                        prev_caller_segment=prev_caller,
                        dialog_acts=seg.get("dialog_acts", []),
                        session_task_intent=intent,
                        labels=[],
                    )
                )
            else:
                last_agent_text = seg["transcript"]
            prev_seg = seg
        label_conversation(conv_rows)
        rows.extend(conv_rows)
    if skipped:
        print(f"ingest: skipped {skipped} sessions (missing/unknown task or transcript)")
    return rows
