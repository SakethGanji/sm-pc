"""Ingest adapter: raw corpus -> list[GoldRow] (one row per CUSTOMER turn).

=============================================================================
THIS IS THE FILE YOU REWRITE TO PORT THE PIPELINE TO YOUR DATA.
=============================================================================
Everything downstream (embed, train, calibrate, threshold, serve, diagnostics)
speaks only `GoldRow` and does not know or care where the data came from. So get
this one file right and the rest of the pipeline "just works".

The body below is the Harper Valley (HVB) rehearsal reference: it reads a folder
of per-call metadata + transcript JSON. To port, REPLACE the body to read YOUR
source (e.g. a CSV — see ../../AGENTS.md §3 for the column contract) and emit
the same `GoldRow`s. Then point `paths.py` at your data and wire `cli.py`'s
`ingest()` to call this. Run it with `poc ingest` (never `python ingest.py` — the
relative imports need the package context).

The contract every row must honor (full column defs: ../../AGENTS.md §3):
  * ONE row per CUSTOMER turn. Agent turns are context, not rows — their text
    rides along on the next customer turn as `previous_agent_utterance`.
  * `raw_transcript` is UNCLEANED ASR. You serve on raw, so you train/eval on raw.
  * Emit NEGATIVE turns too (greetings, "yes", read-back digits) with `labels=[]`,
    or the model never learns when to stay silent.
  * `labels` starts EMPTY here. Real labels come from your labeling pass
    (`poc promote`), never from ingest.
"""

import json
from pathlib import Path

from .schema import GoldRow, PrevCallerSegment
from .weak_labels import label_conversation  # HVB-only weak seed — drop at port


def ingest_corpus(raw_dir: Path, task_type_map: dict[str, str]) -> list[GoldRow]:
    rows: list[GoldRow] = []
    skipped = 0
    # HVB stores one metadata JSON + one transcript JSON per call. Your source
    # will iterate differently (e.g. group CSV rows by conversation_id).
    for meta_path in sorted((raw_dir / "metadata").glob("*.json")):
        meta = json.loads(meta_path.read_text())
        sid = meta["sid"]                                  # -> conversation_id
        # HVB's call-level task type is a coarse CANDIDATE filter, not a label;
        # it only seeds `session_task_intent` (a weak hint). Your real per-turn
        # labels come later from `poc promote`.
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
        segments.sort(key=lambda s: s["index"])            # ensure chronological

        conv_rows: list[GoldRow] = []
        last_agent_text = ""     # most recent agent turn seen; context for next caller turn
        prev_seg = None          # previous segment of ANY speaker (for stitch-gap math)
        for seg in segments:
            if seg["speaker_role"] == "caller":
                # Stitch fragment: only when the immediately preceding segment was
                # ALSO the caller (a split utterance, no agent turn between). Lets
                # the stitcher optionally glue "i lost my" + "card" at serve time.
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
                # One CUSTOMER turn -> one GoldRow. labels=[] (a negative for now);
                # request turns get their intent later via `poc promote`.
                conv_rows.append(
                    GoldRow(
                        conversation_id=sid,
                        turn_index=seg["index"],
                        timestamp_ms=seg["start_timestamp_ms"],
                        raw_transcript=seg["transcript"],        # never clean this
                        human_transcript=seg["human_transcript"],
                        previous_agent_utterance=last_agent_text,
                        prev_caller_segment=prev_caller,
                        dialog_acts=seg.get("dialog_acts", []),
                        session_task_intent=intent,              # weak hint only
                        labels=[],                               # filled by labeling, not here
                    )
                )
            else:
                # Agent turn: not a row, just remembered as context for the next
                # customer turn.
                last_agent_text = seg["transcript"]
            prev_seg = seg
        # HVB-only: seed weak labels from dialog acts. DROP this at port — your
        # labels come from `poc promote`, so leave conv_rows with labels=[].
        label_conversation(conv_rows)
        rows.extend(conv_rows)
    if skipped:
        print(f"ingest: skipped {skipped} sessions (missing/unknown task or transcript)")
    return rows
