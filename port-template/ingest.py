"""
=============================================================================
INGEST ADAPTER — the ONE file you rewrite to run this pipeline on your data.
=============================================================================

WHAT THIS DOES
--------------
Reads YOUR raw corpus (call transcripts from wherever they live — a DB export, a
folder of JSON/CSV, a data-warehouse dump) and emits a flat list of `GoldRow`
objects, one per CUSTOMER turn. Nothing downstream (embedding, training,
calibration, thresholds, serving, diagnostics) knows or cares where the data
came from — it only speaks `GoldRow`. So: get this file right and the rest of
the pipeline "just works."

HOW TO USE
----------
1. Copy this file to `trainer/semantic_poc/ingest.py` (replacing the HVB one).
2. Fill in every `# TODO:` below with your data's specifics.
3. Point `cli.py`'s `ingest()` command at your raw path (see NOTE at the bottom).
4. Run `uv run poc ingest` and eyeball `data/gold/gold.jsonl`.

THE CONTRACT (what a valid GoldRow needs — see trainer/semantic_poc/schema.py)
-----------------------------------------------------------------------------
  conversation_id         str   groups turns; a whole call stays in one split
  turn_index              int   order of this turn within the call (0,1,2,...)
  timestamp_ms            int   real event time in ms; drives time-ordered splits
  raw_transcript          str   ASR output, UNCLEANED — this is what you serve on
  human_transcript        str   human-corrected text; copy raw if you have none
  previous_agent_utterance str  the agent turn right before this customer turn ("" if none)
  prev_caller_segment     obj|None  the customer fragment right before (for stitching); optional
  dialog_acts             list[str] any per-turn tags you have; [] is fine
  session_task_intent     str   call-level reason code if you have one, else ""  (a weak seed only)
  labels                  list[str] LEAVE EMPTY HERE ([]). Real labels come from your
                                    labeling/adjudication pass, NOT from ingest.

RULES YOU MUST NOT BREAK
------------------------
- Do NOT clean `raw_transcript`. You serve on raw ASR, so you train/eval on it.
- Emit CUSTOMER turns only as rows. Agent turns are context, not rows.
- Emit NEGATIVE turns too (greetings, "yes", read-back numbers, small talk) with
  labels=[]. If you only emit request turns, the model never learns to stay silent.
- `labels` stays [] here. Labeling is a separate, human-gated step (see §3 of the
  playbook and port-template/bootstrap_labels.py).
"""

import json
from pathlib import Path

# Import the shared schema. When this file lives at trainer/semantic_poc/ingest.py
# the relative import resolves; the fallback lets the smoke test import it
# standalone from the template folder. Delete the try/except once it's in place.
try:
    from .schema import GoldRow, PrevCallerSegment
except ImportError:  # standalone (smoke test)
    from semantic_poc.schema import GoldRow, PrevCallerSegment


# ---------------------------------------------------------------------------
# TODO 1: Map YOUR call-reason codes to taxonomy intent names (optional).
# If your calls carry a coarse "reason for call" tag, map it here to seed
# `session_task_intent`. This is a WEAK hint only (used by some labeling
# bootstraps), NEVER a final label. If you have no such tag, delete this and
# pass session_task_intent="".
# ---------------------------------------------------------------------------
REASON_CODE_MAP: dict[str, str] = {
    # "CARD_REPLACE_v2": "replace_card",
    # "BILLPAY":         "pay_bill",
    # ... your ~50 mappings ...
}

# What your data calls the two speakers. Adjust to your schema's values.
CUSTOMER_ROLE = "customer"   # TODO 2: your value, e.g. "caller", "cust", "0"
AGENT_ROLE = "agent"         # TODO 2: your value, e.g. "agent", "rep", "1"

# Max gap (ms) between two consecutive customer fragments to still treat the
# earlier one as a stitchable prefix. Tune to your ASR's segmentation. 2000ms is
# a reasonable default lifted from the rehearsal.
STITCH_MAX_GAP_MS = 2000


def _load_conversations(raw_dir: Path):
    """
    TODO 3: Yield one conversation at a time from YOUR corpus.

    A "conversation" here is: (conversation_id, list_of_turns) where each turn is
    a dict you can read below. Adapt this to however your data is stored — a
    folder of per-call JSON files, rows in a CSV grouped by call id, a Parquet
    export, a DB cursor, etc.

    Each turn dict should expose (rename to your columns in _read_turn):
      - speaker role (customer vs agent)
      - the ASR text
      - (optional) the human-corrected text
      - a start timestamp in ms
      - (optional) a duration in ms  (used for stitch gap math)
      - (optional) any per-turn tags

    The example below assumes a folder of JSON files, one per call, each a list
    of turn objects. REPLACE IT.
    """
    for call_path in sorted(raw_dir.glob("*.json")):
        payload = json.loads(call_path.read_text())
        conversation_id = call_path.stem            # TODO: your call id
        turns = payload                             # TODO: your list of turns
        # Ensure chronological order; adjust the sort key to your timestamp field.
        turns = sorted(turns, key=lambda t: t["start_ms"])
        yield conversation_id, turns


def _read_turn(turn: dict) -> dict:
    """
    TODO 4: Normalize ONE raw turn from your schema into the fields the loop below
    expects. This is the only place your column names appear. Return a dict with
    exactly these keys.
    """
    return {
        "role": turn["speaker"],                       # TODO: your speaker field
        "raw": turn["asr_text"],                       # TODO: your ASR text field
        # If you have no human transcript yet, fall back to raw (diagnostics that
        # compare raw vs human will simply show no delta until you add corrections):
        "human": turn.get("human_text", turn["asr_text"]),
        "start_ms": int(turn["start_ms"]),             # TODO: your start-time field (ms)
        "duration_ms": int(turn.get("duration_ms", 0)),# TODO: your duration field (ms), 0 if none
        "acts": turn.get("tags", []),                  # TODO: any per-turn tags, or []
    }


def ingest_corpus(raw_dir: Path) -> list[GoldRow]:
    """Turn your whole corpus into gold rows (one per customer turn)."""
    rows: list[GoldRow] = []
    skipped = 0

    for conversation_id, raw_turns in _load_conversations(raw_dir):
        turns = [_read_turn(t) for t in raw_turns]

        # Optional call-level reason seed (weak hint only).
        session_intent = ""
        # TODO 5 (optional): if your call carries a reason code, map it:
        #   session_intent = REASON_CODE_MAP.get(call_reason_code, "")

        last_agent_text = ""          # most recent agent utterance seen so far
        prev_turn = None              # previous turn of ANY speaker (for stitch gap)

        for idx, t in enumerate(turns):
            if t["role"] == AGENT_ROLE:
                # Agent turns are context, not rows. Remember the text so the next
                # customer turn can carry it as previous_agent_utterance.
                last_agent_text = t["raw"]
                prev_turn = t
                continue

            if t["role"] != CUSTOMER_ROLE:
                # Unknown speaker role — skip, but count it so you notice.
                skipped += 1
                prev_turn = t
                continue

            # Build the optional stitch fragment: the immediately preceding turn,
            # only if it was ALSO the customer (a split utterance) and close in time.
            prev_caller = None
            if prev_turn is not None and prev_turn["role"] == CUSTOMER_ROLE:
                gap = t["start_ms"] - (prev_turn["start_ms"] + prev_turn["duration_ms"])
                if 0 <= gap <= STITCH_MAX_GAP_MS:
                    prev_caller = PrevCallerSegment(
                        turn_index=idx - 1,
                        text=prev_turn["raw"],
                        gap_ms=max(0, gap),
                    )

            rows.append(
                GoldRow(
                    conversation_id=conversation_id,
                    turn_index=idx,
                    timestamp_ms=t["start_ms"],
                    raw_transcript=t["raw"],            # NEVER clean this
                    human_transcript=t["human"],
                    previous_agent_utterance=last_agent_text,
                    prev_caller_segment=prev_caller,
                    dialog_acts=t["acts"],
                    session_task_intent=session_intent,
                    labels=[],                          # filled later by adjudication
                )
            )
            prev_turn = t

    if skipped:
        print(f"ingest: skipped {skipped} turns (unknown speaker role)")
    print(f"ingest: {len(rows)} customer-turn rows from your corpus")
    return rows


# ---------------------------------------------------------------------------
# NOTE — wiring into the CLI:
# The `ingest` command in trainer/semantic_poc/cli.py currently calls the HVB
# signature `ingest_corpus(RAW_HVB, taxonomy["task_type_map"])`. Change ONLY
# that one line — keep the `_preserve_labeled` call, it's what stops a re-run
# of `poc ingest` from wiping out your adjudicated labels:
#
#     from .ingest import ingest_corpus
#     rows = ingest_corpus(YOUR_RAW_DIR)          # e.g. paths.RAW_DIR
#     rows = _preserve_labeled(rows)              # keeps labels across re-ingests
#     save_gold(rows)
#
# and point YOUR_RAW_DIR at your corpus (add it to paths.py). Everything after
# ingest (partition, train, serve, replay) is unchanged.
# ---------------------------------------------------------------------------
