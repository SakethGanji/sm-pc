from pydantic import BaseModel


class PrevCallerSegment(BaseModel):
    """Immediately preceding caller segment (stitcher input), when one exists
    with no agent turn in between."""

    turn_index: int
    text: str
    gap_ms: int


class GoldRow(BaseModel):
    """One caller ASR segment. Raw transcript is never cleaned (§ design
    principles); the human-corrected transcript rides along to separate ASR
    failures from classifier failures."""

    conversation_id: str
    turn_index: int
    timestamp_ms: int
    raw_transcript: str
    human_transcript: str
    previous_agent_utterance: str  # raw ASR text of most recent agent segment, "" if none
    prev_caller_segment: PrevCallerSegment | None = None
    dialog_acts: list[str]
    session_task_intent: str  # the conversation-level task, mapped to taxonomy
    labels: list[str]  # weak turn-level labels ([] = no actionable intent)
    # Later request-bearing turns are ambiguous under weak labeling (could be a
    # genuine restatement or dialog-act noise) — excluded from training and eval.
    ambiguous: bool = False
    split: str = ""  # train | calibration | policy | test (set by partition step)
