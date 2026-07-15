"""Turn-level weak labels derived from HVB dialog acts.

HVB labels the task per conversation; the classifier needs per-turn labels.
Corpus inspection shows the FIRST caller turn tagged with a request-bearing
dialog act is reliably the caller stating the task; LATER request-tagged turns
are mostly dialog-act noise with occasional genuine restatements. Rule:

- first request-bearing caller turn  -> [session task intent]
- later request-bearing caller turns -> unlabeled AND marked ambiguous
  (excluded from training/eval; must not become false no-intent negatives)
- everything else                    -> no actionable intent
"""

from .schema import GoldRow

REQUEST_BEARING_ACTS = {"gridspace_problem_description"}


def is_request_bearing(dialog_acts: list[str]) -> bool:
    return any(a in REQUEST_BEARING_ACTS for a in dialog_acts)


def label_conversation(rows: list[GoldRow]) -> None:
    """Assign labels/ambiguous in place. `rows` = one conversation, turn order."""
    seen_request = False
    for r in rows:
        if is_request_bearing(r.dialog_acts):
            if not seen_request:
                r.labels = [r.session_task_intent]
                seen_request = True
            else:
                r.labels = []
                r.ambiguous = True
        else:
            r.labels = []
