"""
=============================================================================
LABEL BOOTSTRAP — how unlabeled gold rows become TRUSTED labels.
=============================================================================

WHY THIS EXISTS
---------------
`ingest.py` emits rows with `labels=[]`. Something has to fill them in. In the
HVB rehearsal that seed came from dialog-act tags (weak_labels.py) — you almost
certainly DON'T have those. This file is the generalized replacement: the
human + AI-critic adjudication loop that produced the 0.52 -> 0.92 jump.

THE LOOP (playbook §3)
----------------------
  1. BUILD BATCHES   : dump the turns that need labeling into small files.
  2. ADJUDICATE      : two independent, oppositely-biased critics label each turn
                       blind (not shown any proposed label). Keep a label ONLY
                       when both agree; disagreements -> `ambiguous` (excluded).
                       Critics may be LLMs (fast, cheap first pass) but a HUMAN
                       is the final authority, and you gate on inter-annotator
                       agreement (Cohen's kappa >= ~0.75) before scaling.
  3. APPLY           : write the agreed labels back into gold.jsonl.

CRITICAL DISCIPLINE
-------------------
- Clean the EVALUATION rows (calibration/policy/test) first — noisy eval labels
  lie to you about how good you are (playbook §3.3).
- The measurement splits must be adjudicated EXHAUSTIVELY (every turn), or your
  coverage denominator is biased. You may pre-filter TRAINING rows by a cheap
  score/keyword to save effort, but never the eval splits.
- Never let a model's own guess become its label without human sign-off — that's
  how you launder errors into ground truth.

This file stubs steps 1 and 3 (deterministic Python). Step 2 — the actual critic
runner — depends on your tooling (LLM API, annotation vendor, in-house tool), so
it's left as a clearly-marked TODO with the exact input/output contract.
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "trainer"))
from semantic_poc.cli import load_gold, save_gold          # noqa: E402
from semantic_poc.paths import GOLD_DIR                     # noqa: E402

BATCH_DIR = Path("/tmp/adjudication")
BATCH_SIZE = 40

# TODO 1: your intent names (or import from the taxonomy config).
INTENTS = [
    "replace_card", "order_checks", "transfer_money", "pay_bill",
    "check_balance", "reset_password", "schedule_appointment", "get_branch_hours",
]


# ---------------------------------------------------------------------------
# STEP 1 — build adjudication batches
# ---------------------------------------------------------------------------
def build_batches(splits=("policy", "calibration", "test", "train")) -> None:
    """Write the turns that need labeling into /tmp/adjudication/batch_*.jsonl.

    Each line: {"id", "split", "agent", "caller"} — everything a critic needs to
    label the turn blind. Eval splits are dumped EXHAUSTIVELY; you may pre-filter
    train (see TODO 2)."""
    rows = load_gold()
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    for old in BATCH_DIR.glob("batch_*.jsonl"):
        old.unlink()

    cands = []
    for split in splits:
        pool = [r for r in rows if r.split == split and not r.labels and not r.ambiguous]
        if split == "train":
            # TODO 2 (optional, TRAIN ONLY): to save labeling budget you may keep
            # only train turns likely to be positive — e.g. a keyword/regex match
            # or a cheap model score above a low bar. NEVER do this to eval splits.
            # pool = [r for r in pool if my_cheap_prefilter(r)]
            pass
        for r in pool:
            cands.append({
                "id": f"{r.conversation_id}#{r.turn_index}",
                "split": split,
                "agent": r.previous_agent_utterance,
                "caller": r.raw_transcript,
            })

    for i in range(0, len(cands), BATCH_SIZE):
        with open(BATCH_DIR / f"batch_{i // BATCH_SIZE:03d}.jsonl", "w") as f:
            for c in cands[i:i + BATCH_SIZE]:
                f.write(json.dumps(c) + "\n")
    print(f"wrote {len(cands)} candidates to {BATCH_DIR} "
          f"({(len(cands)+BATCH_SIZE-1)//BATCH_SIZE} batches)")


# ---------------------------------------------------------------------------
# STEP 2 — adjudicate (YOUR TOOLING GOES HERE)
# ---------------------------------------------------------------------------
#
# Run TWO independent, oppositely-biased critics over each batch file:
#   - Critic A (precision-biased): "when borderline, prefer 'none'."
#   - Critic B (recall-biased):    "recover the intent through ASR noise if plausibly there."
# Show each critic ONLY {agent, caller} — never a proposed label.
#
# Each critic returns, for every id, one of: an INTENTS name, "none", or "unsure".
# Merge rule (this is what makes the labels trustworthy):
#   - both critics agree on the SAME intent   -> that intent
#   - both agree "none"                        -> negative (labels=[])
#   - anything else (incl. any "unsure")       -> "ambiguous" (excluded)
#
# Write the merged result to VERDICTS_PATH as:
#   {"verdicts": [ {"id": "...", "label": "<intent>|none|ambiguous"}, ... ]}
#
# The critics can be LLMs for a fast first pass, but route disagreements AND a QA
# sample of agreements to HUMAN annotators, and compute kappa before you trust
# the guide. See scripts/adj_prep.py / adj_apply.py in this repo for the exact
# workflow we ran (an agent workflow with model:'sonnet' critics).
VERDICTS_PATH = BATCH_DIR / "verdicts.json"


# ---------------------------------------------------------------------------
# STEP 3 — apply verdicts back into gold.jsonl
# ---------------------------------------------------------------------------
def apply_verdicts() -> None:
    verdicts = {v["id"]: v["label"] for v in json.load(open(VERDICTS_PATH))["verdicts"]}
    # Back up before mutating — always keep a pre-labeling snapshot.
    import shutil
    shutil.copy(GOLD_DIR / "gold.jsonl", GOLD_DIR / "gold.pre-label.jsonl")

    rows = load_gold()
    stats = Counter()
    for r in rows:
        v = verdicts.get(f"{r.conversation_id}#{r.turn_index}")
        if v is None:
            continue
        # SAFETY: never relabel the frozen test split from an automated pass while
        # you're still iterating. Only open test for the final verdict, deliberately.
        if r.split == "test":
            stats["skipped_test"] += 1
            continue
        if r.labels or r.ambiguous:
            stats["skipped_already_set"] += 1
            continue
        if v == "ambiguous":
            r.ambiguous = True
            stats["ambiguous"] += 1
        elif v == "none":
            stats["negative"] += 1                 # labels stays []
        elif v in INTENTS:
            r.labels = [v]
            stats["positive"] += 1
        else:
            raise ValueError(f"unknown verdict label {v!r} for {r.conversation_id}")
    save_gold(rows)
    print(json.dumps(dict(stats), indent=1))
    print("applied. now run `uv run poc partition` then `uv run poc train`.")


if __name__ == "__main__":
    # Usage:
    #   python bootstrap_labels.py build      # step 1
    #   ... run your critics -> write verdicts.json ... (step 2)
    #   python bootstrap_labels.py apply      # step 3
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    {"build": build_batches, "apply": apply_verdicts}[cmd]()
