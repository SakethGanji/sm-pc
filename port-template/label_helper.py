"""
=============================================================================
LABEL HELPER — interactive, chatbot-assisted labeling (no API calls).
=============================================================================

WHAT THIS IS FOR
-----------------
You have a raw call export (one row per CALL, with an existing main-intent tag
and the full conversation text) and you want to label sub-intents WITHOUT
paying for LLM API calls. You have an internal chatbot you can paste text into
by hand. This tool drives that workflow:

  1. Filters your export to calls tagged with a target MAIN intent (the
     "stratify" step — pulls only calls likely to contain what you need).
  2. Shows you one call at a time.
  3. You paste the call into your chatbot, ask it to find the request turn(s)
     and sub-intent(s) (see PROMPT_TEMPLATE below), and paste the answer back.
  4. It records a clean, GoldRow-shaped labeled row for every positive AND lets
     you harvest a couple of negative turns from the same call for free.
  5. It tracks a live running count per sub-intent and tells you when you've
     hit your target (default 150 each) so you know when to stop.
  6. It's resumable — Ctrl+C anytime, state is saved, rerun to continue.

WHY "VERIFIED" MATTERS
-----------------------
Every row asks: did you personally confirm this against the transcript, or
are you just trusting the chatbot's answer? This gets baked into the row's
`dialog_acts` field so it's never lost. Since you CAN'T control which
conversation lands in your eval split later (that's assigned automatically by
time + conversation), the safe habit is: verify generously, every time — it's
one extra glance and it's free. Playbook rule: eval labels must be clean;
train tolerates a little noise. This tool defaults to asking every time so
verification becomes a habit, not an afterthought.

OUTPUT
------
Writes GoldRow-shaped JSON lines to `labeled_pool.jsonl` (default). These are
valid gold rows already — see merge_labeled_pool.py to append them into your
real data/gold/gold.jsonl.

USAGE
-----
    python label_helper.py --main cards --input calls_export.csv

Resume anytime — it skips calls already processed (tracked in
label_progress_<main>.json next to your pool file).
"""

import argparse
import csv
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# TODO 1: your export's column names. Adjust to match your file.
# ---------------------------------------------------------------------------
CALL_ID_COL = "call_id"
MAIN_INTENT_COL = "main_intent"      # your EXISTING classification column
TEXT_COL = "conversation"            # the full call transcript, one big string

# ---------------------------------------------------------------------------
# TODO 2: your sub-intents per main intent, and your target count each.
# Only the mains you're actively labeling need to be here.
# ---------------------------------------------------------------------------
SUB_INTENTS_BY_MAIN = {
    "cards": ["replace_card", "unlock_card", "activate_card", "dispute_charge"],
    "transactions": ["transfer_money", "pay_bill"],
    # ... add your other mains as you get to them ...
}
TARGET_PER_SUBINTENT = 150

PROMPT_TEMPLATE = """You are labeling a bank customer-service call transcript.
This call was tagged as "{main_intent}". Read it and identify EVERY turn where
the customer makes an actual request (not just mentions the topic). For each
request turn, tell me: the exact turn text, and which specific sub-intent it
is, from this list: {sub_intents}.
If there is no actual request in this call, say "no request found".

Transcript:
{transcript}
"""


def load_calls(path: Path):
    """TODO 3: adapt if your export isn't a simple CSV with the columns above."""
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            yield row[CALL_ID_COL], row[MAIN_INTENT_COL], row[TEXT_COL]


# ---------------------------------------------------------------------------
# State (resumable) + output writer
# ---------------------------------------------------------------------------
class State:
    def __init__(self, path: Path):
        self.path = path
        if path.exists():
            self.data = json.loads(path.read_text())
        else:
            self.data = {"processed_call_ids": [], "counts": {}, "negatives": 0}

    def save(self):
        self.path.write_text(json.dumps(self.data, indent=1))

    def is_done(self, call_id: str) -> bool:
        return call_id in self.data["processed_call_ids"]

    def mark_done(self, call_id: str):
        self.data["processed_call_ids"].append(call_id)

    def bump(self, sub_intent: str):
        self.data["counts"][sub_intent] = self.data["counts"].get(sub_intent, 0) + 1

    def count(self, sub_intent: str) -> int:
        return self.data["counts"].get(sub_intent, 0)


def write_row(pool_file, call_id: str, turn_idx: int, raw: str, prev_agent: str,
              labels: list, verified: bool, sub_intent_tag: str) -> None:
    """One GoldRow-shaped JSON line — directly mergeable into gold.jsonl."""
    provenance = "verified" if verified else "chatbot_only"
    row = {
        "conversation_id": call_id,
        # Offset avoids colliding with real turn indices if this call is ever
        # fully ingested later via ingest.py.
        "turn_index": 1000 + turn_idx,
        "timestamp_ms": (1000 + turn_idx) * 10_000,
        "raw_transcript": raw.strip().lower(),
        "human_transcript": raw.strip().lower(),
        "previous_agent_utterance": (prev_agent or "").strip().lower(),
        "prev_caller_segment": None,
        "dialog_acts": [f"manual:{provenance}"],
        "session_task_intent": sub_intent_tag,
        "labels": labels,
        "ambiguous": False,
        "split": "",
    }
    pool_file.write(json.dumps(row) + "\n")
    pool_file.flush()


def ask(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except EOFError:
        return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", required=True, help="main intent to stratify on, e.g. cards")
    ap.add_argument("--input", required=True, type=Path, help="your call export (CSV)")
    ap.add_argument("--pool", type=Path, default=Path("labeled_pool.jsonl"))
    args = ap.parse_args()

    if args.main not in SUB_INTENTS_BY_MAIN:
        sys.exit(f"'{args.main}' not in SUB_INTENTS_BY_MAIN — add it (TODO 2) first.")
    sub_intents = SUB_INTENTS_BY_MAIN[args.main]

    state = State(args.pool.with_suffix("").with_name(f"label_progress_{args.main}.json"))
    pool_file = open(args.pool, "a")

    def print_tally():
        print("\n--- progress ---")
        for si in sub_intents:
            n = state.count(si)
            bar = "#" * min(20, n * 20 // TARGET_PER_SUBINTENT)
            flag = " <- TARGET HIT" if n >= TARGET_PER_SUBINTENT else ""
            print(f"  {si:22s} {n:4d}/{TARGET_PER_SUBINTENT} [{bar:20s}]{flag}")
        print(f"  {'negatives':22s} {state.data['negatives']:4d} (no target, more is fine)")
        print("----------------\n")

    print_tally()

    try:
        for call_id, main_intent_tag, transcript in load_calls(args.input):
            if main_intent_tag != args.main:
                continue
            if state.is_done(call_id):
                continue
            if all(state.count(si) >= TARGET_PER_SUBINTENT for si in sub_intents):
                print(f"All sub-intents under '{args.main}' hit target. Stopping.")
                break

            print("=" * 78)
            print(f"CALL {call_id}  (tagged: {main_intent_tag})")
            print("=" * 78)
            print(transcript)
            print("-" * 78)
            print("Paste this into your chatbot with the prompt below, then enter results.\n")
            print(PROMPT_TEMPLATE.format(main_intent=main_intent_tag,
                                        sub_intents=", ".join(sub_intents),
                                        transcript="<paste transcript above>"))
            print("-" * 78)

            turn_idx = 0
            n_pos = 0
            while True:
                raw = ask("Request turn text (blank = no more requests in this call): ")
                if not raw:
                    break
                prev_agent = ask("  Previous agent line (optional, Enter to skip): ")
                print(f"  Sub-intents: {', '.join(sub_intents)}")
                sub = ask("  Sub-intent: ")
                if sub not in sub_intents:
                    print(f"  '{sub}' not in the list for {args.main} — skipping this turn.")
                    continue
                verified = ask("  Did you personally confirm this against the transcript? [y/n]: ").lower() == "y"
                write_row(pool_file, call_id, turn_idx, raw, prev_agent, [sub], verified, sub)
                state.bump(sub)
                turn_idx += 1
                n_pos += 1
                print(f"  -> recorded as {sub} ({'verified' if verified else 'chatbot-only'}). "
                      f"{sub} now at {state.count(sub)}/{TARGET_PER_SUBINTENT}.\n")

            if n_pos == 0:
                print("No request found in this call.")

            while True:
                neg = ask("Paste a clearly NON-request turn to bank as a negative (blank to stop): ")
                if not neg:
                    break
                write_row(pool_file, call_id, turn_idx, neg, "", [], True, "")
                state.data["negatives"] += 1
                turn_idx += 1

            state.mark_done(call_id)
            state.save()
            print_tally()

    except KeyboardInterrupt:
        print("\nInterrupted — progress saved. Rerun the same command to resume.")
    finally:
        pool_file.close()
        state.save()

    print(f"\nDone (or paused). Labeled rows so far -> {args.pool}")
    print("Next: python merge_labeled_pool.py  (appends into data/gold/gold.jsonl)")


if __name__ == "__main__":
    main()
