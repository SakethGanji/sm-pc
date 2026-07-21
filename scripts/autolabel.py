"""LLM auto-labeler — dual blind critics over unlabeled turns in gold.duckdb.

The accelerator from docs/method-guide.md §3, wired to the DuckDB store. For each unlabeled
turn it asks TWO Gemini critics with opposing biases (one precision-lean, one
recall-lean) what the customer is requesting, then:
  - both agree on an intent   -> promote that intent onto the turn (audited)
  - both agree on "none"       -> leave it OOS (labels stay [])
  - they disagree, or "unclear" -> mark the turn `ambiguous` (excluded from train+eval)

Guardrails (do not skip):
  * LLM labels are a strong FIRST PASS, not ground truth. Spot-check them.
  * NEVER trust an auto-labeled EVAL set. Run this BEFORE `poc partition`, then
    human-verify the positives that land in the test split.
  * Snapshot before a big run: `poc snapshot --label pre-autolabel`.

Usage (from trainer/, venv active, GEMINI_API_KEY set):
  python ../scripts/autolabel.py [--limit N] [--batch 15] [--model gemini-2.5-flash] [--dry-run]
  POC_GOLD_PATH selects the store (default data/gold/gold.duckdb).
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "trainer"))

from semantic_poc import store
from semantic_poc.config import load_config
from semantic_poc.paths import GOLD_DIR

CRITICS = {
    "precision": "Be conservative: when a turn is only topically related or borderline, "
                 "label it \"none\". Assign an intent ONLY when the customer is clearly, "
                 "actively requesting that action.",
    "recall": "Be generous in recovering intent: if the customer is plausibly requesting "
              "an action even through ASR noise, filler, or indirect phrasing, assign that "
              "intent. Still use \"none\" for genuine non-requests (greetings, yes/no, "
              "giving their name, answering the agent).",
}


def _taxonomy_lines(tax: dict) -> str:
    lines = []
    for name, spec in tax["intents"].items():
        spec = spec or {}
        desc = spec.get("desc") or spec.get("description") or ""
        fam = spec.get("family", "")
        lines.append(f"- {name}" + (f" [{fam}]" if fam else "") + (f" — {desc}" if desc else ""))
    return "\n".join(lines)


def _prompt(tax_lines: str, bias: str, turns: list[dict]) -> str:
    payload = [{"i": i, "agent": t["agent"], "customer": t["customer"]}
               for i, t in enumerate(turns)]
    return (
        "You label customer turns from call-center transcripts. For each turn decide "
        "what the CUSTOMER is requesting. Choose exactly one label per turn:\n"
        "  - one of the intent names below, if the customer is clearly making that request\n"
        "  - \"none\" if not a supported request (greeting, name, yes/no, answering the "
        "agent, chit-chat, out-of-scope)\n"
        "  - \"unclear\" if you genuinely cannot tell\n\n"
        f"Supported intents:\n{tax_lines}\n\n"
        f"{bias}\n\n"
        "The \"agent\" field is the agent's turn just before (context). "
        "Return ONLY a JSON array, one object per turn: "
        "[{\"i\": <index>, \"label\": \"<intent|none|unclear>\"}].\n\n"
        f"Turns:\n{json.dumps(payload, ensure_ascii=False)}"
    )


def _critic(client, types, model, prompt) -> list[dict]:
    resp = client.models.generate_content(
        model=model, contents=prompt,
        config=types.GenerateContentConfig(temperature=0.0,
                                           response_mime_type="application/json"))
    return json.loads(resp.text)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="cap turns (0 = all unlabeled)")
    ap.add_argument("--batch", type=int, default=15, help="turns per critic call")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--skip-split", default="",
                    help="comma-sep splits to leave untouched, e.g. 'test' or 'test,policy'. "
                         "Run `poc partition` FIRST, then the LLM never labels your eval set "
                         "(you human-label those). Empty = label everything (verify eval after).")
    ap.add_argument("--dry-run", action="store_true", help="report, don't write")
    args = ap.parse_args(argv)
    skip = {s.strip() for s in args.skip_split.split(",") if s.strip()}

    gold = Path(os.environ.get("POC_GOLD_PATH", GOLD_DIR / "gold.duckdb"))
    if gold.suffix != ".duckdb":
        sys.exit("autolabel needs a .duckdb store (set POC_GOLD_PATH)")
    valid = set(load_config("taxonomy")["intents"])
    tax_lines = _taxonomy_lines(load_config("taxonomy"))

    # Unlabeled, non-ambiguous turns. With --skip-split (after `poc partition`),
    # the named splits are left for humans — the LLM never touches your eval answer key.
    todo = [(f"{r.conversation_id}#{r.turn_index}",
             {"agent": r.previous_agent_utterance, "customer": r.raw_transcript})
            for r in store.iter_gold(gold)
            if not r.labels and not r.ambiguous and r.split not in skip]
    if skip:
        print(f"skipping split(s) {sorted(skip)} — label those by hand (eval integrity)")
    if args.limit:
        todo = todo[:args.limit]
    if not todo:
        print("nothing to label (no unlabeled, non-ambiguous turns)")
        return 0
    print(f"labeling {len(todo)} turns via dual {args.model} critics "
          f"(batch {args.batch}){' [DRY RUN]' if args.dry_run else ''}")

    from google import genai
    from google.genai import types
    client = genai.Client()

    promote = defaultdict(list)   # intent -> [ids]
    ambiguous, none_ct, agree_ct, fail = [], 0, 0, 0
    for start in range(0, len(todo), args.batch):
        batch = todo[start:start + args.batch]
        ids = [rid for rid, _ in batch]
        turns = [t for _, t in batch]
        try:
            r1 = {d["i"]: d["label"] for d in _critic(client, types, args.model,
                                                       _prompt(tax_lines, CRITICS["precision"], turns))}
            r2 = {d["i"]: d["label"] for d in _critic(client, types, args.model,
                                                       _prompt(tax_lines, CRITICS["recall"], turns))}
        except Exception as e:  # noqa: BLE001 — a bad batch shouldn't kill the run
            print(f"  batch @{start}: critic error ({e}); skipped")
            fail += len(batch)
            continue
        for i, rid in enumerate(ids):
            l1, l2 = r1.get(i), r2.get(i)
            if l1 == l2 and l1 in valid:
                promote[l1].append(rid); agree_ct += 1
            elif l1 == l2 == "none":
                none_ct += 1; agree_ct += 1
            else:
                ambiguous.append(rid)   # disagreement, "unclear", or an off-menu label
        print(f"  {min(start + args.batch, len(todo))}/{len(todo)} labeled")

    total = len(todo) - fail
    print(f"\nagreement: {agree_ct}/{total} ({agree_ct / total:.0%})" if total else "")
    print(f"positives: {sum(len(v) for v in promote.values())}  "
          f"none/OOS: {none_ct}  ambiguous: {len(ambiguous)}  failed: {fail}")
    for intent, rids in sorted(promote.items(), key=lambda kv: -len(kv[1])):
        print(f"  {intent:22s} {len(rids)}")

    if args.dry_run:
        print("\n[dry run] nothing written. Drop --dry-run to apply.")
        return 0
    for intent, rids in promote.items():
        store.promote(gold, intent, rids, note=f"autolabel ({args.model})")
    if ambiguous:
        store.mark_ambiguous(gold, ambiguous, note=f"autolabel disagreement ({args.model})")
    print(f"\nwritten to {gold}. Next: poc counts → poc partition → HUMAN-VERIFY the "
          f"test-split positives → poc train.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
