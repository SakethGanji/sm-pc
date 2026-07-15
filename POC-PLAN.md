# POC plan — cards + transactions (two main intents)

Concrete, weekend-scoped plan for the first office POC, tied to the code already
in this repo. Goal: prove the mechanism works on real data and get a certified
per-sub-intent coverage number at a precision floor for two full mains.

## Scope

- **Two main intents: `cards` and `transactions`**, with their ~7–10 sub-intents each.
- Prove the pipeline end-to-end and **certify the sub-intents that have enough data**;
  the rare tail shows as "needs more data" (expected, not a failure).
- This is a *mechanism* proof, not a launch. The number won't be our 0.92 — it's
  yours to discover, per sub-intent.

## Data targets

| | target | notes |
|---|---|---|
| Positives per sub-intent | **~150** | from **many different conversations**, not a few calls repeated |
| Negatives | **one shared pile, ~10k is great** | diverse + hard negatives; NOT random; serves every sub-intent |
| Total | **~5,000 labeled turns** | (150 × ~16 subs) + shared negative pile |
| Eval ratio | **natural** | keep the real positive:negative ratio in calibration/policy/test |

Positive-count reality check: what matters is **distinct conversations per sub**,
not turn count. ~150 positives from ~150 calls = good; from 10 calls = thin.

## The one rule that makes or breaks it

**Never label negatives by blind-dumping.** Buried real requests in the negative
pile are exactly what faked the rehearsal down to 0.52. Route every candidate
through the dual-critic pass; only both-agree-"none" becomes a negative. Verify
**every eval positive** (adjudicate eval splits exhaustively) — a single
positive mislabeled negative there corrupts your precision number.

## Steps (each maps to a file already in the repo)

0. **Prove the machine** — `cd trainer && uv run python ../port-template/smoke_test.py`.
   Green means the pipeline shape works before you touch real data.

1. **Ingest adapter** — copy `port-template/ingest.py` → `trainer/semantic_poc/ingest.py`,
   fill the 5 TODOs to read your prod transcripts, wire `cli.py`'s `ingest()` to
   your path. `uv run poc ingest`; eyeball `data/gold/gold.jsonl` (customer turns
   only, raw text uncleaned, negatives present, labels empty).

2. **Pull candidates from prod** — use your **existing prod classifications** to
   find candidate cards/transactions turns (fast head start) + a broad sample of
   other turns for negatives. These are *candidates*, not final labels.

3. **Critic-label** — `port-template/bootstrap_labels.py` build → run your two
   blind critics (implement step 2 with your LLM API; batching mechanics: ~25
   rows/call, structured JSON, one verdict per id) → apply. This step
   simultaneously **cleans negatives and surfaces the buried positives**. Verify
   eval splits with a human; gate on agreement.

4. **Configs** — `port-template/taxonomy.yaml` → `configs/taxonomy.yaml` (cards +
   transactions subs, each with its `family`). `port-template/policy.yaml` →
   `configs/policy.yaml` (precision_floor, keep accept_floor 0.50,
   exclusive_groups for the confusable subs).

5. **Train + serve + verify** — `uv run poc partition` → `uv run poc train` →
   `cd server && go test ./...` (parity) → `go run ./cmd/serve -artifact <newest>`.

6. **Read the result, honestly** — run `scripts/held_out_eval.py` for held-out
   per-sub coverage at the floor, and `scripts/ceilings.py` + `scripts/fp_audit.py`
   to see whether any gap is a label / audio / model problem. Report **per
   sub-intent**, at the precommitted floor, on the held-out split.

## Code status — what's ready, what you build, what's optional

**Ready as-is (no changes):**
- Whole training/calibration/threshold/serving pipeline (dataset-agnostic).
- `accept_floor` guard (already committed) — no near-zero threshold surprises.
- Class imbalance is handled — `class_weight="balanced"` copes with 10k neg : 150 pos.
- Conversation-grouped, time-ordered splits + frozen-test hash — handles your
  "many turns per conversation" data correctly (no leakage).
- All diagnostics (`scripts/ceilings.py`, `fp_audit.py`, `held_out_eval.py`).

**You build (expected fill-in-the-blanks, not changes to existing code):**
- The ingest adapter (`ingest.py`) for your data.
- The critic runner (`bootstrap_labels.py` step 2) with your LLM API.
- The two config files.

**Optional — NOT needed for this POC:**
- The main→sub **fallback** in the decision policy (emit "cards, unspecified"
  when no sub clears but the main is confident). For the POC, the flat sub-heads
  + `exclusive_groups` already handle the confusable-sibling case, so skip it.
  When you want it later, it's a small addition to `trainer/semantic_poc/policy.py`
  (aggregate a family confidence; if no sub accepted but family ≥ floor, emit the
  family). Say the word and it's a quick add.

**Bottom line: the committed code is good for this POC as-is.** Your weekend is
the three fill-ins above (ingest, critic runner, configs) + the labeling — not
changes to the pipeline.
