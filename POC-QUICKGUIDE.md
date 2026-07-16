# POC quick guide — build it in a weekend

Everything you need to build the first POC on your own data, in one page.

## 1. What you're building

Two intents working end-to-end: prove the pipeline classifies real customer
turns at a precision floor you commit to, and get a per-intent coverage number.
Pick either scope:
- **Weekend version:** 2 sub-intents (ideally two in the *same* main, e.g.
  `replace_card` + `unlock_card` — tests the hard confusable boundary).
- **Bigger version:** 2 full mains (`cards` + `transactions`, ~16 subs).

## 2. How much data

The unit that matters is **positives per sub-intent**, from **many different
conversations** (not a few calls repeated).

| | per sub-intent | notes |
|---|---|---|
| Solid | **~150 positives** | ~90 land in train, ~15 in test — usable, wide error bars |
| Tighter certification | ~300 positives | ~30 in test → better precision estimate |
| Bare minimum | ~50 positives | shows the mechanism only |

- **Negatives: ONE shared pile, not per-intent.** The same pile is the negative
  for every intent. A big pile is good — **10k is fine.**
- **Totals:** 2 subs ≈ 300 positives + your negative pile. 2 mains ≈ 2,400
  positives + pile ≈ ~5,000 turns. The total is mostly negatives — that's realistic.

## 3. The columns (one row per customer turn)

| Column | Required? | What to put |
|---|---|---|
| `conversation_id` | **Yes** | call ID; groups turns of one call |
| `turn_index` | **Yes** | position in the call (0,1,2…) |
| `timestamp_ms` | **Yes** | turn start in ms (real time, or use turn order) |
| `raw_transcript` | **Yes** | customer ASR text — **never cleaned** |
| `previous_agent_utterance` | **Yes** | agent's line just before; `""` if none |
| `human_transcript` | recommended | corrected text; copy raw if you don't have it |
| `labels` | leave blank | filled by the critic pass (intent name, or empty = negative) |
| `session_task_intent` | optional | your prod tag can seed it (weak hint only) |
| `dialog_acts` | optional | any tags, or empty |
| `split` | **don't add** | the pipeline assigns train/cal/policy/test |

Minimum to start = the 5 **Yes** columns + blank `labels`.

## 4. How to get the labels (don't hand-label 5k)

1. **Find candidates fast** using your existing prod classifications — pull
   likely cards/transactions turns + a broad sample of everything else.
2. **Run everything through two blind AI critics** (opposite biases: one strict,
   one lenient). Keep a label only when **both agree**; disagreements →
   `ambiguous`, excluded. This cleans your negatives AND surfaces the positives
   buried in them.
3. **Humans verify every EVAL positive** (adjudicate the eval splits exhaustively;
   training can be lighter). ~800 LLM calls handles 10k turns — cheap.

## 5. One dataset in → four splits out (automatic)

You build **one** file. `uv run poc partition` splits it into
train / calibration / policy / test — **by whole conversation**, **time-ordered**
(old→train, new→test), test frozen + hashed. You do NOT make four files, and you
do NOT balance it — training auto-handles imbalance; eval keeps the natural ratio.

## 6. The 3 rules that make or break it

1. **Never blind-dump negatives.** Buried real requests mislabeled as negatives
   are what fake your number down. Route every turn through the critics; only
   both-agree-"none" becomes a negative.
2. **Clean your EVAL labels first, by hand.** A single positive mislabeled
   negative in the eval set corrupts your precision. The eval set is small on
   purpose — verify it exhaustively.
3. **Diversity = conversations, not turns.** ~150 positives from ~150 calls is
   strong; from 10 calls is thin. Make negatives diverse and include OTHER
   intents (hard negatives), not just random filler.

## 7. Run it

```
cd trainer && uv run python ../port-template/smoke_test.py   # prove the machine
# rewrite ingest.py for your data; fill taxonomy.yaml + policy.yaml
uv run poc ingest        # your data -> gold.jsonl (labels blank)
# ... critic labeling pass fills labels ...
uv run poc partition     # auto-split, freeze test
uv run poc train         # embed(cached) -> heads -> calibrate -> threshold -> artifact
cd ../server && go test ./...                 # parity check
go run ./cmd/serve -artifact ../artifacts/<newest>
cd ../trainer && uv run python ../scripts/held_out_eval.py <artifact>   # per-intent result
```

## 8. Does the code need changing? No — it's ready.

- **Ready as-is:** train / calibrate / threshold / serve, the accept-floor guard,
  class-imbalance handling, conversation-grouped splits, all diagnostics.
- **You build (fill-ins, not edits):** the `ingest.py` adapter for your data, the
  critic runner (`bootstrap_labels.py` step 2) with your LLM API, and the two
  config files.
- **Optional, NOT needed for the POC:** the main→sub fallback in the decision
  policy. Flat sub-heads + `exclusive_groups` already handle confusable siblings.

## 9. What "done" looks like

A per-sub-intent coverage number at your precision floor, on the held-out set,
for the intents that had enough data — and a clear read (from the diagnostics) of
whether any gap is a label / audio / model problem. That's the POC: proof of the
mechanism + the evidence for how much data the full build needs. The number is
yours — it won't match the rehearsal's 0.92, and it shouldn't.
