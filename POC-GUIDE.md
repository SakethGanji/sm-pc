# POC guide — build it on your data in a weekend

The single build guide. Deeper reasoning lives in `docs/office-playbook.md`;
this is the do-it version.

## Reality check (read once)

The **machine is turnkey; the labeling is the real work.** You can have the
pipeline running on your data in ~a day of engineering — rewrite one file, edit
two configs, run. But a *good number* comes from label quality, which is a
labeling project (weeks for the full taxonomy), not a code project. That's by
design: fixing labels alone moved the rehearsal 0.52 → 0.92 with zero model
change. You will **not** reproduce 0.92 — you get your own number, lower and
messier, per intent. The transferable thing is the method, not the figure.

## Scope

Start small: **2 intents** (ideally a confusable pair, to test the hard
boundary), or **2 full mains** (`cards` + `transactions`, ~16 subs). Prove the
mechanism and certify what has enough data; the rare tail shows as "needs more
data" — expected, not failure.

## How much data

Unit that matters: **positives per sub-intent, from many different conversations.**

| | per sub-intent | notes |
|---|---|---|
| Solid | **~150** | ~15 land in test — usable, wide error bars |
| Tighter | ~300 | ~30 in test → better precision estimate |
| Floor | ~50 | shows the mechanism only |

- **Negatives: ONE shared pile, not per-intent. 10k is great.** The same pile is
  the negative for every intent. Make it **diverse + hard** (include OTHER
  intents, not just random filler) — NOT random.
- **Totals:** 2 subs ≈ 300 positives + your pile; 2 mains ≈ ~5,000 turns. Mostly
  negatives — that's realistic.
- Diversity = **conversations, not turns**: ~150 positives from ~150 calls is
  strong; from 10 calls is thin.

## The columns (one row per customer turn)

| Column | Required? | What to put |
|---|---|---|
| `conversation_id` | **Yes** | call ID; groups turns of one call |
| `turn_index` | **Yes** | position in the call (0,1,2…) |
| `timestamp_ms` | **Yes** | turn start in ms (real time, or use turn order) |
| `raw_transcript` | **Yes** | customer ASR text — **never cleaned** |
| `previous_agent_utterance` | **Yes** | agent's line just before; `""` if none |
| `human_transcript` | recommended | corrected text; copy raw if you don't have it |
| `labels` | leave blank | filled by your labeling pass (`poc promote`); intent name, or empty = negative |
| `session_task_intent` | optional | your prod tag can seed it (weak hint only) |
| `split` | **don't add** | the pipeline assigns train/cal/policy/test |

Minimum = the 5 **Yes** columns + blank `labels`. Full schema (every column,
types, the `prev_caller_segment` struct, how to add/label rows in DuckDB):
**`docs/gold-schema.md`**.

## Labeling (don't hand-label 5k)

1. **Find candidates fast** with your existing prod classifications (pull likely
   cards/transactions turns + a broad "other" sample). Candidates, not labels.
2. **Two blind AI critics** (one strict, one lenient) label every turn; keep only
   where **both agree**; disagreements → `ambiguous`, excluded. This cleans
   negatives AND surfaces buried positives. ~800 calls handles 10k turns.
3. **Humans verify every EVAL positive** (adjudicate eval splits exhaustively;
   training can be lighter). Never blind-dump negatives — that's what fakes the
   number down.

If your data is already main-classified: keep the main label (it becomes the
family AND scopes the sub-labeling to ~7–10 options), let critics still flag
"wrong main / none," and verify the eval set.

## One dataset → four splits (automatic)

You build **one** store (`gold.duckdb`). `poc partition` splits it into
train / calibration / policy / test — by whole conversation, time-ordered, test
frozen + hashed. Don't make four files; don't balance it (training auto-handles
imbalance; eval keeps the natural ratio).

## The 3 rules that make or break it

1. **Never blind-dump negatives** — route every turn through the critics.
2. **Clean EVAL labels first, by hand** — one mislabeled positive there corrupts
   your precision.
3. **Diversity = conversations + hard negatives** — not turns, not random filler.

## Run it

```
cd trainer && source .venv/bin/activate    # see README.md Setup if this venv doesn't exist yet
export POC_GOLD_PATH=../data/gold/gold.duckdb   # canonical store (docs/gold-schema.md)
bash ../scripts/e2e_once.sh               # prove the machine end-to-end
# rewrite ingest.py for your CSV; fill taxonomy.yaml + policy.yaml
poc ingest        # your data -> gold.duckdb (labels blank)
poc counts        # check intent balance / OOS pile
poc promote <intent> ids.txt   # label a batch (audited); repeat per intent
poc partition     # auto-split, freeze test
poc train         # embed(cached) -> heads -> calibrate -> threshold -> artifact
python ../scripts/serve_py.py --port 8081                        # SERVE (Python-only, no Go)
python ../scripts/held_out_eval.py                              # per-intent result
python ../scripts/stability_report.py --label "added <intent>"  # regression gate as you add intents
poc snapshot --label "<intent> in"                             # Parquet restore point
```

## Locked design decisions (from planning)

- **Serve from Python** (`scripts/serve_py.py`) — implements the full `/classify`
  contract (stitching, decision policy, latency breakdown). A Go server is part
  of the original design-doc target architecture for production but has been
  removed from this repo for now; nothing here depends on it. Python is fine for
  the POC: the work is I/O-bound (the embedding call), so async/batching scale it.
- **Context = token-budgeted backward window** (robust to ASR splitting), not a
  fixed turn count. Start ~96–128 tokens of preceding context, speaker-tagged;
  sweep to find the smallest W that captures the coverage. One function change
  (`context.py`), rest of pipeline unchanged.
- **LLM only on the abstain path, as a "condense to a clear ask" step** — not a
  classifier. Cheap classifier handles the confident bulk; on abstain, one
  *generic* prompt condenses the (history-aware) context into a clean request,
  which re-runs through the classifier. Keeps certification, one generic prompt
  (not per-intent), covers multi-intent + rambling + history. Measure rewrite
  fidelity + precision on this path before trusting it.
- **Hierarchy (main→sub) is a decision-policy fallback, not a per-family model**
  — optional, NOT needed for the POC.

## Code status: ready as-is

- **No changes:** train / calibrate / threshold / serve (Python), stitching,
  accept-floor guard, imbalance handling, conversation-grouped splits, all
  diagnostics.
- **You build (fill-ins):** the `ingest.py` CSV adapter, your labeling pass
  (`poc promote` into the store; the automated dual-critic runner is recoverable
  from git history if you want it at scale), the two config files.
- **Optional, not for the POC:** main→sub fallback, a Go server for production
  latency (same artifact, no retrain — not part of this repo right now), the
  LLM rewrite path, the token-window widening (add these as your data demands).

## Your first week

- **Day 1** — `bash scripts/e2e_once.sh` → prove the machine. Rewrite `ingest.py`;
  `poc ingest`; eyeball the store (`poc counts`, or query `gold.duckdb`).
- **Day 2–3** — label ~200 turns across your top intents + negatives (`poc promote`);
  your seed and your feel for the guide.
- **Day 3** — fill configs; `poc partition` + `poc train`; a thin first artifact.
- **Day 4** — `held_out_eval.py` for the per-intent number; read whether gaps are
  label / audio / model.
- **Day 5** — scale labeling one intent at a time; `stability_report.py` after each
  add to catch regressions.

## Done looks like

A per-sub-intent coverage number at your precision floor, on the held-out set,
for the intents with enough data — plus a diagnostic read of whether any gap is
a label / audio / model problem. That's the POC: proof of the mechanism + the
evidence for how much data the full build needs.
