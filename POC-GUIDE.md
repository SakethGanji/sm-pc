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
| `labels` | leave blank | filled by the critic pass (intent name, or empty = negative) |
| `session_task_intent` | optional | your prod tag can seed it (weak hint only) |
| `split` | **don't add** | the pipeline assigns train/cal/policy/test |

Minimum = the 5 **Yes** columns + blank `labels`.

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

You build **one** `gold.jsonl`. `poc partition` splits it into
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
cd trainer && uv run python ../port-template/smoke_test.py     # prove the machine
# rewrite ingest.py for your data; fill taxonomy.yaml + policy.yaml
uv run poc ingest        # your data -> gold.jsonl (labels blank)
# ... critic labeling pass fills labels ...
uv run poc partition     # auto-split, freeze test
uv run poc train         # embed(cached) -> heads -> calibrate -> threshold -> artifact
uv run python ../scripts/serve_py.py --port 8081               # SERVE (Python-only, no Go)
uv run python ../scripts/held_out_eval.py <artifact>           # per-intent result
```

## Locked design decisions (from planning)

- **Serve from Python for the POC** (`scripts/serve_py.py`) — no Go needed. Same
  `/classify` contract, identical results. Bring in the Go server later for
  production speed against the *same* artifact (no retrain). Python is fine here:
  the work is I/O-bound (the embedding call), so async/batching scale it; Go is
  for high-concurrency, tight-tail-latency production.
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

- **No changes:** train / calibrate / threshold / serve (Python), accept-floor
  guard, imbalance handling, conversation-grouped splits, all diagnostics.
- **You build (fill-ins):** the `ingest.py` adapter, the critic runner
  (`port-template/bootstrap_labels.py` step 2), the two config files.
- **Optional, not for the POC:** main→sub fallback, the Go server, the LLM
  rewrite path, the token-window widening (add these as your data demands).

## Your first week

- **Day 1** — smoke test → prove the machine. Rewrite `ingest.py`; `poc ingest`;
  eyeball `gold.jsonl`.
- **Day 2–3** — hand-label ~200 turns across your top intents + negatives (your
  seed and your feel for the guide).
- **Day 3** — fill configs; `poc partition` + `poc train`; a thin first artifact.
- **Day 4** — run diagnostics (`ceilings.py`, `fp_audit.py`, `held_out_eval.py`);
  read whether gaps are label / audio / model.
- **Day 5** — stand up the critic loop (`bootstrap_labels.py`) and scale labeling.

## Done looks like

A per-sub-intent coverage number at your precision floor, on the held-out set,
for the intents with enough data — plus a diagnostic read of whether any gap is
a label / audio / model problem. That's the POC: proof of the mechanism + the
evidence for how much data the full build needs.
