# Agent brief — porting this pipeline to our corpus

**This file is your only context. Read it fully before touching anything.**

`docs/method-guide.md` (method and lessons) and `docs/model-design.md` (how the
model works) were written during a finished rehearsal on a public sample corpus.
They are useful background but describe a data path that is being **replaced**.
Where they conflict with this file, this file wins.

---

## 1. What this system is

A non-generative intent classifier for customer-service phone turns. It reads one
customer utterance (plus the agent's preceding turn as context) and either names
the intent, suggests nothing, or abstains.

It exists to replace a per-utterance generative LLM call at high volume — so it
must be cheap, low-latency, deterministic, and auditable. A confident wrong
answer is worse than silence.

**Architecture.** A frozen, rented embedding does the language understanding; the
only trained parts are small linear models on top:

```
customer turn
  → stitch a dangling previous fragment (deterministic, config rules)
  → build context string:  [AGENT] <prev agent turn>\n[CUSTOMER] <raw text>
  → embed once via Gemini (frozen, cached on disk)   ← the only network call
  → semantic head:  per-intent logistic regression over the 768-dim embedding
  → lexical head:   per-intent logistic regression over TF-IDF of the raw text
  → fusion:         per-intent LR over [all semantic logits ‖ all lexical logits ‖ meta]
  → calibration:    per-intent Platt scaling (score → real probability)
  → thresholds:     per-intent cut certified to hold the precision floor
  → decision policy: floors, exclusive groups, abstain
```

Heads are **one-vs-rest**: each intent is an independent binary classifier, so
each gets its own regularization, calibration, and threshold. Everything trains
on conversation-grouped folds; fusion trains on out-of-fold logits. One embedding
call per turn regardless of intent count.

Output is exactly one of: `accepted`, `multi_accepted`, `abstained`,
`no_supported_intent`. (Infrastructure failures return `degraded` — never a
semantic decision.)

**The metric.** We do not optimize F1. Precision is fixed at a floor committed in
advance (`precision_floor`, currently 0.90) and coverage is maximized under it.
The headline is "X% coverage at ≥90% precision, held-out, **per intent**."

---

## 2. The one architectural idea

Everything downstream of ingest speaks exactly one type: **`GoldRow`**
(`trainer/semantic_poc/schema.py`). Partition, labeling, training, calibration,
thresholds, serving, and diagnostics never learn where the data came from.

**`trainer/semantic_poc/ingest.py` is the only corpus-aware file in the
pipeline.** Get it right and the rest works unchanged. Any change that leaks
corpus-specific assumptions past that boundary is a bug.

---

## 3. Your task: the ingest port

A sample CSV of our corpus is provided. Map it onto `GoldRow`.

| File | Change | Size |
|---|---|---|
| `trainer/semantic_poc/ingest.py` | **Rewrite the body.** Read our CSV → one `GoldRow` per CUSTOMER turn, `labels=[]` | the real work |
| `trainer/semantic_poc/paths.py` | Repoint the raw-corpus constant (currently `RAW_HVB`) at our CSV under `data/raw/`. Rename it `RAW_CORPUS` and update its importers | 1 line + refs |
| `trainer/semantic_poc/cli.py` | Wire the `ingest()` command to your new function signature | ~1 line |
| `configs/taxonomy.yaml` | Our real intents + family roll-up. Add a `desc:` line per intent — the auto-labeler reads these | a list |
| `trainer/semantic_poc/schema.py` | Give `dialog_acts` and `session_task_intent` defaults (`[]` and `""`) so rows validate without them | 2 lines |
| `trainer/semantic_poc/weak_labels.py` | Drop the call from ingest; the file can go. It derives labels from per-turn dialog-act metadata our corpus does not have | delete |

Nothing else should need to change. **If a task seems to require editing
`train.py`, `policy.py`, `runtime.py`, `features.py`, or `store.py`, stop and
ask** — that means the ingest boundary is being violated.

### The ingest contract — five rules, none optional

1. **One row per CUSTOMER turn.** Agent turns are *not* rows. An agent turn's
   text rides along on the *next* customer turn as `previous_agent_utterance`.
2. **Never clean `raw_transcript`.** No normalizing, punctuating, spell-fixing,
   or casing changes. We serve on raw ASR output, so we train and evaluate on raw
   ASR output. Corrected text, if any, goes in `human_transcript` — copy
   `raw_transcript` into it if no correction exists.
3. **Emit negative turns too.** Greetings, "yes", "uh huh", read-back digits,
   small talk — every one of them, with `labels=[]`. Without negatives the model
   never learns to stay silent, and it will fire on everything.
4. **`labels` starts empty.** Ingest never assigns a final label. Labeling is a
   separate, human-gated step. Every row comes out of ingest with `labels=[]` —
   that is expected and correct, not a gap to fill.
5. **`previous_agent_utterance` matters.** Many hard cases are fragments that only
   parse in context ("yes, it still hasn't arrived"). Populate it; `""` when the
   customer turn opens the call or follows another customer turn.

### `GoldRow` — the full column contract

| Field | Type | Meaning |
|---|---|---|
| `conversation_id` | str | Groups all turns of one call. A whole call always stays in one split |
| `turn_index` | int | Order within the call (0,1,2,…) |
| `timestamp_ms` | int | Real event time, milliseconds. Drives time-ordered splits |
| `raw_transcript` | str | ASR output, **UNCLEANED** |
| `human_transcript` | str | Human-corrected text; copy `raw_transcript` if none |
| `previous_agent_utterance` | str | Raw ASR of the agent turn immediately before; `""` if none |
| `prev_caller_segment` | struct or None | `{turn_index, text, gap_ms}` — the immediately preceding **caller** fragment, set **only** when no agent turn sits between the two. `None` otherwise. Feeds the stitcher |
| `dialog_acts` | list[str] | Per-turn tags if you have them; `[]` is fine |
| `session_task_intent` | str | Call-level reason code if available; `""` otherwise. A weak hint only — **never** a final label |
| `labels` | list[str] | Turn-level intents. `[]` = no actionable intent. May hold more than one |
| `ambiguous` | bool | `True` excludes the row from training **and** eval. Default `False` |
| `split` | str | `""` until `poc partition` sets `train`\|`calibration`\|`policy`\|`test` |

Example of a negative and a positive row:

```jsonc
// greeting — a negative, and these are essential
{ "conversation_id":"C0002f7","turn_index":4,"timestamp_ms":1591056796175,
  "raw_transcript":"hi","human_transcript":"hi",
  "previous_agent_utterance":"how can i help you today",
  "prev_caller_segment":null,"dialog_acts":[],"session_task_intent":"",
  "labels":[],"ambiguous":false,"split":"" }

// a request turn — labels still [] at ingest; the label comes later
{ "conversation_id":"C0002f7","turn_index":6,"timestamp_ms":1591056800705,
  "raw_transcript":"i lost my debit card","human_transcript":"i lost my debit card",
  "previous_agent_utterance":"how can i help you today",
  "prev_caller_segment":{"turn_index":5,"text":"my name is patricia brown","gap_ms":2260},
  "dialog_acts":[],"session_task_intent":"","labels":[],"ambiguous":false,"split":"" }
```

`prev_caller_segment` exists so the stitcher can glue an ASR-split utterance
("i lost my" + "card") at serve time. Set it only when the previous segment was
*also* the caller; `gap_ms` is the silence between them, floored at 0.

---

## 4. Rules that must not be broken

1. **Never run `poc train` unless explicitly asked.** It sends every train,
   calibration, and policy row to the Gemini embedding API and costs real money.
   Results cache by text hash so re-runs are free, but the first run on a full
   corpus is not.
2. **Never run `scripts/autolabel.py` unless explicitly asked.** Same reason.
3. **`poc partition` must run before any auto-labeling**, and auto-labeling must
   be invoked with `--skip-split test`. The test split is the only honest number
   this project will produce. An LLM that writes labels onto test rows destroys it
   silently and unrecoverably.
4. **Ingest the WHOLE corpus before `poc partition` runs.** Partition freezes the
   test set by time. Ingesting more data later and re-partitioning changes the
   test set and voids the "touched once" guarantee.
5. **Never invent, infer, or synthesize labels** to make a step run or a number
   look better. `labels: []` is meaningful data, not a missing value.
6. **Do not evaluate against the test split while iterating.** Use `calibration`
   and `policy`. Test is read once, for reporting.
7. **Do not weaken `configs/policy.yaml` floors** (`precision_floor`,
   `accept_floor`, `tau_low`) to improve a metric. They are precommitted safety
   parameters.
8. **Do not commit data.** `.gitignore` excludes `data/raw/`, `data/gold/*`,
   `data/cache/`, `artifacts/`, `.venv/`. Customer transcripts stay local.
   Put our CSV in `data/raw/`.
9. **Do not modify `data/gold/test_verdicts.json`** — curated human ground truth,
   version-controlled deliberately.
10. **Synthetic or augmented rows, if ever added, are train-only forever.**
    Precision floors are certified on real data only.

---

## 5. Running things

```sh
cd trainer && source .venv/bin/activate          # required every session
export POC_GOLD_PATH=../data/gold/gold.duckdb    # required every session (see below)
export GEMINI_API_KEY=...                        # or: set -a; source ../.env; set +a
```

`POC_GOLD_PATH` selects the on-disk gold format by file extension — `.duckdb`
(canonical, mutable, required for `poc promote`), `.parquet`, or `.jsonl`
(default if unset). Keep it pointed at the DuckDB store.

**Two ways to run things, don't mix them:**

| Kind | How | Example |
|---|---|---|
| pipeline stage (in `semantic_poc/`) | the `poc` command | `poc ingest` |
| side tool (in `scripts/`) | `python` | `python ../scripts/held_out_eval.py` |

**Never run `python semantic_poc/ingest.py`** — it is a package module and its
relative imports will break. Always go through `poc`.

`poc` is a [Typer](https://typer.tiangolo.com/) app defined in
`trainer/semantic_poc/cli.py`. Each `@app.command()`-decorated function is one
subcommand, named after the function; its docstring is the `--help` text. It is
installed editable (`pip install -e .`), so source edits take effect immediately
with no reinstall — you only re-install if `pyproject.toml` changes.

### Commands

| Command | What it does |
|---|---|
| `poc ingest` | corpus → gold store. Re-running preserves any existing labels/splits, keyed on `(conversation_id, turn_index)` |
| `poc counts` | positives per intent, OOS/labeled/total, ambiguous, per-split. **Free, no API call.** The best checkpoint |
| `poc partition` | conversation-grouped, time-ordered splits; freezes and hashes test |
| `poc promote <intent> <ids.txt>` | add `<intent>` to each `conv#turn` in the file. Audited, idempotent. Needs the DuckDB store |
| `poc snapshot --label <name>` | Parquet restore point |
| `poc train` | **costs money.** embed → heads → fusion → calibrate → threshold → artifact |
| `poc replay` | stream test turns through a running server, diff vs the offline forward pass |
| `pytest` (from `trainer/`) | unit tests — pure functions, no API key needed |

Standard first-pass order:

```
poc ingest → poc counts → poc partition → autolabel (--skip-split test)
   → human-label the test rows → poc train → held_out_eval
```

### Module map (`trainer/semantic_poc/`)

| Role | Modules |
|---|---|
| Contract | `schema.py` — `GoldRow`, the one type everything speaks |
| Data in / I/O | `ingest.py` (corpus → `GoldRow`), `store.py` (gold I/O seam), `partitions.py` (splits), `paths.py`, `config.py` |
| Featurize | `context.py` (`build_c1`), `embeddings.py` (Gemini, cached), `features.py` (TF-IDF) |
| Train | `train.py` (heads → fusion → Platt → thresholds), `fixtures.py` |
| Serve / decide | `runtime.py` (`Scorer`), `policy.py` (`decide`), `stitch.py` |
| Out | `artifact.py` (immutable hash-named artifact) |
| Entry | `cli.py` (the `poc` commands) |

Side tools live in `scripts/`: `serve_py.py` (inference server), `autolabel.py`
(LLM dual-critic bulk labeling), `held_out_eval.py` (cleaned held-out number),
`stability_report.py` (per-intent regression gate), `e2e_once.sh`,
`first_pass.sh`.

---

## 6. Labeling and the gold store

Ingest emits every turn with `labels=[]`. Labels arrive afterwards, and every
change is audited in the same file and transaction.

| Operation | How |
|---|---|
| Bulk LLM labeling | `python ../scripts/autolabel.py --skip-split test` — dual blind critics with opposing biases; agreements promote, `none` stays OOS, disagreements → `ambiguous` |
| Label a batch by hand | `poc promote <intent> ids.txt` — one `conv#turn` per line. Union semantics, idempotent |
| Fix one row exactly | `store.relabel(path, "conv#turn", labels, note)` |
| Exclude a row | `store.mark_ambiguous(path, ids, note)` — drops it from train **and** eval |
| Restore point | `poc snapshot --label <name>` → Parquet in `snapshots/` |

The DuckDB store holds a second table, `label_audit`, appended by
`relabel`/`promote` in the same transaction as the change — columns
`id` (`conv#turn`), `ts`, `old_labels`, `new_labels`, `note`. History cannot
drift from the data and no external diff file is needed.

Storage tiers: `gold.duckdb` is canonical and mutable; Parquet snapshots are
regenerable exports, **not** a second source of truth; small curated JSON
(e.g. `test_verdicts.json`) is git-tracked eval truth.

To add rows programmatically, build `GoldRow` objects and let the store handle
ordering and atomic writes:

```python
from semantic_poc.schema import GoldRow, PrevCallerSegment
from semantic_poc import store

rows = store.load_gold("../data/gold/gold.duckdb")
rows.append(GoldRow(conversation_id="C123", turn_index=6, timestamp_ms=...,
                    raw_transcript="...", human_transcript="...",
                    previous_agent_utterance="...", prev_caller_segment=None,
                    dialog_acts=[], session_task_intent="", labels=[]))
store.save_gold(rows, "../data/gold/gold.duckdb")   # full rewrite, row order re-derived
```

`save_gold` to a `.duckdb` path **replaces** the store and resets `label_audit`.
Use it to build or migrate, not to append to a live store.

The store maintains a hidden `_ord` column so reads are always in stable
original order — TF-IDF fitting and the conversation-grouped CV folds are
order-sensitive. Don't expose it in `GoldRow` and don't reuse a value.

### The add-an-intent loop

```
poc promote <intent> ids.txt                          # label the batch (audited)
poc train                                             # retrain
python ../scripts/stability_report.py --label "added X"   # regression gate, non-zero exit on drop
poc snapshot --label "X in"                           # restore point
```

Because heads are one-vs-rest, a turn already present as OOS is already a
negative for every other intent — so a representative OOS pool up front keeps
existing intents stable when new ones are added. Only same-family competitors
shift, and only at the `decide()` policy layer.

---

## 7. Config files

| File | Holds | Touch? |
|---|---|---|
| `configs/taxonomy.yaml` | intents + family roll-up + per-intent `desc:` | **yes, rewrite** |
| `configs/policy.yaml` | `precision_floor` 0.90, `accept_floor` 0.50, `tau_low` 0.30, `max_accepted` 3, `exclusive_groups` | only when asked |
| `configs/features.yaml` | TF-IDF params, tokenizer regex, context tags | no |
| `configs/embedding.yaml` | pinned provider/model/dim | only if we move off Gemini |
| `configs/stitcher.yaml` | fragment-merge rules (`max_gap_ms`, cue words) | later, once we see real ASR |

Config values are hashed into every trained artifact, so changing one forces a
new artifact. That is the provenance guarantee — don't work around it.

`exclusive_groups` lists sibling intents that must not co-fire; within a group
only the argmax survives. Currently empty.

---

## 8. Context that will save you from bad suggestions

- **Labels are the bottleneck, not the model.** In the rehearsal, coverage went
  **0.52 → 0.92 with zero model changes**, purely by cleaning a noisy evaluation
  set. Assume the same here.
- **Per-intent AUC was ~0.999.** The embedding already separates the intents
  linearly. Low coverage means a threshold or label problem, not a model-capacity
  problem. Do not propose a bigger model, a fine-tuned encoder, an MLP, or a
  neural architecture change. A fine-tuned encoder was measured and **lost** to
  the linear model.
- **More raw volume did nothing** (flat learning curve). Synthetic augmentation
  measured null once labels were clean.
- **The binding number is labeled positives per intent per split**, from
  *different conversations* — 150 positives from 150 calls ≫ 150 from 10 calls.
  Roughly: <30 positives, the metric is meaningless; ~150, first real per-intent
  number; ~300, solid enough to certify a floor.
- **Clean the EVALUATION set before the training set.** Noisy training labels cost
  some accuracy; noisy eval labels *lie to you* and mis-set the thresholds.

---

## 9. Deliberately not implemented

A **family-level fallback** decision ("confident it's a card issue, unsure which
action") is discussed in `docs/method-guide.md` and `docs/model-design.md` as a design
option. It is **not implemented** — `policy.py` returns only `accepted`,
`multi_accepted`, `abstained`, `no_supported_intent`. Do not implement it; the
decision is deferred until there is real data showing how often "confident
family, unsure leaf" actually occurs.

---

## 10. Corpus specifics

<!-- FILL IN before starting. Describe the sample CSV: file location, one row
     per what, exact column names, how customer vs agent turns are distinguished,
     timestamp units and origin, conversation boundary field, whether a corrected
     transcript exists, whether ASR confidence is present. -->

- Sample CSV location:
- One row per:
- Column names and meanings:
- Customer vs agent turn is identified by:
- Timestamp column, units, and epoch:
- Conversation boundary field:
- Corrected/human transcript available:
- ASR confidence available:

**Before writing code**, inspect the sample and show the proposed field-by-field
mapping onto `GoldRow` for review. Ask about anything ambiguous — especially
speaker identification, timestamp units, and conversation boundaries. Guessing
there produces a pipeline that runs and is silently wrong.

**Definition of done for the port:** `poc ingest && poc counts` runs clean, and
you report the counts output plus 3 sample `GoldRow`s. Expected shape of a
correct first run: `total` = number of customer turns, `oos` ≈ `total`,
`labeled` = 0, and the intents from `taxonomy.yaml` listed with zero positives.
Stop there and report before anything else runs.
