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

Exactly six files change. Line numbers are current as of this writing.

| # | File | Change |
|---|---|---|
| 1 | `trainer/semantic_poc/ingest.py` | **Rewrite the body.** Read our CSV → one `GoldRow` per CUSTOMER turn, `labels=[]`. Also remove the `weak_labels` import (`:31`) and call (`:100`), and drop the now-unused `task_type_map` parameter (`:34`, `:47`, `:50`) |
| 2 | `trainer/semantic_poc/paths.py` | `RAW_HVB` (`:8`) → rename `RAW_CORPUS`, repointed at our CSV under `data/raw/` |
| 3 | `trainer/semantic_poc/cli.py` | Two lines: the import (`:20`) and the `ingest_corpus(...)` call (`:83`, which currently passes `taxonomy["task_type_map"]`) |
| 4 | `configs/taxonomy.yaml` | Our real intents + family roll-up. Add a `desc:` line per intent — the auto-labeler reads these. Delete `task_type_map` (`:13`) |
| 5 | `trainer/semantic_poc/schema.py` | Give `dialog_acts` and `session_task_intent` defaults (`[]` and `""`, `:30-31`) so rows validate without them |
| 6 | `trainer/semantic_poc/weak_labels.py` | **Delete the file** (see below for why) |

Everything else stays untouched — `store.py`, `partitions.py`, `train.py`,
`features.py`, `runtime.py`, `policy.py`, `artifact.py`, `embeddings.py`,
`context.py`, `stitch.py`, `config.py`. That is the `GoldRow` seam working as
designed. **If a task seems to require editing any of them, stop and ask** —
that means the ingest boundary is being violated.

### Why `weak_labels.py` is deleted, not ported

It assigns turn-level labels inside ingest by a two-part heuristic: find the
first caller turn tagged with a request-bearing **dialog act**, then label it
with the call-level **`session_task_intent`**. Our corpus has neither field, so
the function is inert — `is_request_bearing()` would return `False` for every
turn and every row would end up `labels=[]` regardless.

But "harmless no-op" is not why it goes. It goes because it **violates ingest
contract rule 4: ingest never assigns a label.** It existed only to synthesize
labels for a rehearsal that had no human labelers, and leaving it in place keeps
a dormant code path that would start silently writing labels the moment anyone
populated `dialog_acts` with something. Labels come from the human-gated
labeling step (§6) and from nowhere else.

### A trap that was already fixed — don't re-break it

`poc adjudicate` used to read `r.labels = [r.session_task_intent]`, which only
worked while ingest seeded that field from call-level metadata. With
`session_task_intent` defaulting to `""` (change #5 above), a `yes` verdict would
have silently written the label `[""]` — no crash, corrupted labels.

It now takes an explicit, taxonomy-validated intent:

```sh
poc adjudicate <intent> <verdicts.tsv>     # lines: conv_id#turn_index<TAB>yes|no
```

Nothing further is needed here. It is not part of the port and you should not
run it — the note exists so the signature change isn't mistaken for a bug and
reverted.

Our corpus details are provided separately; see §10 for what you must establish
before writing any code. Changing how embeddings are fetched is a **separate**
task with its own details still to come — see §7. Do not bundle the two.

### How far you can validate — read this before planning

**You have no access to the embedding API or any LLM call.** That is a hard
environment limit, not a permission you can request. It bounds what you can run:

| Stage | You | Why |
|---|---|---|
| `poc ingest` | **run it** | pure local file reading |
| `poc counts` | **run it** | reads the store, no network |
| `poc partition` | **run it** | pure local computation |
| `pytest` (from `trainer/`) | **run it** | all pure functions, no key needed |
| `scripts/autolabel.py` | **cannot** | LLM calls |
| `poc train` | **cannot** | embeds every row via the API |
| `scripts/serve_py.py`, `poc replay`, `held_out_eval.py` | **cannot** | need an artifact, which needs training |

So your job is to get the **local** part of the flow completely working and
verified, and hand the rest back. Do not fake, stub, mock, or monkey-patch the
embedding layer to get further — a pipeline that "runs" on synthetic vectors
proves nothing and risks a fake artifact being mistaken for a real one.

When you have taken it as far as it goes, report:

1. Confirmation that `poc ingest`, `poc counts`, `poc partition`, and `pytest`
   all run clean, with their output
2. The exact commands to run next, in order, for the stages you could not reach
3. Anything you could not verify and what specifically you would check about it

Do not treat the unreachable stages as failures or try to work around them.
Getting ingest, the schema, the taxonomy, and the splits correct **is** the
deliverable.

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

1. **Never run `poc train`.** It sends every train, calibration, and policy row
   to the embedding API and costs real money. It is also outside your
   environment's reach (§3).
2. **Never run `scripts/autolabel.py`.** Same reason.
3. **Never fake, stub, or mock the embedding layer** to get past those limits.
   A pipeline that runs on synthetic vectors validates nothing and risks a
   meaningless artifact being taken for a real one.
4. **`poc partition` must run before any auto-labeling**, and auto-labeling must
   be invoked with `--skip-split test`. The test split is the only honest number
   this project will produce. An LLM that writes labels onto test rows destroys it
   silently and unrecoverably.
5. **Ingest the WHOLE corpus before `poc partition` runs.** Partition freezes the
   test set by time. Ingesting more data later and re-partitioning changes the
   test set and voids the "touched once" guarantee.
6. **Never invent, infer, or synthesize labels** to make a step run or a number
   look better. `labels: []` is meaningful data, not a missing value.
7. **Do not evaluate against the test split while iterating.** Use `calibration`
   and `policy`. Test is read once, for reporting.
8. **Do not weaken `configs/policy.yaml` floors** (`precision_floor`,
   `accept_floor`, `tau_low`) to improve a metric. They are precommitted safety
   parameters.
9. **Do not commit data.** `.gitignore` excludes `data/raw/`, `data/gold/*`,
   `data/cache/`, `artifacts/`, `.venv/`. Customer transcripts stay local.
   Put our CSV in `data/raw/`.
10. **Do not modify `data/gold/test_verdicts.json`** — curated human ground truth,
   version-controlled deliberately.
11. **Synthetic or augmented rows, if ever added, are train-only forever.**
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

### Embedding provider — a second, separate task

**Our organization does not call the embedding API the way this code currently
does.** The details of our endpoint will be provided separately. Do not change
anything here until you have them, and treat this as a task distinct from the
ingest port — do not bundle the two.

When you do have the details, the change is deliberately contained:

**The seam is one method: `GeminiEmbedder._fetch_batch` in
`trainer/semantic_poc/embeddings.py`.** Replace its body. Everything else in that
file — the sqlite cache, dedup, order preservation, batching, retry/backoff — is
provider-agnostic and must not change.

Its contract:

```python
def _fetch_batch(self, texts: list[str]) -> list[np.ndarray]:
    # one vector per input text, SAME ORDER,
    # dtype float32, length == self.dim, L2-normalized
```

Invariants that must hold in any replacement:

1. **Keep the L2 normalization.** Downstream math assumes unit vectors, and
   normalizing here means nothing depends on whether the provider normalizes.
2. **Keep `float32`.** That is what the cache blobs store; training widens to
   float64 later.
3. **Keep the retry/backoff loop.** Internal gateways rate-limit and return
   transient 5xx too.
4. **Preserve input order.** Callers zip results back against their inputs.

**The cache-poisoning trap.** The cache key is
`sha256(f"{model}|{task_type}|{dim}|{text}")` — it does **not** include the
provider. Pointing `_fetch_batch` at a different endpoint while leaving
`configs/embedding.yaml`'s `model:` string unchanged means every previously
cached vector is a silent hit, and training would mix two providers' embeddings
with no error raised. When switching provider: change the `model:` string to
something provider-qualified **and** delete `data/cache/embeddings.sqlite`.

**Switching provider invalidates any trained artifact.** The embedding is the
model's input space; a different provider is a different space, so coefficients,
calibration, and thresholds are all void and everything must be re-embedded and
retrained. Flag this rather than assuming an existing artifact still applies.

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

## 10. Corpus specifics — to be provided

A sample CSV and a description of our corpus will be given to you separately.
This section is intentionally open: **do not assume a shape, and do not infer one
from the rehearsal code you find in `ingest.py`.** That body reads a completely
different source and is reference only.

Before writing any code, establish answers to all of the following. Inspect the
sample for what you can determine yourself; **ask for anything you cannot
confirm.** Guessing here produces a pipeline that runs and is silently wrong —
which is far worse than a pipeline that fails.

| Must establish | Why it matters |
|---|---|
| File location and format | — |
| One row per *what* — turn, segment, utterance, or call | Determines whether you group, split, or map 1:1 |
| Exact column names and meanings | — |
| **How a customer turn is distinguished from an agent turn** | Getting this backwards produces a model trained on the wrong speaker. Highest-risk item |
| Timestamp column, units (s / ms / µs), and epoch | Drives time-ordered splits; wrong units silently scramble split order |
| Conversation boundary field | A whole call must stay in one split. Wrong grouping means leakage and inflated scores |
| Whether turns are already in chronological order, or need sorting | `turn_index` must reflect true order |
| Whether a corrected/human transcript exists | If not, copy `raw_transcript` into `human_transcript` |
| Whether ASR confidence is present | Logged upstream, never gates a decision. Optional |
| How to tell a *split* customer segment from a new turn | Governs whether `prev_caller_segment` is populated |

**Then, before writing code:** present the proposed field-by-field mapping onto
`GoldRow` (§3) and get it confirmed. State explicitly which fields you inferred
from the sample versus which you were told, and flag anything you are less than
confident about.

**Sanity checks to run on your own output before reporting:**

- Row count equals the number of customer turns, not total turns
- `conversation_id` cardinality matches the number of calls in the sample
- `turn_index` is strictly increasing within each conversation
- `previous_agent_utterance` is non-empty for a plausible majority of rows — if
  it is empty nearly everywhere, speaker identification is probably wrong
- `timestamp_ms` values are plausible epoch milliseconds (13 digits for recent
  dates), and ordering by them reproduces conversation order
- Every row has `labels == []`

**Definition of done for the port:** `poc ingest && poc counts` runs clean, and
you report the counts output plus 3 sample `GoldRow`s and the checks above.
Expected shape of a correct first run: `total` = number of customer turns,
`oos` ≈ `total`, `labeled` = 0, and the intents from `taxonomy.yaml` listed with
zero positives. Stop there and report before anything else runs.
