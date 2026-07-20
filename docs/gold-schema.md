# Gold dataset — schema & how to add rows

The gold dataset is the labeled corpus the pipeline trains and evaluates on.
Canonically it lives as a single **mutable DuckDB file, `data/gold/gold.duckdb`**
(select it with `POC_GOLD_PATH=…/gold.duckdb`). Everything downstream speaks
`GoldRow` (`trainer/semantic_poc/schema.py`) and never opens the file directly —
all I/O goes through `trainer/semantic_poc/store.py`, so the on-disk format is a
swappable detail (`.duckdb` / `.parquet` / `.jsonl`, dispatched by extension).

**One row = one CUSTOMER turn.** Agent turns are context, not rows.

## Storage tiers
| File | Role |
|---|---|
| `gold.duckdb` | **canonical, mutable.** Label / relabel here (`INSERT`/`UPDATE`). |
| `gold-<label>-<ts>.parquet` (in `snapshots/`) | regenerable export — restore point / archive / frozen training input. **Not** a second source of truth. |
| `*.json` (e.g. `test_verdicts.json`) | small, git-tracked, curated eval truth. |

## The `gold` table

| Column | Type | Definition | Rules / example |
|---|---|---|---|
| `conversation_id` | VARCHAR | Groups all turns of one call. A whole call stays in one split. | `"0002f70f7386445b"` |
| `turn_index` | BIGINT | Order of this turn within the call (0,1,2,…). | `6` |
| `timestamp_ms` | BIGINT | Real event time in ms. Drives time-ordered splits. | `1591056800705` |
| `raw_transcript` | VARCHAR | ASR output, **UNCLEANED**. You serve on this, so you train/eval on it. | **Never normalize.** `"i lost my debit card"` |
| `human_transcript` | VARCHAR | Human-corrected text. Copy `raw_transcript` if you have no correction. | `"i lost my debit card"` |
| `previous_agent_utterance` | VARCHAR | Raw ASR of the agent turn right before this customer turn; `""` if none. | `"how can i help you today"` |
| `prev_caller_segment` | STRUCT(`turn_index` BIGINT, `text` VARCHAR, `gap_ms` BIGINT) or NULL | The immediately preceding **caller** fragment (for stitching a split utterance), only when no agent turn sits between. NULL otherwise. | `{turn_index: 5, text: 'my name is patricia brown', gap_ms: 2260}` |
| `dialog_acts` | VARCHAR[] | Any per-turn tags you have. `[]` is fine. | `['gridspace_problem_description']` |
| `session_task_intent` | VARCHAR | Call-level reason code, `""` if none. **Weak seed only, never a final label.** | `"replace_card"` |
| `labels` | VARCHAR[] | **Turn-level intent labels. This is what you label.** `[]` = OOS / no actionable intent. A turn may carry more than one intent. | `['replace_card']` or `[]` |
| `ambiguous` | BOOLEAN | If true, the row is excluded from training and eval. | `false` |
| `split` | VARCHAR | `""` until the partition step sets `train` \| `calibration` \| `policy` \| `test`. | `"policy"` |
| `_ord` | BIGINT | **Hidden, store-managed.** A unique monotonic ordinal that preserves exact row order. Reads always `ORDER BY _ord` because TF-IDF and the conversation-grouped CV folds are order-sensitive (the corpus is NOT sorted by conversation_id/turn_index). Don't expose it in `GoldRow`; don't reuse a value. | `0,1,2,…` |

### `label_audit` table (relabel history, in the same file)
`store.relabel` / `store.promote` append one row here per change, in the same
transaction — so history can't drift from the data, and no external diff file is
needed.

| Column | Type | Meaning |
|---|---|---|
| `id` | VARCHAR | `"<conversation_id>#<turn_index>"` |
| `ts` | VARCHAR | UTC timestamp of the change |
| `old_labels` / `new_labels` | VARCHAR | JSON label arrays before/after |
| `note` | VARCHAR | free-text reason (e.g. `"promote card_status"`) |

## Contract you must not break (from `ingest.py`)
1. **One row per CUSTOMER turn.** Agent turns are context (they populate
   `previous_agent_utterance`), not rows.
2. **Do not clean `raw_transcript`.** You serve on raw ASR. Corrections go in
   `human_transcript`.
3. **Emit negatives too.** Greetings, "yes", read-back digits, small talk — all
   with `labels=[]`. Without them the model never learns to stay silent.
4. **`labels` starts `[]`.** Labeling is a separate, human-gated step (`promote`);
   ingest never sets a final label.
5. **Set `_ord`** on every row (see below). A NULL `_ord` breaks ordered reads.

## Example rows
```jsonc
// OOS turn (greeting) — a negative
{ "conversation_id":"0002f70f7386445b","turn_index":4,"timestamp_ms":1591056796175,
  "raw_transcript":"hi","human_transcript":"hi",
  "previous_agent_utterance":"how can i help you today",
  "prev_caller_segment":null,"dialog_acts":["gridspace_greeting"],
  "session_task_intent":"replace_card","labels":[],"ambiguous":false,"split":"policy" }

// Labeled request turn — a positive
{ "conversation_id":"0002f70f7386445b","turn_index":6,"timestamp_ms":1591056800705,
  "raw_transcript":"i lost my debit card","human_transcript":"i lost my debit card",
  "previous_agent_utterance":"how can i help you today",
  "prev_caller_segment":{"turn_index":5,"text":"my name is patricia brown","gap_ms":2260},
  "dialog_acts":["gridspace_problem_description"],
  "session_task_intent":"replace_card","labels":["replace_card"],"ambiguous":false,"split":"policy" }
```

## How to add rows

### Path A — through the store API (recommended)
Build `GoldRow` objects and let the store handle `_ord`, schema, and atomic write:
```python
from semantic_poc.schema import GoldRow, PrevCallerSegment
from semantic_poc import store

rows = store.load_gold("data/gold/gold.duckdb")        # existing rows (or [])
rows.append(GoldRow(
    conversation_id="C123", turn_index=6, timestamp_ms=1591056800705,
    raw_transcript="i lost my debit card", human_transcript="i lost my debit card",
    previous_agent_utterance="how can i help you today",
    prev_caller_segment=PrevCallerSegment(turn_index=5, text="my name is …", gap_ms=2260),
    dialog_acts=[], session_task_intent="", labels=[],   # labels=[] — label later
))
store.save_gold(rows, "data/gold/gold.duckdb")          # full rewrite, _ord re-derived
```
`save_gold` to a `.duckdb` path REPLACES the store (fresh `gold` table + empty
`label_audit`) and re-derives `_ord` from row order. Use it to (re)build or
migrate. For incremental appends to a live store, use Path B.

### Path B — raw SQL (you add rows yourself)
Insert straight into the `gold` table. **You must set `_ord`** to a unique value
past the current max; list/struct columns use DuckDB literal syntax:
```sql
INSERT INTO gold BY NAME
SELECT
  'C123' AS conversation_id, 6 AS turn_index, 1591056800705 AS timestamp_ms,
  'i lost my debit card' AS raw_transcript, 'i lost my debit card' AS human_transcript,
  'how can i help you today' AS previous_agent_utterance,
  {'turn_index': 5, 'text': 'my name is …', 'gap_ms': 2260} AS prev_caller_segment,
  []::VARCHAR[] AS dialog_acts, '' AS session_task_intent,
  []::VARCHAR[] AS labels, false AS ambiguous, '' AS split,
  (SELECT COALESCE(max(_ord), -1) + 1 FROM gold) AS _ord;
-- prev_caller_segment when none:  NULL
-- labels when it's a positive:    ['card_status']
```

## Labeling & lifecycle (the intent-by-intent loop)
All exposed as `store.*` functions and `poc` subcommands
(`POC_GOLD_PATH=…/gold.duckdb`):

| Command | What it does |
|---|---|
| `poc counts` | positives per intent, OOS / labeled / total, ambiguous, per-split — check balance |
| `poc promote <intent> <ids.txt>` | add `<intent>` to each `conv#turn` listed (union, audited, one txn); idempotent |
| `poc snapshot --label <name>` | write a Parquet restore point into `snapshots/` |
| `store.relabel(path, "conv#turn", labels, note)` | set a row's labels exactly (fix a mistake), audited |
| `poc partition` | assign `split` (whole call stays in one split) |
| `poc train` | retrain the full model → hash-named artifact |
| `python scripts/held_out_eval.py` | cleaned held-out precision/recall/coverage |
| `python scripts/stability_report.py --label "added X"` | diff per-intent metrics vs last run; non-zero exit on regression |

Typical add-an-intent cycle:
```
poc counts
poc promote card_status ids.txt          # label the batch
poc train
python scripts/stability_report.py --label "added card_status"
poc snapshot --label "card_status in"
```
Because heads are one-vs-rest, an utterance already present as OOS is already a
negative for every other intent — so a **representative OOS pool up front** keeps
existing intents stable when you add new ones (only same-family competitors shift,
at the `decide()` policy layer). See `docs/feasibility-verdict.md` for the
mechanism.

## Scripts kept in `scripts/`
| Script | Purpose |
|---|---|
| `e2e_once.sh` | full smoke test: train → unit tests → serve → replay |
| `serve_py.py` | Python inference server (loads one artifact) |
| `held_out_eval.py` | cleaned held-out evaluation (importable; used by the stability report) |
| `stability_report.py` | per-intent metrics tracked run-over-run; regression gate |

The rehearsal-era diagnostics (ASR-damage, ceilings, FP audit, learning curve,
miss analysis, adjudication prep/apply, showcase) were removed — reconstruct from
git history if a specific analysis is needed again.
