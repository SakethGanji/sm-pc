# QUICKSTART — zero to your first result

Follow top to bottom. The *why* behind each step is in `POC-GUIDE.md`; the data
contract is `docs/gold-schema.md`; the code map is `trainer/README.md`.

## 0. Set up the machine (once)
Get the repo onto the box (clone from your remote, or copy the folder **excluding
`data/` and `.venv/`**), then:
```bash
git checkout parquet-datasets
python -m venv trainer/.venv && source trainer/.venv/bin/activate
pip install -r trainer/requirements.txt
echo 'GEMINI_API_KEY=<your key, with billing>' > .env   # required for training
export POC_GOLD_PATH=../data/gold/gold.duckdb            # canonical store; keep this set every session
cd trainer && poc --help && pytest -q                    # smoke test → help + "23 passed"
```

## 1. Point the pipeline at your data (the only files you edit)

| File | What | Effort |
|---|---|---|
| `trainer/semantic_poc/ingest.py` | **Rewrite** to read your CSV → `GoldRow` per customer turn, `labels=[]` | **the real work** |
| `trainer/semantic_poc/paths.py` | Point `RAW_HVB` at your CSV | 1 line |
| `configs/taxonomy.yaml` | Your intents + families | a list (no code) |
| `trainer/semantic_poc/cli.py` | Wire the `ingest()` call to your new signature | ~1 line |
| `configs/policy.yaml` | Set your precision floor (+ `exclusive_groups`) | ~2 lines (has defaults) |

Everything else (`features.yaml`, `stitcher.yaml`, `embedding.yaml`, `train.py`, …)
— leave as defaults. Delete `weak_labels.py` once ingest stops seeding weak labels.
Drop your CSV in **`data/raw/`** (gitignored — customer transcripts stay local).

## 2. Ingest → store
```bash
poc ingest      # CSV -> gold.duckdb, every turn labels=[]
poc counts      # SANITY: total = your turn count, oos ≈ total, labeled = 0, your intents listed
```
If `counts` looks wrong (missing turns, labels already set, wrong intents), fix
`ingest.py` before going further.

## 3. Label — two ways (pick one, or mix)

**A. LLM auto-label (bulk, recommended accelerator).** Dual Gemini critics label
every unlabeled turn across all intents at once; `none` becomes your OOS pool for
free; disagreements are quarantined `ambiguous`.
```bash
poc snapshot --label pre-autolabel           # restore point first
python ../scripts/autolabel.py --dry-run --limit 50   # preview: agreement rate + counts
python ../scripts/autolabel.py                # apply to all unlabeled turns
poc counts                                    # positives per intent + OOS should populate
```
Guardrails: LLM labels are a first pass, not truth — spot-check them, and **never
trust an auto-labeled eval set.** Two ways to keep the eval honest (pick one):
- **verify-after** (less work): label everything → `poc partition` → human-check the
  `test` rows' labels.
- **skip-eval** (purest): `poc partition` **first**, then
  `python ../scripts/autolabel.py --skip-split test` so the LLM never touches the
  answer key → you hand-label the `test` rows.

Adding a one-line `desc:` per intent in `taxonomy.yaml` improves accuracy.

**B. Manual promote (small / targeted).** Pull the calls tagged with an intent,
write the request turns' ids (`conversation_id#turn_index`, one per line) to
`ids.txt`, then:
```bash
poc promote card_status ids.txt   # unions the intent onto those turns (audited)
```

Either way, make sure plenty of turns from *other* call types sit at `labels=[]` —
those are your representative negatives (auto-label produces them as `none`).

## 4. Train + read the number
```bash
poc partition                                     # split by conversation, time-ordered, freeze test
poc train                                         # embed (cached) -> heads -> calibrate -> threshold -> artifact
python ../scripts/held_out_eval.py                # per-intent precision/recall/coverage on held-out test
python ../scripts/stability_report.py --label "card_status"   # record + diff vs previous run
poc snapshot --label "card_status-in"             # Parquet restore point
```
The **first** train embeds every unique text via Gemini (one-time API cost); later
trains reuse the cache for $0.

## 5. Loop — one intent at a time
Repeat 3–4 per intent. Every add: **retrain the full model, then run
`stability_report`** to confirm earlier intents didn't regress (only same-family
siblings should move). Snapshot at milestones.

---

## How much data until an ACTUAL result?
The binding number is **positives for that intent, from *different* conversations**
(150 from 150 calls ≫ 150 turns from 10 calls — diversity is the point). Since test
is ~10% of a conversation-grouped split:

| positives (one intent) | ~land in test | what you can honestly say |
|---|---|---|
| < 30 | ~0–3 | **nothing** — head too noisy; keep labeling, don't read the metric |
| ~50 | ~5 | **smoke** — does it run end-to-end, is precision roughly holding? directional only |
| **~150** | ~15 | **first real per-intent number** (wide error bars) — this is your "actual result" |
| ~300 | ~30 | **solid** estimate; certify the floor here |

Also: don't `partition`/`train` until you have enough **conversations** to form four
non-degenerate splits — rough floor ~50–100 calls with a spread of intents + OOS.
Below that the splits are too thin to mean anything.

## Testing cadence — do NOT test every 10 positives
With ~10 positives, roughly **one** lands in test, so the number is pure noise and
jumps around — testing that often wastes time and misleads. Instead:

- **Label in batches, test at milestones:** train + look at **~50** (smoke), **~150**
  (real), **~300** (tight). Between those, just keep labeling.
- **Always run `stability_report` when you ADD a new intent** — that's the regression
  gate, independent of the milestone rhythm. It's the one check you run every time.
- **Trust the test-split positive count, not the headline:** < ~10 test positives →
  don't believe the number; ~15 → usable with wide bars; ~30 → solid.

## The one-glance loop
```
poc counts                                   # check balance / OOS pool
poc promote <intent> ids.txt                 # label a batch (audited)
poc train                                    # retrain full model
python ../scripts/stability_report.py --label "added <intent>"   # regression gate
poc snapshot --label "<intent> in"           # restore point
# → read held_out_eval.py at ~50 / ~150 / ~300 positives; certify intents that clear the floor,
#   tier the rare/confusable tail that can't.
```
