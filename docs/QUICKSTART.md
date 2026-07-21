# QUICKSTART — zero to your first result

Follow top to bottom. The *why* is in `POC-GUIDE.md`; the data contract is
`docs/gold-schema.md`; the code map is `trainer/README.md`.

The flow: **ingest your whole corpus → freeze the eval set → LLM-label the bulk
(not eval) → human-label the eval set → train → measure → iterate.**

## 0. Set up the machine (once)
Get the repo onto the box (clone from your remote, or copy the folder **excluding
`data/` and `.venv/`**), then:
```bash
git checkout parquet-datasets
python -m venv trainer/.venv && source trainer/.venv/bin/activate
pip install -r trainer/requirements.txt
echo 'GEMINI_API_KEY=<your key, with billing>' > .env    # embeddings + auto-labeling
export POC_GOLD_PATH=../data/gold/gold.duckdb             # the store; keep set every session
cd trainer && poc --help && pytest -q                      # smoke → help + "23 passed"
```

## 1. Point the pipeline at your data (the only files you edit)

| File | What | Effort |
|---|---|---|
| `trainer/semantic_poc/ingest.py` | **Rewrite** to read your CSV → `GoldRow` per customer turn, `labels=[]` | **the real work** |
| `trainer/semantic_poc/paths.py` | Point `RAW_HVB` at your CSV | 1 line |
| `configs/taxonomy.yaml` | Your intents + families (add a `desc:` per intent — helps the auto-labeler) | a list |
| `trainer/semantic_poc/cli.py` | Wire the `ingest()` call to your new signature | ~1 line |
| `configs/policy.yaml` | Set your precision floor (+ `exclusive_groups`) | ~2 lines |

Everything else (`features.yaml`, `stitcher.yaml`, `embedding.yaml`, `train.py`, …)
— defaults. Delete `weak_labels.py` once ingest stops seeding weak labels.
Drop your CSV in **`data/raw/`** (gitignored — customer transcripts stay local).

## 2. Build the store & FREEZE the eval set
Ingest your **whole** corpus, then partition **once** — this freezes the test set
by time so the LLM can skip it and you never re-shuffle your answer key.
```bash
poc ingest        # ALL calls → gold.duckdb, every turn labels=[]
poc counts        # SANITY: total = turn count, oos ≈ total, labeled = 0, your intents listed
poc partition     # split by conversation + time; freezes + hashes the test set
```
If `counts` looks wrong (missing turns, labels set, wrong intents) — fix `ingest.py`
before going on.

## 3. Auto-label the bulk (LLM, everything EXCEPT eval)
Two Gemini critics label train/calibration/policy; `--skip-split test` keeps them
**off your eval answer key**. `none` becomes your OOS pool for free.
```bash
poc snapshot --label pre-autolabel                                    # restore point
python ../scripts/autolabel.py --skip-split test --dry-run --limit 50  # preview: agreement rate + counts
python ../scripts/autolabel.py --skip-split test                       # apply (costs Gemini Flash calls)
poc counts        # positives per intent + OOS should now populate
```
Agreements → the intent; disagreements/"unclear" → `ambiguous` (excluded). For extra
rigor add `,policy` (`--skip-split test,policy`) to keep the LLM off the
threshold-certification split too.

## 4. Own the eval set (human — non-negotiable)
The `test` rows are still `labels=[]` (the LLM skipped them). **A human labels them**
— that's the answer key `held_out_eval` scores against. Pull them for review:
```bash
python - <<'PY'
import sys; sys.path.insert(0,".")
from semantic_poc import store
for r in store.iter_gold("../data/gold/gold.duckdb"):
    if r.split == "test":
        print(f"{r.conversation_id}#{r.turn_index}\t{r.previous_agent_utterance!r}\t{r.raw_transcript!r}")
PY
```
Decide each turn's intent, collect ids per intent into `ids.txt` files, and write them:
```bash
poc promote <intent> ids.txt        # per intent; leave true negatives at []
```
(Also spot-check the auto-labels from step 3 and fix any with `poc promote` /
`python -c "from semantic_poc import store; store.relabel(...)"`.)

## 5. Train + read the number
```bash
poc train                                          # embed (cached) → heads → calibrate → threshold → artifact
python ../scripts/held_out_eval.py                 # per-intent precision/recall/coverage vs your human eval
python ../scripts/stability_report.py --label first-pass
poc snapshot --label first-pass
```
The **first** train embeds every unique text via Gemini (one-time API cost); later
trains reuse the cache for $0.

## 6. Iterate
Fix bad labels (`poc promote` / `store.relabel` / `store.mark_ambiguous`), retrain,
and **certify intents in tiers** — ship the ones that clear the precision floor;
hold the thin/confusable tail. Run `stability_report` after each labeling wave to
catch regressions (only same-family siblings should move).

---

## How much data until an ACTUAL result?
The binding number is **positives for that intent, from *different* conversations**
(150 from 150 calls ≫ 150 turns from 10 calls). Test is ~10% of a
conversation-grouped split:

| positives (one intent) | ~land in test | what you can honestly say |
|---|---|---|
| < 30 | ~0–3 | **nothing** — head too noisy; keep labeling, don't read the metric |
| ~50 | ~5 | **smoke** — does it run, is precision holding? directional only |
| **~150** | ~15 | **first real per-intent number** (wide bars) — your "actual result" |
| ~300 | ~30 | **solid** estimate; certify the floor here |

Don't `train` until you have ~50–100 **conversations** with a spread of intents +
OOS — below that the splits are too thin to mean anything.

## Testing cadence — do NOT test every 10 positives
With ~10 positives, roughly **one** lands in test, so the number is pure noise.
- **Test at milestones:** ~50 (smoke) → ~150 (real) → ~300 (tight). Between those, label.
- **Always run `stability_report` when you add a new intent** — the regression gate.
- **Trust the test-split positive count:** < ~10 → don't believe it; ~15 → usable; ~30 → solid.

## Two things to internalize
1. **Ingest your whole corpus before `poc partition`.** Partition freezes the test
   set by time; ingesting more data later and re-partitioning changes the test set and
   breaks the "touched once" guarantee. Load it all, partition once, label *within* it.
2. **`--skip-split test` is what enforces eval integrity** — the LLM never writes a
   label to a test row, so your answer key stays 100% human. A model graded against
   its own kind's guesses tells you nothing.

## The one-glance loop
```
# once: poc ingest → poc counts → poc partition           (freeze eval)
poc snapshot --label pre-autolabel
python ../scripts/autolabel.py --skip-split test           # bulk-label (not eval)
# human-label the test rows (§4), spot-check the rest
poc train
python ../scripts/held_out_eval.py                         # read at ~50 / ~150 / ~300 positives
python ../scripts/stability_report.py --label "wave N"     # regression gate
poc snapshot --label "wave N"
# certify intents that clear the floor; tier the rest
```
