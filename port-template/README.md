# port-template — fill-in-the-blanks stubs for running this pipeline on your data

These are **templates**, not live code. The working HVB rehearsal is untouched.
Copy each file to its real location, fill in every `# TODO:`, and follow the
order below. Full rationale for each step is in `docs/office-playbook.md`
(this folder maps 1:1 to Appendix B there).

## What's here

| template file | copy to | what you change |
|---|---|---|
| `ingest.py` | `trainer/semantic_poc/ingest.py` | **the main rewrite** — read your corpus → `GoldRow`s |
| `taxonomy.yaml` | `configs/taxonomy.yaml` | your ~50 intents + families |
| `policy.yaml` | `configs/policy.yaml` | precision floor, accept-floor, exclusive groups |
| `bootstrap_labels.py` | `scripts/bootstrap_labels.py` | the label adjudication loop, **API-based** (batched LLM calls) |
| `label_helper.py` | `scripts/label_helper.py` | the label loop, **manual/chatbot-assisted** (no API calls) |
| `merge_labeled_pool.py` | `scripts/merge_labeled_pool.py` | appends `label_helper.py`'s output into `gold.jsonl` |

Not templated because they change little or not at all: the embedding provider
(`trainer/semantic_poc/embeddings.py` — only if you're not on Gemini), and
everything downstream (train / calibrate / threshold / serve / diagnostics),
which is dataset-agnostic and runs as-is.

## Order of operations

1. **Ingest** — rewrite `ingest.py` (TODOs 1–5), wire `cli.py`'s `ingest()` to
   your raw path, then `uv run poc ingest`. Eyeball `data/gold/gold.jsonl`:
   customer turns only, raw text uncleaned, negatives present, `labels` empty.
2. **Taxonomy** — put your intents + families in `configs/taxonomy.yaml`.
3. **Label** — two ways, pick one (or mix — e.g. manual now, API later at scale):
   - **API-based:** `python scripts/bootstrap_labels.py build`, run your two
     blind critics + human adjudication (step 2 in that file), then
     `python scripts/bootstrap_labels.py apply`. Clean the eval splits first;
     gate on kappa ≥ ~0.75.
   - **Manual/chatbot-assisted (no API cost):** stratify by pulling calls
     already tagged with a target MAIN intent from your export, walk them one
     at a time with `python scripts/label_helper.py --main cards --input
     your_export.csv` — it prompts you to paste each call into your internal
     chatbot, records the sub-intent + a verified/chatbot-only flag, and tracks
     live progress toward ~150/sub-intent. Then
     `python scripts/merge_labeled_pool.py` appends the results straight into
     `gold.jsonl`. **Verify generously** (say yes to "did you personally
     confirm this?") — you can't control which conversation lands in your eval
     split later, so unverified rows there are a real risk.
4. **Policy** — set safety knobs in `configs/policy.yaml` (do this before you
   look at any result).
5. **Partition** — `uv run poc partition` (conversation-grouped, time-ordered,
   freezes + hashes the test split).
6. **Train** — `uv run poc train` (embeds cached → heads → fusion → calibrate →
   threshold → artifact).
7. **Verify** — `uv run pytest` (incl. the stitcher), start
   `uv run python ../scripts/serve_py.py`, then `uv run poc replay` (online vs
   offline agree on your artifact).
8. **Diagnose before trusting** — run `scripts/ceilings.py`, `scripts/fp_audit.py`,
   `scripts/held_out_eval.py` (playbook §7). These tell you whether any gap is a
   label, audio, or model problem — run them *before* proposing any upgrade.

## The one thing to keep repeating to yourself

`ingest.py` emits `labels=[]`. **Labels are earned in step 3, by humans, gated on
agreement — not produced by ingest and not laundered from a model's own guess.**
That discipline is what moved the rehearsal from 0.52 to 0.92.
