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
| `bootstrap_labels.py` | `scripts/bootstrap_labels.py` | the label adjudication loop (replaces HVB weak labels) |

Not templated because they change little or not at all: the embedding provider
(`trainer/semantic_poc/embeddings.py` — only if you're not on Gemini), and
everything downstream (train / calibrate / threshold / serve / diagnostics),
which is dataset-agnostic and runs as-is.

## Order of operations

1. **Ingest** — rewrite `ingest.py` (TODOs 1–5), wire `cli.py`'s `ingest()` to
   your raw path, then `uv run poc ingest`. Eyeball `data/gold/gold.jsonl`:
   customer turns only, raw text uncleaned, negatives present, `labels` empty.
2. **Taxonomy** — put your intents + families in `configs/taxonomy.yaml`.
3. **Label** — `python scripts/bootstrap_labels.py build`, run your two blind
   critics + human adjudication (step 2 in that file), then
   `python scripts/bootstrap_labels.py apply`. Clean the eval splits first;
   gate on kappa ≥ ~0.75.
4. **Policy** — set safety knobs in `configs/policy.yaml` (do this before you
   look at any result).
5. **Partition** — `uv run poc partition` (conversation-grouped, time-ordered,
   freezes + hashes the test split).
6. **Train** — `uv run poc train` (embeds cached → heads → fusion → calibrate →
   threshold → artifact).
7. **Verify** — `cd server && go test ./...` (Python↔Go parity on your artifact),
   `uv run poc replay` (online vs offline agree).
8. **Diagnose before trusting** — run `scripts/ceilings.py`, `scripts/fp_audit.py`,
   `scripts/held_out_eval.py` (playbook §7). These tell you whether any gap is a
   label, audio, or model problem — run them *before* proposing any upgrade.

## The one thing to keep repeating to yourself

`ingest.py` emits `labels=[]`. **Labels are earned in step 3, by humans, gated on
agreement — not produced by ingest and not laundered from a model's own guess.**
That discipline is what moved the rehearsal from 0.52 to 0.92.
