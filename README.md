# semantic-poc — intent-classification POC rehearsal

Home rehearsal (design doc §14) of the Agent Assist non-generative intent
classifier: **Python trains and serves, real Gemini embeddings throughout.**
Data is the public HarperValleyBank corpus (1,446 bank call-center
conversations with machine-ASR + human transcripts and dialog acts).

A Go server (§5 contract, same math) is part of the original design-doc target
architecture but has been removed from this repo for now — see
[docs/design-doc.md](docs/design-doc.md) for the intended production shape.
The POC serves entirely from Python (`scripts/serve_py.py`); nothing here
depends on Go.

## Pipeline

```
gold rows (per caller ASR segment, weak labels from dialog acts)
  └─ conversation-grouped time-ordered splits: train / calibration / policy / test
train:  TF-IDF (deterministic, explicit math)  +  Gemini C1 embeddings (sqlite cache)
        semantic OVR LR + lexical OVR LR (per-intent C, grouped folds)
        fusion LR on out-of-fold logits  →  Platt (calibration split)
        →  precision-floor thresholds (policy split)
export: artifacts/artifact-poc-hvb-001-<hash8>/  (JSON + float64 .npy + fixtures)
serve:  Python — stitcher → embedding + lexical → semantic → fusion → Platt
        → thresholds → decision policy;  §5 request/response contract
```

## Setup

```sh
export GEMINI_API_KEY=...          # https://aistudio.google.com/apikey
# corpus (once): git clone https://github.com/cricketclub/gridspace-stanford-harper-valley data/raw/harper-valley
```

## Run

```sh
cd trainer
uv run poc ingest        # HVB -> data/gold/gold.jsonl
uv run poc partition     # splits + manifest + frozen-test hash
uv run poc train         # embed (cached) -> 3 models -> calibrate -> thresholds -> artifact

uv run pytest            # unit tests, incl. stitcher
uv run python ../scripts/serve_py.py &     # POST /classify on :8081, loads newest artifact
uv run poc replay        # stream test conversations through the server,
                         # compare with the Python offline forward pass
```

Tests: `uv run pytest` (trainer).

Decisions: `accepted | multi_accepted | no_supported_intent | abstained`;
infrastructure failures return `degraded`, never a semantic decision.
Every numeric knob lives in `configs/*.yaml`; changed configs or thresholds
mean a new artifact. Raw ASR text is never cleaned.

## Docs

- **[POC-GUIDE.md](POC-GUIDE.md)** — build it on your own data (start here).
- **[docs/office-playbook.md](docs/office-playbook.md)** — the full method + reasoning.
- **[docs/feasibility-verdict.md](docs/feasibility-verdict.md)** — this rehearsal's result (GREEN, 0.92 held-out).
- **[docs/design-doc.md](docs/design-doc.md)** — the work spec this rehearses.
- **[port-template/](port-template/)** — fill-in-the-blanks stubs + sample data + smoke test.
- [docs/synthetic-data-plan.md](docs/synthetic-data-plan.md) — record of the synthetic-data experiment (tried, null result, dropped).

Serve: `cd trainer && uv run python ../scripts/serve_py.py`.
