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
cd trainer
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .                   # registers the `poc` command
export GEMINI_API_KEY=...          # https://aistudio.google.com/apikey
# corpus (once): git clone https://github.com/cricketclub/gridspace-stanford-harper-valley ../data/raw/harper-valley
```

Requires Python 3.12 (see `trainer/.python-version`).

## Run

```sh
cd trainer
source .venv/bin/activate          # if not already active

poc ingest        # HVB -> data/gold/gold.jsonl
poc partition     # splits + manifest + frozen-test hash
poc train         # embed (cached) -> 3 models -> calibrate -> thresholds -> artifact

pytest             # unit tests, incl. stitcher
python ../scripts/serve_py.py &     # POST /classify on :8081, loads newest artifact
poc replay        # stream test conversations through the server,
                  # compare with the Python offline forward pass
```

Tests: `pytest` (trainer, venv activated).

Decisions: `accepted | multi_accepted | no_supported_intent | abstained`;
infrastructure failures return `degraded`, never a semantic decision.
Every numeric knob lives in `configs/*.yaml`; changed configs or thresholds
mean a new artifact. Raw ASR text is never cleaned.

## Docs

- **[docs/QUICKSTART.md](docs/QUICKSTART.md)** — zero to your first result: setup, the files to edit, commands, how much data (do-it-now, start here).
- **[POC-GUIDE.md](POC-GUIDE.md)** — the build/port guide: method, data, scaling, rollout (the *why*).
- **[docs/model-design.md](docs/model-design.md)** — how the model works: the layers, how each is trained, a worked example.
- **[docs/gold-schema.md](docs/gold-schema.md)** — gold DuckDB schema + how to add/label rows (the data contract).
- **[trainer/README.md](trainer/README.md)** — package map + how to run (`poc` vs scripts).
- **[docs/feasibility-verdict.md](docs/feasibility-verdict.md)** — the rehearsal's result (GREEN, 0.92 held-out) + caveats.
- **[docs/design-doc.md](docs/design-doc.md)** — the original work spec this rehearses.

Serve: `cd trainer && source .venv/bin/activate && python ../scripts/serve_py.py`.
