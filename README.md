# semantic-poc — non-generative intent classification

Classifies one customer turn (in the context of the agent's preceding turn) into
a fixed intent taxonomy, at a precommitted precision floor — a cheap,
deterministic, auditable alternative to a per-utterance generative LLM call.

A frozen Gemini embedding does the language understanding; everything trained is
a stack of small linear models on top:

```
turn → stitch → context string → embed (frozen, cached)
     → semantic head + lexical head → fusion → Platt calibration
     → per-intent certified thresholds → decision policy
```

Decisions: `accepted` · `multi_accepted` · `abstained` · `no_supported_intent`.
Infrastructure failures return `degraded`, never a semantic decision. Every
numeric knob lives in `configs/*.yaml`; a changed config means a new artifact.
Raw ASR text is never cleaned.

## Status

Validated as a rehearsal on a public sample corpus — verdict GREEN, 0.92
held-out coverage at the 0.90 precision floor. **Currently being ported to our
production corpus**; the rehearsal data path is being replaced.

## Setup

```sh
cd trainer
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .                                 # registers the `poc` command
export POC_GOLD_PATH=../data/gold/gold.duckdb    # every session
export GEMINI_API_KEY=...                        # every session
```

Requires Python 3.12 (see `trainer/.python-version`). `pytest` from `trainer/`
runs the unit tests — pure functions, no API key needed.

## Run

```sh
poc ingest        # corpus -> gold store
poc counts        # label balance + splits. Free, no API call
poc partition     # conversation-grouped, time-ordered splits; freezes test
poc train         # embed -> heads -> fusion -> calibrate -> threshold -> artifact

python ../scripts/serve_py.py &   # POST /classify on :8081, loads newest artifact
poc replay                        # stream test turns through it, diff vs offline
```

`poc --help` lists everything. Pipeline stages run through `poc`; side tools in
`scripts/` run with `python`, e.g. `python ../scripts/held_out_eval.py`.

## Docs

**[AGENTS.md](AGENTS.md)** — start here. Operational truth: architecture, the
`GoldRow` data contract, the port task, setup, commands, labeling lifecycle, and
the rules that must not be broken.

**[docs/method-guide.md](docs/method-guide.md)** — the method: how much data and
of what, the labeling standard, splits hygiene, certification, scaling to ~50
intents, rollout, and the checklist of things the rehearsal tripped on.

**[docs/model-design.md](docs/model-design.md)** — how the model works, layer by
layer, with a worked example.
