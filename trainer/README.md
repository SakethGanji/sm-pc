# trainer — the pipeline package

`semantic_poc/` is the pipeline library. `scripts/` (repo root) holds standalone
side-tools. Two different ways to run things:

| You run… | with… | e.g. |
|---|---|---|
| a **pipeline stage** (in `semantic_poc/`) | the **`poc`** command | `poc ingest`, `poc train` |
| a **side-tool** (in `../scripts/`) | `python` | `python ../scripts/held_out_eval.py` |

**Entry point:** `poc` is a console command on your PATH (installed from
`pyproject.toml`: `poc = "semantic_poc.cli:app"` → `semantic_poc/cli.py`). You
never invoke a package module by path — `python semantic_poc/ingest.py` breaks its
relative imports. Always go through `poc`.

```bash
# setup: see ../README.md; then, from this dir:
source .venv/bin/activate
export POC_GOLD_PATH=../data/gold/gold.duckdb   # canonical store (../docs/gold-schema.md)
poc --help
```

## Data flow
```
your CSV ──ingest──▶ gold.duckdb ──partition──▶ splits ──train──▶ artifact ──serve
                        ▲  (label with: poc promote / relabel)
```

## Module map (`semantic_poc/`)
| Role | Modules |
|---|---|
| **Contract** | `schema.py` — `GoldRow`, the one type everything speaks |
| **Data in / I/O** | `ingest.py` (your corpus → `GoldRow`), `store.py` (gold I/O seam: jsonl/parquet/duckdb + snapshot/promote/counts), `partitions.py` (splits), `paths.py`, `config.py` |
| **Featurize** | `context.py` (`build_c1`), `embeddings.py` (Gemini, cached), `features.py` (TF-IDF) |
| **Train** | `train.py` (per-intent heads → fusion → Platt → thresholds), `fixtures.py` |
| **Serve / decide** | `runtime.py` (`Scorer`), `policy.py` (`decide`), `stitch.py` |
| **Out** | `artifact.py` (immutable, hash-named artifact) |
| **Entry** | `cli.py` (the `poc` commands) |
| **HVB-only** | `weak_labels.py` — rehearsal weak labels; drop at port |

## `poc` commands
`ingest` · `counts` · `promote` · `snapshot` · `partition` · `train` ·
`adjudicate` · `replay` — see `poc --help`.

## Porting to your data
Rewrite `ingest.py` (CSV → `GoldRow` per customer turn, `labels=[]`), set
`../configs/taxonomy.yaml` + `policy.yaml`, then `poc ingest`. Everything
downstream is data-agnostic. Full guide: `../docs/gold-schema.md` (schema +
how to add/label rows) and `../POC-GUIDE.md`.
