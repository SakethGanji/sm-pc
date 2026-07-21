"""Canonical filesystem locations, derived from the repo root. Import these
rather than hard-coding paths anywhere else."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]   # .../semantic-poc
CONFIGS = REPO_ROOT / "configs"                    # YAML configs
RAW_HVB = REPO_ROOT / "data" / "raw" / "harper-valley" / "data"  # PORT: repoint at your raw corpus (data/raw is gitignored)
GOLD_DIR = REPO_ROOT / "data" / "gold"             # gold store lives here (gold.duckdb / .jsonl / .parquet)
CACHE_DIR = REPO_ROOT / "data" / "cache"           # embeddings.sqlite (embedding cache — reused across runs)
ARTIFACTS_DIR = REPO_ROOT / "artifacts"            # hash-named trained artifacts
