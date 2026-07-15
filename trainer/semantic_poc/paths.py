from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS = REPO_ROOT / "configs"
RAW_HVB = REPO_ROOT / "data" / "raw" / "harper-valley" / "data"
GOLD_DIR = REPO_ROOT / "data" / "gold"
CACHE_DIR = REPO_ROOT / "data" / "cache"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
