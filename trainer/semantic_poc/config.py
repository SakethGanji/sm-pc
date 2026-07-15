import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from .paths import CONFIGS


def load_config(name: str) -> dict[str, Any]:
    with open(CONFIGS / f"{name}.yaml") as f:
        return yaml.safe_load(f)


def config_hash(cfg: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
