"""Config loading + hashing. Configs are YAML in ../configs (taxonomy, policy,
features, embedding, stitcher). Their hashes go into the artifact manifest, so a
changed config means a new artifact — that's the provenance guarantee."""

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from .paths import CONFIGS


def load_config(name: str) -> dict[str, Any]:
    """Load configs/<name>.yaml, e.g. load_config("taxonomy")."""
    with open(CONFIGS / f"{name}.yaml") as f:
        return yaml.safe_load(f)


def config_hash(cfg: dict[str, Any]) -> str:
    """Short stable digest of a config dict (for the artifact manifest)."""
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
