"""Immutable, hash-named artifact of language-neutral files (JSON + .npy).
The Go server loads exactly one of these; any parameter change = new artifact."""

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from .config import config_hash, file_sha256, load_config
from .paths import ARTIFACTS_DIR, GOLD_DIR
from .train import Trained

NPY_FILES = {
    "sem_coef": "sem_coef.npy", "sem_int": "sem_int.npy",
    "lex_coef": "lex_coef.npy", "lex_int": "lex_int.npy",
    "fus_coef": "fus_coef.npy", "fus_int": "fus_int.npy",
}


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def export(models: Trained, families: dict[str, str], policy_cfg: dict,
           eval_summary: dict, fixtures: list[dict]) -> Path:
    tmp = ARTIFACTS_DIR / "artifact-tmp"
    tmp.mkdir(parents=True, exist_ok=True)

    for attr, fname in NPY_FILES.items():
        np.save(tmp / fname, getattr(models, attr).astype(np.float64))
    np.save(tmp / "tfidf_idf.npy", models.tfidf.idf.astype(np.float64))
    (tmp / "tfidf_vocab.json").write_text(json.dumps(models.tfidf.vocab))

    model_json = {
        "intents": models.intents,
        "families": families,
        "platt_a": models.platt_a.tolist(),
        "platt_b": models.platt_b.tolist(),
        "thresholds": models.thresholds,
        "tau_low": policy_cfg["tau_low"],
        "max_accepted": policy_cfg["max_accepted"],
        "exclusive_groups": policy_cfg["exclusive_groups"],
        "meta_features": ["token_count_over_100", "context_available"],
        "sem_c": models.sem_cs,
        "lex_c": models.lex_cs,
    }
    (tmp / "model.json").write_text(json.dumps(model_json, indent=1))
    feat_cfg = load_config("features")
    runtime_cfg = {
        "stitcher": load_config("stitcher"),
        "context": feat_cfg["context"],
    }
    (tmp / "runtime_config.json").write_text(json.dumps(runtime_cfg, indent=1))
    (tmp / "fixtures.json").write_text(json.dumps(fixtures, indent=1))
    (tmp / "eval.json").write_text(json.dumps(eval_summary, indent=2))

    manifest = {
        "name": "poc-hvb-001",
        "git_sha": _git_sha(),
        "gold_sha256": file_sha256(GOLD_DIR / "gold.jsonl"),
        "embedding": load_config("embedding"),
        "taxonomy_version": load_config("taxonomy")["version"],
        "config_hashes": {n: config_hash(load_config(n))
                          for n in ["taxonomy", "features", "policy", "embedding"]},
    }
    (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2))

    h = hashlib.sha256()
    for f in sorted(tmp.iterdir()):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    final = ARTIFACTS_DIR / f"artifact-poc-hvb-001-{h.hexdigest()[:8]}"
    if final.exists():
        for f in (ARTIFACTS_DIR / "artifact-tmp").iterdir():
            f.unlink()
        tmp.rmdir()
    else:
        tmp.rename(final)
    return final
