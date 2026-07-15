import json
from collections import Counter

import typer

from .config import load_config
from .paths import GOLD_DIR, RAW_HVB
from .schema import GoldRow

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)

GOLD_PATH = GOLD_DIR / "gold.jsonl"
MANIFEST_PATH = GOLD_DIR / "partitions.json"


def load_gold() -> list[GoldRow]:
    with open(GOLD_PATH) as f:
        return [GoldRow.model_validate_json(line) for line in f]


def save_gold(rows: list[GoldRow]) -> None:
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    with open(GOLD_PATH, "w") as f:
        for r in rows:
            f.write(r.model_dump_json() + "\n")


@app.command()
def ingest() -> None:
    """HVB corpus -> data/gold/gold.jsonl with weak turn labels."""
    from .ingest import ingest_corpus

    taxonomy = load_config("taxonomy")
    rows = ingest_corpus(RAW_HVB, taxonomy["task_type_map"])
    save_gold(rows)
    label_counts = Counter(l for r in rows for l in r.labels)
    n_labeled = sum(1 for r in rows if r.labels)
    typer.echo(f"gold rows: {len(rows)} caller segments from "
               f"{len({r.conversation_id for r in rows})} conversations")
    typer.echo(f"labeled (request-bearing): {n_labeled}; no-intent: {len(rows) - n_labeled}")
    for intent, c in label_counts.most_common():
        typer.echo(f"  {intent:22s} {c}")


@app.command()
def train() -> None:
    """Embed, train branches + fusion, calibrate, threshold, export artifact."""
    import numpy as np

    from .artifact import export
    from .embeddings import GeminiEmbedder
    from .features import TfIdf
    from .fixtures import build_fixtures
    from .runtime import Scorer
    from .train import (build_split, fit_branch_full, fit_fusion, fit_platt,
                        fused_logits, pick_thresholds, sweep_and_oof)
    from .train import Trained

    import os

    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        typer.echo("GEMINI_API_KEY is not set — real Gemini embeddings are required "
                   "(get a key at https://aistudio.google.com/apikey)")
        raise typer.Exit(1)

    taxonomy = load_config("taxonomy")
    feat_cfg = load_config("features")
    emb_cfg = load_config("embedding")
    policy_cfg = load_config("policy")
    intents = list(taxonomy["intents"].keys())
    families = {i: v["family"] for i, v in taxonomy["intents"].items()}

    rows = [r for r in load_gold() if not r.ambiguous]
    parts = {s: [r for r in rows if r.split == s] for s in ["train", "calibration", "policy"]}
    typer.echo(f"rows: train={len(parts['train'])} cal={len(parts['calibration'])} "
               f"policy={len(parts['policy'])} (ambiguous excluded)")

    typer.echo("fitting TF-IDF on train ...")
    tfidf = TfIdf.fit([r.raw_transcript for r in parts["train"]],
                      min_df=feat_cfg["tfidf"]["min_df"],
                      max_vocab=feat_cfg["tfidf"]["max_vocab"])
    typer.echo(f"vocab size: {len(tfidf.vocab)}")

    embedder = GeminiEmbedder(emb_cfg)
    typer.echo("embedding splits (cache-first, Gemini for misses) ...")
    splits = {name: build_split(rs, intents, tfidf, embedder) for name, rs in parts.items()}
    tr = splits["train"]

    models = Trained(intents=intents, tfidf=tfidf)
    typer.echo("semantic branch: per-intent C sweep on conversation-grouped folds")
    models.sem_cs, oof_sem = sweep_and_oof(tr.X_sem, tr.y, tr.groups, intents)
    typer.echo("lexical branch: per-intent C sweep")
    models.lex_cs, oof_lex = sweep_and_oof(tr.X_lex, tr.y, tr.groups, intents)

    typer.echo("fusion on out-of-fold logits; branches retrained on full train")
    models.fus_coef, models.fus_int = fit_fusion(oof_sem, oof_lex, tr.meta, tr.y, intents)
    models.sem_coef, models.sem_int = fit_branch_full(tr.X_sem, tr.y, intents, models.sem_cs)
    models.lex_coef, models.lex_int = fit_branch_full(tr.X_lex, tr.y, intents, models.lex_cs)

    typer.echo("Platt calibration on calibration split")
    models.platt_a, models.platt_b = fit_platt(
        fused_logits(models, splits["calibration"]), splits["calibration"].y, intents)

    typer.echo("thresholds on policy split")
    from .runtime import sigmoid
    pol = splits["policy"]
    probs = sigmoid(models.platt_a * fused_logits(models, pol) + models.platt_b)
    models.thresholds = pick_thresholds(probs, pol.y, intents,
                                        policy_cfg["precision_floor"])

    eval_summary = {"per_intent": {}}
    for k, intent in enumerate(intents):
        t = models.thresholds[intent]
        sel = probs[:, k] >= t
        y = pol.y[:, k]
        eval_summary["per_intent"][intent] = {
            "threshold": t,
            "accepted": int(sel.sum()),
            "precision": round(float(y[sel].mean()), 4) if sel.sum() else None,
            "recall": round(float(sel[y == 1].mean()), 4) if y.sum() else None,
            "positives_in_policy": int(y.sum()),
        }
    typer.echo(json.dumps(eval_summary, indent=1))

    scorer = Scorer(
        intents=intents, families=families, tfidf=tfidf,
        sem_coef=models.sem_coef, sem_int=models.sem_int,
        lex_coef=models.lex_coef, lex_int=models.lex_int,
        fus_coef=models.fus_coef, fus_int=models.fus_int,
        platt_a=models.platt_a, platt_b=models.platt_b,
        thresholds=models.thresholds, tau_low=policy_cfg["tau_low"],
        max_accepted=policy_cfg["max_accepted"],
        exclusive_groups=policy_cfg["exclusive_groups"],
    )
    fixtures = build_fixtures(pol.rows, pol.X_sem, scorer)
    path = export(models, families, policy_cfg, eval_summary, fixtures)
    typer.echo(f"artifact: {path}")


@app.command()
def partition() -> None:
    """Assign conversation-grouped time-ordered splits; write manifest."""
    from .partitions import assign_splits

    rows = load_gold()
    manifest = assign_splits(rows)
    save_gold(rows)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    typer.echo(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    app()
