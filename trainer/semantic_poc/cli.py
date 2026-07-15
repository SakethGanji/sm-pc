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
