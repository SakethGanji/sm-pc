"""Gold dataset I/O — the single seam that knows how gold is stored on disk.

Everything else in the pipeline speaks `GoldRow` and never opens the gold file
itself (mirrors the ingest contract: "nothing downstream knows where the data
came from"). Because of that, the on-disk format is a swappable detail, chosen
here by file extension:

  .jsonl   — newline-delimited JSON. Human-readable, greppable, diff-friendly,
             trivially appendable. Best while hand-labeling / adjudicating and
             through a data-shape-still-moving port.
  .parquet — columnar, compressed (~10x smaller), typed, with column/predicate
             pushdown on read. Best for large, read-mostly corpora. Read and
             written via DuckDB (no pandas/pyarrow dependency).

Row order is preserved EXACTLY across formats. Parquet carries a hidden `_ord`
column and reads always `ORDER BY _ord`, so a parallel or multi-file scan still
reconstructs the original order. This matters: TF-IDF fitting and the
conversation-grouped CV folds are order-sensitive, so an unordered scan would
silently change the trained model at scale.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .schema import GoldRow

_ORD = "_ord"  # hidden ordering column, internal to the columnar representations
_BATCH = 10_000
_GOLD_TABLE = "gold"          # the rows, inside a .duckdb store
_AUDIT_TABLE = "label_audit"  # in-file relabel history (id, ts, old, new, note)


def content_sha256(path: str | Path) -> str:
    """Hash the gold CONTENT (canonical row JSON, in order), not the file bytes.

    Format-independent by design: JSONL and Parquet holding identical rows yield
    the same digest, so the artifact provenance — and the artifact's own content
    hash — stay stable across a format switch. Streams, so it is O(1) memory.
    """
    h = hashlib.sha256()
    for r in iter_gold(path):
        h.update(r.model_dump_json().encode())
        h.update(b"\n")
    return h.hexdigest()


def load_gold(path: str | Path) -> list[GoldRow]:
    """Load the whole gold set into memory, in original order."""
    return list(iter_gold(path))


def iter_gold(path: str | Path):
    """Stream gold rows in original order (constant memory)."""
    path = Path(path)
    if path.suffix == ".parquet":
        yield from _iter_parquet(path)
    elif path.suffix == ".duckdb":
        yield from _iter_duckdb(path)
    else:
        yield from _iter_jsonl(path)


def save_gold(rows, path: str | Path) -> None:
    """Persist rows atomically (temp file + rename — never a half-written gold)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = {".parquet": _write_parquet, ".duckdb": _write_duckdb}.get(
        path.suffix, _write_jsonl)
    _atomic(path, lambda tmp: writer(rows, tmp))


# --- JSONL -----------------------------------------------------------------

def _iter_jsonl(path: Path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield GoldRow.model_validate_json(line)


def _write_jsonl(rows, tmp: Path) -> None:
    with open(tmp, "w") as f:
        for r in rows:
            f.write(r.model_dump_json() + "\n")


# --- Parquet (via DuckDB) --------------------------------------------------

def _iter_parquet(path: Path):
    import duckdb

    con = duckdb.connect()
    src = f"read_parquet('{path.as_posix()}')"
    cols = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {src}").fetchall()]
    order = f"ORDER BY {_ORD}" if _ORD in cols else ""
    keep = [c for c in cols if c != _ORD]
    sel = ", ".join(f'"{c}"' for c in keep)
    cur = con.execute(f"SELECT {sel} FROM {src} {order}")
    while True:
        batch = cur.fetchmany(_BATCH)
        if not batch:
            break
        for r in batch:
            yield GoldRow.model_validate(dict(zip(keep, r)))


def _bridge_jsonl(rows, dir_: Path) -> Path:
    """Dump rows to a temp JSONL with an added `_ord`, so DuckDB's read_json_auto
    infers the nested schema (prev_caller_segment struct, list columns) for free.
    Shared by the Parquet and DuckDB writers — keeps duckdb the only extra dep."""
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", dir=dir_, delete=False) as jf:
        bridge = Path(jf.name)
        for i, r in enumerate(rows):
            obj = r.model_dump()
            obj[_ORD] = i
            jf.write(json.dumps(obj) + "\n")
    return bridge


def _write_parquet(rows, tmp: Path) -> None:
    """Write rows to Parquet, adding `_ord` to preserve their exact order."""
    import duckdb

    bridge = _bridge_jsonl(rows, tmp.parent)
    try:
        duckdb.connect().execute(
            f"COPY (SELECT * FROM read_json_auto('{bridge.as_posix()}', "
            f"maximum_object_size=100000000)) TO '{tmp.as_posix()}' (FORMAT PARQUET)"
        )
    finally:
        bridge.unlink(missing_ok=True)


# --- DuckDB database file (mutable working store) ---------------------------

def _iter_duckdb(path: Path):
    import duckdb

    con = duckdb.connect(str(path), read_only=True)
    try:
        cols = [r[0] for r in con.execute(f"DESCRIBE {_GOLD_TABLE}").fetchall()]
        order = f"ORDER BY {_ORD}" if _ORD in cols else ""
        keep = [c for c in cols if c != _ORD]
        sel = ", ".join(f'"{c}"' for c in keep)
        cur = con.execute(f"SELECT {sel} FROM {_GOLD_TABLE} {order}")
        while True:
            batch = cur.fetchmany(_BATCH)
            if not batch:
                break
            for r in batch:
                yield GoldRow.model_validate(dict(zip(keep, r)))
    finally:
        con.close()


def _write_duckdb(rows, tmp: Path) -> None:
    """Write rows into a fresh .duckdb database: a `gold` table (with `_ord`) plus
    an empty `label_audit` history table, so relabels are auditable in-file."""
    import duckdb

    tmp.unlink(missing_ok=True)  # mkstemp left an empty file; let duckdb create the db
    bridge = _bridge_jsonl(rows, tmp.parent)
    try:
        con = duckdb.connect(str(tmp))
        con.execute(
            f"CREATE TABLE {_GOLD_TABLE} AS SELECT * FROM read_json_auto("
            f"'{bridge.as_posix()}', maximum_object_size=100000000)"
        )
        con.execute(
            f"CREATE TABLE {_AUDIT_TABLE} (id VARCHAR, ts VARCHAR, "
            f"old_labels VARCHAR, new_labels VARCHAR, note VARCHAR)"
        )
        con.execute("CHECKPOINT")
        con.close()
    finally:
        bridge.unlink(missing_ok=True)


def relabel(path: str | Path, row_id: str, new_labels: list[str], note: str = "") -> None:
    """Update one row's labels in a .duckdb gold store and record the change in
    label_audit — same transaction, so history can never drift from the data.
    row_id is 'conversation_id#turn_index'."""
    import duckdb

    conv, _, turn = row_id.rpartition("#")
    con = duckdb.connect(str(path))
    try:
        con.execute("BEGIN")
        old = con.execute(
            f"SELECT labels FROM {_GOLD_TABLE} WHERE conversation_id=? AND turn_index=?",
            [conv, int(turn)]).fetchone()
        if old is None:
            raise KeyError(f"no such row: {row_id}")
        con.execute(
            f"UPDATE {_GOLD_TABLE} SET labels=? WHERE conversation_id=? AND turn_index=?",
            [new_labels, conv, int(turn)])
        con.execute(
            f"INSERT INTO {_AUDIT_TABLE} VALUES (?, now()::VARCHAR, ?, ?, ?)",
            [row_id, json.dumps(old[0]), json.dumps(new_labels), note])
        con.execute("COMMIT")
        con.execute("CHECKPOINT")
    finally:
        con.close()


# --- working-store verbs (snapshot / promote / counts) ---------------------

def snapshot(src: str | Path, label: str = "") -> Path:
    """Export a point-in-time Parquet copy of the gold store into a `snapshots/`
    dir next to it: gold-<label>-<ts>.parquet. A restore point / open-format
    archive / frozen-for-reproducibility training input — regenerable, NOT a
    second source of truth. Streams, so it's cheap on a large store."""
    src = Path(src)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = f"{label}-" if label else ""
    dst = src.parent / "snapshots" / f"gold-{tag}{ts}.parquet"
    save_gold(iter_gold(src), dst)
    return dst


def promote(path: str | Path, intent: str, row_ids, note: str = "") -> int:
    """Batch "easy add": add `intent` to the labels of every turn in row_ids
    (union — multi-intent turns keep their other labels), auditing each change,
    all in one transaction. row_ids are 'conversation_id#turn_index'. Returns how
    many rows changed (already-labeled rows are skipped). Requires a .duckdb store."""
    path = Path(path)
    if path.suffix != ".duckdb":
        raise ValueError("promote needs a mutable .duckdb store; label there, "
                         "then snapshot to Parquet")
    import duckdb

    con = duckdb.connect(str(path))
    changed = 0
    try:
        con.execute("BEGIN")
        for rid in row_ids:
            conv, _, turn = rid.rpartition("#")
            row = con.execute(
                f"SELECT labels FROM {_GOLD_TABLE} WHERE conversation_id=? AND turn_index=?",
                [conv, int(turn)]).fetchone()
            if row is None:
                raise KeyError(f"no such row: {rid}")
            old = list(row[0] or [])
            if intent in old:
                continue
            new = sorted(set(old) | {intent})
            con.execute(
                f"UPDATE {_GOLD_TABLE} SET labels=? WHERE conversation_id=? AND turn_index=?",
                [new, conv, int(turn)])
            con.execute(
                f"INSERT INTO {_AUDIT_TABLE} VALUES (?, now()::VARCHAR, ?, ?, ?)",
                [rid, json.dumps(old), json.dumps(new), note or f"promote {intent}"])
            changed += 1
        con.execute("COMMIT")
        con.execute("CHECKPOINT")
    except BaseException:
        con.execute("ROLLBACK")
        raise
    finally:
        con.close()
    return changed


def counts(path: str | Path) -> dict:
    """Label balance at a glance: positives per intent, OOS/labeled/total,
    ambiguous, and a per-split breakdown. Backend-agnostic (streams)."""
    per, by_split = Counter(), Counter()
    total = oos = amb = 0
    for r in iter_gold(path):
        total += 1
        by_split[r.split or "(unset)"] += 1
        if r.ambiguous:
            amb += 1
        if r.labels:
            per.update(r.labels)
        else:
            oos += 1
    return {"total": total, "oos": oos, "labeled": total - oos, "ambiguous": amb,
            "per_intent": dict(per.most_common()), "by_split": dict(by_split)}


# --- atomic write ----------------------------------------------------------

def _atomic(path: Path, write_fn) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=path.suffix + ".tmp")
    os.close(fd)
    tmp = Path(tmp)
    try:
        write_fn(tmp)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
