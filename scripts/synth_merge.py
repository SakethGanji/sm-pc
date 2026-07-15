"""Join generated cells with blind verdicts, keep agreements, ASR-corrupt,
dedup, and append as train-only gold rows (docs/synthetic-data-plan.md
stages 2-4). Baseline gold is backed up to gold.real.jsonl first."""

import json
import random
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "trainer"))

from semantic_poc.paths import GOLD_DIR
from semantic_poc.schema import GoldRow

SYNTH_DIR = Path("/tmp/synth")

HOMOPHONES = {
    "card": ["car", "cart"], "cards": ["cars"], "checks": ["chess", "check"],
    "checkbook": ["check book"], "check": ["czech"], "bill": ["bell"],
    "bills": ["bells"], "branch": ["ranch"], "hours": ["ours"],
    "money": ["many"], "lost": ["last"], "password": ["pass word"],
    "balance": ["balances"], "transfer": ["transferred"], "pay": ["hey"],
    "account": ["a count"], "appointment": ["appointments"],
}
FILLERS = ["uh", "um", "you know", "like"]


def corrupt(text: str, rng: random.Random) -> str:
    words = text.split()
    out = []
    substituted = False
    for w in words:
        if w in HOMOPHONES and rng.random() < 0.35 and not substituted:
            out.append(rng.choice(HOMOPHONES[w]))
            substituted = True
        elif rng.random() < 0.04 and len(words) > 4:
            continue  # deletion
        else:
            out.append(w)
    if rng.random() < 0.4:
        out.insert(rng.randrange(len(out) + 1), rng.choice(FILLERS))
    return " ".join(out)


def norm(text: str) -> frozenset:
    return frozenset(re.findall(r"[a-z0-9']+", text.lower()))


def main() -> None:
    gold_rows = [json.loads(l) for l in open(GOLD_DIR / "gold.jsonl")]
    if any(r["conversation_id"].startswith("synth-") for r in gold_rows):
        print("gold.jsonl already contains synthetic rows — aborting")
        sys.exit(1)
    shutil.copy(GOLD_DIR / "gold.jsonl", GOLD_DIR / "gold.real.jsonl")
    real_texts = {r["raw_transcript"] for r in gold_rows}

    survivors, stats = [], {}
    for gen_path in sorted(SYNTH_DIR.glob("gen_*.jsonl")):
        cell = gen_path.stem[len("gen_"):]
        verdict_path = SYNTH_DIR / f"verdict_{cell}.jsonl"
        if not verdict_path.exists():
            stats[cell] = "NO VERDICT FILE"
            continue
        gen = {r["idx"]: r for r in map(json.loads, open(gen_path)) if "idx" in r}
        verdicts = {r["idx"]: set(r["labels"]) for r in map(json.loads, open(verdict_path))}
        kept = dropped_label = dropped_dup = 0
        seen: list[frozenset] = []
        for idx, row in gen.items():
            if idx not in verdicts or set(row.get("labels", [])) != verdicts[idx]:
                dropped_label += 1
                continue
            text = row["text"].strip().lower()
            key = norm(text)
            if text in real_texts or any(
                len(key & s) / max(1, len(key | s)) > 0.85 for s in seen
            ):
                dropped_dup += 1
                continue
            seen.append(key)
            survivors.append({**row, "text": text, "cell": cell})
            kept += 1
        stats[cell] = f"kept {kept}, label-mismatch {dropped_label}, dup {dropped_dup}"

    for cell, s in sorted(stats.items()):
        print(f"  {cell:28s} {s}")

    new_rows = []
    for n, row in enumerate(survivors):
        rng = random.Random(n)
        variants = [row["text"], corrupt(row["text"], rng)]
        for v_i, text in enumerate(dict.fromkeys(variants)):  # skip no-op corruption dupes
            new_rows.append(GoldRow(
                conversation_id=f"synth-b1-{n}-{v_i}",
                turn_index=1,
                timestamp_ms=0,
                raw_transcript=text,
                human_transcript=row["text"],
                previous_agent_utterance=row.get("agent_said", "") or "",
                dialog_acts=[f"synthetic:{row['cell']}"],
                session_task_intent=(row["labels"][0] if row["labels"] else "no_intent"),
                labels=row["labels"],
                split="train",
            ))

    with open(GOLD_DIR / "gold.jsonl", "a") as f:
        for r in new_rows:
            f.write(r.model_dump_json() + "\n")
    n_pos = sum(1 for r in new_rows if r.labels)
    print(f"\nsurvivors {len(survivors)} -> {len(new_rows)} train rows appended "
          f"({n_pos} positive, {len(new_rows) - n_pos} negative)")
    print("baseline preserved at data/gold/gold.real.jsonl")


if __name__ == "__main__":
    main()
