"""Bucket-2 synthetic batch (docs/next-session.md step 4): wrap LLM-generated
core requests in the observed call-opening pattern — [noise] tokens, name
preambles, garbled agent greetings — plus ASR corruption. Emits verification
batches (same format as the adjudication workflow) + an intended-labels sidecar;
merge happens after blind-critic verification in synth_b2_merge.py.

Usage: python synth_b2.py <workflow_journal.jsonl>
"""

import json
import random
import re
import sys
from pathlib import Path

OUT = Path("/tmp/synth-b2")
OUT.mkdir(exist_ok=True)
BATCH = 50
SEED = 20260715

FIRST = ["mary", "james", "john", "patricia", "robert", "jennifer", "michael",
         "linda", "elizabeth", "david", "susan", "richard", "jessica", "william"]
LAST = ["smith", "johnson", "williams", "brown", "jones", "garcia", "miller",
        "davis", "rodriguez", "martinez", "wilson", "anderson"]

GREETINGS = [
    "hello this is harper valley national bank my name is {a} how can i help you today",
    "hello this is a upper valley national bank my name is {a} how can i help you",
    "hello this is how providing national bank my name is {a} how can i help you today",
    "hello thank you for calling harper valley national bank my name is {a}",
    "hello yes ella harper valley national bank {a} how can i help you today",
    "harper valley national bank how can i help you",
    "how can i help you today",
    "my name is {a} how can i help you today",
    "mm hmm",
    "",
]

WOULD_LIKE = ["will like to", "will lie to", "don't like to", "will let me", "would love to"]
HOMOPHONES = {
    "card": ["car", "cart"], "cards": ["cars"], "checks": ["chess", "check"],
    "checkbook": ["check book"], "check": ["czech"], "bill": ["bell"],
    "bills": ["bells"], "branch": ["ranch"], "hours": ["ours", "showers"],
    "money": ["many"], "lost": ["last"], "password": ["pass word"],
    "balance": ["balances"], "transfer": ["transferred"], "pay": ["hey"],
    "account": ["a count"], "appointment": ["appointments"],
}
FILLERS = ["uh", "um", "you know"]


def corrupt(text: str, rng: random.Random) -> str:
    if "would like to" in text and rng.random() < 0.35:
        text = text.replace("would like to", rng.choice(WOULD_LIKE), 1)
    words = text.split()
    out, substituted = [], False
    for w in words:
        if w in HOMOPHONES and not substituted and rng.random() < 0.25:
            out.append(rng.choice(HOMOPHONES[w]))
            substituted = True
        elif rng.random() < 0.03 and len(words) > 5:
            continue
        else:
            out.append(w)
    if rng.random() < 0.3:
        out.insert(rng.randrange(len(out) + 1), rng.choice(FILLERS))
    return " ".join(out)


def wrap(core: str, rng: random.Random) -> str:
    name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
    pre = rng.choice([f"hi my name is {name}", f"hi my name is {name}",
                      f"my name is {name}", f"hi my name's {name}",
                      f"hi this is {name}", f"[noise] hi my name is {name}"])
    text = f"{pre} {core}"
    if rng.random() < 0.25 and not text.startswith("[noise]"):
        text = f"[noise] {text}"
    return text


def main(result_path: str) -> None:
    # result_path is the workflow's saved task output JSON: {result: {cells, negatives}}.
    doc = json.load(open(result_path))
    res = doc["result"] if "result" in doc else doc
    cells = [(c["cell"], c["requests"]) for c in res["cells"]]
    negatives = res["negatives"]

    assert cells and negatives, f"parse failed: {len(cells)} cells, {len(negatives)} negatives"
    print("cells:", [(c, len(r)) for c, r in cells], "negatives:", len(negatives))

    rows, intended = [], {}
    rng = random.Random(SEED)

    def emit(text: str, agent: str, labels: list[str], kind: str) -> None:
        rid = f"b2-{len(rows):04d}"
        rows.append({"id": rid, "agent": agent, "caller": text})
        intended[rid] = {"labels": labels, "kind": kind}

    for cell, reqs in cells:
        for core in reqs:
            core = re.sub(r"[^a-z0-9' \[\]]", "", core.lower()).strip()
            agent_g = rng.choice(GREETINGS).format(a=rng.choice(FIRST))
            emit(corrupt(wrap(core, rng), rng), agent_g, [cell], "wrapped")
            if rng.random() < 0.6:  # second variant, different wrapping draw
                agent_g2 = rng.choice(GREETINGS).format(a=rng.choice(FIRST))
                emit(corrupt(wrap(core, rng), rng), agent_g2, [cell], "wrapped")

    for core in negatives:
        core = re.sub(r"[^a-z0-9' \[\]]", "", core.lower()).strip()
        agent_g = rng.choice(GREETINGS).format(a=rng.choice(FIRST))
        text = wrap(core, rng) if rng.random() < 0.5 else core
        emit(corrupt(text, rng), agent_g, [], "hard_negative")

    for _ in range(45):  # preamble-only openings: the pure no-request shell
        name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        text = rng.choice([f"hi my name is {name}", f"[noise] hi my name is {name}",
                           f"hi my name is {name} how are you doing today",
                           f"hello this is {name} calling", f"hi my name's {name}"])
        agent_g = rng.choice(GREETINGS).format(a=rng.choice(FIRST))
        emit(corrupt(text, rng), agent_g, [], "preamble_only")

    for old in OUT.glob("batch_*.jsonl"):
        old.unlink()
    for i in range(0, len(rows), BATCH):
        with open(OUT / f"batch_{i // BATCH:03d}.jsonl", "w") as f:
            for r in rows[i : i + BATCH]:
                f.write(json.dumps(r) + "\n")
    json.dump(intended, open(OUT / "intended.json", "w"))
    n_pos = sum(1 for v in intended.values() if v["labels"])
    print(f"{len(rows)} wrapped rows ({n_pos} positive) in "
          f"{(len(rows) + BATCH - 1) // BATCH} batches under {OUT}")


if __name__ == "__main__":
    main(sys.argv[1])
