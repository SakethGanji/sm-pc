"""Fine-tuned-encoder challenger (design doc §12 growth path). Fine-tunes a
small transformer as an 8-way multi-label head on the SAME train split and
context format (C1) the LR-on-frozen-Gemini stack uses, then certifies
precision-floor thresholds on the SAME real policy split and reports coverage
at the 0.90 floor. Head-to-head number is directly comparable to the LR
baseline's in-sample policy coverage. Promotion bar: +3-5 pts at equal precision.

Usage: challenger-env/bin/python scripts/challenger.py [model_name]
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "trainer"))
from semantic_poc.cli import load_gold  # noqa: E402
from semantic_poc.config import load_config  # noqa: E402
from semantic_poc.context import build_c1  # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else "microsoft/deberta-v3-small"
FLOOR = 0.90
MAX_LEN = 64
EPOCHS = 6
BATCH = 16
LR = 2e-5
SEED = 20260715
DEVICE = "cuda"

torch.manual_seed(SEED)
np.random.seed(SEED)

taxonomy = load_config("taxonomy")
INTENTS = list(taxonomy["intents"].keys())
families = {i: v["family"] for i, v in taxonomy["intents"].items()}
policy_cfg = load_config("policy")
K = len(INTENTS)

rows = [r for r in load_gold() if not r.ambiguous]
# Synthetic rows are train-only; include them (parity with LR which trains on
# gold.jsonl incl. synth). Keep negatives — the model needs to learn to abstain.
parts = {s: [r for r in rows if r.split == s or (s == "train" and r.conversation_id.startswith("synth-"))]
         for s in ["train", "policy"]}
parts["train"] = [r for r in rows if r.split == "train"]

def encode_row(r):
    return build_c1(r.previous_agent_utterance, r.raw_transcript)

def labels_row(r):
    y = np.zeros(K, dtype=np.float32)
    for lab in r.labels:
        y[INTENTS.index(lab)] = 1.0
    return y

tok = AutoTokenizer.from_pretrained(MODEL)

class DS(Dataset):
    def __init__(self, rs):
        self.texts = [encode_row(r) for r in rs]
        self.y = np.array([labels_row(r) for r in rs])
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, i):
        enc = tok(self.texts[i], truncation=True, max_length=MAX_LEN,
                  padding="max_length", return_tensors="pt")
        return {"input_ids": enc["input_ids"][0],
                "attention_mask": enc["attention_mask"][0],
                "labels": torch.tensor(self.y[i])}

train_ds, pol_ds = DS(parts["train"]), DS(parts["policy"])
print(f"train rows={len(train_ds)} (pos turns={int(train_ds.y.sum())}), "
      f"policy rows={len(pol_ds)} (pos={int(pol_ds.y.sum())})")

# Positive class weighting: pos_weight = neg/pos per intent (rare positives).
pos = train_ds.y.sum(0)
neg = len(train_ds) - pos
pos_weight = torch.tensor(np.clip(neg / np.maximum(pos, 1), 1, 25), dtype=torch.float32).to(DEVICE)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL, num_labels=K, problem_type="multi_label_classification").to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=LR)
train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
sched = get_linear_schedule_with_warmup(opt, int(0.1 * len(train_dl) * EPOCHS),
                                        len(train_dl) * EPOCHS)
lossfn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)


@torch.no_grad()
def policy_probs():
    model.eval()
    dl = DataLoader(pol_ds, batch_size=64)
    out = []
    for b in dl:
        logits = model(input_ids=b["input_ids"].to(DEVICE),
                       attention_mask=b["attention_mask"].to(DEVICE)).logits
        out.append(torch.sigmoid(logits).cpu().numpy())
    return np.vstack(out)


def coverage_at_floor(P, Y):
    """Certify per-intent thresholds at FLOOR on the policy split (point
    estimate, same rule as pick_thresholds), then coverage under argmax-style
    acceptance. Coverage = caught positives / total positives."""
    caught = pos_tot = 0
    per = {}
    for k, intent in enumerate(INTENTS):
        p, y = P[:, k], Y[:, k]
        best = 2.0
        for t in sorted(set(p[y == 1])):
            sel = p >= t
            if sel.sum() and y[sel].mean() >= FLOOR:
                best = float(t)
                break
        sel = p >= best
        prec = float(y[sel].mean()) if sel.sum() else None
        rec = int(y[sel].sum())
        per[intent] = {"t": round(best, 3), "prec": None if prec is None else round(prec, 3),
                       "caught": rec, "pos": int(y.sum())}
        caught += rec
        pos_tot += int(y.sum())
    return caught, pos_tot, per


Y_pol = pol_ds.y
best_cov = -1
for ep in range(EPOCHS):
    model.train()
    tot = 0.0
    for b in train_dl:
        opt.zero_grad()
        logits = model(input_ids=b["input_ids"].to(DEVICE),
                       attention_mask=b["attention_mask"].to(DEVICE)).logits
        loss = lossfn(logits, b["labels"].to(DEVICE))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        tot += loss.item()
    P = policy_probs()
    c, pt, per = coverage_at_floor(P, Y_pol)
    best_cov = max(best_cov, c / pt)
    print(f"epoch {ep+1}: loss={tot/len(train_dl):.4f} "
          f"policy coverage@{FLOOR} = {c}/{pt} = {c/pt:.3f}")

print(f"\n=== {MODEL} best in-sample policy coverage@{FLOOR}: {best_cov:.3f} ===")
P_pol_final = policy_probs()
c, pt, per = coverage_at_floor(P_pol_final, Y_pol)
for intent, d in per.items():
    print(f"  {intent:22s} t={d['t']} prec={d['prec']} caught={d['caught']}/{d['pos']}")

# Honest held-out: certify thresholds on policy, evaluate on the cleaned test
# split (dual-critic verdicts), same protocol as the LR held_out_eval.py.
test_verdicts = {v["id"]: v["label"] for v in json.load(open("/tmp/adj-test/verdicts.json"))["verdicts"]}
test_rows = [r for r in load_gold() if r.split == "test"]
tclean = []
for r in test_rows:
    v = test_verdicts.get(f"{r.conversation_id}#{r.turn_index}")
    if v == "ambiguous":
        continue
    labels = set(r.labels) if v is None else (set() if v == "none" else {v})
    tclean.append((r, labels))

# policy-certified thresholds (point estimate at FLOOR on policy)
policy_th = {}
for k, intent in enumerate(INTENTS):
    p, y = P_pol_final[:, k], Y_pol[:, k]
    best = 2.0
    for t in sorted(set(p[y == 1])):
        sel = p >= t
        if sel.sum() and y[sel].mean() >= FLOOR:
            best = float(t); break
    policy_th[intent] = best

class TDS(Dataset):
    def __init__(self, rs): self.rs = rs
    def __len__(self): return len(self.rs)
    def __getitem__(self, i):
        enc = tok(encode_row(self.rs[i][0]), truncation=True, max_length=MAX_LEN,
                  padding="max_length", return_tensors="pt")
        return {"input_ids": enc["input_ids"][0], "attention_mask": enc["attention_mask"][0]}

model.eval()
tprobs = []
with torch.no_grad():
    for b in DataLoader(TDS(tclean), batch_size=64):
        logits = model(input_ids=b["input_ids"].to(DEVICE),
                       attention_mask=b["attention_mask"].to(DEVICE)).logits
        tprobs.append(torch.sigmoid(logits).cpu().numpy())
TP = np.vstack(tprobs)
caught = accd = tot = 0
per_t = {}
for k, intent in enumerate(INTENTS):
    acc = TP[:, k] >= policy_th[intent]
    tp = sum(1 for (r, l), s in zip(tclean, acc) if s and intent in l)
    pos = sum(1 for _, l in tclean if intent in l)
    per_t[intent] = {"t": round(policy_th[intent], 3),
                     "prec": None if not acc.sum() else round(tp / acc.sum(), 3),
                     "rec": None if not pos else round(tp / pos, 3), "pos": pos}
    caught += tp; tot += pos; accd += int(acc.sum())
print(f"\n=== {MODEL} HELD-OUT test coverage@{FLOOR}: {caught}/{tot} = {caught/tot:.3f} "
      f"(overall precision {caught}/{accd} = {caught/max(accd,1):.3f}) ===")
for intent, d in per_t.items():
    print(f"  {intent:22s} t={d['t']} prec={d['prec']} rec={d['rec']} pos={d['pos']}")
json.dump({"model": MODEL, "policy_coverage": best_cov, "heldout_coverage": caught / tot,
           "heldout_precision": caught / max(accd, 1), "per_intent_heldout": per_t},
          open(f"/tmp/challenger_{MODEL.split('/')[-1]}.json", "w"), indent=1)
