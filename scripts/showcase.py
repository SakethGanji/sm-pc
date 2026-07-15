"""Surface real corpus turns that exhibit ASR damage / rambling / call-opening
noise, run them through the winning artifact, and show raw ASR vs clean human
transcript, the adjudicated truth, and the model's decision + probability."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "trainer"))

from semantic_poc.artifact import NPY_FILES
from semantic_poc.cli import load_gold
from semantic_poc.config import load_config
from semantic_poc.context import build_c1
from semantic_poc.embeddings import GeminiEmbedder
from semantic_poc.features import TfIdf
from semantic_poc.paths import ARTIFACTS_DIR
from semantic_poc.runtime import Scorer, sigmoid

art = max(ARTIFACTS_DIR.glob("artifact-*"), key=lambda p: p.stat().st_mtime)
model = json.loads((art / "model.json").read_text())
mats = {a: np.load(art / f) for a, f in NPY_FILES.items()}
intents = model["intents"]
scorer = Scorer(
    intents=intents, families=model["families"],
    tfidf=TfIdf(json.loads((art / "tfidf_vocab.json").read_text()), np.load(art / "tfidf_idf.npy")),
    sem_coef=mats["sem_coef"], sem_int=mats["sem_int"],
    lex_coef=mats["lex_coef"], lex_int=mats["lex_int"],
    fus_coef=mats["fus_coef"], fus_int=mats["fus_int"],
    platt_a=np.array(model["platt_a"]), platt_b=np.array(model["platt_b"]),
    thresholds=model["thresholds"], tau_low=model["tau_low"],
    max_accepted=model["max_accepted"], exclusive_groups=model["exclusive_groups"],
)
embedder = GeminiEmbedder(load_config("embedding"))

# apply the test verdicts as truth where available (clean ground truth)
tv = {v["id"]: v["label"] for v in json.load(open("/tmp/adj-test/verdicts.json"))["verdicts"]}

rows = [r for r in load_gold() if r.split in ("policy", "test")]
def truth(r):
    v = tv.get(f"{r.conversation_id}#{r.turn_index}")
    if v is None:
        return set(r.labels)
    if v in ("none", "ambiguous"):
        return set()
    return {v}

texts = [build_c1(r.previous_agent_utterance, r.raw_transcript) for r in rows]
E = embedder.embed(texts).astype(np.float64)
A, B = np.array(model["platt_a"]), np.array(model["platt_b"])

def run(r, e):
    sem = scorer.sem_coef @ e + scorer.sem_int
    lex = np.asarray(scorer.tfidf.transform([r.raw_transcript]) @ scorer.lex_coef.T).ravel() + scorer.lex_int
    meta = scorer.meta_features(r.raw_transcript, r.previous_agent_utterance)
    fused = scorer.fus_coef @ np.concatenate([sem, lex, meta]) + scorer.fus_int
    probs = sigmoid(A * fused + B)
    probs_d = {i: float(p) for i, p in zip(intents, probs)}
    from semantic_poc.policy import decide
    out = decide(probs_d, model["thresholds"], model["families"], model["tau_low"],
                 model["max_accepted"], model["exclusive_groups"])
    return out, probs_d

recs = []
for r, e in zip(rows, E):
    out, probs = run(r, e)
    recs.append((r, out, probs, truth(r)))


def show(title, filt, n=6, key=None):
    print(f"\n{'='*70}\n{title}\n{'='*70}")
    sel = [x for x in recs if filt(x)]
    if key:
        sel.sort(key=key)
    for r, out, probs, t in sel[:n]:
        top = max(probs, key=probs.get)
        acc = [i["name"] for i in out["intents"]] or ["—"]
        print(f"\n  raw ASR : {r.raw_transcript[:110]}")
        if r.human_transcript.strip().lower() != r.raw_transcript.strip().lower():
            print(f"  clean   : {r.human_transcript[:110]}")
        if r.previous_agent_utterance:
            print(f"  agent   : {r.previous_agent_utterance[:90]}")
        tl = ", ".join(t) if t else "no request"
        print(f"  truth={tl:18s} model={out['decision']}({','.join(acc)}) "
              f"p[{top}]={probs[top]:.2f}")


ntok = lambda r: len(r.raw_transcript.split())
has = lambda r, *w: any(x in r.raw_transcript for x in w)

# 1. Heavy ASR mangling but caught correctly
show("1. ASR mangled the words — model still recovered the request",
     lambda x: x[3] and list(x[3])[0] in {i["name"] for i in x[1]["intents"]}
               and (has(x[0], "lie", "czech", "chess", "ranch", "showers", "bell", "many", "a count", "pass word")
                    or "unk" in x[0].raw_transcript))

# 2. Call-opening wrapping (noise + name + garbled greeting)
show("2. Call-opening noise — buried request after [noise] + name + greeting",
     lambda x: x[3] and "noise" in x[0].raw_transcript and "my name is" in x[0].raw_transcript
               and list(x[3])[0] in {i["name"] for i in x[1]["intents"]})

# 3. Rambling / long turns
show("3. Rambling — long, meandering turns",
     lambda x: ntok(x[0]) >= 22, key=lambda x: -ntok(x[0]))

# 4. Correctly stayed silent (no request / small talk)
show("4. Correctly said nothing — greetings, data answers, small talk",
     lambda x: not x[3] and x[1]["decision"] in ("no_supported_intent", "abstained")
               and ntok(x[0]) >= 5, key=lambda x: -ntok(x[0]))

# 5. Genuinely hard ASR — model abstains where truth is a request
show("5. Too mangled to call — model abstains (truth WAS a request)",
     lambda x: x[3] and list(x[3])[0] not in {i["name"] for i in x[1]["intents"]},
     key=lambda x: (x[0].raw_transcript))

# 6. Confusable pair handled
show("6. Transfer vs pay-bill (the confusable pair)",
     lambda x: x[3] & {"transfer_money", "pay_bill"}
               and list(x[3])[0] in {i["name"] for i in x[1]["intents"]})
