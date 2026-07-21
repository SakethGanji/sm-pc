# Model design — how it works

This is not one model. It's a **pipeline of small, single-purpose layers**, each
doing one job and handing off to the next. The heavy lifting (understanding
language) is done by a **frozen, pretrained embedding** you rent; everything you
*train* is a stack of cheap linear models on top. This document walks the layers,
how each is trained, and a worked example.

## The big picture

```
                                    ┌─────────── you TRAIN these (cheap, on your labels) ───────────┐
customer turn ─▶ stitch ─▶ context ─▶ [EMBED] ─▶ semantic head ┐
                                        (frozen)                ├─▶ fusion ─▶ calibration ─▶ threshold ─▶ decision
                              raw text ────────▶ lexical head  ┘                                            policy ─▶ answer
                                                 + meta features
                        └── not trained ──┘  └───────────────── one-vs-rest, per intent ─────────────────┘
```

Two worlds:
- **Frozen / rented:** the embedding model (Gemini). Pretrained on huge data by
  someone else; you never train it — you call it and cache the result.
- **Trained by you:** the linear heads, fusion, calibration, thresholds — all fit
  on *your* labels. Cheap: a handful of logistic regressions.

## The layers, in order (the inference path)

**1. Stitcher** (`stitch.py`) — deterministic config rules, not learned. If a
customer utterance was split by the ASR into two fragments with a small gap and no
agent turn between, it optionally glues them ("i lost my" + "card"). Pure function,
tuned by `stitcher.yaml`.

**2. Context builder** (`context.py`) — formats one string:
`[AGENT] <previous agent utterance>\n[CUSTOMER] <raw transcript>`. The agent line
is included because half the hard cases only make sense in context ("yes it still
hasn't arrived"). This exact format is part of the contract — training and serving
must build it identically.

**3. Embedding** (`embeddings.py`) — the **frozen** layer. Sends the context string
to Gemini (`gemini-embedding-001`), gets back a 768-dimensional vector, L2-normalized.
This is the only network call (~200–400ms) and the only place "language understanding"
happens. Cached on disk keyed by `(model, task_type, dim, text)` so repeated text is
never re-billed. **Not trained.**

**4. Semantic head** — a per-intent, one-vs-rest **logistic regression over the
768-dim embedding**. For each intent it learns the direction in embedding space that
separates "this intent" from everything else. Output: one **semantic logit per intent**.

**5. Lexical head** — a per-intent one-vs-rest logistic regression over the **TF-IDF**
of the raw text (keyword signal). Catches exact words the embedding might smooth over.
Output: one **lexical logit per intent**. (The TF-IDF vectorizer is fit once on the
training text and shared across all intents.)

**Meta features** — two small hand-built numbers per turn: normalized token count and
whether there was a preceding agent utterance. Cheap side signal for the fusion.

**6. Fusion** — a per-intent logistic regression over the **concatenation of every
branch's output**: `[all K semantic logits ∥ all K lexical logits ∥ meta]`. Because
each intent's fusion head sees *all* branches' logits (not just its own), it can learn
cross-intent corrections — e.g. "down-weight me when a sibling is also firing." Output:
one **fused logit per intent**.

**7. Calibration (Platt)** — a per-intent 1-parameter logistic (`a·logit + b`) that
turns the fused logit into a **real probability**. Without this, "0.8" is just a score;
with it, 0.8 means ~80% likely. Fit on a held-out calibration split.

**8. Thresholds** — a per-intent cutoff. A turn is a *candidate* accept for an intent
only if its calibrated probability ≥ that intent's threshold. Each threshold is
**certified** as the lowest cut that still holds your precision floor (max coverage
under the safety bar), then clamped up by an absolute `accept_floor` so a perfectly
separable intent can't certify a near-zero cut that accepts garbage.

**9. Decision policy** (`policy.py`) — turns per-intent candidates into the final
answer, with safety built in:
- **`tau_low`:** if nothing clears a minimum confidence → `no_supported_intent`.
- **Exclusive groups:** within a family, only the argmax fires (siblings don't co-fire).
- **`max_accepted`:** cap how many intents a single turn can carry.
- **Abstain:** genuinely ambiguous (above `tau_low`, but nothing clears its
  certified threshold) → abstain and hand off.

The output is one of: accepted intent(s), `abstained`, or `no_supported_intent`.

> A **family fallback** — confident it's a *card* issue but unsure which action,
> so suggest the family — is a design option that is **not implemented**.
> `policy.py` uses the family map only to annotate `main_intent` on the output.
> Deferred until real data shows how often "confident family, unsure leaf"
> occurs.

## How it's trained

Input: your labeled `GoldRow`s, split **by conversation, time-ordered** into
`train` / `calibration` / `policy` / `test` (a whole call stays in one split; test is
frozen + hashed). Then, in order (`train.py`):

1. **Fit TF-IDF** on the training text.
2. **Embed** every split's context strings (cache-first — Gemini only for new text).
3. **Semantic branch:** for each intent, sweep the regularization `C` over a grid using
   **conversation-grouped cross-validation**, pick the best `C` by out-of-fold average
   precision, and record the **out-of-fold (OOF) logits**. Then retrain that head on the
   full training set.
4. **Lexical branch:** identical procedure over TF-IDF.
5. **Fusion:** fit the per-intent fusion heads on the **OOF logits** from steps 3–4 plus
   meta. Training on out-of-fold predictions (not in-fold) prevents the fusion from
   overfitting to scores the branches already "saw."
6. **Calibration:** compute fused logits on the **calibration** split, fit per-intent Platt.
7. **Thresholds:** compute calibrated probabilities on the **policy** split, pick each
   intent's threshold at the precision floor (+ accept-floor clamp).
8. **Evaluate once** on `test` for the honest number.
9. **Export** every coefficient + config into one immutable, hash-named artifact.

Principles that make the numbers trustworthy:
- **One-vs-rest:** each intent is an independent binary classifier — that's why there
  are K of each head.
- **Conversation-grouped folds:** a call never appears in both train and validation, so
  no leakage inflates the score.
- **Out-of-fold fusion:** the combiner learns on honest, unseen-branch predictions.
- **Separate calibration + policy splits:** probabilities and cutoffs are certified on
  data the model didn't train on.
- **Everything is supervised** — it all learns from your labels. (The one auto-tuned
  thing is `C`, chosen by CV. You tune nothing by hand.)

For **K intents** the trained artifact holds **4K logistic regressions** (K each of
semantic, lexical, fusion, Platt) **+ 1 shared TF-IDF**. For 50 intents that's 200 tiny
linear models — at serve time, coefficient matrices and a few matrix multiplies.

## Worked example

Turn: customer says `"i lost my card"`, agent had said `"how can i help you today"`.

1. **Stitch:** no dangling fragment → unchanged.
2. **Context:** `[AGENT] how can i help you today\n[CUSTOMER] i lost my card`.
3. **Embed:** → a 768-vector (frozen Gemini).
4. **Semantic head:** high logit for `replace_card`, low for the rest.
5. **Lexical head:** "lost", "card" → high logit for `replace_card`.
6. **Fusion:** combines both branches → high fused logit for `replace_card`.
7. **Calibration:** fused logit → probability ≈ 0.98.
8. **Threshold:** 0.98 ≥ `replace_card`'s certified cut → candidate.
9. **Policy:** no exclusive-group sibling outscores it, under the cap → **accept
   `replace_card`**.

A greeting like `"hi"` runs the same path, scores low on every intent, fails `tau_low`,
and returns `no_supported_intent` — the model correctly stays silent.

## Why layered this way

- **Frozen embedding + linear heads** (transfer learning): the expensive part
  (understanding language) is pretrained and rented; you train only the cheap part.
  In the rehearsal this **beat a fine-tuned encoder** and is orders of magnitude cheaper
  to run — and it's why the number is gated on *label quality*, not model horsepower.
- **Two branches (semantic + lexical), then fusion:** the embedding catches meaning,
  TF-IDF catches exact keywords; fusion learns when to trust which, per intent.
- **Calibration separate from scoring:** you can't set safe thresholds on uncalibrated
  scores.
- **Thresholds + policy separate from the model:** safety (precision floor, abstain,
  no-co-fire) is a governed decision layer, not something the classifier is trusted
  to get right implicitly.
- **One embedding call regardless of intent count:** 8 or 200 intents is the same single
  embed + a taller matmul — latency is embedding-bound and ~constant, which is what makes
  it scale to a large taxonomy and high call volume.

## What's trained vs frozen vs configured

| | Examples | Changes when |
|---|---|---|
| **Frozen** | the Gemini embedding | you switch embedding provider/model (→ re-embed) |
| **Trained** | semantic/lexical/fusion heads, Platt, thresholds | you `poc train` on new/relabeled data |
| **Configured** | taxonomy, precision floor, accept floor, exclusive groups, TF-IDF params, context format | you edit `configs/*.yaml` (→ new artifact) |
