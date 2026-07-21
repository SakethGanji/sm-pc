# Build guide — intent classification on your own data

The single guide for rebuilding this system on real office data. It captures what
the home rehearsal (HarperValleyBank, 8 intents) taught the hard way, so you
don't rediscover it on a corpus that will look nothing like HVB.

**The one lesson that reorders everything below:** the model was never the
bottleneck — the *labels* were. Coverage went **0.52 → 0.92 with zero model
changes**, purely by fixing a noisy evaluation set. Assume the same until you've
measured otherwise. Budget for data quality first, model sophistication last.

You will **not** reproduce 0.92 — you get your own number, per intent, lower and
messier. The transferable thing is the method, not the figure.

## 0. Mental model — where effort actually pays off

| Lever | Rehearsal impact | Effort |
|---|---|---|
| Label quality (esp. the eval set) | **the whole gap** (0.52→0.92) | high, unglamorous |
| More raw training volume | ~0 (flat learning curve) | medium |
| Synthetic augmentation | measured null once labels were clean | medium |
| Bigger / fine-tuned model | *lost* to the linear model | high |
| Decision-policy guards (floors, abstain) | fragile demo → safe tool | low |

Spend top-down by that first column. Everything below serves it.

## 1. What your data needs to look like

The unit of prediction is **one customer turn, in context** (the preceding agent
utterance). One row per customer turn. Full schema — every column, types,
examples, how to add rows to the store — is in **`AGENTS.md`**; the field
intuitions that trip people up:

- **`raw_transcript` — never clean it.** You serve on raw ASR, so you train and
  eval on raw ASR. The clean version lives in `human_transcript` only to
  *diagnose* whether a miss is the recognizer's fault or the model's (§7).
- **`previous_agent_utterance` is not optional.** Half the hard cases are
  fragments that only make sense given the agent's question ("yes it still hasn't
  arrived"). It goes into the embedded context string.
- **`labels` is a list**, and **`[]` (empty) is a real, important class.** Most
  turns are *not* requests — greetings, "yes", account numbers read aloud. If you
  only collect request turns, the model fires on everything. Collect the negatives.
- **`ambiguous`** excludes a turn from train AND eval — your escape valve for
  turns even a human can't confidently label. Use it; don't force a guess.

If your ASR vendor can emit both a raw hypothesis and you can afford a
human-correction pass on a sample, do it — the `raw`↔`human` delta is what proves
"this slice is an *audio* problem, not a *classifier* problem."

## 2. Volume and composition — how much, and of what

The binding number is **labeled positives per intent per split**, not total rows.

- **Training:** ~100–300 real positives per common intent for a stable linear
  head. Below ~30 it's noisy; below ~10 it barely learns.
- **Certification (calibration + policy + test):** the real constraint, chronically
  under-budgeted. To *prove* an intent holds a 90% precision floor you need ~30+
  adjudicated positives **per intent per eval split**; fewer and your precision has
  a confidence interval wider than the decision. In the rehearsal `transfer_money`
  had 9–16 and simply could not be certified — a data wall, not a model failure.
- **Composition beats size.** Stratify to include plain requests, confusable
  pairs, agent-context-dependent fragments, corrections ("no I meant…"),
  negations, multi-intent turns, hard negatives (adjacent non-requests), and
  ASR-mangled examples. A balanced 5k beats a lopsided 50k.

**Denominator honesty:** headline coverage is over *request-bearing turns in the
families you actually support*. A card-only model correctly saying "nothing here"
on a mortgage call is not low coverage.

## 3. Labeling — the part that actually determines your ceiling

**Gold standard: dual human annotation with an agreement gate.** Two independent
humans label each row; measure Cohen's κ. If κ < ~0.75 your guidelines are
ambiguous — fix the guide before labeling more, because every downstream number
is void if the labels are noise. This gate is the highest-leverage control you have.

**Accelerate without giving up the gate** (what the rehearsal did):
1. **Dual *blind* AI critics** relabel each row from text + context alone, not
   shown the proposed label. Give them opposing biases — one toward precision
   ("prefer 'no request' when borderline"), one toward recall ("recover the intent
   through ASR noise").
2. **Keep only agreements.** Trust a row only when both critics independently
   produce the same label; disagreements → `ambiguous`, excluded, not guessed.
3. **Humans adjudicate the disagreements + a sample of agreements.** The AI pass
   collapses human workload onto the genuinely hard rows; it doesn't replace the
   human as final authority. (LLM critics over-promote some edge cases — a caller
   *answering* "what's the amount?" got mislabeled a fresh `pay_bill`. The κ gate
   and human adjudication exist to catch this.)

**Clean your EVALUATION set first, not your training set.** Noisy *training*
labels cost some accuracy; noisy *evaluation* labels **lie to you about how good
you are** and mis-set your thresholds. The rehearsal plateau was entirely an
eval-label artifact.

**How labels land in this codebase:** ingest emits every turn with `labels=[]`
into `gold.duckdb`; you promote the request turns with `poc promote <intent>
<ids.txt>` (batched, audited in-file). See `AGENTS.md` §6.

## 4. Splits and hygiene — how to not fool yourself

- **Group by conversation.** Every turn of a call goes to the *same* split, or you
  leak and inflate your numbers. (`poc partition` enforces this.)
- **Order by time.** Oldest → train, newest → test; mimics production and catches drift.
- **Four splits:** `train` (fit), `calibration` (fit probability calibration, held
  out from train), `policy` (certify thresholds — iterate here), `test` (touch
  **once**, at the end, for the honest number).
- **Freeze the test set with a hash** and don't look while iterating.
- **Synthetic data is train-only, forever.** Precision floors must be certified on
  real data only.

## 5. The pipeline (what runs, in order)

One immutable, hash-named artifact per config. The math is deliberately simple so
it can be mirrored exactly in a fast serving language.

```
raw turn + prev agent turn
  → (optional) stitch a dangling previous fragment
  → build context string:  [AGENT] <prev>\n[CUSTOMER] <raw>
  → embed once (frozen provider)                ← the only network call, ~200-400ms
  → semantic head:  per-intent linear model over the embedding
  → lexical head:   per-intent linear model over TF-IDF of raw text
  → fusion:         small linear model over [semantic ∥ lexical ∥ meta] logits
  → calibration:    per-intent Platt scaling (scores → real probabilities)
  → thresholds:     per-intent cut certified to hold the precision floor
  → decision policy: floors + abstain + exclusive groups
```

- **One embedding call per turn**, regardless of intent count (8 or 50 or 200 =
  same embed + a taller matmul, still sub-millisecond). Latency is embedding-bound
  and ~constant.
- **Train heads out-of-fold, fit fusion on the out-of-fold scores** (in-fold
  overfits); retrain heads on full train after hyperparameter selection.
- **Calibrate on its own split.** Probabilities are meaningless if calibrated on
  training data.
- **Pin the embedding model + version.** Treat a provider model change as a planned
  re-embed + retrain event. Embeddings are cached on disk keyed by (model, version,
  dim, text) so you never re-bill the same text.

## 6. Certification — turning scores into a safe yes/no

- **Precommit the precision floor before any result** (rehearsal used 0.90). A
  governance act: set the safety bar first, then measure coverage against it.
- **Per-intent thresholds on the `policy` split** — the *lowest* cut whose precision
  still meets the floor (max coverage subject to safety).
- **Add an absolute accept floor** (0.50). On separable data the per-intent cut can
  collapse to ~0 and accept garbage — a live test caught a mangled non-request
  clearing a ~4e-8 threshold. Clamp every cut up to a sane minimum (raising a
  threshold only *increases* precision). Two-line guard; do it day one.
- **Report the *held-out* number.** In-sample coverage is optimistic (0.97 in-sample
  vs 0.92 held-out in the rehearsal). The held-out number is the one you promise.

## 7. Diagnostics — is it the labels, the audio, or the model?

When an intent underperforms, localize the cause before spending (all cheap, use
cached embeddings):

1. **Representation ceiling** — per-intent AUC/AP of the fused score on `policy`.
   High AUC (rehearsal saw ~0.999) → the embedding already separates the intent, so
   low coverage is a *threshold/label* problem, not a model one. **Run this before
   ever proposing to fine-tune.**
2. **ASR ceiling** — re-score using `human_transcript` instead of `raw`. If most
   misses vanish on clean text, the fix is upstream (ASR), not your classifier.
3. **False-positive / mislabel audit** — list the highest-scoring turns the model
   accepted that are labeled negative. If they're obviously real requests, your
   *labels* are wrong. This one check cracked the rehearsal open.

Kept scripts: `scripts/held_out_eval.py` (held-out coverage per intent) and
`scripts/stability_report.py` (§9). The rehearsal's one-off diagnostic scripts
(representation-ceiling, ASR-damage, FP-audit) live in git history — recover one if
you need that specific analysis.

## 8. Decision policy and safety nets — never confidently mislead

A confident wrong answer is worse than silence. Build in:
- **Two no-accept outcomes, implemented:** `no_supported_intent` (nothing
  relevant) and `abstained` (something's here, not sure enough).
- **A family-level fallback** (confident it's a *card* issue, unsure which
  action) is a design option, **not implemented**. Deferred until real data shows
  how often "confident family, unsure leaf" actually occurs — measure the family
  composition of the abstain bucket before building it.
- **Abstain on genuine ambiguity** — a non-generative classifier can't resolve "did
  you mean replace or unlock?"; the correct move is a calibrated abstain + hand-off.
- **Exclusive groups / argmax within a family** so siblings don't both fire.
- **Cap accepted intents per turn** (rehearsal used 3).

## 9. Scaling to ~50 intents (your real situation)

Mechanically fine — one embedding call, taller matrix, no latency change. The hard
parts are statistical:

- **Confusability grows ~quadratically.** 8 intents → 28 possible confusion pairs;
  50 → 1,225, mostly fine-grained siblings. Mine the confusion matrix for the pairs
  that actually confuse and pour hard-negative data at exactly those.
- **The rare-intent tail is a certification problem.** Many of 50 will have too few
  positives to certify a floor. Don't ship those on equal footing: (a) hold back
  until they have data, (b) use retrieval/prototypes (nearest-neighbor over the
  embedding, far fewer examples than a trained head), or (c) roll up into a
  family-level suggestion until the leaf earns certification.
- **Roll out in tiers, certify per-intent.** Report coverage **per intent**, never
  one headline number — uneven coverage is the correct, honest shape at 50.

**Add intents one at a time, and re-check the ones you already had.** Because the
heads are one-vs-rest, a turn already present as OOS is already a negative for every
other intent — so a **representative OOS pool up front** keeps existing intents
stable when you add new ones. Only *same-family* competitors shift, and only at the
`decide()` policy layer (verified: adding an 8th intent to a 7-intent model left
6/7 at Δ0.000). The loop:

```
poc promote <intent> ids.txt                         # label the batch (audited)
poc train                                            # retrain full model
python scripts/stability_report.py --label "added X" # per-intent diff vs last run; non-zero exit on regression
poc snapshot --label "X in"                          # Parquet restore point
```

**Hierarchy (main→sub): do it in the decision policy, not as a cascade.** Keep flat
leaf heads (every leaf scores every turn); layer the family logic on the decision —
aggregate leaf scores into family confidence, argmax within family, fall back to the
family suggestion when no leaf clears its floor. A hard cascade locks in a wrong
family decision with no recovery. Hierarchy is a decision-structure win, not a
representation win.

## 10. When to escalate to a fine-tuned encoder (and when not to)

Promote a fine-tuned-encoder challenger only on **measured** evidence. In the
rehearsal the fine-tuned encoder (MiniLM) *lost* to the linear model on held-out
data, and DeBERTa wouldn't train stably. Promotion bar: **+3–5 points coverage at
equal precision, a critical-intent win, or lower total cost.** Trigger to even try:
a §7 representation-ceiling check showing genuinely low AUC on an intent you must
support. Absent that signal, the cheap stack wins — don't spend the money.

## 11. Rollout and shadow mode

- **Shadow first.** Score real production turns with zero live impact; sample weekly
  for human adjudication. This is how you get the *real* coverage number — lower and
  messier than any offline figure. Don't commit to a coverage SLA before shadow data.
- **Measure the incumbent too.** "We beat the baseline (LLM/human) at lower cost" is
  the argument that ships; a raw accuracy number in a vacuum is not.
- **Report per-slice:** ASR-damaged, stitched, long, negation/correction,
  multi-intent, rare, confusable pairs. Label context/state-dependent slices as
  *expected scope gaps*, not unexplained failures.

## 12. The checklist (things the rehearsal tripped on)

- [ ] Raw ASR text is never cleaned anywhere in train/serve.
- [ ] Negatives (`[]`-label turns) are collected and labeled, not just requests.
- [ ] Conversations never split across train/calibration/policy/test.
- [ ] Splits are time-ordered (old→train, new→test).
- [ ] Test set is hashed and touched exactly once.
- [ ] Synthetic rows are hard-pinned to train and can never leak into eval.
- [ ] Calibration is fit on its own held-out split.
- [ ] Precision floor is precommitted before any result is seen.
- [ ] An absolute accept-floor guard exists (no near-zero thresholds).
- [ ] Held-out (not in-sample) coverage is what you report.
- [ ] Embedding model+version is pinned; embeddings are cached on disk.
- [ ] κ ≥ ~0.75 gate passed before scaling labeling.
- [ ] Eval labels cleaned *before* chasing training-set size or a bigger model.
- [ ] Rare/confusable intents rolled out in tiers, certified per-intent.

## 13. Porting this codebase — what changes

**Almost none of the code.** Training, calibration, thresholding, decision policy,
serving, and the store are all dataset-agnostic. You rewrite **one adapter** and
edit **a few configs**. Full detail: `AGENTS.md` (data contract, how to add and
label rows, module map, how to run). The map:

- **`trainer/semantic_poc/ingest.py`** — the one rewrite: read your CSV → one
  `GoldRow` per customer turn, `labels=[]`. (Its header spells out the contract.)
- **`trainer/semantic_poc/paths.py`** — point `RAW_HVB` at your corpus (`data/raw`
  is gitignored — real transcripts stay local).
- **`configs/taxonomy.yaml`** — your ~50 intents + families.
- **`configs/policy.yaml`** — `precision_floor`, `accept_floor` (~0.50), `tau_low`,
  `max_accepted`, `exclusive_groups` (sibling intents that shouldn't co-fire).
- **`configs/embedding.yaml`** (+ `embeddings.py` `_fetch_batch` only if not Gemini).
- **`configs/features.yaml`, `stitcher.yaml`** — usually fine; tune if needed.
- **`weak_labels.py`** — HVB-only; delete once ingest stops seeding weak labels.

Run (from `trainer/`, venv active, `POC_GOLD_PATH=../data/gold/gold.duckdb`):
```
poc ingest → poc counts → poc promote <intent> ids.txt → poc partition → poc train
python ../scripts/serve_py.py --artifact <newest>
python ../scripts/held_out_eval.py        # per-intent number
```

## Appendix — the metrics, in plain terms

Imagine the model makes 100 suggestions to agents:
- **Precision** = of the suggestions it *made*, how many were right. High precision
  = it rarely misleads the agent.
- **Recall / coverage** = of all the requests that existed, how many it caught. High
  = it rarely stays silent when it should speak.
- **F1** = a blended precision/recall score.

**We do NOT optimize F1.** F1 treats a false suggestion and a missed request as
equally bad; for agent-assist they are not — confidently telling an agent the wrong
thing is far worse than silence. So we **fix precision at a floor committed up
front** (e.g. ≥90% right) and **maximize coverage** under it. The headline you
report is: **"X% coverage at ≥90% precision, on the holdout set, per intent."**

## TL;DR for the exec conversation

Buy **good labels** (especially a clean evaluation set), keep the **cheap model**,
roll out the intents **gradually with a safety net**, and measure **each intent
honestly** instead of promising a single accuracy number. The rehearsal's proof:
fixing labels alone moved coverage 52% → 92% with no model change, and a fine-tuned
neural network *lost* to the simple one. The money belongs in data quality and
disciplined rollout, not model horsepower.
