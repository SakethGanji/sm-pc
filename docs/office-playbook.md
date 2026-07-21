# Production build playbook — intent classification on your own data

A field guide for rebuilding this system on real office data. It captures what
we learned the hard way in the home rehearsal (HarperValleyBank, 8 intents) so
you don't have to rediscover it on a corpus that will look nothing like HVB.

**The one lesson that reorders everything below:** in the rehearsal, the model
was never the bottleneck — the *labels* were. Coverage went 0.52 → 0.92 with
**zero** model changes, purely by fixing a noisy evaluation set. Assume the same
is true for you until measured otherwise. Budget for data quality first, model
sophistication last.

---

## 0. Mental model — where effort actually pays off

| Lever | Rehearsal impact | Effort |
|---|---|---|
| Label quality (esp. the eval set) | **the whole gap** (0.52→0.92) | high, unglamorous |
| More raw training volume | ~0 (flat learning curve) | medium |
| Synthetic augmentation | measured null once labels were clean | medium |
| Bigger/fine-tuned model | *lost* to the linear model | high |
| Decision-policy guards (floors, abstain) | turned a fragile demo into a safe one | low |

Spend top-down by that first column. Everything else in this doc serves it.

---

## 1. What your data needs to look like

### 1.1 The unit of prediction is one customer turn, in context

You classify a single customer utterance, given the immediately preceding agent
utterance. Store one row per customer turn. Minimum viable row:

```jsonc
{
  "conversation_id": "string",        // groups turns; NEVER split a conversation across train/test
  "turn_index": 0,                     // order within the conversation
  "timestamp_ms": 0,                   // real event time — drives time-ordered splits
  "raw_transcript": "string",          // ASR output, NEVER cleaned. This is what you serve on.
  "human_transcript": "string",        // human-corrected text. Rides along to separate ASR vs model failure.
  "previous_agent_utterance": "string",// the agent turn just before this one, "" if none
  "prev_customer_segment": {           // optional: immediately preceding customer fragment (for stitching)
    "turn_index": 0, "text": "string", "gap_ms": 0
  },
  "labels": ["intent_name"],           // 0, 1, or 2+ intents. [] means "no actionable request".
  "session_task_intent": "intent_name",// the conversation-level task, if known (weak-label seed)
  "ambiguous": false,                  // true = excluded from train AND eval (see §3)
  "split": ""                          // train | calibration | policy | test (assigned, not hand-set)
}
```

Field-by-field, why each matters:

- **`raw_transcript` — never clean it.** You will be tempted to fix ASR typos.
  Don't. You serve on raw ASR, so you must train and evaluate on raw ASR. The
  clean version lives in `human_transcript` only to *diagnose* whether a miss is
  the recognizer's fault or the model's (see §7, ASR ceiling).
- **`previous_agent_utterance` is not optional.** Half the hard cases are
  fragments that only make sense given the agent's question ("yes it still
  hasn't arrived" → depends on what the agent asked). Feed the agent turn into
  the text you embed.
- **`labels` is a list.** Turns can carry two intents ("transfer money and
  what are your hours") or zero. Model it as multi-label from day one.
- **`[]` (empty) is a real, important class.** Most turns are *not* requests —
  greetings, "yes", account numbers read aloud, small talk. If you only collect
  request turns, the model never learns to stay silent and will fire on
  everything. Collect and label the negatives.
- **`ambiguous`** is your escape valve for turns even a human can't confidently
  label. Excluded from both training and scoring. Use it; don't force a guess.

### 1.2 The two transcripts are the single most valuable thing HVB gave us

If your ASR vendor can emit both the raw hypothesis and you can afford a human
correction pass on a sample, do it. The `raw`↔`human` delta is what lets you
prove "this slice is an *audio* problem, not a *classifier* problem" — otherwise
every miss looks like your model's fault and you'll over-invest in the model.

---

## 2. Volume and composition — how much, and of what

There is no single "N rows" answer; the binding number is **labeled positives
per intent in each split**, not total rows.

- **Training:** aim for enough positives per intent that a linear head is stable
  — very roughly 100–300 real positives per common intent. Below ~30 the head
  is noisy; below ~10 it barely learns. Rare intents will fall short — that's
  the tail problem (§9), solve it with retrieval/prototypes, not by faking data.
- **Certification (calibration + policy/eval + test):** this is the real
  constraint and the one people under-budget. To *prove* an intent holds a 90%
  precision floor you need enough graded positives that the estimate isn't
  noise. ~30+ real, adjudicated positives per intent per eval split is a working
  minimum; fewer and your precision number has a confidence interval wider than
  the decision. In the rehearsal, `transfer_money` had 9–16 and we simply could
  not certify it — that was a data wall, not a model failure.
- **Composition matters more than raw size.** Stratify collection to include:
  plain requests, confusable pairs, agent-context-dependent fragments,
  corrections ("no I meant…"), negations, multi-intent turns, hard negatives
  (adjacent non-requests), and ASR-mangled examples. A balanced 5k beats a
  lopsided 50k. (This is the design doc's "difficult streams" sampling.)

**Denominator honesty:** headline coverage should be over *request-bearing turns
in the families you actually support*. A card-only model correctly saying
"nothing here" on a mortgage call is not low coverage — don't let it be counted
as such.

---

## 3. Labeling and adjudication — the part that actually determines your ceiling

### 3.1 The gold standard is dual human annotation with an agreement gate

Two independent humans label each row; measure inter-annotator agreement
(Cohen's κ). If κ < ~0.75 your guidelines are ambiguous — fix the guide before
labeling more, because every downstream model number is void if the labels are
noise. This gate is not optional; it is the single highest-leverage control you
have.

### 3.2 How to accelerate it without giving up the gate (what we did)

Humans are slow. The rehearsal used a scalable accelerator that you can mirror:

1. **Dual *blind* AI critics.** Two LLM graders relabel each row from the text +
   context alone, *not* shown the proposed label. Give them opposing biases —
   one instructed toward precision ("prefer 'no request' when borderline"), one
   toward recall ("recover the intent through ASR noise when it's plausibly
   there").
2. **Keep only agreements.** A row is trusted only when both critics
   independently produce the *same* label. Disagreements are marked `ambiguous`
   and excluded, not guessed.
3. **Humans adjudicate the disagreements and a sample of the agreements.** The
   AI pass collapses the human workload onto the genuinely hard rows and a QA
   sample; it does not replace the human as final authority.

This is exactly how we moved the rehearsal from 0.52 to 0.92: the crude weak
labels (from dialog-act heuristics) had silently marked hundreds of real
requests as "no request," and the dual-critic pass surfaced them. Expect the
same class of error in any heuristic or single-pass labeling.

**Caveat we hit and you must respect:** the AI critics over-promote a few edge
cases (e.g., a caller *answering* the agent's "what's the amount?" got labeled
as a fresh `pay_bill` request). LLM labels are a starting point, not ground
truth. The human adjudication and κ gate exist precisely to catch this.

### 3.3 Clean your **evaluation** set first, not your training set

Counterintuitive but decisive: noisy *training* labels cost you some accuracy;
noisy *evaluation* labels **lie to you about how good you are** and mis-set your
thresholds. The rehearsal plateau was entirely an eval-label artifact. Prioritize
a clean, human-adjudicated test/policy set over a bigger training set.

---

## 4. Splits and hygiene — how to not fool yourself

- **Group by conversation.** Every turn of a conversation goes to the *same*
  split. If turns from one call land in both train and test, you leak and your
  numbers are inflated.
- **Order by time.** Oldest calls → train; newest → test. This mimics
  production (you train on the past, serve the future) and catches drift.
- **Four splits:** `train` (fit models), `calibration` (fit probability
  calibration — must be held out from training), `policy` (certify thresholds),
  `test` (touch **once**, at the very end, for the honest number).
- **Freeze the test set with a hash** and don't look at it while iterating. We
  hashed the test rows and verified the hash was unchanged before every step.
  Iterate on `policy`; report on `test`.
- **Synthetic data is train-only, forever.** If you generate synthetic rows,
  hard-pin them to `train` and make the partitioner refuse to ever place them in
  calibration/policy/test. Precision floors must be certified on *real* data only.

---

## 5. The pipeline (what runs, in order)

One immutable artifact per config. The math is deliberately simple so it can be
mirrored exactly in a fast serving language (we used Go).

```
raw turn + prev agent turn
  → (optional) stitch a dangling previous fragment
  → build context string:  [AGENT] <prev>\n[CUSTOMER] <raw>
  → embed once (frozen embedding provider)      ← the only network call, ~200-400ms
  → semantic head:  per-intent linear model over the embedding
  → lexical head:   per-intent linear model over TF-IDF of raw text
  → fusion:         small linear model over [semantic ∥ lexical ∥ meta] logits
  → calibration:    per-intent Platt scaling (turns scores into real probabilities)
  → thresholds:     per-intent cut certified to hold the precision floor
  → decision policy: floors + abstain + family fallback + exclusive groups
```

Key points:
- **One embedding call per turn.** Number of intents does not change this. 8 or
  50 or 200 intents = same single embedding + a taller matrix multiply (still
  sub-millisecond). Latency is embedding-bound and roughly constant.
- **Train the two heads out-of-fold, then fit fusion on the out-of-fold
  scores.** Fitting fusion on in-fold scores overfits. Retrain the heads on full
  train after selecting hyperparameters.
- **Calibration on its own split.** Probabilities are meaningless if calibrated
  on training data. Hold `calibration` out.
- **Pin the embedding model + version.** Treat a provider model change as a
  planned re-embed + retrain event, not a silent dependency. Cache embeddings on
  disk keyed by (model, version, dim, text) so you never re-bill for the same
  text.

---

## 6. Certification — turning scores into a safe yes/no

- **Precommit the precision floor before you see any result** (we used 0.90).
  This is a governance act: you decide the safety bar first, then measure
  coverage against it. Don't let a good coverage number tempt you to lower it
  after the fact.
- **Per-intent thresholds, chosen on the `policy` split** as the *lowest* cut
  whose precision still meets the floor (maximizes coverage subject to safety).
- **Add an absolute accept floor** (we used 0.50). On clean, separable data the
  per-intent cut can collapse to ~0, which then accepts garbage. A live test
  caught exactly this: a mangled non-request cleared a ~0.00000004 threshold and
  false-fired. Clamp every certified cut up to a sane minimum (raising a
  threshold only *increases* precision, so it never breaks the floor). Do this
  from day one; it is a two-line change that prevents an embarrassing prod bug.
- **Report the *held-out* number.** In-sample coverage (certify and evaluate on
  the same split) is optimistic — ours read 0.97 in-sample vs 0.92 on held-out
  test. The held-out number is the one you promise.

---

## 7. The diagnostic playbook — is it the labels, the audio, or the model?

When an intent underperforms, don't guess. Run these three cheap checks (all use
cached embeddings) to localize the cause before spending:

1. **Representation ceiling — per-intent AUC/AP of the fused score on `policy`.**
   High AUC (we saw ~0.999) means the embedding *already* separates the intent;
   low coverage is then a *threshold/label* problem, not a model problem. Low
   AUC on a specific intent means the representation genuinely can't see it —
   that's your one real case for a fancier model. **Run this before ever
   proposing to fine-tune.**
2. **ASR ceiling — re-score using `human_transcript` instead of `raw`.** The
   coverage delta bounds what *any* text model could do given your audio. If
   most of your misses vanish on clean text, the fix is upstream (ASR), not in
   your classifier. In the rehearsal this was small (~3 pts); yours may differ.
3. **False-positive / mislabel audit.** List the highest-scoring turns the model
   accepted that are labeled negative. If they're obviously real requests
   (they were, for us), your *labels* are wrong, not your model. This one check
   is what cracked the whole rehearsal open.

Only after these point at the model do you reach for §11.

---

## 8. Decision policy and safety nets — never confidently mislead

The agent sees your suggestion; a confident wrong answer is worse than silence.
Build these behaviors in:

- **Three no-accept outcomes, not one:** `no_supported_intent` ("heard you,
  nothing relevant"), `abstained` ("something's here but I'm not sure enough"),
  and a **family-level fallback** ("confident it's a *card* issue, unsure which
  card action"). The fallback turns vague/hard inputs into something still
  useful instead of a dead end.
- **Abstain on genuine ambiguity.** A non-generative classifier *cannot* resolve
  "did you mean replace or unlock?" — its correct move is a calibrated abstain
  and hand-off. Don't try to force a leaf label on truly ambiguous turns.
- **Exclusive groups / argmax within a family** so two sibling intents don't
  both fire on one turn.
- **Cap accepted intents per turn** (we used 3) and surface overflow.

---

## 9. Scaling to ~50 intents (your real situation)

Mechanically fine — one embedding call, taller matrix, no latency change. The
hard parts are statistical and they are the ones from §2–3, amplified:

- **Confusability grows ~quadratically.** 8 intents → 28 possible confusion
  pairs; 50 → 1,225, most of them fine-grained siblings. The near-perfect
  separation you'll see on coarse, distinct intents will *not* hold uniformly.
  Mine the confusion matrix for the pairs that actually confuse and pour
  hard-negative labeled data at exactly those.
- **The rare-intent tail is a certification problem.** Many of 50 intents will
  have too few real positives to certify a floor. Do **not** ship those on the
  same footing as your common intents. Options: (a) hold them back until they
  have data, (b) use **retrieval/prototypes** (nearest-neighbor over the
  embedding) which need far fewer examples than a trained head, (c) roll them up
  into a family-level suggestion until the leaf earns its own certification.
- **Roll out in tiers, certify per-intent.** Ship the confident, high-volume,
  distinct intents now; gate the fine/rare tail behind more labeling. Report
  coverage **per intent**, never as one headline number — coverage will be
  uneven and that is the correct, honest shape at 50.

### 9.1 Hierarchy (main-intent → sub-intent) — do it in the decision policy, not as a cascade

Worth doing, mostly for **graceful degradation** (the family fallback above) and
for **containing confusion within families**. But:

- **Don't build a hard cascade** (family head gates which sub-heads run) — a
  wrong family decision locks in the error with no recovery.
- **Do keep flat leaf heads** (every leaf scores every turn, no cascade) **and
  layer the hierarchy on the decision:** aggregate leaf scores into a family
  confidence, resolve within-family competition by argmax, and fall back to the
  family suggestion when no leaf clears its floor but the family is confident.
- Hierarchy does **not** fix the data/certification wall and does **not** give
  the embedding more separating power. It's a decision-structure win, not a
  representation win. Keep expectations there.

---

## 10. When to escalate to a fine-tuned encoder (and when not to)

The design's growth path allows a fine-tuned-encoder challenger, but promote it
only on **measured** evidence, not vibes. In the rehearsal the fine-tuned
encoder (MiniLM) *lost* to the linear model on held-out data, and DeBERTa
wouldn't even train stably. Promotion bar: **+3–5 points of coverage at equal
precision, or a critical-intent win, or lower total cost.** Trigger to even try
it: a §7 representation-ceiling check showing genuinely low AUC on an intent you
must support. Absent that signal, the cheap stack wins — don't spend the money.

---

## 11. Rollout and shadow mode

- **Shadow first.** Score real production turns with zero live impact; sample
  weekly for human adjudication. This is how you get the *real* coverage number —
  which is unknowable in advance and will be lower and messier than any offline
  or rehearsal figure. Don't commit to a coverage SLA before shadow data exists.
- **Measure the incumbent too.** If an LLM (or a human) is the current baseline,
  measure *its* accuracy against human truth on the same turns. "We beat the
  baseline at lower cost" is the argument that ships; a raw accuracy number in a
  vacuum is not.
- **Report per-slice:** ASR-damaged, stitched, long, negation/correction,
  multi-intent, rare, confusable pairs. Label context-dependent and
  state-dependent slices as *expected scope gaps*, not unexplained failures, or
  you'll get judged on ground you deliberately didn't contest.

---

## 12. The checklist (things we actually tripped on)

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
- [ ] κ ≥ ~0.75 gate passed on the labeling guide before scaling labeling.
- [ ] Eval labels cleaned *before* chasing training-set size or a bigger model.
- [ ] Ran the 3 diagnostics (§7) before proposing any model upgrade.
- [ ] Rare/confusable intents rolled out in tiers, certified per-intent.

---

## Appendix A — the data you need and the metrics, in plain terms

If the rest of this doc assumed too much, start here. Every kind of data you
need, and every number you'll hear, explained from zero.

### The kinds of examples you must collect

1. **Positive examples** — turns where the customer *is* asking for something you
   support. `"i lost my card"` → label `replace_card`. You need a healthy pile of
   these **for each intent** (roughly 100–300 per common intent to train; ~30+
   per intent in each test bucket to *prove* it works). This is what teaches the
   model each intent.

2. **Negative examples** — turns where the customer is **not** making a supported
   request. `"yes that's right"`, `"my name is john smith"`, `"how are you
   today"`, or an out-of-scope ask like `"what's my interest rate"`. Label these
   with an **empty label `[]`**. These teach the model *when to stay silent*. If
   you skip them, the model fires on everything. You need lots — in real calls,
   most turns are negatives.

   - **Hard negatives** are the tricky negatives that *look* like requests but
     aren't: `"i'm calling about my card"` (topic, not a request), or a customer
     answering the agent's question. Deliberately collect these; they're what
     stops false alarms.

3. **The context that rides with each turn** — for every example above, also
   store the **previous agent utterance** and both the **raw ASR text** (what the
   machine heard) and the **human-corrected text** (what was actually said).
   You serve on the raw text; the clean text is only for diagnosis.

### The buckets you split the data into ("train / holdout")

Think of it like studying for an exam. You split your labeled data into four
piles, and a conversation only ever lands in one pile:

- **Train** (~60%) — the model *learns* from these. Like practice problems.
- **Calibration** (~15%) — used to make the model's confidence scores honest
  (turn a raw score into a real "80% sure"). Must be separate from train.
- **Policy** (~15%) — used to *set the accept/reject cutoffs* at your safety bar.
  This is the pile you tune against while iterating.
- **Holdout / Test** (~10%) — the **final exam**. You look at it **once**, at the
  very end, to get the honest score. If you peek and tune against it, you're
  grading yourself on the answer key you memorized — the number becomes a lie.

"Holdout" and "test" are the same idea: data held back so the score reflects how
the model does on calls it has **never seen**.

### The numbers you'll hear (precision, recall, F1, coverage)

Imagine the model makes 100 suggestions to agents:

- **Precision** = of the suggestions it *made*, how many were right. High
  precision = it rarely misleads the agent. (If 95 of 100 were correct →
  precision 0.95.)
- **Recall** = of all the requests that *existed*, how many it caught. High
  recall = it rarely stays silent when it should speak. (We call this
  **coverage** in this doc — same idea.)
- **F1** = a single blended score of precision and recall (their harmonic mean).
  One number, handy for quick comparisons.

**Why we do NOT optimize F1 here — this matters.** F1 treats a *false suggestion*
and a *missed request* as equally bad. For an agent-assist tool they are **not**
equal: confidently telling an agent the wrong thing is far worse than quietly
saying nothing. So instead of chasing the best F1, we **fix precision at a floor
you commit to up front** (e.g. "never below 90% right") and then **maximize
coverage** under that constraint. Precision is the safety bar you refuse to drop;
coverage is what you push up. F1 is still fine as a *secondary* health check, but
it is not the number you steer by.

So the headline you report is: **"X% coverage at ≥90% precision, on the holdout
set, measured per intent."** Not a single F1.

---

## Appendix B — porting this codebase to your dataset (what to change, what not to)

The good news: **almost none of the code changes.** The training math,
calibration, thresholding, decision policy, serving, parity, and diagnostics are
all dataset-agnostic. To run on your data you mainly write **one adapter** and
edit **a few config files**. Here's the map.

### The one file you rewrite: the ingest adapter

`trainer/semantic_poc/ingest.py` is the *only* piece that knows your raw data
format. It reads your corpus and emits a list of gold rows. Replace its body so
it reads *your* files (DB export, CSV, JSON, whatever) and produces one
`GoldRow` per customer turn. A gold row must come out with these fields filled
(schema in `trainer/semantic_poc/schema.py`):

- `conversation_id`, `turn_index`, `timestamp_ms` — for grouping + time-ordered
  splits.
- `raw_transcript` (ASR, uncleaned), `human_transcript` (corrected, or copy raw
  if you don't have corrections yet).
- `previous_agent_utterance` — the agent turn immediately before this one.
- `prev_caller_segment` — optional, for stitching split fragments.
- `labels` — start as `[]`; real labels come from your labeling pass, not ingest.
- `session_task_intent` — optional weak-label seed if you have a call-reason tag.

That's the whole contract. Everything downstream consumes `GoldRow`, so if your
adapter emits valid rows, the rest of the pipeline just works.

**Labels don't come from ingest.** In the rehearsal, weak labels came from HVB's
dialog-act tags (`weak_labels.py`) — you almost certainly won't have those. Your
labels come from §3's process (human + AI-critic adjudication). Ingest emits
`[]`; your adjudication fills the labels in. If you have *no* label seed at all,
you can bootstrap: embed, cluster, or use the AI critics to propose first-pass
labels, then human-adjudicate. Delete or replace `weak_labels.py` accordingly.

### The config files you edit (no code, just YAML in `configs/`)

- **`taxonomy.yaml`** — replace the 8 intents with your ~50, each with its
  `family`. This one file defines your label space and the family hierarchy.
- **`policy.yaml`** — set `precision_floor` (your safety bar), `accept_floor`
  (keep ~0.50), `tau_low`, `max_accepted`, and `exclusive_groups` (list the
  sibling intents that shouldn't co-fire, e.g. the card-action family).
- **`embedding.yaml`** — your embedding provider + pinned model/version/dim. If
  you use a provider other than Gemini, that's the one other code touch (see
  below).
- **`features.yaml`, `stitcher.yaml`** — usually fine as-is; tune only if needed.

### The one conditional code touch: a different embedding provider

If you don't use Gemini, edit `trainer/semantic_poc/embeddings.py` (the
`_fetch_batch` call) to hit your provider. Keep the on-disk cache and the
L2-normalization. If you use Gemini, change nothing here.

### What you run (same commands, your data)

```
cd trainer && source .venv/bin/activate   # see README.md Setup
export POC_GOLD_PATH=../data/gold/gold.duckdb   # canonical store — schema in docs/gold-schema.md
poc ingest        # your adapter -> gold.duckdb (labels blank)
poc counts        # intent balance / OOS pile
poc promote <intent> ids.txt   # label a batch (audited); repeat per intent
poc partition     # conversation-grouped time-ordered splits + frozen-test hash
poc train         # embed (cached) -> heads -> fusion -> calibrate -> threshold -> artifact
pytest             # unit tests, incl. the stitcher
python ../scripts/serve_py.py --artifact <newest>   # serve
poc replay        # sanity-check online vs offline agree
```

### The diagnostic + labeling scripts, mapped to this doc

Kept in this repo, pointed at your data:

- **§6 honest scoring** — `scripts/held_out_eval.py` (held-out coverage at the
  floor, per intent) and `scripts/stability_report.py` (per-intent metrics tracked
  run-over-run; the regression gate as you add intents).
- **labeling** — `poc promote` / `store.relabel` write labels into `gold.duckdb`
  with an in-file audit trail; `poc counts` / `poc snapshot` for balance and
  restore points. Full flow in `docs/gold-schema.md`.

Kept only in git history (rehearsal-era harnesses — recover a file if you want to
rerun that specific analysis):

- **§3 labeling/adjudication** — the dual-blind-critic method ran as an agent
  workflow (`adj_prep.py` built the batches, `adj_apply.py` applied the verdicts);
  bootstrap stubs lived in `port-template/`. Port the *method*; your runner differs.
- **§7 diagnostics** — `ceilings.py` (representation AUC/AP, ASR ceiling,
  certification), `fp_audit.py` (the mislabel audit that cracked the rehearsal),
  `miss_analysis.py` (per-miss breakdown), `asr_damage.py` (catch rate vs ASR damage).
- **§10 challenger** — `challenger.py` fine-tuned-encoder head-to-head (needed a
  separate `challenger-env/`; didn't clear the bar — see `docs/feasibility-verdict.md`).

### Reproduce our *result*, not our numbers

Following the above on your data reproduces the **method and the outcome shape**
— a certified, per-intent coverage number at your precision floor, with the
diagnostics to know whether any gap is labels/audio/model. It will **not**
reproduce our 0.92; your number is yours to discover (and lower/messier — §11).
The transferable promise is: *if you get the labels right and follow the
discipline, this cheap architecture will tell you honestly how far it can go on
your data, per intent.*

### Minimal port checklist

- [ ] Rewrote `ingest.py` to emit valid `GoldRow`s from your corpus.
- [ ] Replaced `taxonomy.yaml` with your intents + families.
- [ ] Set `policy.yaml` (floor, accept_floor, exclusive_groups).
- [ ] Pointed `embedding.yaml` (and provider code, if not Gemini) at your embedder.
- [ ] Ran your labeling pass to fill `labels` (`poc promote`; not weak heuristics).
- [ ] `poc partition` → `poc train` → `pytest` green on your data.
- [ ] `held_out_eval.py` for the per-intent number; `stability_report.py` as you add intents.

---

## 13. TL;DR for the exec conversation

Buy **good labels** (especially a clean evaluation set), keep the **cheap
model**, roll out the 50 intents **gradually with a safety net**, and measure
**each intent honestly** instead of promising a single accuracy number. The
rehearsal's proof: fixing labels alone moved coverage from 52% to 92% with no
model change, and a fine-tuned neural network *lost* to the simple one. The
money belongs in data quality and disciplined rollout, not model horsepower.
