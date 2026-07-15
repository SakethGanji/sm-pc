# Synthetic training-data track (rehearsal)

## Premise and honest expectations

Measured so far: label *quality* moved coverage 0.13 → 0.40; training *volume* is
saturated (flat learning curve); the binding constraints are (a) the
transfer_money/pay_bill confusable pair and (b) real policy-split positives for
threshold certification.

Synthetic data therefore targets (a) and the design doc's challenge-table
phenomena — it cannot touch (b): **synthetic rows go into train ONLY, never
calibration/policy/test.** Precision floors are only ever certified on real,
adjudicated data. Expected outcome: better class separation on confusables and
hard slices → higher probabilities on *real* positives → more intents clear
their floors on the real policy split. Modest macro gain, biggest on the
confusable pair; a null result is possible and the ablation gate handles it.

(At work this entire track is "generative model used offline only" — same
category as LLM-proposed labels, but synthetic *data generation* still needs
its own policy sign-off. At home: no constraint.)

## Stage 1 — Generation (agent fan-out)

One generator agent per (intent × phenomenon) cell, each producing ~30–60
utterances with high diversity (personas, phrasing, verbosity). Phenomena per
the doc's challenge table:

| phenomenon | notes |
|---|---|
| plain requests | varied phrasings beyond the scripted corpus |
| confusable stress | transfer_money vs pay_bill pairs written to be *minimally* different ("pay my electric bill" / "send money to my landlord") — labeled carefully, ~300 per side |
| agent-context-dependent | agent question + short caller fragment ("yes it still hasn't arrived") where the label needs C1 |
| agent-rejection/correction | agent guesses wrong, caller corrects — label = corrected intent (§6.2 requires these so the model doesn't parrot the agent) |
| negation | "i don't need to order checks i need to check my balance" → check_balance only |
| multi-intent | genuine two-label turns ("transfer money and also what are your hours") — exercises multilabel |
| hard negatives | adjacent-sounding non-requests + out-of-scope banking asks → no label |
| fragment pairs | prev-segment + completion pairs for stitcher-consistent training text |

Each generator emits JSONL: clean utterance, agent context (or ""), labels
(0, 1, or 2 intents), phenomenon tag.

## Stage 2 — ASR corruption (make it sound like Chirp, not like an LLM)

Every clean utterance gets 1–2 corrupted variants; corrupted variants are what
we train on (raw ASR text is never cleaned — same principle, mirrored):
- lowercase, strip punctuation (matches corpus)
- disfluency injection (uh / um / false starts) — programmatic
- valid-word phonetic substitutions ("card"→"car", "checks"→"chex"→"checks"?
  agent-generated homophone map per vocabulary, applied programmatically at
  ~5–10% word rate)
- word deletions at ~3–5% rate
Deterministic seed per row (reproducible; no Date/random drift in artifacts).

## Stage 3 — Adversarial verification (agents police agents)

Independent critic agents re-label every synthetic row blind (see text +
context, NOT the intended label). Keep a row only if critic label == intended
label; genuine-ambiguity rejects are logged (they often indicate a bad
template, not a bad row). Near-duplicate filter (n-gram overlap) within and
against the real corpus. Target survival ≈ 80%; ~2.5–4k rows kept.

## Stage 4 — Merge, embed, retrain

- New gold rows: `conversation_id="synth-<batch>-<n>"`, `split="train"`
  (hard-coded; partitioner never reassigns synth conversations),
  `sample_reasons=["synthetic", "<phenomenon>"]`.
- Embed via the normal cached Gemini path (~4k texts ≈ pennies).
- `poc train` unchanged; artifact manifest records synthetic row counts.

## Stage 5 — Ablation gate (the doc's keep/drop discipline)

Train two artifacts from identical real data: with-synth vs without-synth.
Judged ONLY on the real policy split:
- coverage at the 0.90 floor (macro + per intent)
- the confusable pair specifically (does transfer_money's floor become
  attainable? does pay_bill hold?)
- no regression on any currently-passing intent
Keep synthetic data only if it wins. Report either way — a measured null is a
useful Part-1 input ("synthetic augmentation tried, gain was X").

## What this deliberately does not claim

- It does not add real-world variety — prod coverage numbers still come from
  prod shadow data only (doc risk #4).
- It does not grow measurement data — the path to relaxed thresholds remains
  real adjudicated calibration/policy examples.
- LLM-generated text differs from real Chirp output even after corruption;
  the ablation on real policy data is the only verdict that counts.
