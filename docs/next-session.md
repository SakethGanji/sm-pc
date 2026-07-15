# Next session: ceiling analysis — can this architecture reach very high performance?

## Where we are (end of 2026-07-15 session)

- Full rehearsal pipeline works e2e: Python trainer -> language-neutral artifact ->
  Go server, real Gemini embeddings, parity 1e-9, replay 0 mismatches, ~150-350 ms.
- Live artifact: `artifact-poc-hvb-001-7ec9fd8a` (server :8080).
- Coverage at the 0.90-precision floor: 0.13 (weak labels) -> 0.40 (adjudicated
  labels) -> 0.52 (+938 verified synthetic train rows). Precision held everywhere.
- Learning curve on real train volume: flat. transfer_money floor unattainable
  (9 policy positives — certification wall, model itself scores 0.99 on its
  slam dunks).
- Miss analysis (`scripts/miss_analysis.py`, 77/165 policy positives missed):
  ~40% threshold-locked slam dunks (certification data too small), ~40%
  "call-opening wrapping" pattern ([noise] + "hi my name is X" preamble +
  garbled agent greeting — under-trained, synthetically fixable), ~20% hard
  ASR mangling ("i will lie to reset my password") where abstention is arguably
  correct.

User verdict: 52% is not good enough. Next session decides whether THIS setup
(linear heads over frozen Gemini embeddings + calibration/threshold policy) can
reach very high performance, or whether it is architecturally capped.

## Step 0 — define "very high performance" (precommit, §3 discipline)

Ask the user for concrete targets BEFORE running anything, e.g. "X% coverage of
request-bearing turns at 0.90/0.95 precision." Everything below reports against
that number.

## The feasibility program (each step isolates one candidate ceiling)

1. **Certification ceiling** — oracle-threshold experiment: recall at floor if
   thresholds were set with unlimited grading data (point-estimate on the full
   policy+test pool vs the certified thresholds). Gap = coverage recoverable
   purely by labeling more real evaluation data. (Cheap: cached embeddings.)
2. **Representation ceiling** — per-intent ranking quality (AP/AUC on policy
   split) of the current fused scores. High AP + low certified coverage =>
   data problem, architecture fine. Low AP on any intent => representation
   problem on that intent.
3. **ASR ceiling** — re-run eval scoring `human_transcript` instead of
   `raw_transcript` (both are in every gold row). The delta bounds what any
   text classifier can do given Chirp-quality input; that slice belongs to the
   doc's ASR-failure category, not classifier failure.
4. **Training-gap closure** — bucket-2 synthetic batch: wrap requests in
   [noise] tokens, name preambles, garbled agent greetings (workflow pattern
   from docs/synthetic-data-plan.md; generators + blind critics). Expected to
   be the cheapest remaining coverage win.
5. **Challenger** — fine-tuned encoder head-to-head (doc §12 growth path,
   promotion bar: +3-5 pts coverage at equal precision). RTX 3080 available;
   e.g. MiniLM/DeBERTa-v3-small fine-tuned on the same train split, judged on
   the same real policy split. If the challenger crushes LR-on-embeddings =>
   current setup is the cap. If it doesn't => data remains the cap and the
   cheap architecture is vindicated.
6. **Synthesis** — write the feasibility verdict: for each ceiling, how many
   coverage points it explains, and the Green/Yellow/Red call per doc Gate 4
   (with the caveat that home-corpus numbers bound mechanisms, not prod
   coverage).

## Session hygiene

- Frozen test split still untouched (hash in data/gold/partitions.json) — keep
  it that way until a final verdict artifact; do all iteration on policy.
- gold.real.jsonl = adjudicated real-only baseline; gold.jsonl currently
  includes synth-b1 train rows.
- GEMINI_API_KEY (paid) + GEMINI_API_KEY_ALT in gitignored .env; embedding
  cache data/cache/embeddings.sqlite (~13k texts, avoid re-billing).
- Go at ~/.local/go/bin; artifacts don't sort lexically — newest by mtime or
  ask /healthz.
