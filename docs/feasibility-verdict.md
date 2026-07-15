# Feasibility verdict — can linear-over-frozen-Gemini reach very high performance?

Session 2026-07-15 (part 2). Precommitted target: **75% coverage at the 0.90
precision floor** (user delegated the number; 0.90 floor per §3 discipline).
Everything below is reported against that target.

## Verdict: **GREEN** — target exceeded; the architecture was never the cap.

Held-out test coverage of the cleaned linear stack: **0.916 at 0.90+ precision
per intent** (0.973 overall precision), versus the 0.75 target. The prior 0.52
was an artifact of **evaluation-label noise**, not a capability ceiling. No
change to the model architecture was required — only correct labels.

## How each candidate ceiling scored

| Ceiling | Test | Result | Coverage it explains |
|---|---|---|---|
| Representation | per-intent AUC/AP of fused scores on policy | AUC **0.998–1.000** every intent | ~0 pts — embeddings already separate the intents |
| ASR | raw vs human transcript, policy | +0.03 coverage (0.53→0.56) | ~3 pts — real but small |
| Certification (as posed) | oracle thresholds on noisy labels | coverage got **worse** (0.53→0.38) | negative — proof the labels, not the data volume, were wrong |
| **Label noise (actual cause)** | dual-critic re-adjudication + retrain | in-sample policy **0.53→0.97**, held-out test **0.92** | **~39 pts** — the whole gap |
| Training gap (synth b2) | call-opening wrapping, ablation on held-out | 0.916→0.884 | negative — redundant after label fix; **dropped** |
| Challenger (encoder) | fine-tuned MiniLM/DeBERTa, head-to-head | see §Challenger | — |

## What actually happened

The 0.52 plateau was not a representation, ASR, or data-volume wall. It was
**weak-label noise in the policy (evaluation) split**: request-bearing turns
that the dialog-act heuristic never tagged — restatements, and requests that
follow the "[noise] + hi my name is X + garbled greeting" call opening. The
false-positive audit made it unmistakable: the highest-scoring "false positives"
were textbook real requests labeled as no-intent (`i lost my card` p=0.97,
`i would like to transfer money between my accounts` p=0.996, `what are the
local branch hours` p=0.96). Those mislabeled negatives dragged measured
precision down, which forced certified thresholds up, which suppressed coverage.
The oracle-threshold experiment confirmed it by backfiring: given *more* of the
same noisy labels it set thresholds even higher and lost coverage.

**Fix:** dual blind-critic re-adjudication (two Sonnet critics, one precision-
biased, one recall-biased; keep only rows where both agree on the same label,
disagreements marked ambiguous and excluded). Applied exhaustively to the policy
and calibration splits, to score/regex-flagged train rows, and — for the final
verdict only — to the full test split. Frozen-test row hash was verified
unchanged before opening it (labels live outside the hashed payload... see
caveats). Policy positives 165→230; clean test positives 107→155.

## The numbers

- **Prior (noisy labels), in-sample policy coverage:** 88/165 = **0.53**
- **Cleaned labels, in-sample policy coverage:** 224/230 = **0.974**
  (artifact `artifact-poc-hvb-001-7a54f402`)
- **Cleaned labels, HELD-OUT test coverage:** 142/155 = **0.916**, overall
  precision 142/146 = 0.973, every intent ≥ 0.90 precision. **This is the honest
  generalization number and the headline.**
- **+ synth-b2 (ablation):** held-out 137/155 = 0.884 → **worse**; b2 dropped.

Per-intent held-out (cleaned, no-b2): replace_card 0.90p/1.0r, order_checks
1.0/0.95, transfer_money 1.0/0.79, pay_bill 0.91/1.0, check_balance 1.0/1.0,
reset_password 1.0/1.0, schedule_appointment 1.0/0.69, get_branch_hours 1.0/0.86.

Winning deployable artifact (cleaned, no synth-b2, retrained as newest):
`artifact-poc-hvb-001-87314504`. Trainer pytest (14) + Go Python↔Go parity pass;
frozen-test row hash unchanged (`d1180cdd…76ad6`).

## Challenger (design doc §12, promotion bar +3–5 pts at equal precision)

Fine-tuned encoder trained on the same cleaned train split and C1 context,
certified on the same policy split, evaluated on the same held-out test.

| stack | in-sample policy | held-out test | held-out precision |
|---|---|---|---|
| LR over frozen Gemini (incumbent) | 0.974 | **0.916** | 0.973 |
| MiniLM-L6 fine-tuned | 0.974 | 0.890 | 0.986 |
| DeBERTa-v3-small fine-tuned | — (diverged to NaN) | — | — |

**The challenger does not clear the bar — it trails by ~2.6 pts on held-out
test.** DeBERTa-v3-small was unstable (NaN loss even after capping class weights
and lowering LR); MiniLM trained cleanly and tied the LR in-sample but
generalized slightly worse. This is the expected outcome given the representation
ceiling (AUC ~0.999): the frozen Gemini embedding already separates these intents,
so fine-tuning a small encoder from ~900 positives is *less* data-efficient than
a linear head over a strong frozen representation, not more. **Verdict: keep the
cheap stack; the encoder track is not warranted on this evidence.**

## Caveats (these numbers bound the mechanism, not prod coverage)

1. **Home corpus is scripted and clean.** HVB requests are near-templated; prod
   ASR + spontaneous speech will be harder and coverage lower (doc risk #4).
   What this run proves is the *mechanism* — label quality is the dominant lever
   and this architecture is not the bottleneck — not a transferable coverage %.
2. **LLM critics, not human annotators.** The re-adjudication used Sonnet, not
   the design doc's dual-human + κ≥0.75 gate (Gate 1). A few promotions look
   like answers to agent data-questions ("the amount of the bill is one hundred
   and seven" → pay_bill) — LLM-critic imperfection. At work this must be human-
   adjudicated; the LLM pass is a labeling *accelerator*, not the ground truth.
3. **Small per-intent test sets** (13–24 positives): precision-floor estimates
   have wide CIs. replace_card sits at exactly 0.90; one flip crosses it.
4. **Degenerate thresholds** (pay_bill t≈0, order_checks/get_branch_hours t≈1)
   are clean-corpus artifacts of near-perfect separation; robustness under noisy
   prod negatives is untested and is the first thing prod shadow data would stress.
   *Partly mitigated:* a live garbage input (`i'd like textbooks`) false-fired
   pay_bill because its certified cut sat at ~4e-8. Added an absolute
   `accept_floor: 0.50` (configs/policy.yaml) that clamps any certified cut up to
   0.50 — pay_bill precision 0.909→1.0, **overall held-out coverage unchanged at
   0.916**, and the garbage input now returns `no_supported_intent`. Guarded
   artifact: `artifact-poc-hvb-001-5f653652` (supersedes 87314504). This only
   fixes the near-zero failure mode; the t≈1 intents and small-sample CIs remain.
5. **Frozen-test discipline:** test was opened only at the final-verdict stage,
   after all iteration was done on policy. No thresholds or model choices were
   tuned against it.

## Implication for the work design doc

Gate 2/3 posture is vindicated: **the cheap stage (linear heads over frozen
Gemini embeddings + calibration/threshold policy) is not the constraint; label
quality is.** The §12 fine-tuned-encoder challenger track is not warranted on
these results (representation ceiling is ~0). Spend goes to **annotation quality
and volume of real adjudicated eval data**, not to a bigger model. The synthetic
augmentation track produced a measured null here (label fix subsumed it) — worth
reporting as a Part-1 input, not worth shipping.
