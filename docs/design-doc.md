# Agent Assist Intent Classification — POC Part 1
## End-to-End Handoff: Problem Statement, Technical Design, Execution Plan

**Document status:** Final for Part 1 implementation. Self-contained — no prior design docs required.

> **Frozen record — do not edit.** Kept as evidence of the problem statement,
> scope decisions, and the precision floor as committed *before* any result was
> seen. Current operational truth is `AGENTS.md`.

> Note: this repo is the §14 rehearsal track of this document — same pipeline
> mechanics on the public HarperValleyBank corpus, real Gemini embeddings as
> the provider, Python trainer + Go server. Scope per project decisions: the
> three trained models (semantic LR, lexical LR, fusion LR) end-to-end; no kNN
> baseline, no bootstrap-LCB gating (point-estimate thresholds), no
> long-utterance path yet.

---

# 1. Problem statement

Our customer-service Agent Assist platform classifies every customer utterance into a fixed banking-intent taxonomy (8 main-intent families, each with sub-intents). Today this is done by sending every utterance, with full chat history, to a generative LLM.

**Why this is a problem:** at ~10 million utterances/day, runtime LLM classification is a major recurring cost, adds generative-model latency and variability to a real-time assist experience, and puts a generative model in a decision path that our risk environment prefers to keep deterministic and auditable.

**Why this is hard to replace:** production transcripts (Google Speech-to-Text, Chirp 2) are not clean sentences. The classifier must handle:

| Challenge | Example |
|---|---|
| Valid-word ASR substitutions | "status of the replacement **car**" (card) |
| Word deletions/insertions, missing punctuation | "replacment card still not here need new one" |
| Split final utterances | "I want to…" / "…lock my card" as two ASR segments |
| Long narratives, one relevant sentence | 60-token complaint ending in "…anyway where is my card" |
| Context-dependent fragments | Agent: "Are you asking about the replacement card?" → Customer: "Yes, it still hasn't arrived" |
| Confusable sub-intents | card_delivery_status vs card_replacement vs card_declined — near-identical vocabulary |
| Negation / correction | "I do **not** need another card, mine keeps getting declined" |
| Multiple intents in one turn | "card hasn't arrived and I need to change my address" |
| No actionable intent | "okay, thank you", "can you hold on" |
| Out-of-scope requests | products we don't support |

**Central POC question:** *At predefined per-intent precision levels, what percentage of real production customer turns can a lightweight, non-generative classifier safely classify?*

The POC runs in **shadow mode** beside the existing LLM. It changes nothing in the live experience. The LLM is used offline only: proposing labels for human review, prioritizing examples, and serving as a comparison baseline. **Human-adjudicated labels are the sole ground truth** — including for measuring the incumbent LLM itself, whose true accuracy is currently unmeasured.

---

# 2. Scope

**In scope (Part 1):** one main-intent family (**cards**), 5–7 sub-intents trained flat with main-intent derived by roll-up; multilabel outputs; fragment stitching; immediate agent context; bounded long-utterance chunking; per-intent calibration and precision-floor thresholds; abstention; shadow deployment; full evaluation report.

**Explicitly out of scope (preserved as growth paths, nothing here blocks them):** persistent conversation state; retrieval/prototype models; separate stage-2 models; generative rewrite loop (POC-2 candidate, evaluated later offline); other intent families (Part 2); runtime LLM fallback; locally hosted transformers (later challenger).

**Sub-intent set (Phase 0 confirms):** card_delivery_status, card_replacement, card_activation, card_declined, card_lost_or_stolen, card_lock. Must include the confusable pair delivery_status + replacement.

---

# 3. Precommitted success criteria (Phase 0 deliverable — signed before any model result exists)

- Minimum accepted precision **P_i per sub-intent** (critical intents higher)
- Maximum false suggestions per million turns
- Minimum commercially useful coverage within cards traffic
- p95 / p99 latency limits; sustainable peak RPS
- Maximum cost per million utterances

Precision floors are enforced as **lower confidence bounds** (conversation-level bootstrap LCB₉₅ ≥ P_i), not point estimates — an intent with 3 accepted examples and zero errors is unmeasured, not safe. Targets are **not renegotiable after results are seen.**

---

# 4. Architecture

```
Final Chirp 2 customer segment
        │
        ▼
[1] Fragment stitcher (deterministic, config rules)
        │
        ▼
[2] Context builder:  [AGENT] prev turn \n [CUSTOMER] current
        │
        ├────────────────────────────┐
        ▼                            ▼
[3] Google embedding (async)   [4] Word TF-IDF (local, parallel)
        │                            │
        ▼                            ▼
[5] Semantic OVR logistic reg  [6] Lexical OVR logistic reg (optional)
        │                            │
        └────────────┬───────────────┘
                     ▼
[7] Fusion LR (optional — kept only if ablation proves value)
                     ▼
[8] Long turns only: ≤4 overlapping chunks → same semantic LR →
    conservative fixed combination (whole-utterance support floor)
                     ▼
[9] Per-intent Platt calibration
                     ▼
[10] Per-intent thresholds (LCB-gated) + compatibility rules
                     ▼
decision: accepted | multi_accepted | no_supported_intent | abstained
(infrastructure failures: system_error / degraded — never semantic abstention)
```

**Trained models: three** (semantic LR, lexical LR, fusion LR — the latter two optional pending ablation). Google's embedding model is the fourth component in the loop but is not trained by us. No softmax — independent sigmoid per intent (multilabel: zero, one, or several intents per turn).

**Design principles:** raw ASR text is never cleaned (clean the labels, not the customer's language); every numeric parameter is config, tuned on the policy period, never hard-coded; components are pure functions over configs so offline evaluation and runtime execute identical code.

---

# 5. Contracts

### 5.1 Request
```json
{
  "conversation_id": "c-123",
  "turn_index": 17,
  "timestamp": "2026-07-13T14:04:00Z",
  "current_customer_transcript": "the replacement car still isn't here",
  "previous_agent_utterance": "When was your replacement card expected?",
  "previous_customer_segment": {"text": "I want to", "turn_index": 16,
      "end_ts_ms": 1720881830120, "decision": "abstained"},
  "asr_confidence": 0.81,
  "asr_is_final": true
}
```
Upstream conversation owner serializes turns per conversation and supplies `previous_customer_segment` so the service stays stateless.

### 5.2 Response
```json
{
  "decision": "accepted",
  "intents": [{"name": "card_delivery_status", "probability": 0.96,
               "threshold": 0.91, "main_intent": "cards"}],
  "supersedes_turn_index": 16,
  "stitched": true,
  "long_path_used": false,
  "model_version": "poc-cards-001-a3f9",
  "embedding_model_version": "<pinned id>",
  "latency_ms": {"stitch": 0, "embedding": 74, "lexical": 1,
                 "semantic": 1, "fusion": 1, "total": 81}
}
```
`supersedes_turn_index` appears whenever stitching merged a previously answered segment — consumers must void the earlier result (prevents double-counting in metrics and UI).

---

# 6. Component specifications

### 6.1 Fragment stitcher
Merge previous customer segment into current iff ALL: same conversation + speaker, no agent turn between; gap ≤ 2000 ms (config); previous segment short (≤6 tokens) OR ends in an incompleteness cue (config list: to, and, my, the, because, uh, um…) OR previously decided abstained/no_supported_intent; previous segment triggered no committed downstream action; combined ≤ 120 tokens. ASR confidence is logged, never gating. Log originals, stitched text, gap, rule fired. Offline: every stitched turn also classified unstitched → stitcher net effect (abstentions reduced vs false merges) is a Phase-2 deliverable.

### 6.2 Context representations
`C0 = [CUSTOMER] current` · `C1 = [AGENT] prev [CUSTOMER] current`. Missing agent turn → empty string + `context_available=false` flag. C0-vs-C1 is a required ablation, judged overall AND on the previous_agent_required slice. Training data must include agent-rejection/correction rows so the model doesn't parrot the agent's question.

### 6.3 Embedding provider
Single module owning all Google calls: pinned model version + classification task config (as approved in-region); async with pooling, deadline, limited jittered retries, circuit breaker; text-hash × model-version cache for training; L2-normalization applied here so downstream code never depends on provider behavior. Interface is swappable (local stand-in for rehearsal, Google for production) with zero downstream change.

### 6.4 Semantic branch
Embedding → per-intent binary LR (OVR), class_weight=balanced, L2, C swept on conversation-grouped inner folds. Output: K raw logits. A kNN baseline over the same vectors is trained alongside (near-free) — LR beating kNN on the confusable pair is a Gate-2 check.

### 6.5 Lexical branch (optional)
Word 1–2-gram TF-IDF (min_df=2, sublinear TF, vocab ≤200k) on raw current transcript → OVR LR. Runs locally during the embedding round trip. Char 2–5-grams only as an ablation, retained only if they improve real Chirp ASR slices (voice errors are valid-word substitutions, not typos — word features + context are the default bet).

### 6.6 Fusion (optional)
LR over [sem logits ⊕ lex logits ⊕ meta(token_count, context_available, stitched, long_path_used, asr_conf)]. Never receives raw embeddings or sparse vectors. **Out-of-fold training mandatory:** GroupKFold(5) by conversation → branches trained on 4 folds emit logits on the 5th → fusion fits on honest logits → branches retrained on full train for deployment. Fusion kept only if the sem/lex/fused ablation improves coverage-at-floor, a critical intent, or the ASR slice.

### 6.7 Long-utterance path
Trigger: token_count > 60 (config). ≤4 overlapping windows (24 tokens, stride 12; sentence splits when punctuation exists, <5-token fragments merged). Chunks keep the agent prefix; embedded concurrently; scored by the SAME semantic LR. Combination (fixed rule selected on policy period): `combined_i = max(whole_i, chunk_max_i)` subject to `whole_i ≥ support_floor` — a chunk alone cannot accept an intent the whole utterance contradicts ("I do NOT need another card…"). Chunk-driven acceptance uses threshold offset +δ. Worst case 1+4 concurrent embedding calls ≈ one call of wall-clock; fires only on the long minority.

### 6.8 Calibration & thresholds
Per-intent Platt on the calibration period (isotonic only ≥300 positives). Thresholds on the later untouched policy period: maximize coverage s.t. LCB₉₅(precision_i) ≥ P_i via conversation-level bootstrap. Report reliability/ECE/Brier per intent; unstable calibration → raise threshold, abstain more, flag.

### 6.9 Decision policy
Candidates = intents ≥ own threshold → drop within exclusive groups (keep argmax; compatibility.yaml is small and auditable) → cap at 3 (overflow logged) → 1=accepted, ≥2=multi_accepted, 0 with max p < τ_low = no_supported_intent, else abstained. `no_supported_intent` deliberately pools no-intent + unsupported + out-of-scope for Part 1. Main intent = roll-up (family probability = max over children; "confident family / uncertain leaf" logged as a diagnostic).

---

# 7. Data

### 7.1 Gold row
```json
{
  "conversation_id": "c-123", "turn_index": 17, "timestamp": "...",
  "raw_chirp_transcript": "the replacement car isn't here",
  "human_reference_transcript": "the replacement card isn't here",
  "previous_agent_utterance": "When was your replacement card expected?",
  "adjudicated_labels": ["card_delivery_status"],
  "context_requirement": "previous_agent",
  "mention_status": "current_request",
  "sample_reasons": ["representative", "asr_substitution", "context_required"]
}
```
Raw transcript preserved always; human-corrected transcript when available (separates ASR failure from classifier failure). Annotators distinguish current request / historical mention / negated / resolved / hypothetical / agent-suggested / correction / genuine multi-intent. One dataset serves all three models.

### 7.2 Composition & volume
Tranche 1: **3,000–5,000 adjudicated rows** (directional baseline). Full Part 1: **10,000–20,000**. Mix ≈ 60% cards sub-intents (300–500 per normal sub-intent, 1,000+ for critical/confusable), 25% hard negatives from other families ("my transfer hasn't arrived", "refund pending", "deposit missing" — things that *sound* adjacent), 15% no-intent / out-of-scope / ambiguous. Sample whole conversations, never isolated turns. Difficult-case streams: low ASR confidence, long turns, fragments, LLM-low-confidence, negations, corrections.

### 7.3 Quality gate
Dual-annotate 10%; require κ ≥ 0.75 before scaling. Failure means the taxonomy/guide is ambiguous — fix that before any model conclusion is valid.

### 7.4 Partitions
By complete conversation AND time: **train → calibration → policy-tuning → frozen test** (newest). No conversation crosses partitions. Frozen test embedded only after full model+policy freeze; evaluated exactly once; hash recorded.

---

# 8. Training pipeline (CLI → one immutable artifact)

```
1 load gold rows → partitions
2 fit TF-IDF on train only; bulk-embed train/cal/policy for C0+C1 (cached)
3 inner-fold sweeps (LR C; C0 vs C1)
4 OOF logits → fusion fit → branch retrain on full train
5 calibrators on calibration period
6 policy period: thresholds (LCB gate), chunk rule + δ + long_threshold,
  stitcher params, compatibility rules
7 freeze: artifact-{ver}-{hash8}/ = manifest.json (git sha, data hashes,
  embedding model id/task/dims, taxonomy version, config hashes) + semantic.pkl
  + tfidf.pkl + lexical.pkl + fusion.pkl + calibrators.pkl + configs + eval_report
8 embed frozen test NOW; evaluate once; slice report; load test
```
Artifact rules: immutable, hash-named, service loads exactly one, version echoed in every response; threshold change = new artifact; embedding model change = new artifact + full re-embed/retrain. **Deprecation playbook:** raw text is the source of truth; steps 2–8 are automated and rehearsed once during the POC, so a Google embedding retirement is a planned, frozen-test-gated event.

---

# 9. Service, throughput, telemetry

Async FastAPI: stitch (µs) → embedding dispatch + lexical in parallel → semantic on arrival (~1 ms) → fusion (µs) → calibrate/threshold (µs). p50 ≈ embedding RTT (~50–150 ms); long path ~200–400 ms; both far under the 1 s budget. Throughput: 10M/day ≈ 116 rps flat / ~347 rps peak-window — load-test at observed peak; embedding demand = turns × (1 + r×4) with r = long-path rate → **confirm quota with Google in Phase 0** (likely provisioned throughput). Deadline exceeded → `degraded`, never a forced intent. Stateless horizontal replicas, CPU-only (few GB RAM each; measured). Telemetry asynchronous: every request logs decision/scores/flags/versions/latency; full diagnostics sampled + all abstained/disagreement turns; transcript-bearing logs inherit the transcript pipeline's data-governance controls.

---

# 10. Evaluation & shadow mode

Both systems score selected production turns; zero live impact; weekly human-adjudication sample. Report three cuts:
1. **Overall** LLM-vs-POC-vs-human-truth (including the LLM's own accuracy vs human truth — measure the incumbent).
2. **Matched-information:** LLM restricted to prev-agent + current turn where feasible (the fair comparison).
3. **Context-requirement breakdown:** current_only / previous_agent / older_history / persistent_state / ambiguous — the last two labeled *expected Part-1 scope gaps*, included in totals, never presented as unexplained failures.

**Headline metric: coverage at required precision within cards-family traffic** (denominator from human adjudication). All-traffic numbers reported separately — a cards-only model correctly saying no_supported_intent on non-cards traffic is not low coverage. Secondary: false suggestions/million, abstention rate, per-intent P/R, per-slice performance (ASR substitutions/deletions, stitched, long, negation/correction, multi-intent, rare, confusable pairs), multilabel set metrics, calibration on accepted traffic, p50/p95/p99, calls/turn, cost/million.

---

# 11. Execution plan

### Phase 0 — Foundations (Week 1) — blocking
Scope lock (sub-intent list) · **signed precommitted targets (§3)** · annotation guide v1 · policy checks in parallel: embedding model/version/quota approval, pretrained-weights source policy (for later transformer challenger), telemetry governance sign-off.
**Gate 0:** targets signed, guide drafted, embedding access confirmed.

### Phase 1 — Data spine (Weeks 1–3)
Conversation sampler (representative + stratified + difficult streams) · gold schema + LLM-proposed/human-adjudicated flow · dual-annotation κ check · tranche 1 (3–5k rows) · partitions + frozen-test hash.
**Gate 1:** κ ≥ 0.75; tranche 1 partitioned.

### Phase 2 — Baselines (Weeks 3–4)
Embedding provider · stitcher (+ with/without analysis) · semantic LR (C0+C1) · lexical LR · kNN baseline · first report (per-intent, confusable confusion, slices, learning curves).
**Gate 2 (the big one):** confusable pair separates workably; C1 wins the context slice; LR ≥ kNN. Fail → escalate with diagnostics before further spend.

### Phase 3 — Fusion, long turns, policy (Weeks 4–6)
OOF fusion + keep/drop ablation · chunking path + negation protections · calibration + LCB thresholds + compatibility + decision assembly · targeted data expansion toward 10–20k (mine disagreements/abstentions) · retrain.
**Gate 3:** calibration stable; a meaningful sub-intent subset clears floors with nonzero coverage.

### Phase 4 — Service + shadow (Weeks 6–8)
Production-shaped service + artifact + telemetry · peak load test + quota confirmation + deprecation-playbook rehearsal · shadow deployment · **final Part-1 report** (§10 cuts + cost/latency vs signed targets).
**Gate 4 — Part-1 decision:**
- **Green** (floors met, useful coverage, cost win) → Part 2: add most-confusable second family; stage-2 candidate bake-off — non-generative enhanced stage vs generative rewrite loop (POC-2) — offline, on the same routed/abstained turns, judged on recovery rate AND rewrite fidelity.
- **Yellow** (floors met on subset, coverage thin) → targeted data expansion + immediate-context/state growth path before scaling families.
- **Red** (representation cannot separate confusables at any threshold) → tuned-embedding / fine-tuned-encoder challenger track, diagnostic evidence attached.

---

# 12. Growth paths (designed-in, not built)
Persistent active-intent state (continuation/switch/decay, causal training) · second→eighth families with cross-family confusion tracking · out_of_scope split from no_supported_intent once a diverse negative set exists · learned chunk aggregation · retrieval/prototypes for rare intents · fine-tuned encoder challenger (promoted only on ~3–5 pts coverage at equal precision, a critical-intent win, or lower total cost) · POC-2 rewrite loop as stage-2 candidate (requires reopening the no-generative-runtime decision with measured evidence and its own re-entry calibration).

# 13. Standing risks
1. **Label quality is the #1 determinant** — the κ gate is not optional; every model conclusion downstream of bad labels is void.
2. **Embedding lifecycle** — pin the version; rehearse re-embed/retrain; treat provider deprecation as a planned event.
3. **Quota at peak** — confirm provisioned throughput early, not at load test.
4. **Coverage is unknowable in advance** — that is the number the POC exists to produce; resist commitments to it before shadow data exists.
5. **Scope-gap optics** — context/state slices will lose to the LLM by design in Part 1; the report labels them as scope gaps or the POC gets judged on ground it deliberately didn't contest.

# 14. Rehearsal track (optional, parallel)
Same pipeline exercised on the public HarperValleyBank corpus + generic banking taxonomy + local embedding stand-in: proves schema, partitions, stitcher, training CLI, replay, and API before office data exists. No production data leaves the office; rehearsal code is rebuilt through approved channels at work.
