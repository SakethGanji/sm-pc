# Getting started — read this first

If you're picking up this project to build it on real (office) data, start here,
then follow `docs/office-playbook.md` for the detail. This page sets honest
expectations and gives you a concrete first week.

## The one-paragraph reality check

The **machine is turnkey**; the **labeling is not**, and the labeling is what
determines whether you get a good result. You can have the full pipeline running
on your data in about a day of engineering — one file to rewrite, two configs to
edit, a smoke test to confirm, then repoint at real data. Everything downstream
(embedding, training, calibration, thresholds, serving, diagnostics) is
dataset-agnostic and runs unchanged. But getting a *good number* out of it is a
**labeling project, not a code project** — realistically weeks of human
adjudication, not hours. That is by design: this whole codebase's founding
lesson is that label quality is the ceiling (it moved the rehearsal from 0.52 to
0.92 with zero model change). No setup can shortcut it, because it needs your
data and your people.

## What's easy (already done for you)

- **The port surface is tiny.** Rewrite `trainer/semantic_poc/ingest.py` (numbered
  TODOs), edit `configs/taxonomy.yaml` + `configs/policy.yaml`. That's it for code.
- **Downstream is dataset-agnostic.** Train/calibrate/threshold/serve/parity and
  all diagnostics run as-is.
- **The discipline is written down.** `docs/office-playbook.md` has the data spec,
  labeling method, split hygiene, diagnostics, 50-intent + hierarchy guidance, a
  metrics glossary, and a code-port map. Checklists catch the traps.
- **Reference implementations exist.** Every diagnostic/experiment from the
  rehearsal is a script in `scripts/` you point at your data.
- **A runnable example ships.** `port-template/` has commented stubs, fake sample
  data, and a green smoke test — see the shape work before touching real data.

## What's still on you (cannot be shortcut)

1. **Labeling — the real work, ~80% of the effort.** Collect positives AND
   negatives, run the two-critic adjudication, human-review disagreements, gate
   on inter-annotator agreement (κ ≥ ~0.75). Weeks, not hours, especially across
   ~50 intents with a rare tail.
2. **The ingest adapter against your real schema.** The stub is a skeleton; real
   transcripts bring quirks (diarization errors, missing timestamps, encodings).
   Expect to iterate.
3. **Embedding provider approval, quota, and cost** at your call volume — an
   ops/governance task.
4. **The confusable-pair and rare-intent tail** at 50 intents — per-intent
   labeling mined from the confusion matrix.

## Your first week (concrete)

- **Day 1 — prove the machine.** `cd trainer && uv run python ../port-template/smoke_test.py`.
  Watch valid gold rows come out of the fake corpus. Read `port-template/README.md`.
- **Day 1–2 — write the adapter.** Repoint `ingest.py`'s `_load_conversations` and
  `_read_turn` at your data. Run `uv run poc ingest`; eyeball `data/gold/gold.jsonl`
  — customer turns only, raw text uncleaned, negatives present, labels empty.
- **Day 2–3 — label a starter set BY HAND.** Take ~200 turns spanning your top
  intents + plenty of negatives, and label them carefully yourself. Small, but
  real. This is your seed and your feel for the guidelines.
- **Day 3 — first train.** Fill `taxonomy.yaml`, set `policy.yaml`, then
  `uv run poc partition && uv run poc train`. You'll get a (thin, noisy) first
  artifact. Don't trust the number yet.
- **Day 4 — run the diagnostics, not your hopes.** `scripts/ceilings.py`,
  `scripts/fp_audit.py`, `scripts/held_out_eval.py`. Read whether your gaps are
  label / audio / model. On this little data it's mostly "need more labels" —
  that's expected and it's the signal to scale labeling.
- **Day 5 — stand up the adjudication loop.** `port-template/bootstrap_labels.py`
  build → your two-critic pass → apply. This is the engine you'll run for weeks
  to grow clean labels. Get it working small, then scale it.

## The single rule that, if broken, wastes all of the above

**Clean your evaluation labels first, gate on agreement, and never let a model's
own guess become its own label.** Noisy *eval* labels don't just lower your score
— they lie to you about how good you are and mis-set your thresholds (that was
the entire 0.52 plateau). If someone skips this to "move faster," you will
rediscover the plateau from scratch and blame the model.

## What to expect from the result

You will **not** reproduce the rehearsal's 0.92 — that was 8 clean, scripted
intents. You'll get your own number: lower, messier, uneven across intents, and
only trustworthy after shadow-mode data on real traffic. Report it **per intent,
at a precommitted precision floor, on the held-out set** — never as a single
headline accuracy. The transferable promise is the method, not the figure.
