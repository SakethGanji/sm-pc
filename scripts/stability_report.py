"""Track per-intent held-out metrics run-over-run, so adding an intent has a
one-command "did anything regress?" check.

Each run scores an artifact on the cleaned held-out set (via held_out_eval),
appends its per-intent precision/recall to a rolling history file, and prints a
diff vs the previous run: stable / moved / NEW / DROPPED, flagging any precision
or recall drop worse than --eps. Exits non-zero if any prior intent regressed,
so it can gate a retrain.

Usage (from trainer/, venv activated):
  python ../scripts/stability_report.py [--label "added dispute"] [--eps 0.02] [<artifact_name>]
History file: data/gold/stability_history.json (override POC_STABILITY_HISTORY).
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import held_out_eval as hoe  # noqa: E402
from semantic_poc.paths import ARTIFACTS_DIR, GOLD_DIR  # noqa: E402

HISTORY = Path(os.environ.get("POC_STABILITY_HISTORY", GOLD_DIR / "stability_history.json"))


def _fmt(x):
    return "  -  " if x is None else f"{x:.3f}"


def _delta(now, prev):
    if now is None or prev is None:
        return None
    return now - prev


def main(argv):
    label, eps, name = "", 0.02, None
    it = iter(argv)
    for a in it:
        if a == "--label":
            label = next(it, "")
        elif a == "--eps":
            eps = float(next(it, "0.02"))
        else:
            name = a
    if name is None:
        arts = sorted(ARTIFACTS_DIR.glob("artifact-*"), key=lambda p: p.stat().st_mtime)
        if not arts:
            sys.exit("no artifacts found — run `poc train` first")
        name = arts[-1].name

    clean = hoe.build_clean_test()
    E = hoe.embed_clean(clean)
    res = hoe.evaluate_artifact(name, clean, E)
    cur = {i: {"precision": d["precision"], "recall": d["recall"], "pos": d["pos"]}
           for i, d in res["per_intent"].items()}

    history = json.loads(HISTORY.read_text()) if HISTORY.exists() else []
    prev = history[-1] if history else None
    prev_pi = prev["per_intent"] if prev else {}

    print(f"\n=== stability: {name}"
          + (f'  [{label}]' if label else "")
          + (f'  vs previous [{prev.get("label") or prev["ts"]}]' if prev else "  (first run — baseline)")
          + " ===")
    print(f"  {'intent':22s} {'prec':>6s} {'Δprec':>7s}  {'rec':>6s} {'Δrec':>7s}  status")
    regressed = []
    for i in sorted(cur):
        p, r = cur[i]["precision"], cur[i]["recall"]
        dp = _delta(p, prev_pi.get(i, {}).get("precision"))
        dr = _delta(r, prev_pi.get(i, {}).get("recall"))
        if i not in prev_pi:
            status = "NEW"
        elif (dp is not None and dp < -eps) or (dr is not None and dr < -eps):
            status = "REGRESSED <<"
            regressed.append(i)
        elif (dp and abs(dp) > 1e-9) or (dr and abs(dr) > 1e-9):
            status = "moved"
        else:
            status = "stable"
        ds = lambda d: "   -   " if d is None else f"{d:+.3f}"
        print(f"  {i:22s} {_fmt(p):>6s} {ds(dp):>7s}  {_fmt(r):>6s} {ds(dr):>7s}  {status}")
    for i in sorted(set(prev_pi) - set(cur)):
        print(f"  {i:22s} {'':>6s} {'':>7s}  {'':>6s} {'':>7s}  DROPPED")

    cov = res["coverage"]
    print(f"  overall coverage: {'n/a' if cov is None else round(cov, 3)}"
          + (f"  (prev {round(prev['coverage'], 3)})" if prev and prev.get("coverage") is not None else ""))
    if regressed:
        print(f"\n  !! REGRESSION: {regressed} dropped > {eps} vs previous run")

    history.append({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "label": label, "artifact": name, "coverage": cov, "per_intent": cur})
    HISTORY.parent.mkdir(parents=True, exist_ok=True)
    HISTORY.write_text(json.dumps(history, indent=1))
    print(f"  history -> {HISTORY} ({len(history)} runs)")
    return 1 if regressed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
