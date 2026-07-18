"""
Smoke test for the ingest adapter — proves the template produces valid gold rows
from the fake corpus in sample_data/ BEFORE you wire in real data.

Run from the repo root:
    cd trainer && source .venv/bin/activate && python ../port-template/smoke_test.py

What it checks (the invariants your real adapter must also satisfy):
  - only CUSTOMER turns become rows (agent turns are context)
  - raw text is carried through uncleaned; human text rides alongside
  - previous_agent_utterance is populated from the preceding agent turn
  - consecutive customer fragments get a prev_caller_segment (stitch candidate)
  - labels come out EMPTY (labeling is a separate, human-gated step)
"""

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "trainer"))   # make semantic_poc importable

# Load the template ingest.py directly from this folder.
spec = importlib.util.spec_from_file_location("port_ingest", HERE / "ingest.py")
ingest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ingest)

rows = ingest.ingest_corpus(HERE / "sample_data" / "calls")

print(f"\n{'='*72}\nProduced {len(rows)} gold rows from sample_data/\n{'='*72}")
for r in rows:
    stitch = f"  (+stitch: {r.prev_caller_segment.text!r})" if r.prev_caller_segment else ""
    print(f"\n[{r.conversation_id} turn {r.turn_index}]  labels={r.labels}")
    print(f"   agent : {r.previous_agent_utterance!r}")
    print(f"   raw   : {r.raw_transcript!r}{stitch}")
    if r.human_transcript != r.raw_transcript:
        print(f"   human : {r.human_transcript!r}   <- ASR fixed a word here")

# --- invariants ---
assert len(rows) == 9, f"expected 9 customer rows, got {len(rows)}"
assert all(r.labels == [] for r in rows), "ingest must leave labels empty"
assert all(r.raw_transcript for r in rows), "raw_transcript must be present"
lost = next(r for r in rows if "lost my debit" in r.raw_transcript)
assert lost.previous_agent_utterance, "previous_agent_utterance should be populated"
assert lost.prev_caller_segment is not None, "consecutive customer turn should stitch"
assert lost.human_transcript != lost.raw_transcript, "human transcript should differ (cart->card)"
print(f"\n{'='*72}\nALL INVARIANTS PASSED — the adapter shape is correct.\n"
      f"Next: point _load_conversations/_read_turn at your real data and re-run.\n{'='*72}")
