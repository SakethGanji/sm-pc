"""Cross-language parity fixtures: real policy-split rows with their real
Gemini embeddings and every intermediate value of the forward pass. The Go
server's parity test replays these and must match."""

from .runtime import Scorer
from .schema import GoldRow


def build_fixtures(rows: list[GoldRow], embeddings, scorer: Scorer,
                   n_labeled: int = 12, n_unlabeled: int = 8) -> list[dict]:
    picked: list[tuple[GoldRow, int]] = []
    lab = unlab = 0
    for idx, r in enumerate(rows):
        if r.labels and lab < n_labeled:
            picked.append((r, idx))
            lab += 1
        elif not r.labels and not r.ambiguous and unlab < n_unlabeled:
            picked.append((r, idx))
            unlab += 1
    out = []
    for r, idx in picked:
        emb = embeddings[idx]
        fwd = scorer.forward(r.raw_transcript, r.previous_agent_utterance, emb)
        out.append({
            "conversation_id": r.conversation_id,
            "turn_index": r.turn_index,
            "raw_transcript": r.raw_transcript,
            "previous_agent_utterance": r.previous_agent_utterance,
            "labels": r.labels,
            "embedding": [float(x) for x in emb],
            "expected": fwd,
        })
    return out
