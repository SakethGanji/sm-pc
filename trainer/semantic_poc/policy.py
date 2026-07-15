"""§6.9 decision policy — pure function; mirrored in Go and parity-tested."""

from typing import Any


def decide(
    probs: dict[str, float],
    thresholds: dict[str, float],
    families: dict[str, str],
    tau_low: float,
    max_accepted: int,
    exclusive_groups: list[list[str]],
) -> dict[str, Any]:
    cands = {i: p for i, p in probs.items() if p >= thresholds[i]}
    for group in exclusive_groups:
        members = [i for i in group if i in cands]
        if len(members) > 1:
            keep = max(members, key=lambda i: cands[i])
            for i in members:
                if i != keep:
                    del cands[i]
    ordered = sorted(cands, key=lambda i: -cands[i])
    overflow = len(ordered) > max_accepted
    ordered = ordered[:max_accepted]

    if not ordered:
        decision = "no_supported_intent" if max(probs.values()) < tau_low else "abstained"
    else:
        decision = "accepted" if len(ordered) == 1 else "multi_accepted"
    return {
        "decision": decision,
        "intents": [
            {"name": i, "probability": round(cands[i], 6),
             "threshold": thresholds[i], "main_intent": families[i]}
            for i in ordered
        ],
        "overflow": overflow,
    }
