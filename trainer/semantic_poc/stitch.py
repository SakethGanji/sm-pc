"""§6.1 fragment stitcher: deterministic config rules, pure function. Formerly
mirrored by the Go server (server/internal/stitch); now the only copy since
the POC serves from Python. ASR confidence is logged upstream, never gating."""

from dataclasses import dataclass, field

from .features import tokenize


@dataclass
class PrevSegment:
    text: str
    turn_index: int
    end_ts_ms: int
    decision: str


@dataclass
class StitcherConfig:
    max_gap_ms: int
    prev_max_tokens: int
    incompleteness_cues: list[str] = field(default_factory=list)
    combined_max_tokens: int = 120

    @classmethod
    def from_config(cls, cfg: dict) -> "StitcherConfig":
        return cls(
            max_gap_ms=cfg["max_gap_ms"],
            prev_max_tokens=cfg["prev_max_tokens"],
            incompleteness_cues=cfg["incompleteness_cues"],
            combined_max_tokens=cfg["combined_max_tokens"],
        )


@dataclass
class Outcome:
    text: str  # text to classify (stitched or original)
    stitched: bool = False
    supersedes_index: int | None = None  # set only when stitched
    gap_ms: int = 0
    rule_fired: str = ""


def stitch(prev: PrevSegment | None, cur_text: str, cur_start_ms: int,
           cfg: StitcherConfig) -> Outcome:
    """Merges prev into cur iff all §6.1 conditions hold. The caller only
    supplies prev when same conversation + speaker and no agent turn
    intervenes, so those conditions are given by the contract."""
    out = Outcome(text=cur_text)
    if prev is None:
        return out
    gap = cur_start_ms - prev.end_ts_ms
    out.gap_ms = gap
    if gap < 0 or gap > cfg.max_gap_ms:
        return out
    prev_toks = tokenize(prev.text)
    if len(prev_toks) <= cfg.prev_max_tokens:
        rule = "prev_short"
    elif prev_toks and prev_toks[-1] in cfg.incompleteness_cues:
        rule = "incompleteness_cue"
    elif prev.decision in ("abstained", "no_supported_intent"):
        rule = "prev_unresolved"
    else:
        return out
    combined = prev.text + " " + cur_text
    if len(tokenize(combined)) > cfg.combined_max_tokens:
        return out
    out.text = combined
    out.stitched = True
    out.supersedes_index = prev.turn_index
    out.rule_fired = rule
    return out
