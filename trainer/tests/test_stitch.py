from semantic_poc.stitch import PrevSegment, StitcherConfig, stitch

CFG = StitcherConfig(
    max_gap_ms=2000,
    prev_max_tokens=6,
    incompleteness_cues=["to", "and", "my", "the", "um"],
    combined_max_tokens=120,
)


def prev(text: str, decision: str, end_ts: int) -> PrevSegment:
    return PrevSegment(text=text, turn_index=16, end_ts_ms=end_ts, decision=decision)


def test_no_prev_segment():
    out = stitch(None, "lock my card", 1000, CFG)
    assert not out.stitched
    assert out.text == "lock my card"


def test_short_prev_within_gap_merges():
    out = stitch(prev("i want to", "abstained", 10_000), "lock my card", 11_000, CFG)
    assert out.stitched
    assert out.text == "i want to lock my card"
    assert out.supersedes_index == 16
    assert out.rule_fired == "prev_short"


def test_gap_too_large_rejects():
    out = stitch(prev("i want to", "abstained", 10_000), "lock my card", 12_500, CFG)
    assert not out.stitched


def test_negative_gap_rejects():
    out = stitch(prev("i want to", "abstained", 12_000), "lock my card", 11_000, CFG)
    assert not out.stitched


def test_incompleteness_cue_on_long_prev():
    p = prev("i was hoping you could help me change the", "accepted", 10_000)
    out = stitch(p, "mailing address", 10_500, CFG)
    assert out.stitched
    assert out.rule_fired == "incompleteness_cue"


def test_unresolved_prev_decision_merges():
    p = prev("something about the card thing being broken", "no_supported_intent", 10_000)
    out = stitch(p, "i need a replacement", 10_500, CFG)
    assert out.stitched
    assert out.rule_fired == "prev_unresolved"


def test_resolved_long_prev_without_cue_rejects():
    p = prev("i already told you about my address change", "accepted", 10_000)
    out = stitch(p, "and the card", 10_500, CFG)
    assert not out.stitched


def test_combined_token_cap_rejects():
    small = StitcherConfig(max_gap_ms=CFG.max_gap_ms, prev_max_tokens=CFG.prev_max_tokens,
                           incompleteness_cues=CFG.incompleteness_cues, combined_max_tokens=5)
    out = stitch(prev("i want to", "abstained", 10_000), "lock my card now", 10_500, small)
    assert not out.stitched


def test_from_config_matches_stitcher_yaml_shape():
    cfg = StitcherConfig.from_config({
        "max_gap_ms": 2000, "prev_max_tokens": 6,
        "incompleteness_cues": ["to", "and"], "combined_max_tokens": 120,
    })
    assert cfg.max_gap_ms == 2000
    assert cfg.incompleteness_cues == ["to", "and"]
