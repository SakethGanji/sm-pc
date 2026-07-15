package stitch

import (
	"testing"

	"semantic-poc/server/internal/artifact"
)

var cfg = artifact.StitcherConfig{
	MaxGapMs:           2000,
	PrevMaxTokens:      6,
	IncompletenessCues: []string{"to", "and", "my", "the", "um"},
	CombinedMaxTokens:  120,
}

func prev(text, decision string, endTs int64) *PrevSegment {
	return &PrevSegment{Text: text, TurnIndex: 16, EndTsMs: endTs, Decision: decision}
}

func TestNoPrevSegment(t *testing.T) {
	out := Stitch(nil, "lock my card", 1000, cfg)
	if out.Stitched || out.Text != "lock my card" {
		t.Fatalf("unexpected stitch: %+v", out)
	}
}

func TestShortPrevWithinGapMerges(t *testing.T) {
	out := Stitch(prev("i want to", "abstained", 10_000), "lock my card", 11_000, cfg)
	if !out.Stitched || out.Text != "i want to lock my card" {
		t.Fatalf("expected merge, got %+v", out)
	}
	if out.SupersedesIndex != 16 || out.RuleFired != "prev_short" {
		t.Fatalf("bad metadata: %+v", out)
	}
}

func TestGapTooLargeRejects(t *testing.T) {
	out := Stitch(prev("i want to", "abstained", 10_000), "lock my card", 12_500, cfg)
	if out.Stitched {
		t.Fatalf("expected no merge on 2500ms gap, got %+v", out)
	}
}

func TestNegativeGapRejects(t *testing.T) {
	out := Stitch(prev("i want to", "abstained", 12_000), "lock my card", 11_000, cfg)
	if out.Stitched {
		t.Fatalf("expected no merge on negative gap, got %+v", out)
	}
}

func TestIncompletenessCueOnLongPrev(t *testing.T) {
	p := prev("i was hoping you could help me change the", "accepted", 10_000)
	out := Stitch(p, "mailing address", 10_500, cfg)
	if !out.Stitched || out.RuleFired != "incompleteness_cue" {
		t.Fatalf("expected cue merge, got %+v", out)
	}
}

func TestUnresolvedPrevDecisionMerges(t *testing.T) {
	p := prev("something about the card thing being broken", "no_supported_intent", 10_000)
	out := Stitch(p, "i need a replacement", 10_500, cfg)
	if !out.Stitched || out.RuleFired != "prev_unresolved" {
		t.Fatalf("expected unresolved merge, got %+v", out)
	}
}

func TestResolvedLongPrevWithoutCueRejects(t *testing.T) {
	p := prev("i already told you about my address change", "accepted", 10_000)
	out := Stitch(p, "and the card", 10_500, cfg)
	if out.Stitched {
		t.Fatalf("expected no merge, got %+v", out)
	}
}

func TestCombinedTokenCapRejects(t *testing.T) {
	small := cfg
	small.CombinedMaxTokens = 5
	out := Stitch(prev("i want to", "abstained", 10_000), "lock my card now", 10_500, small)
	if out.Stitched {
		t.Fatalf("expected cap rejection, got %+v", out)
	}
}
