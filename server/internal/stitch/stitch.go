// Package stitch implements the §6.1 fragment stitcher: deterministic config
// rules, pure function. ASR confidence is logged upstream, never gating.
package stitch

import (
	"slices"

	"semantic-poc/server/internal/artifact"
	"semantic-poc/server/internal/features"
)

type PrevSegment struct {
	Text      string `json:"text"`
	TurnIndex int    `json:"turn_index"`
	EndTsMs   int64  `json:"end_ts_ms"`
	Decision  string `json:"decision"`
}

type Outcome struct {
	Text            string // text to classify (stitched or original)
	Stitched        bool
	SupersedesIndex int // valid only when Stitched
	GapMs           int64
	RuleFired       string
}

// Stitch merges prev into cur iff all §6.1 conditions hold. The upstream
// conversation owner only supplies prev when same conversation + speaker and
// no agent turn intervenes, so those conditions are given by the contract.
func Stitch(prev *PrevSegment, curText string, curStartMs int64, cfg artifact.StitcherConfig) Outcome {
	out := Outcome{Text: curText}
	if prev == nil {
		return out
	}
	gap := curStartMs - prev.EndTsMs
	out.GapMs = gap
	if gap < 0 || gap > cfg.MaxGapMs {
		return out
	}
	prevToks := features.Tokenize(prev.Text)
	rule := ""
	switch {
	case len(prevToks) <= cfg.PrevMaxTokens:
		rule = "prev_short"
	case len(prevToks) > 0 && slices.Contains(cfg.IncompletenessCues, prevToks[len(prevToks)-1]):
		rule = "incompleteness_cue"
	case prev.Decision == "abstained" || prev.Decision == "no_supported_intent":
		rule = "prev_unresolved"
	default:
		return out
	}
	combined := prev.Text + " " + curText
	if len(features.Tokenize(combined)) > cfg.CombinedMaxTokens {
		return out
	}
	out.Text = combined
	out.Stitched = true
	out.SupersedesIndex = prev.TurnIndex
	out.RuleFired = rule
	return out
}
