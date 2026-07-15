// Package model is the forward pass over artifact parameters — the same
// computation as trainer/semantic_poc/runtime.py (parity-tested). Steps are
// exported separately so the API layer can run lexical work during the
// embedding round trip and time each stage; Score composes them.
package model

import (
	"math"

	"semantic-poc/server/internal/artifact"
	"semantic-poc/server/internal/features"
	"semantic-poc/server/internal/policy"
)

type Forward struct {
	SemLogits   []float64
	LexLogits   []float64
	Meta        []float64
	FusedLogits []float64
	Probs       map[string]float64
	Result      policy.Result
}

func sigmoid(z float64) float64 { return 1 / (1 + math.Exp(-z)) }

func denseLogits(coef [][]float64, intercept, x []float64) []float64 {
	out := make([]float64, len(coef))
	for k, w := range coef {
		s := intercept[k]
		for j, v := range x {
			s += w[j] * v
		}
		out[k] = s
	}
	return out
}

// SemLogits scores the L2-normalized C1 embedding with the semantic branch.
func SemLogits(a *artifact.Artifact, embedding []float64) []float64 {
	return denseLogits(a.SemCoef, a.SemInt, embedding)
}

// LexLogits featurizes the (possibly stitched) raw transcript and scores it
// with the lexical branch.
func LexLogits(a *artifact.Artifact, rawText string) []float64 {
	vec := features.Transform(rawText, a.Vocab, a.Idf)
	out := make([]float64, len(a.LexCoef))
	for k, w := range a.LexCoef {
		s := a.LexInt[k]
		for j, v := range vec {
			s += w[j] * v
		}
		out[k] = s
	}
	return out
}

func Meta(rawText, prevAgent string) []float64 {
	ctx := 0.0
	if prevAgent != "" {
		ctx = 1.0
	}
	return []float64{float64(len(features.Tokenize(rawText))) / 100.0, ctx}
}

// Fuse combines branch logits + meta, applies Platt, and decides.
func Fuse(a *artifact.Artifact, sem, lex, meta []float64) ([]float64, map[string]float64, policy.Result) {
	fusIn := make([]float64, 0, len(sem)+len(lex)+len(meta))
	fusIn = append(fusIn, sem...)
	fusIn = append(fusIn, lex...)
	fusIn = append(fusIn, meta...)
	fused := denseLogits(a.FusCoef, a.FusInt, fusIn)

	probs := make(map[string]float64, len(a.Model.Intents))
	for k, intent := range a.Model.Intents {
		probs[intent] = sigmoid(a.Model.PlattA[k]*fused[k] + a.Model.PlattB[k])
	}
	res := policy.Decide(probs, a.Model.Thresholds, a.Model.Families,
		a.Model.TauLow, a.Model.MaxAccepted, a.Model.ExclusiveGroups)
	return fused, probs, res
}

// Score is the full forward pass (used by the parity test and replay).
func Score(a *artifact.Artifact, rawText, prevAgent string, embedding []float64) Forward {
	sem := SemLogits(a, embedding)
	lex := LexLogits(a, rawText)
	meta := Meta(rawText, prevAgent)
	fused, probs, res := Fuse(a, sem, lex, meta)
	return Forward{sem, lex, meta, fused, probs, res}
}
