// Package policy mirrors trainer/semantic_poc/policy.py (§6.9).
package policy

import (
	"math"
	"sort"
)

type Intent struct {
	Name        string  `json:"name"`
	Probability float64 `json:"probability"`
	Threshold   float64 `json:"threshold"`
	MainIntent  string  `json:"main_intent"`
}

type Result struct {
	Decision string   `json:"decision"`
	Intents  []Intent `json:"intents"`
	Overflow bool     `json:"-"`
}

func Decide(probs, thresholds map[string]float64, families map[string]string,
	tauLow float64, maxAccepted int, exclusiveGroups [][]string) Result {

	cands := map[string]float64{}
	maxP := 0.0
	for i, p := range probs {
		if p > maxP {
			maxP = p
		}
		if p >= thresholds[i] {
			cands[i] = p
		}
	}
	for _, group := range exclusiveGroups {
		var members []string
		for _, i := range group {
			if _, ok := cands[i]; ok {
				members = append(members, i)
			}
		}
		if len(members) > 1 {
			keep := members[0]
			for _, i := range members[1:] {
				if cands[i] > cands[keep] {
					keep = i
				}
			}
			for _, i := range members {
				if i != keep {
					delete(cands, i)
				}
			}
		}
	}
	ordered := make([]string, 0, len(cands))
	for i := range cands {
		ordered = append(ordered, i)
	}
	sort.Slice(ordered, func(x, y int) bool { return cands[ordered[x]] > cands[ordered[y]] })
	overflow := len(ordered) > maxAccepted
	if overflow {
		ordered = ordered[:maxAccepted]
	}

	res := Result{Overflow: overflow}
	switch {
	case len(ordered) == 0 && maxP < tauLow:
		res.Decision = "no_supported_intent"
	case len(ordered) == 0:
		res.Decision = "abstained"
	case len(ordered) == 1:
		res.Decision = "accepted"
	default:
		res.Decision = "multi_accepted"
	}
	for _, i := range ordered {
		res.Intents = append(res.Intents, Intent{
			Name:        i,
			Probability: math.Round(cands[i]*1e6) / 1e6,
			Threshold:   thresholds[i],
			MainIntent:  families[i],
		})
	}
	return res
}
