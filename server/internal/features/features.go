// Package features mirrors trainer/semantic_poc/features.py exactly
// (parity-tested): tokenizer, 1-2 grams, sublinear TF-IDF, L2 norm.
package features

import (
	"math"
	"regexp"
	"strings"
)

var tokenRe = regexp.MustCompile(`[a-z0-9']+`)

func Tokenize(text string) []string {
	return tokenRe.FindAllString(strings.ToLower(text), -1)
}

func Terms(text string) []string {
	toks := Tokenize(text)
	terms := make([]string, 0, 2*len(toks))
	terms = append(terms, toks...)
	for i := 0; i+1 < len(toks); i++ {
		terms = append(terms, toks[i]+" "+toks[i+1])
	}
	return terms
}

// Transform returns the L2-normalized sublinear TF-IDF vector as index->value.
func Transform(text string, vocab map[string]int, idf []float64) map[int]float64 {
	counts := map[int]int{}
	for _, t := range Terms(text) {
		if j, ok := vocab[t]; ok {
			counts[j]++
		}
	}
	row := make(map[int]float64, len(counts))
	var sq float64
	for j, c := range counts {
		v := (1 + math.Log(float64(c))) * idf[j]
		row[j] = v
		sq += v * v
	}
	norm := math.Sqrt(sq)
	if norm == 0 {
		norm = 1
	}
	for j := range row {
		row[j] /= norm
	}
	return row
}
