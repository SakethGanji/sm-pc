package features

import (
	"math"
	"testing"
)

func TestTokenizeLowercaseApostrophes(t *testing.T) {
	got := Tokenize("Card's NOT here, ok?")
	want := []string{"card's", "not", "here", "ok"}
	if len(got) != len(want) {
		t.Fatalf("%v", got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("%v", got)
		}
	}
}

func TestTermsIncludeBigrams(t *testing.T) {
	got := Terms("replace my card")
	want := []string{"replace", "my", "card", "replace my", "my card"}
	if len(got) != len(want) {
		t.Fatalf("%v", got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("%v", got)
		}
	}
}

func TestTransformL2NormalizedAndSublinear(t *testing.T) {
	vocab := map[string]int{"card": 0, "lost": 1, "my card": 2}
	idf := []float64{1.2, 1.5, 1.1}
	vec := Transform("my card is lost lost", vocab, idf)
	var sq float64
	for _, v := range vec {
		sq += v * v
	}
	if math.Abs(math.Sqrt(sq)-1.0) > 1e-12 {
		t.Fatalf("not L2-normalized: %v", vec)
	}
	// lost appears twice: tf = 1+ln(2); card once: tf = 1.
	wantRatio := ((1 + math.Log(2)) * idf[1]) / (1 * idf[0])
	if math.Abs(vec[1]/vec[0]-wantRatio) > 1e-12 {
		t.Fatalf("sublinear tf wrong: %v", vec)
	}
}

func TestTransformUnknownTermsEmpty(t *testing.T) {
	if len(Transform("completely unseen", map[string]int{"card": 0}, []float64{1})) != 0 {
		t.Fatal("expected empty vector")
	}
}
