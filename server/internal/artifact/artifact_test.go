package artifact

import (
	"strings"
	"testing"
)

func valid() *Artifact {
	a := &Artifact{
		Model: Model{
			Intents:    []string{"a", "b"},
			Families:   map[string]string{"a": "f1", "b": "f1"},
			PlattA:     []float64{1, 1},
			PlattB:     []float64{0, 0},
			Thresholds: map[string]float64{"a": 0.9, "b": 0.9},
		},
		SemCoef: [][]float64{{1, 2, 3}, {4, 5, 6}},
		SemInt:  []float64{0, 0},
		LexCoef: [][]float64{{1, 2}, {3, 4}},
		LexInt:  []float64{0, 0},
		FusCoef: [][]float64{{1, 2, 3, 4, 5, 6}, {1, 2, 3, 4, 5, 6}}, // 2K+2 = 6
		FusInt:  []float64{0, 0},
		Vocab:   map[string]int{"card": 0, "my card": 1},
		Idf:     []float64{1.2, 1.5},
	}
	a.Manifest.Embedding.OutputDim = 3
	return a
}

func TestValidateAcceptsConsistentArtifact(t *testing.T) {
	if err := valid().validate(); err != nil {
		t.Fatal(err)
	}
}

func TestValidateCatchesMissingThreshold(t *testing.T) {
	a := valid()
	delete(a.Model.Thresholds, "b")
	err := a.validate()
	if err == nil || !strings.Contains(err.Error(), "threshold") {
		t.Fatalf("expected missing-threshold error, got %v", err)
	}
}

func TestValidateCatchesPlattLengthMismatch(t *testing.T) {
	a := valid()
	a.Model.PlattA = a.Model.PlattA[:1]
	if a.validate() == nil {
		t.Fatal("expected platt length error")
	}
}

func TestValidateCatchesWidthMismatches(t *testing.T) {
	a := valid()
	a.Manifest.Embedding.OutputDim = 4
	if a.validate() == nil {
		t.Fatal("expected semantic width error")
	}

	a = valid()
	a.FusCoef = [][]float64{{1, 2, 3}, {1, 2, 3}}
	if a.validate() == nil {
		t.Fatal("expected fusion width error")
	}

	a = valid()
	a.Idf = a.Idf[:1]
	if a.validate() == nil {
		t.Fatal("expected idf/vocab length error")
	}
}
