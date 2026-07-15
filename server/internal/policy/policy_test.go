package policy

import "testing"

var (
	fam = map[string]string{"a": "f1", "b": "f1", "c": "f2", "d": "f2"}
	th  = map[string]float64{"a": 0.9, "b": 0.8, "c": 0.85, "d": 0.7}
)

func run(probs map[string]float64, groups [][]string) Result {
	return Decide(probs, th, fam, 0.3, 3, groups)
}

func TestSingleAccept(t *testing.T) {
	r := run(map[string]float64{"a": 0.95, "b": 0.1, "c": 0.1, "d": 0.1}, nil)
	if r.Decision != "accepted" || r.Intents[0].Name != "a" || r.Intents[0].MainIntent != "f1" {
		t.Fatalf("%+v", r)
	}
}

func TestMultiAcceptSortedDesc(t *testing.T) {
	r := run(map[string]float64{"a": 0.91, "b": 0.99, "c": 0.1, "d": 0.1}, nil)
	if r.Decision != "multi_accepted" || r.Intents[0].Name != "b" || r.Intents[1].Name != "a" {
		t.Fatalf("%+v", r)
	}
}

func TestNoSupportedIntentBelowTauLow(t *testing.T) {
	r := run(map[string]float64{"a": 0.2, "b": 0.1, "c": 0.25, "d": 0.05}, nil)
	if r.Decision != "no_supported_intent" || len(r.Intents) != 0 {
		t.Fatalf("%+v", r)
	}
}

func TestAbstainedBetweenTauLowAndThreshold(t *testing.T) {
	r := run(map[string]float64{"a": 0.5, "b": 0.1, "c": 0.1, "d": 0.1}, nil)
	if r.Decision != "abstained" {
		t.Fatalf("%+v", r)
	}
}

func TestExclusiveGroupKeepsArgmax(t *testing.T) {
	r := run(map[string]float64{"a": 0.95, "b": 0.85, "c": 0.1, "d": 0.1},
		[][]string{{"a", "b"}})
	if len(r.Intents) != 1 || r.Intents[0].Name != "a" {
		t.Fatalf("%+v", r)
	}
}

func TestCapAndOverflow(t *testing.T) {
	r := Decide(map[string]float64{"a": 0.95, "b": 0.9, "c": 0.9, "d": 0.8},
		th, fam, 0.3, 3, nil)
	if len(r.Intents) != 3 || !r.Overflow {
		t.Fatalf("%+v", r)
	}
}

func TestEqualProbabilitiesTieBreakByName(t *testing.T) {
	// Same rule as the Python implementation: prob desc, then name asc. Run
	// repeatedly so random map iteration order cannot hide nondeterminism.
	for range 50 {
		r := run(map[string]float64{"a": 0.92, "b": 0.92, "d": 0.92, "c": 0.1}, nil)
		got := []string{r.Intents[0].Name, r.Intents[1].Name, r.Intents[2].Name}
		if got[0] != "a" || got[1] != "b" || got[2] != "d" {
			t.Fatalf("tie order not deterministic: %v", got)
		}
	}
}
