// Parity: replay the trainer's fixtures (real texts + real Gemini embeddings
// + every expected intermediate) through the Go forward pass.
package parity

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"semantic-poc/server/internal/artifact"
	"semantic-poc/server/internal/model"
)

type fixture struct {
	ConversationID         string    `json:"conversation_id"`
	TurnIndex              int       `json:"turn_index"`
	RawTranscript          string    `json:"raw_transcript"`
	PreviousAgentUtterance string    `json:"previous_agent_utterance"`
	Embedding              []float64 `json:"embedding"`
	Expected               struct {
		C1Text      string             `json:"c1_text"`
		SemLogits   []float64          `json:"sem_logits"`
		LexLogits   []float64          `json:"lex_logits"`
		Meta        []float64          `json:"meta"`
		FusedLogits []float64          `json:"fused_logits"`
		Probs       map[string]float64 `json:"probs"`
		Decision    string             `json:"decision"`
		Intents     []struct {
			Name string `json:"name"`
		} `json:"intents"`
	} `json:"expected"`
}

const tol = 1e-9

func close(a, b float64) bool { return math.Abs(a-b) <= tol }

func vecClose(t *testing.T, name string, got, want []float64, fx *fixture) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s turn %d: %s length %d != %d", fx.ConversationID, fx.TurnIndex, name, len(got), len(want))
	}
	for i := range got {
		if !close(got[i], want[i]) {
			t.Errorf("%s turn %d: %s[%d] = %.12g, want %.12g",
				fx.ConversationID, fx.TurnIndex, name, i, got[i], want[i])
		}
	}
}

func TestForwardMatchesPythonFixtures(t *testing.T) {
	dir := os.Getenv("ARTIFACT_DIR")
	if dir == "" {
		var err error
		dir, err = artifact.Newest("../../../artifacts")
		if err != nil {
			t.Skip("no trained artifact yet — run `poc train` first (needs GEMINI_API_KEY)")
		}
	}
	art, err := artifact.Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(filepath.Join(dir, "fixtures.json"))
	if err != nil {
		t.Fatal(err)
	}
	var fixtures []fixture
	if err := json.Unmarshal(raw, &fixtures); err != nil {
		t.Fatal(err)
	}
	if len(fixtures) == 0 {
		t.Fatal("empty fixtures")
	}
	t.Logf("artifact %s: %d fixtures", art.Version, len(fixtures))

	for i := range fixtures {
		fx := &fixtures[i]
		fwd := model.Score(art, fx.RawTranscript, fx.PreviousAgentUtterance, fx.Embedding)

		c1 := art.Runtime.Context.AgentTag + " " + fx.PreviousAgentUtterance +
			"\n" + art.Runtime.Context.CustomerTag + " " + fx.RawTranscript
		if c1 != fx.Expected.C1Text {
			t.Errorf("%s turn %d: c1 text mismatch", fx.ConversationID, fx.TurnIndex)
		}
		vecClose(t, "sem_logits", fwd.SemLogits, fx.Expected.SemLogits, fx)
		vecClose(t, "lex_logits", fwd.LexLogits, fx.Expected.LexLogits, fx)
		vecClose(t, "meta", fwd.Meta, fx.Expected.Meta, fx)
		vecClose(t, "fused_logits", fwd.FusedLogits, fx.Expected.FusedLogits, fx)
		for intent, want := range fx.Expected.Probs {
			if !close(fwd.Probs[intent], want) {
				t.Errorf("%s turn %d: prob[%s] = %.12g, want %.12g",
					fx.ConversationID, fx.TurnIndex, intent, fwd.Probs[intent], want)
			}
		}
		if fwd.Result.Decision != fx.Expected.Decision {
			t.Errorf("%s turn %d: decision %q, want %q",
				fx.ConversationID, fx.TurnIndex, fwd.Result.Decision, fx.Expected.Decision)
		}
		if len(fwd.Result.Intents) != len(fx.Expected.Intents) {
			t.Errorf("%s turn %d: %d intents, want %d", fx.ConversationID,
				fx.TurnIndex, len(fwd.Result.Intents), len(fx.Expected.Intents))
			continue
		}
		for j, in := range fwd.Result.Intents {
			if in.Name != fx.Expected.Intents[j].Name {
				t.Errorf("%s turn %d: intent[%d] = %s, want %s", fx.ConversationID,
					fx.TurnIndex, j, in.Name, fx.Expected.Intents[j].Name)
			}
		}
	}
}
