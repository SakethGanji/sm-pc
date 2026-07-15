// Package artifact loads one immutable trainer artifact (§8): model
// parameters, TF-IDF vocab/idf, runtime config, manifest.
package artifact

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"

	"semantic-poc/server/internal/npy"
)

type Model struct {
	Intents         []string           `json:"intents"`
	Families        map[string]string  `json:"families"`
	PlattA          []float64          `json:"platt_a"`
	PlattB          []float64          `json:"platt_b"`
	Thresholds      map[string]float64 `json:"thresholds"`
	TauLow          float64            `json:"tau_low"`
	MaxAccepted     int                `json:"max_accepted"`
	ExclusiveGroups [][]string         `json:"exclusive_groups"`
}

type StitcherConfig struct {
	MaxGapMs           int64    `json:"max_gap_ms"`
	PrevMaxTokens      int      `json:"prev_max_tokens"`
	IncompletenessCues []string `json:"incompleteness_cues"`
	CombinedMaxTokens  int      `json:"combined_max_tokens"`
}

type RuntimeConfig struct {
	Stitcher StitcherConfig `json:"stitcher"`
	Context  struct {
		AgentTag    string `json:"agent_tag"`
		CustomerTag string `json:"customer_tag"`
	} `json:"context"`
}

type EmbeddingConfig struct {
	Model       string `json:"model"`
	TaskType    string `json:"task_type"`
	OutputDim   int    `json:"output_dim"`
	L2Normalize bool   `json:"l2_normalize"`
}

type Manifest struct {
	Name      string          `json:"name"`
	Embedding EmbeddingConfig `json:"embedding"`
}

type Artifact struct {
	Version  string // directory base name
	Model    Model
	Runtime  RuntimeConfig
	Manifest Manifest
	SemCoef  [][]float64
	SemInt   []float64
	LexCoef  [][]float64
	LexInt   []float64
	FusCoef  [][]float64
	FusInt   []float64
	Vocab    map[string]int
	Idf      []float64
}

func loadJSON(path string, v any) error {
	b, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	return json.Unmarshal(b, v)
}

func Load(dir string) (*Artifact, error) {
	a := &Artifact{Version: filepath.Base(dir)}
	if err := loadJSON(filepath.Join(dir, "model.json"), &a.Model); err != nil {
		return nil, err
	}
	if err := loadJSON(filepath.Join(dir, "runtime_config.json"), &a.Runtime); err != nil {
		return nil, err
	}
	if err := loadJSON(filepath.Join(dir, "manifest.json"), &a.Manifest); err != nil {
		return nil, err
	}
	if err := loadJSON(filepath.Join(dir, "tfidf_vocab.json"), &a.Vocab); err != nil {
		return nil, err
	}
	var err error
	if a.SemCoef, err = npy.ReadMatrix(filepath.Join(dir, "sem_coef.npy")); err != nil {
		return nil, err
	}
	if a.SemInt, err = npy.ReadVector(filepath.Join(dir, "sem_int.npy")); err != nil {
		return nil, err
	}
	if a.LexCoef, err = npy.ReadMatrix(filepath.Join(dir, "lex_coef.npy")); err != nil {
		return nil, err
	}
	if a.LexInt, err = npy.ReadVector(filepath.Join(dir, "lex_int.npy")); err != nil {
		return nil, err
	}
	if a.FusCoef, err = npy.ReadMatrix(filepath.Join(dir, "fus_coef.npy")); err != nil {
		return nil, err
	}
	if a.FusInt, err = npy.ReadVector(filepath.Join(dir, "fus_int.npy")); err != nil {
		return nil, err
	}
	if a.Idf, err = npy.ReadVector(filepath.Join(dir, "tfidf_idf.npy")); err != nil {
		return nil, err
	}
	k := len(a.Model.Intents)
	if len(a.SemCoef) != k || len(a.LexCoef) != k || len(a.FusCoef) != k {
		return nil, fmt.Errorf("artifact %s: coefficient rows do not match %d intents", dir, k)
	}
	return a, nil
}

// Newest returns the lexically-last artifact-* directory under root.
func Newest(root string) (string, error) {
	matches, err := filepath.Glob(filepath.Join(root, "artifact-*"))
	if err != nil || len(matches) == 0 {
		return "", fmt.Errorf("no artifacts under %s", root)
	}
	sort.Strings(matches)
	return matches[len(matches)-1], nil
}
