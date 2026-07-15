// Package embedding is the §6.3 provider: real Gemini embeddings over REST
// with deadline, limited jittered retries, and client-side L2 normalization.
// Failure is an infrastructure outcome (degraded), never a semantic decision.
package embedding

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"math/rand/v2"
	"net/http"
	"time"

	"semantic-poc/server/internal/artifact"
)

const endpoint = "https://generativelanguage.googleapis.com/v1beta/models/%s:embedContent"

type Client struct {
	cfg    artifact.EmbeddingConfig
	apiKey string
	http   *http.Client
}

func New(cfg artifact.EmbeddingConfig, apiKey string) *Client {
	return &Client{cfg: cfg, apiKey: apiKey, http: &http.Client{Timeout: 10 * time.Second}}
}

type embedRequest struct {
	Content struct {
		Parts []struct {
			Text string `json:"text"`
		} `json:"parts"`
	} `json:"content"`
	TaskType             string `json:"taskType"`
	OutputDimensionality int    `json:"outputDimensionality"`
}

type embedResponse struct {
	Embedding struct {
		Values []float64 `json:"values"`
	} `json:"embedding"`
}

func (c *Client) Embed(ctx context.Context, text string) ([]float64, error) {
	var req embedRequest
	req.Content.Parts = []struct {
		Text string `json:"text"`
	}{{Text: text}}
	req.TaskType = c.cfg.TaskType
	req.OutputDimensionality = c.cfg.OutputDim
	body, _ := json.Marshal(req)

	var lastErr error
	for attempt := 0; attempt < 3; attempt++ {
		if attempt > 0 {
			select {
			case <-time.After(time.Duration(50+rand.IntN(150)) * time.Millisecond):
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		}
		vec, retryable, err := c.once(ctx, body)
		if err == nil {
			return vec, nil
		}
		lastErr = err
		if !retryable {
			break
		}
	}
	return nil, lastErr
}

func (c *Client) once(ctx context.Context, body []byte) (vec []float64, retryable bool, err error) {
	url := fmt.Sprintf(endpoint, c.cfg.Model)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, false, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-goog-api-key", c.apiKey)
	resp, err := c.http.Do(req)
	if err != nil {
		return nil, true, err
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, true, err
	}
	if resp.StatusCode != http.StatusOK {
		return nil, resp.StatusCode == 429 || resp.StatusCode >= 500,
			fmt.Errorf("gemini embed: HTTP %d: %.200s", resp.StatusCode, data)
	}
	var er embedResponse
	if err := json.Unmarshal(data, &er); err != nil {
		return nil, false, err
	}
	v := er.Embedding.Values
	if len(v) != c.cfg.OutputDim {
		return nil, false, fmt.Errorf("gemini embed: got %d dims, want %d", len(v), c.cfg.OutputDim)
	}
	if c.cfg.L2Normalize {
		var sq float64
		for _, x := range v {
			sq += x * x
		}
		if n := math.Sqrt(sq); n > 0 {
			for i := range v {
				v[i] /= n
			}
		}
	}
	return v, false, nil
}
