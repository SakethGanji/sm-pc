// Package api implements the §5 request/response contract.
package api

import (
	"context"
	"encoding/json"
	"log"
	"net/http"
	"time"

	"semantic-poc/server/internal/artifact"
	"semantic-poc/server/internal/embedding"
	"semantic-poc/server/internal/model"
	"semantic-poc/server/internal/policy"
	"semantic-poc/server/internal/stitch"
)

type Request struct {
	ConversationID            string              `json:"conversation_id"`
	TurnIndex                 int                 `json:"turn_index"`
	Timestamp                 string              `json:"timestamp"`
	CurrentCustomerTranscript string              `json:"current_customer_transcript"`
	PreviousAgentUtterance    string              `json:"previous_agent_utterance"`
	PreviousCustomerSegment   *stitch.PrevSegment `json:"previous_customer_segment"`
	AsrConfidence             float64             `json:"asr_confidence"`
	AsrIsFinal                bool                `json:"asr_is_final"`
}

type LatencyMs struct {
	Stitch    int64 `json:"stitch"`
	Embedding int64 `json:"embedding"`
	Lexical   int64 `json:"lexical"`
	Semantic  int64 `json:"semantic"`
	Fusion    int64 `json:"fusion"`
	Total     int64 `json:"total"`
}

type Response struct {
	Decision              string          `json:"decision"`
	Intents               []policy.Intent `json:"intents"`
	SupersedesTurnIndex   *int            `json:"supersedes_turn_index,omitempty"`
	Stitched              bool            `json:"stitched"`
	LongPathUsed          bool            `json:"long_path_used"`
	ModelVersion          string          `json:"model_version"`
	EmbeddingModelVersion string          `json:"embedding_model_version"`
	Error                 string          `json:"error,omitempty"`
	LatencyMs             LatencyMs       `json:"latency_ms"`
}

type Handler struct {
	Art      *artifact.Artifact
	Embedder *embedding.Client
	Deadline time.Duration
}

func (h *Handler) Classify(w http.ResponseWriter, r *http.Request) {
	t0 := time.Now()
	var req Request
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	resp := h.classify(r.Context(), &req, t0)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func (h *Handler) classify(ctx context.Context, req *Request, t0 time.Time) *Response {
	resp := &Response{
		ModelVersion:          h.Art.Version,
		EmbeddingModelVersion: h.Art.Manifest.Embedding.Model,
	}

	curStartMs := int64(0)
	if ts, err := time.Parse(time.RFC3339, req.Timestamp); err == nil {
		curStartMs = ts.UnixMilli()
	}
	st := stitch.Stitch(req.PreviousCustomerSegment, req.CurrentCustomerTranscript,
		curStartMs, h.Art.Runtime.Stitcher)
	resp.Stitched = st.Stitched
	if st.Stitched {
		idx := st.SupersedesIndex
		resp.SupersedesTurnIndex = &idx
	}
	resp.LatencyMs.Stitch = time.Since(t0).Milliseconds()

	c1 := h.Art.Runtime.Context.AgentTag + " " + req.PreviousAgentUtterance +
		"\n" + h.Art.Runtime.Context.CustomerTag + " " + st.Text

	// Embedding round trip and local lexical work run in parallel (§4).
	type embOut struct {
		vec []float64
		ms  int64
		err error
	}
	embCh := make(chan embOut, 1)
	go func() {
		te := time.Now()
		ctx, cancel := context.WithTimeout(ctx, h.Deadline)
		defer cancel()
		vec, err := h.Embedder.Embed(ctx, c1)
		embCh <- embOut{vec, time.Since(te).Milliseconds(), err}
	}()

	tl := time.Now()
	lex := model.LexLogits(h.Art, st.Text)
	meta := model.Meta(st.Text, req.PreviousAgentUtterance)
	resp.LatencyMs.Lexical = time.Since(tl).Milliseconds()

	emb := <-embCh
	resp.LatencyMs.Embedding = emb.ms
	if emb.err != nil {
		// Infrastructure failure -> degraded, never a semantic decision (§6.3).
		log.Printf("embedding degraded conv=%s turn=%d: %v",
			req.ConversationID, req.TurnIndex, emb.err)
		resp.Decision = "degraded"
		resp.Error = "embedding_unavailable"
		resp.LatencyMs.Total = time.Since(t0).Milliseconds()
		return resp
	}

	ts := time.Now()
	sem := model.SemLogits(h.Art, emb.vec)
	resp.LatencyMs.Semantic = time.Since(ts).Milliseconds()

	tf := time.Now()
	_, _, result := model.Fuse(h.Art, sem, lex, meta)
	resp.LatencyMs.Fusion = time.Since(tf).Milliseconds()

	resp.Decision = result.Decision
	resp.Intents = result.Intents
	if result.Overflow {
		log.Printf("accepted-cap overflow conv=%s turn=%d", req.ConversationID, req.TurnIndex)
	}
	resp.LatencyMs.Total = time.Since(t0).Milliseconds()
	return resp
}

func (h *Handler) Health(w http.ResponseWriter, _ *http.Request) {
	json.NewEncoder(w).Encode(map[string]string{
		"status": "ok", "model_version": h.Art.Version})
}
