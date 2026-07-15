package main

import (
	"flag"
	"log"
	"net/http"
	"os"
	"time"

	"semantic-poc/server/internal/api"
	"semantic-poc/server/internal/artifact"
	"semantic-poc/server/internal/embedding"
)

func main() {
	artDir := flag.String("artifact", "", "artifact directory (default: newest under ../artifacts)")
	addr := flag.String("addr", ":8080", "listen address")
	deadline := flag.Duration("embed-deadline", 2*time.Second, "embedding call deadline")
	flag.Parse()

	dir := *artDir
	if dir == "" {
		var err error
		if dir, err = artifact.Newest("../artifacts"); err != nil {
			log.Fatal(err)
		}
	}
	art, err := artifact.Load(dir)
	if err != nil {
		log.Fatalf("load artifact: %v", err)
	}
	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		apiKey = os.Getenv("GOOGLE_API_KEY")
	}
	if apiKey == "" {
		log.Fatal("GEMINI_API_KEY is not set — real Gemini embeddings are required")
	}

	h := &api.Handler{
		Art:      art,
		Embedder: embedding.New(art.Manifest.Embedding, apiKey),
		Deadline: *deadline,
	}
	mux := http.NewServeMux()
	mux.HandleFunc("POST /classify", h.Classify)
	mux.HandleFunc("GET /healthz", h.Health)

	log.Printf("serving %s on %s (embedding=%s)", art.Version, *addr, art.Manifest.Embedding.Model)
	log.Fatal(http.ListenAndServe(*addr, mux))
}
