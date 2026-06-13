// Command basic is a runnable example for the llmleaf Go client: a non-streaming
// chat completion, a streaming chat completion, and a model listing.
//
// Configure it from the environment:
//
//	export LLMLEAF_BASE_URL=https://gateway.example.com
//	export LLMLEAF_API_KEY=sk-...
//	go run ./examples/basic
//
// Optionally set LLMLEAF_MODEL (defaults to "gpt-4o-mini").
package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log"
	"os"
	"time"

	llmleaf "github.com/codefionn/llmleaf/clients/go"
	pb "github.com/codefionn/llmleaf/clients/go/llmleafpb"
)

func main() {
	baseURL := os.Getenv("LLMLEAF_BASE_URL")
	apiKey := os.Getenv("LLMLEAF_API_KEY")
	if baseURL == "" || apiKey == "" {
		log.Fatal("set LLMLEAF_BASE_URL and LLMLEAF_API_KEY")
	}
	model := os.Getenv("LLMLEAF_MODEL")
	if model == "" {
		model = "gpt-4o-mini"
	}

	client := llmleaf.New(baseURL, apiKey, llmleaf.WithTimeout(30*time.Second))
	ctx := context.Background()

	userMessage := func(text string) *pb.ChatMessage {
		return &pb.ChatMessage{
			Role:    pb.Role_USER,
			Content: &pb.ChatMessage_Text{Text: text},
		}
	}

	// 1. Non-streaming chat: print the assembled text.
	fmt.Println("== non-streaming chat ==")
	resp, err := client.CreateChatCompletion(ctx, &pb.ChatRequest{
		Model:    model,
		Messages: []*pb.ChatMessage{userMessage("Say hello in one short sentence.")},
	})
	if err != nil {
		fatal(err)
	}
	if len(resp.GetChoices()) > 0 {
		fmt.Println(resp.GetChoices()[0].GetMessage().GetText())
	}
	if u := resp.GetUsage(); u != nil {
		fmt.Printf("(tokens: %d prompt + %d completion)\n", u.GetPromptTokens(), u.GetCompletionTokens())
	}

	// 2. Streaming chat: print deltas as they arrive.
	fmt.Println("\n== streaming chat ==")
	stream, err := client.CreateChatCompletionStream(ctx, &pb.ChatRequest{
		Model:    model,
		Messages: []*pb.ChatMessage{userMessage("Count from 1 to 5, one number per line.")},
	})
	if err != nil {
		fatal(err)
	}
	defer stream.Close()
	for {
		chunk, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			fatal(err)
		}
		for _, choice := range chunk.GetChoices() {
			fmt.Print(choice.GetDelta().GetContent())
		}
	}
	fmt.Println()

	// 3. List models.
	fmt.Println("\n== models ==")
	models, err := client.ListModels(ctx, &llmleaf.ListModelsOptions{Type: "all"})
	if err != nil {
		fatal(err)
	}
	for i, m := range models.GetData() {
		if i >= 10 {
			fmt.Printf("... and %d more\n", len(models.GetData())-10)
			break
		}
		fmt.Printf("- %s\n", m.GetId())
	}
}

func fatal(err error) {
	var apiErr *llmleaf.ApiError
	if errors.As(err, &apiErr) {
		log.Fatalf("API error: HTTP %d: %s", apiErr.Status, apiErr.Message)
	}
	log.Fatal(err)
}
