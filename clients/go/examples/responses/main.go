// Command responses is a runnable example for the llmleaf Go client's OpenAI
// Responses dialect (POST /v1/responses): a non-streaming call and a streaming
// call over the typed SSE events.
//
// Configure it from the environment:
//
//	export LLMLEAF_BASE_URL=https://gateway.example.com
//	export LLMLEAF_API_KEY=sk-...
//	go run ./examples/responses
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
	"strings"
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

	// 1. Non-streaming response: `input` as a bare string is one user message.
	fmt.Println("== non-streaming response ==")
	resp, err := client.CreateResponse(ctx, &pb.ResponsesRequest{
		Model: model,
		Input: &pb.ResponsesRequest_Text{Text: "Say hello in one short sentence."},
	})
	if err != nil {
		fatal(err)
	}
	fmt.Println(responseText(resp.GetOutput()))
	if u := resp.GetUsage(); u != nil {
		fmt.Printf("(tokens: %d input + %d output)\n", u.GetInputTokens(), u.GetOutputTokens())
	}

	// 2. Streaming response: accumulate output_text deltas; the stream ends on
	//    the terminal response.completed event (there is no [DONE] sentinel).
	fmt.Println("\n== streaming response ==")
	stream, err := client.CreateResponseStream(ctx, &pb.ResponsesRequest{
		Model: model,
		Input: &pb.ResponsesRequest_Text{Text: "Count from 1 to 5, one number per line."},
	})
	if err != nil {
		fatal(err)
	}
	defer stream.Close()
	for {
		ev, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			fatal(err)
		}
		if ev.GetType() == "response.output_text.delta" {
			fmt.Print(ev.GetDelta())
		}
	}
	fmt.Println()
}

// responseText joins the assistant output_text parts of a response's output.
func responseText(output []*pb.ResponseItem) string {
	var b strings.Builder
	for _, item := range output {
		msg := item.GetMessage()
		if msg == nil {
			continue
		}
		if parts := msg.GetParts(); parts != nil {
			for _, p := range parts.GetItems() {
				if ot := p.GetOutputText(); ot != nil {
					b.WriteString(ot.GetText())
				}
			}
		} else {
			b.WriteString(msg.GetText())
		}
	}
	return b.String()
}

func fatal(err error) {
	var apiErr *llmleaf.ApiError
	if errors.As(err, &apiErr) {
		log.Fatalf("API error: HTTP %d: %s", apiErr.Status, apiErr.Message)
	}
	log.Fatal(err)
}
