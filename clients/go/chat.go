package llmleaf

import (
	"context"
	"net/http"

	pb "github.com/codefionn/llmleaf/clients/go/llmleafpb"
)

// CreateChatCompletion performs a non-streaming chat completion
// (POST /v1/chat/completions). The request's stream flag is forced false on the
// wire; use CreateChatCompletionStream for streaming.
func (c *Client) CreateChatCompletion(ctx context.Context, req *pb.ChatRequest) (*pb.ChatResponse, error) {
	body, err := encodeChatRequest(req, ptr(false))
	if err != nil {
		return nil, err
	}
	resp, err := c.doJSON(ctx, http.MethodPost, "/v1/chat/completions", body, "application/json")
	if err != nil {
		return nil, err
	}
	var wr wireChatResponse
	if err := decodeJSONBody(resp, &wr); err != nil {
		return nil, err
	}
	return wr.toPB()
}

// CreateChatCompletionStream opens a streaming chat completion
// (POST /v1/chat/completions with stream:true). It returns a *ChatStream whose
// Recv yields *llmleafpb.ChatCompletionChunk values until io.EOF. The caller
// MUST Close the stream. The `data: [DONE]` sentinel is handled internally.
func (c *Client) CreateChatCompletionStream(ctx context.Context, req *pb.ChatRequest) (*ChatStream, error) {
	body, err := encodeChatRequest(req, ptr(true))
	if err != nil {
		return nil, err
	}
	resp, err := c.doJSON(ctx, http.MethodPost, "/v1/chat/completions", body, "text/event-stream")
	if err != nil {
		return nil, err
	}
	return newChatStream(resp), nil
}
