package llmleaf

import (
	"context"
	"net/http"

	pb "github.com/codefionn/llmleaf/clients/go/llmleafpb"
)

// CreateEmbeddings requests embeddings (POST /v1/embeddings). When
// req.EncodingFormat is "base64" the wire returns base64-encoded little-endian
// f32 bytes, which are decoded into each Embedding.Embedding float vector before
// returning.
func (c *Client) CreateEmbeddings(ctx context.Context, req *pb.EmbeddingRequest) (*pb.EmbeddingResponse, error) {
	body, err := encodeEmbeddingRequest(req)
	if err != nil {
		return nil, err
	}
	resp, err := c.doJSON(ctx, http.MethodPost, "/v1/embeddings", body, "application/json")
	if err != nil {
		return nil, err
	}
	var wr wireEmbeddingResponse
	if err := decodeJSONBody(resp, &wr); err != nil {
		return nil, err
	}
	return wr.toPB()
}
