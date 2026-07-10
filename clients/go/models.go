package llmleaf

import (
	"context"
	"net/http"
	"net/url"

	pb "github.com/codefionn/llmleaf/clients/go/llmleafpb"
)

// ListModelsOptions are the optional query filters for ListModels.
type ListModelsOptions struct {
	// Type filters the catalog: "all", "llm", "tts", "stt", "embedding", or
	// "rerank". Empty means unset (server default).
	Type string
	// Search is a case-insensitive substring filter over model ids/names.
	Search string
}

// ListModels fetches the model catalog (GET /v1/models). If the client was built
// WithAdminToken, each entry additionally carries its provider/fallback
// `endpoints` array. Pass nil opts for no filters.
func (c *Client) ListModels(ctx context.Context, opts *ListModelsOptions) (*pb.ListModelsResponse, error) {
	path := "/v1/models"
	if opts != nil {
		q := url.Values{}
		if opts.Type != "" {
			q.Set("type", opts.Type)
		}
		if opts.Search != "" {
			q.Set("search", opts.Search)
		}
		if encoded := q.Encode(); encoded != "" {
			path += "?" + encoded
		}
	}
	resp, err := c.doJSON(ctx, http.MethodGet, path, nil, "application/json")
	if err != nil {
		return nil, err
	}
	var wr wireListModelsResponse
	if err := decodeJSONBody(resp, &wr); err != nil {
		return nil, err
	}
	return wr.toPB(), nil
}
