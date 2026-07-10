package llmleaf

import (
	"context"
	"encoding/json"
	"net/http"

	pb "github.com/codefionn/llmleaf/clients/go/llmleafpb"
)

// RerankRequest is the body of POST /v1/rerank: rank Documents by their
// relevance to Query for the given Model.
type RerankRequest struct {
	Model string `json:"model"`
	Query string `json:"query"`
	// Documents are the candidate passages to rank. They are modelled as plain
	// strings — the common, portable case. The wire additionally accepts
	// structured/multimodal document objects; that is a JSON-only extension not
	// modelled here (cf. EmbeddingRequest.Input, which is likewise string-only in
	// the typed model). Reach for the raw-JSON Extra passthrough on those fields
	// if you need the richer form.
	Documents []string `json:"documents"`
	// TopN caps the number of results returned (nil = server default, all docs).
	TopN *int `json:"top_n,omitempty"`
	// ReturnDocuments asks the server to echo each ranked document back in the
	// response's Document field (nil = server default).
	ReturnDocuments *bool `json:"return_documents,omitempty"`
}

// RerankResult is one ranked document: its position Index in the request's
// Documents slice and its RelevanceScore. Document is present only when the
// request set ReturnDocuments; it is raw JSON because the wire echoes either a
// bare string or a structured document object.
type RerankResult struct {
	Index          int             `json:"index"`
	RelevanceScore float64         `json:"relevance_score"`
	Document       json.RawMessage `json:"document,omitempty"`
}

// RerankResponse is the POST /v1/rerank result. Results are ordered by
// descending relevance_score.
type RerankResponse struct {
	Object  string         `json:"object"`
	Model   string         `json:"model"`
	Results []RerankResult `json:"results"`
	Usage   *pb.Usage      `json:"usage"`
}

// CreateRerank ranks req.Documents by relevance to req.Query (POST /v1/rerank).
// Results come back ordered by descending relevance_score.
func (c *Client) CreateRerank(ctx context.Context, req RerankRequest) (*RerankResponse, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}
	resp, err := c.doJSON(ctx, http.MethodPost, "/v1/rerank", body, "application/json")
	if err != nil {
		return nil, err
	}
	var out RerankResponse
	if err := decodeJSONBody(resp, &out); err != nil {
		return nil, err
	}
	return &out, nil
}
