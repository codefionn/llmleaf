package llmleaf

import (
	"context"
	"encoding/json"
	"net/http"
)

// DecisionsRequest is the body of POST /v1/decisions. State and every question
// are raw JSON so provider-specific decision schemas round-trip unchanged.
type DecisionsRequest struct {
	Model     string                     `json:"model"`
	State     json.RawMessage            `json:"state"`
	Questions map[string]json.RawMessage `json:"questions"`
	// Extra adds provider-specific top-level fields. Explicit fields above win
	// when an Extra key collides with one of them.
	Extra map[string]json.RawMessage `json:"-"`
}

func (r DecisionsRequest) MarshalJSON() ([]byte, error) {
	model, err := json.Marshal(r.Model)
	if err != nil {
		return nil, err
	}
	questions, err := json.Marshal(r.Questions)
	if err != nil {
		return nil, err
	}
	body := map[string]json.RawMessage{
		"model":     model,
		"state":     r.State,
		"questions": questions,
	}
	for key, value := range r.Extra {
		if _, exists := body[key]; !exists {
			body[key] = value
		}
	}
	return json.Marshal(body)
}

// DecisionsUsage is token and provider cost accounting. Cost is a pointer so
// an explicit zero remains distinct from an omitted provider cost.
type DecisionsUsage struct {
	InputTokens  uint64                     `json:"input_tokens"`
	OutputTokens uint64                     `json:"output_tokens"`
	Cost         *float64                   `json:"cost,omitempty"`
	Extra        map[string]json.RawMessage `json:"-"`
}

func (u *DecisionsUsage) UnmarshalJSON(data []byte) error {
	*u = DecisionsUsage{}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	if err := json.Unmarshal(raw["input_tokens"], &u.InputTokens); err != nil {
		return err
	}
	if err := json.Unmarshal(raw["output_tokens"], &u.OutputTokens); err != nil {
		return err
	}
	if value, ok := raw["cost"]; ok {
		if err := json.Unmarshal(value, &u.Cost); err != nil {
			return err
		}
	}
	delete(raw, "input_tokens")
	delete(raw, "output_tokens")
	delete(raw, "cost")
	u.Extra = raw
	return nil
}

// DecisionsResponse is the result of POST /v1/decisions. Answers and Extra
// retain raw JSON because both OpenRouter and JEV can add structured fields.
type DecisionsResponse struct {
	Model    string                     `json:"model"`
	Answers  map[string]json.RawMessage `json:"answers"`
	Usage    *DecisionsUsage            `json:"usage,omitempty"`
	ID       *string                    `json:"id,omitempty"`
	Provider *string                    `json:"provider,omitempty"`
	Extra    map[string]json.RawMessage `json:"-"`
}

func (r *DecisionsResponse) UnmarshalJSON(data []byte) error {
	*r = DecisionsResponse{}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	if err := json.Unmarshal(raw["model"], &r.Model); err != nil {
		return err
	}
	if err := json.Unmarshal(raw["answers"], &r.Answers); err != nil {
		return err
	}
	if value, ok := raw["usage"]; ok {
		if err := json.Unmarshal(value, &r.Usage); err != nil {
			return err
		}
	}
	if value, ok := raw["id"]; ok {
		if err := json.Unmarshal(value, &r.ID); err != nil {
			return err
		}
	}
	if value, ok := raw["provider"]; ok {
		if err := json.Unmarshal(value, &r.Provider); err != nil {
			return err
		}
	}
	delete(raw, "model")
	delete(raw, "answers")
	delete(raw, "usage")
	delete(raw, "id")
	delete(raw, "provider")
	r.Extra = raw
	return nil
}

// CreateDecisions submits state and named decision questions (POST /v1/decisions).
func (c *Client) CreateDecisions(ctx context.Context, req DecisionsRequest) (*DecisionsResponse, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}
	resp, err := c.doJSON(ctx, http.MethodPost, "/v1/decisions", body, "application/json")
	if err != nil {
		return nil, err
	}
	var out DecisionsResponse
	if err := decodeJSONBody(resp, &out); err != nil {
		return nil, err
	}
	return &out, nil
}
