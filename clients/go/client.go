// Package llmleaf is the official Go client for the llmleaf LLM proxy.
//
// The typed model is generated from clients/proto/llmleaf/v1/llmleaf.proto into
// the llmleafpb package; this package adds a hand-written HTTP transport that
// (de)serialises those types to and from the OpenAI/OpenRouter-shaped JSON the
// llmleaf core speaks (see clients/SPEC.md). The wire is JSON, never
// protobuf-binary.
//
// Construct a Client with New, then call the endpoint methods. Every call takes
// a context.Context. Non-2xx responses surface as *ApiError.
package llmleaf

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// DefaultTimeout is applied when neither WithTimeout nor WithHTTPClient is given.
const DefaultTimeout = 60 * time.Second

// Client talks to a single llmleaf gateway. It is safe for concurrent use.
type Client struct {
	baseURL    string
	apiKey     string
	adminToken string
	http       *http.Client
}

// Option configures a Client at construction time.
type Option func(*clientConfig)

type clientConfig struct {
	timeout    time.Duration
	adminToken string
	httpClient *http.Client
}

// WithTimeout sets the HTTP client timeout. Ignored if WithHTTPClient is also
// supplied (the caller's client owns its own timeout). For streaming calls the
// timeout bounds the whole stream, so prefer a per-call context deadline or
// WithHTTPClient with Timeout: 0 when consuming long streams.
func WithTimeout(d time.Duration) Option {
	return func(c *clientConfig) { c.timeout = d }
}

// WithAdminToken sets the optional `x-admin-token` header. With it, GET
// /v1/models adds each model's provider/fallback `endpoints` array.
func WithAdminToken(token string) Option {
	return func(c *clientConfig) { c.adminToken = token }
}

// WithHTTPClient supplies a custom *http.Client (proxies, transport tuning,
// custom TLS). It takes precedence over WithTimeout.
func WithHTTPClient(hc *http.Client) Option {
	return func(c *clientConfig) { c.httpClient = hc }
}

// New builds a Client for the given base URL and API key. The base URL is the
// operator-provided gateway root, e.g. "https://gateway.example.com"; a trailing
// slash is trimmed. The API key is sent as `Authorization: Bearer <key>`.
func New(baseURL, apiKey string, opts ...Option) *Client {
	cfg := clientConfig{timeout: DefaultTimeout}
	for _, opt := range opts {
		opt(&cfg)
	}
	hc := cfg.httpClient
	if hc == nil {
		hc = &http.Client{Timeout: cfg.timeout}
	}
	return &Client{
		baseURL:    strings.TrimRight(baseURL, "/"),
		apiKey:     apiKey,
		adminToken: cfg.adminToken,
		http:       hc,
	}
}

// ApiError is the typed error parsed from the canonical envelope
// {"error":{"message":"..."}} returned on any non-2xx response.
type ApiError struct {
	// Status is the HTTP status code (400, 401, 403, 404, 429, 502, ...).
	Status int
	// Message is the human-readable message from the error envelope, or the raw
	// body / status text when the body is not the expected envelope.
	Message string
}

func (e *ApiError) Error() string {
	return fmt.Sprintf("llmleaf: HTTP %d: %s", e.Status, e.Message)
}

// newAPIError builds an *ApiError from a non-2xx response body, preferring the
// {"error":{"message":...}} envelope and falling back to the raw body / status.
func newAPIError(status int, body []byte) *ApiError {
	msg := strings.TrimSpace(string(body))
	var env struct {
		Error struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if json.Unmarshal(body, &env) == nil && env.Error.Message != "" {
		msg = env.Error.Message
	}
	if msg == "" {
		msg = http.StatusText(status)
	}
	return &ApiError{Status: status, Message: msg}
}

func (c *Client) url(path string) string {
	return c.baseURL + path
}

// newRequest builds a request with auth headers applied. `body` may be nil.
func (c *Client) newRequest(ctx context.Context, method, path string, body io.Reader) (*http.Request, error) {
	req, err := http.NewRequestWithContext(ctx, method, c.url(path), body)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+c.apiKey)
	if c.adminToken != "" {
		req.Header.Set("x-admin-token", c.adminToken)
	}
	return req, nil
}

// doJSON sends a request whose body is the given JSON bytes (nil for none) and
// returns a non-error response for the caller to read. The caller closes the
// body. On non-2xx it reads the body and returns an *ApiError.
func (c *Client) doJSON(ctx context.Context, method, path string, body []byte, accept string) (*http.Response, error) {
	var reader io.Reader
	if body != nil {
		reader = bytes.NewReader(body)
	}
	req, err := c.newRequest(ctx, method, path, reader)
	if err != nil {
		return nil, err
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	if accept != "" {
		req.Header.Set("Accept", accept)
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		defer resp.Body.Close()
		errBody, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
		return nil, newAPIError(resp.StatusCode, errBody)
	}
	return resp, nil
}

// decodeJSONBody reads and JSON-decodes a successful response body into v, then
// closes it.
func decodeJSONBody(resp *http.Response, v any) error {
	defer resp.Body.Close()
	dec := json.NewDecoder(resp.Body)
	if err := dec.Decode(v); err != nil {
		return fmt.Errorf("llmleaf: decode response: %w", err)
	}
	return nil
}
