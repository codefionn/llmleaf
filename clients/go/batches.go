package llmleaf

import (
	"bufio"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/url"

	pb "github.com/codefionn/llmleaf/clients/go/llmleafpb"
)

// CreateBatch submits a batch of chat requests (POST /v1/batches) and returns
// the initial handle.
func (c *Client) CreateBatch(ctx context.Context, req *pb.BatchCreateRequest) (*pb.BatchHandle, error) {
	body, err := encodeBatchCreateRequest(req)
	if err != nil {
		return nil, err
	}
	resp, err := c.doJSON(ctx, http.MethodPost, "/v1/batches", body, "application/json")
	if err != nil {
		return nil, err
	}
	return decodeBatchHandle(resp)
}

// GetBatch retrieves a batch's current handle (GET /v1/batches/{id}).
func (c *Client) GetBatch(ctx context.Context, id string) (*pb.BatchHandle, error) {
	resp, err := c.doJSON(ctx, http.MethodGet, "/v1/batches/"+url.PathEscape(id), nil, "application/json")
	if err != nil {
		return nil, err
	}
	return decodeBatchHandle(resp)
}

// CancelBatch requests cancellation of a batch (POST /v1/batches/{id}/cancel).
func (c *Client) CancelBatch(ctx context.Context, id string) (*pb.BatchHandle, error) {
	resp, err := c.doJSON(ctx, http.MethodPost, "/v1/batches/"+url.PathEscape(id)+"/cancel", nil, "application/json")
	if err != nil {
		return nil, err
	}
	return decodeBatchHandle(resp)
}

func decodeBatchHandle(resp *http.Response) (*pb.BatchHandle, error) {
	var wh wireBatchHandle
	if err := decodeJSONBody(resp, &wh); err != nil {
		return nil, err
	}
	return wh.toPB(), nil
}

// BatchResultsStream iterates the NDJSON result lines from
// GET /v1/batches/{id}/results — one BatchResultLine per line. Call Recv until
// io.EOF, then Close (or defer Close). Not safe for concurrent use.
type BatchResultsStream struct {
	resp    *http.Response
	scanner *bufio.Scanner
	done    bool
	err     error
}

// GetBatchResults opens the results stream (GET /v1/batches/{id}/results,
// application/x-ndjson).
func (c *Client) GetBatchResults(ctx context.Context, id string) (*BatchResultsStream, error) {
	resp, err := c.doJSON(ctx, http.MethodGet, "/v1/batches/"+url.PathEscape(id)+"/results", nil, "application/x-ndjson")
	if err != nil {
		return nil, err
	}
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 0, 64*1024), 1<<20)
	return &BatchResultsStream{resp: resp, scanner: sc}, nil
}

// Recv returns the next result line, io.EOF when the stream is exhausted, or a
// non-EOF error on transport / decode failure.
func (s *BatchResultsStream) Recv() (*pb.BatchResultLine, error) {
	if s.err != nil {
		return nil, s.err
	}
	if s.done {
		return nil, io.EOF
	}
	for s.scanner.Scan() {
		line := s.scanner.Bytes()
		if len(line) == 0 {
			continue // tolerate blank separator lines
		}
		var wl wireBatchResultLine
		if err := json.Unmarshal(line, &wl); err != nil {
			s.finish(err)
			return nil, err
		}
		return wl.toPB()
	}
	if err := s.scanner.Err(); err != nil {
		s.finish(err)
		return nil, err
	}
	s.finish(io.EOF)
	return nil, io.EOF
}

func (s *BatchResultsStream) finish(err error) {
	s.done = true
	if s.err == nil {
		s.err = err
	}
	s.resp.Body.Close()
}

// Close releases the underlying connection. Safe to call multiple times.
func (s *BatchResultsStream) Close() error {
	s.done = true
	return s.resp.Body.Close()
}
