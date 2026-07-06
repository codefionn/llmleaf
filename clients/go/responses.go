package llmleaf

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"

	pb "github.com/codefionn/llmleaf/clients/go/llmleafpb"
)

// CreateResponse performs a non-streaming Responses request
// (POST /v1/responses) — the OpenAI Responses dialect on the same canonical
// core. The request's stream flag is forced false on the wire; use
// CreateResponseStream for streaming.
func (c *Client) CreateResponse(ctx context.Context, req *pb.ResponsesRequest) (*pb.ResponsesResponse, error) {
	body, err := encodeResponsesRequest(req, ptr(false))
	if err != nil {
		return nil, err
	}
	resp, err := c.doJSON(ctx, http.MethodPost, "/v1/responses", body, "application/json")
	if err != nil {
		return nil, err
	}
	var wr wireResponsesResponse
	if err := decodeJSONBody(resp, &wr); err != nil {
		return nil, err
	}
	return wr.toPB()
}

// CreateResponseStream opens a streaming Responses request (POST /v1/responses
// with stream:true). It returns a *ResponsesStream whose Recv yields
// *llmleafpb.ResponsesStreamEvent values until io.EOF. The caller MUST Close the
// stream. Unlike chat streaming there is NO `data: [DONE]` sentinel — the stream
// ends after the terminal response.completed / response.incomplete /
// response.failed event (or when the connection closes).
func (c *Client) CreateResponseStream(ctx context.Context, req *pb.ResponsesRequest) (*ResponsesStream, error) {
	body, err := encodeResponsesRequest(req, ptr(true))
	if err != nil {
		return nil, err
	}
	resp, err := c.doJSON(ctx, http.MethodPost, "/v1/responses", body, "text/event-stream")
	if err != nil {
		return nil, err
	}
	return newResponsesStream(resp), nil
}

// knownResponsesEventTypes is the set of Responses stream event types this SDK
// recognises. The dialect grows by adding types, so Recv silently SKIPS any type
// not in this set (per SPEC.md — "ignore unrecognised types"). The "error" event
// is handled separately (surfaced as *ApiError), so it is not listed here.
var knownResponsesEventTypes = map[string]struct{}{
	"response.created":                       {},
	"response.in_progress":                   {},
	"response.completed":                     {},
	"response.incomplete":                    {},
	"response.failed":                        {},
	"response.output_item.added":             {},
	"response.output_item.done":              {},
	"response.content_part.added":            {},
	"response.content_part.done":             {},
	"response.output_text.delta":             {},
	"response.output_text.done":              {},
	"response.output_text.annotation.added":  {},
	"response.refusal.delta":                 {},
	"response.refusal.done":                  {},
	"response.function_call_arguments.delta": {},
	"response.function_call_arguments.done":  {},
	"response.reasoning_summary_part.added":  {},
	"response.reasoning_summary_part.done":   {},
	"response.reasoning_summary_text.delta":  {},
	"response.reasoning_summary_text.done":   {},
	"response.reasoning_text.delta":          {},
	"response.reasoning_text.done":           {},
}

// responsesTerminalEventTypes are the events after which the stream is finished.
var responsesTerminalEventTypes = map[string]struct{}{
	"response.completed":  {},
	"response.incomplete": {},
	"response.failed":     {},
}

// ResponsesStream is an iterator over a streaming Responses request's typed SSE
// events. Call Recv until it returns io.EOF, then Close (or just defer Close). It
// is not safe for concurrent use. There is no `[DONE]` sentinel; a mid-stream
// "error" event surfaces from Recv as a typed *ApiError.
type ResponsesStream struct {
	resp    *http.Response
	scanner *bufio.Scanner
	done    bool
	err     error
}

func newResponsesStream(resp *http.Response) *ResponsesStream {
	sc := bufio.NewScanner(resp.Body)
	// Allow large SSE frames (default 64 KiB max token; lift to 1 MiB).
	sc.Buffer(make([]byte, 0, 64*1024), 1<<20)
	return &ResponsesStream{resp: resp, scanner: sc}
}

// Recv returns the next recognised event. It returns io.EOF once the stream is
// exhausted (after a terminal event or connection close), a typed *ApiError when
// the stream carries an "error" event, and a non-EOF error on transport or
// decode failure. Event types the SDK does not recognise are skipped silently.
// After any error (including io.EOF) the stream is finished.
func (s *ResponsesStream) Recv() (*pb.ResponsesStreamEvent, error) {
	if s.err != nil {
		return nil, s.err
	}
	if s.done {
		return nil, io.EOF
	}

	var data bytes.Buffer
	haveData := false

	for s.scanner.Scan() {
		line := s.scanner.Text()

		// A blank line terminates the current SSE event.
		if line == "" {
			if !haveData {
				continue // stray blank line between events
			}
			payload := strings.TrimRight(data.String(), "\n")
			data.Reset()
			haveData = false

			ev, skip, terminal, err := s.decodeEvent(payload)
			if err != nil {
				s.finish(err)
				return nil, err
			}
			if skip {
				continue
			}
			if terminal {
				s.finish(io.EOF)
			}
			return ev, nil
		}

		// SSE comment lines start with ':'.
		if strings.HasPrefix(line, ":") {
			continue
		}

		field, value, found := strings.Cut(line, ":")
		if !found {
			field, value = line, ""
		} else if strings.HasPrefix(value, " ") {
			value = value[1:] // strip a single leading space per the SSE spec
		}
		// Parse `data:` lines only; the `event:` line is redundant (the JSON
		// self-describes via its `type` field).
		if field == "data" {
			data.WriteString(value)
			data.WriteByte('\n')
			haveData = true
		}
		// Other SSE fields (event, id, retry) are ignored.
	}

	// Scanner exhausted. Flush a final event that lacked a trailing blank line.
	if err := s.scanner.Err(); err != nil {
		s.finish(err)
		return nil, err
	}
	if haveData {
		payload := strings.TrimRight(data.String(), "\n")
		if payload != "" {
			ev, skip, terminal, err := s.decodeEvent(payload)
			if err != nil {
				s.finish(err)
				return nil, err
			}
			if !skip {
				if terminal {
					s.finish(io.EOF)
				}
				return ev, nil
			}
		}
	}
	s.finish(io.EOF)
	return nil, io.EOF
}

// decodeEvent decodes one SSE `data:` payload. It reports whether the event
// should be skipped (unrecognised type) and whether it is terminal. An "error"
// event is surfaced as a typed *ApiError. A JSON decode failure is returned as-is.
func (s *ResponsesStream) decodeEvent(payload string) (ev *pb.ResponsesStreamEvent, skip, terminal bool, err error) {
	var we wireResponsesStreamEvent
	if e := json.Unmarshal([]byte(payload), &we); e != nil {
		return nil, false, false, e
	}
	if we.Type == "error" {
		msg := ""
		if we.Message != nil {
			msg = *we.Message
		}
		// Mid-stream error: no HTTP status is associated (the response was 200),
		// so Status is 0. Mirrors surfacing transport errors from Recv.
		return nil, false, false, &ApiError{Status: 0, Message: msg}
	}
	if _, ok := knownResponsesEventTypes[we.Type]; !ok {
		return nil, true, false, nil // unrecognised type -> skip
	}
	pbEv, e := we.toPB()
	if e != nil {
		return nil, false, false, e
	}
	_, terminal = responsesTerminalEventTypes[we.Type]
	return pbEv, false, terminal, nil
}

func (s *ResponsesStream) finish(err error) {
	s.done = true
	if s.err == nil {
		s.err = err
	}
	s.resp.Body.Close()
}

// Close releases the underlying connection. Safe to call multiple times and
// after Recv has returned io.EOF.
func (s *ResponsesStream) Close() error {
	s.done = true
	return s.resp.Body.Close()
}
