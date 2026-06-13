package llmleaf

import (
	"bufio"
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"strings"

	pb "github.com/codefionn/llmleaf/clients/go/llmleafpb"
)

// ChatStream is an iterator over a streaming chat completion's
// ChatCompletionChunk frames. Call Recv until it returns io.EOF, then Close (or
// just defer Close). It is not safe for concurrent use.
type ChatStream struct {
	resp    *http.Response
	scanner *bufio.Scanner
	done    bool
	err     error
}

func newChatStream(resp *http.Response) *ChatStream {
	sc := bufio.NewScanner(resp.Body)
	// Allow large SSE frames (default 64 KiB max token; lift to 1 MiB).
	sc.Buffer(make([]byte, 0, 64*1024), 1<<20)
	return &ChatStream{resp: resp, scanner: sc}
}

// Recv returns the next chunk. It returns io.EOF once the stream is exhausted
// (after the `data: [DONE]` sentinel or EOF) and a non-EOF error on transport or
// decode failure. After any error (including io.EOF) the stream is finished.
func (s *ChatStream) Recv() (*pb.ChatCompletionChunk, error) {
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

			if payload == "[DONE]" {
				s.finish(io.EOF)
				return nil, io.EOF
			}
			var wc wireChunk
			if err := json.Unmarshal([]byte(payload), &wc); err != nil {
				s.finish(err)
				return nil, err
			}
			return wc.toPB(), nil
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
		if field == "data" {
			if value == "[DONE]" && !haveData {
				// Sentinel may arrive without a trailing blank line.
				s.finish(io.EOF)
				return nil, io.EOF
			}
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
		if payload != "" && payload != "[DONE]" {
			var wc wireChunk
			if err := json.Unmarshal([]byte(payload), &wc); err != nil {
				s.finish(err)
				return nil, err
			}
			s.finish(io.EOF)
			return wc.toPB(), nil
		}
	}
	s.finish(io.EOF)
	return nil, io.EOF
}

func (s *ChatStream) finish(err error) {
	s.done = true
	if s.err == nil {
		s.err = err
	}
	s.resp.Body.Close()
}

// Close releases the underlying connection. Safe to call multiple times and
// after Recv has returned io.EOF.
func (s *ChatStream) Close() error {
	s.done = true
	return s.resp.Body.Close()
}
