package llmleaf

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"net/url"
	"strconv"
	"strings"

	pb "github.com/codefionn/llmleaf/clients/go/llmleafpb"
)

// SpeechResult is the raw audio returned by CreateSpeech, plus the Content-Type
// the gateway reported (e.g. "audio/mpeg" for mp3).
type SpeechResult struct {
	Audio       []byte
	ContentType string
}

// CreateSpeech synthesises speech (POST /v1/audio/speech). The response is raw
// audio bytes; the returned ContentType reflects req.ResponseFormat
// (mp3->audio/mpeg, wav->audio/wav, ...).
func (c *Client) CreateSpeech(ctx context.Context, req *pb.SpeechRequest) (*SpeechResult, error) {
	body, err := encodeSpeechRequest(req)
	if err != nil {
		return nil, err
	}
	resp, err := c.doJSON(ctx, http.MethodPost, "/v1/audio/speech", body, "")
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	audio, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("llmleaf: read speech audio: %w", err)
	}
	return &SpeechResult{Audio: audio, ContentType: resp.Header.Get("Content-Type")}, nil
}

// ListVoices fetches the voice catalog for a TTS model
// (GET /v1/audio/voices?model=<id>).
func (c *Client) ListVoices(ctx context.Context, model string) (*pb.VoicesResponse, error) {
	q := url.Values{}
	q.Set("model", model)
	resp, err := c.doJSON(ctx, http.MethodGet, "/v1/audio/voices?"+q.Encode(), nil, "application/json")
	if err != nil {
		return nil, err
	}
	var wr wireVoicesResponse
	if err := decodeJSONBody(resp, &wr); err != nil {
		return nil, err
	}
	return wr.toPB(), nil
}

// TranscriptionFile is the audio payload for a transcription request: the bytes
// and a filename (the filename's extension helps the server sniff the format).
type TranscriptionFile struct {
	Filename string
	Data     io.Reader
}

// TranscriptionResult holds the outcome of CreateTranscription. Exactly one of
// Response (json / verbose_json formats) or Text (text / srt / vtt formats) is
// populated, matching the request's response_format.
type TranscriptionResult struct {
	// Response is set for response_format "json" / "verbose_json".
	Response *pb.TranscriptionResponse
	// Text is set for response_format "text" / "srt" / "vtt": the plain-text body.
	Text string
}

// CreateTranscription transcribes audio (POST /v1/audio/transcriptions,
// multipart/form-data). The audio is sent as the `file` part; req carries the
// accompanying form fields. For response_format json/verbose_json the result's
// Response is populated; for text/srt/vtt the result's Text is the body.
func (c *Client) CreateTranscription(ctx context.Context, file TranscriptionFile, req *pb.TranscriptionRequest) (*TranscriptionResult, error) {
	if file.Data == nil {
		return nil, fmt.Errorf("llmleaf: transcription file data is required")
	}
	filename := file.Filename
	if filename == "" {
		filename = "audio"
	}

	var buf bytes.Buffer
	mw := multipart.NewWriter(&buf)

	part, err := mw.CreateFormFile("file", filename)
	if err != nil {
		return nil, err
	}
	if _, err := io.Copy(part, file.Data); err != nil {
		return nil, fmt.Errorf("llmleaf: copy transcription file: %w", err)
	}

	fields := map[string]string{"model": req.GetModel()}
	if req.Language != nil {
		fields["language"] = *req.Language
	}
	if req.Prompt != nil {
		fields["prompt"] = *req.Prompt
	}
	if req.ResponseFormat != nil {
		fields["response_format"] = *req.ResponseFormat
	}
	if req.Temperature != nil {
		fields["temperature"] = strconv.FormatFloat(float64(*req.Temperature), 'g', -1, 32)
	}
	for k, v := range fields {
		if err := mw.WriteField(k, v); err != nil {
			return nil, err
		}
	}
	if err := mw.Close(); err != nil {
		return nil, err
	}

	httpReq, err := c.newRequest(ctx, http.MethodPost, "/v1/audio/transcriptions", &buf)
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", mw.FormDataContentType())

	resp, err := c.http.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	respBody, err := io.ReadAll(io.LimitReader(resp.Body, 16<<20))
	if err != nil {
		return nil, fmt.Errorf("llmleaf: read transcription response: %w", err)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, newAPIError(resp.StatusCode, respBody)
	}

	// json/verbose_json -> structured; text/srt/vtt -> plain text. Branch on the
	// requested format, falling back to a content-type / shape sniff when unset.
	format := ""
	if req.ResponseFormat != nil {
		format = *req.ResponseFormat
	}
	if isStructuredTranscription(format, resp.Header.Get("Content-Type"), respBody) {
		var wr wireTranscriptionResponse
		if err := unmarshalStrict(respBody, &wr); err != nil {
			return nil, fmt.Errorf("llmleaf: decode transcription response: %w", err)
		}
		return &TranscriptionResult{Response: wr.toPB()}, nil
	}
	return &TranscriptionResult{Text: string(respBody)}, nil
}

// isStructuredTranscription decides whether the transcription body is the JSON
// envelope (json/verbose_json) versus a plain-text body (text/srt/vtt). The
// requested format is authoritative; when unset we sniff the content-type and
// then the body shape.
func isStructuredTranscription(format, contentType string, body []byte) bool {
	switch format {
	case "json", "verbose_json":
		return true
	case "text", "srt", "vtt":
		return false
	}
	if strings.Contains(strings.ToLower(contentType), "application/json") {
		return true
	}
	trimmed := bytes.TrimSpace(body)
	return len(trimmed) > 0 && trimmed[0] == '{'
}

// unmarshalStrict decodes JSON, surfacing an error if the body is not a JSON
// object (so a plain-text body is not silently accepted as an empty struct).
func unmarshalStrict(data []byte, v any) error {
	if t := bytes.TrimSpace(data); len(t) == 0 || t[0] != '{' {
		return fmt.Errorf("expected a JSON object, got %q", truncate(string(data), 64))
	}
	return json.Unmarshal(data, v)
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}
