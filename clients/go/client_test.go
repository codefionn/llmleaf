package llmleaf

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"io"
	"math"
	"mime"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	pb "github.com/codefionn/llmleaf/clients/go/llmleafpb"
)

// readBody reads and parses a request body into a generic map for assertions.
func readBody(t *testing.T, r *http.Request) map[string]any {
	t.Helper()
	b, err := io.ReadAll(r.Body)
	if err != nil {
		t.Fatalf("read body: %v", err)
	}
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatalf("unmarshal body %q: %v", b, err)
	}
	return m
}

func newTestClient(h http.HandlerFunc, opts ...Option) (*Client, *httptest.Server) {
	srv := httptest.NewServer(h)
	return New(srv.URL, "test-key", opts...), srv
}

func TestChatCompletionWireAndAuth(t *testing.T) {
	var gotAuth string
	client, srv := newTestClient(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		if r.URL.Path != "/v1/chat/completions" {
			t.Errorf("path = %q", r.URL.Path)
		}
		body := readBody(t, r)
		if body["model"] != "gpt-4o-mini" {
			t.Errorf("model = %v", body["model"])
		}
		if body["stream"] != false {
			t.Errorf("stream should be false, got %v", body["stream"])
		}
		msgs := body["messages"].([]any)
		first := msgs[0].(map[string]any)
		if first["role"] != "user" {
			t.Errorf("role wire token = %v (want lowercased)", first["role"])
		}
		if first["content"] != "hi" {
			t.Errorf("content = %v (want bare string)", first["content"])
		}
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"id":"c1","object":"chat.completion","created":1,"model":"gpt-4o-mini",
		  "choices":[{"index":0,"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}],
		  "usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3,"cost_usd":0.0001}}`)
	})
	defer srv.Close()

	resp, err := client.CreateChatCompletion(context.Background(), &pb.ChatRequest{
		Model:    "gpt-4o-mini",
		Messages: []*pb.ChatMessage{{Role: pb.Role_USER, Content: &pb.ChatMessage_Text{Text: "hi"}}},
	})
	if err != nil {
		t.Fatalf("CreateChatCompletion: %v", err)
	}
	if gotAuth != "Bearer test-key" {
		t.Errorf("Authorization = %q", gotAuth)
	}
	if got := resp.GetChoices()[0].GetMessage().GetText(); got != "hello" {
		t.Errorf("text = %q", got)
	}
	if resp.GetChoices()[0].GetFinishReason() != pb.FinishReason_STOP {
		t.Errorf("finish_reason = %v (want STOP from wire token)", resp.GetChoices()[0].GetFinishReason())
	}
	if resp.GetUsage().GetCostUsd() != 0.0001 {
		t.Errorf("cost_usd = %v", resp.GetUsage().GetCostUsd())
	}
}

func TestChatExtraMergeAndStopAndToolChoice(t *testing.T) {
	client, srv := newTestClient(func(w http.ResponseWriter, r *http.Request) {
		body := readBody(t, r)
		// extra keys merged at top level
		if body["provider"] == nil {
			t.Errorf("extra key 'provider' not merged at top level: %v", body)
		}
		prov := body["provider"].(map[string]any)
		if prov["order"] == nil {
			t.Errorf("nested extra value not spliced verbatim: %v", prov)
		}
		// single-element stop -> bare string
		if _, ok := body["stop"].(string); !ok {
			t.Errorf("stop should be a bare string for one element, got %T %v", body["stop"], body["stop"])
		}
		// tool_choice mode -> bare string
		if body["tool_choice"] != "auto" {
			t.Errorf("tool_choice = %v (want bare string)", body["tool_choice"])
		}
		// explicit field must win over extra of same name
		if body["model"] != "real-model" {
			t.Errorf("explicit model overridden by extra: %v", body["model"])
		}
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"id":"x","object":"chat.completion","created":1,"model":"m","choices":[]}`)
	})
	defer srv.Close()

	_, err := client.CreateChatCompletion(context.Background(), &pb.ChatRequest{
		Model:      "real-model",
		Messages:   []*pb.ChatMessage{{Role: pb.Role_USER, Content: &pb.ChatMessage_Text{Text: "hi"}}},
		Stop:       []string{"END"},
		ToolChoice: &pb.ToolChoice{Choice: &pb.ToolChoice_Mode{Mode: "auto"}},
		Extra:      ptr(`{"provider":{"order":["a","b"]},"model":"SHOULD_NOT_WIN"}`),
	})
	if err != nil {
		t.Fatalf("CreateChatCompletion: %v", err)
	}
}

func TestChatStreaming(t *testing.T) {
	client, srv := newTestClient(func(w http.ResponseWriter, r *http.Request) {
		body := readBody(t, r)
		if body["stream"] != true {
			t.Errorf("stream should be true, got %v", body["stream"])
		}
		w.Header().Set("Content-Type", "text/event-stream")
		fl := w.(http.Flusher)
		frames := []string{
			`{"id":"c","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":"Hel"}}]}`,
			`{"id":"c","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"content":"lo"}}]}`,
			`{"id":"c","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`,
		}
		for _, f := range frames {
			io.WriteString(w, "data: "+f+"\n\n")
			fl.Flush()
		}
		io.WriteString(w, "data: [DONE]\n\n")
		fl.Flush()
	})
	defer srv.Close()

	stream, err := client.CreateChatCompletionStream(context.Background(), &pb.ChatRequest{
		Model:    "m",
		Messages: []*pb.ChatMessage{{Role: pb.Role_USER, Content: &pb.ChatMessage_Text{Text: "hi"}}},
	})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	defer stream.Close()

	var assembled strings.Builder
	var sawUsage bool
	var sawStop bool
	for {
		chunk, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatalf("Recv: %v", err)
		}
		for _, ch := range chunk.GetChoices() {
			assembled.WriteString(ch.GetDelta().GetContent())
			if ch.GetFinishReason() == pb.FinishReason_STOP {
				sawStop = true
			}
		}
		if chunk.GetUsage() != nil {
			sawUsage = true
		}
	}
	if assembled.String() != "Hello" {
		t.Errorf("assembled = %q (want Hello)", assembled.String())
	}
	if !sawUsage {
		t.Error("usage not seen on terminal chunk")
	}
	if !sawStop {
		t.Error("finish_reason stop not seen")
	}
}

func TestEmbeddingsBase64Decode(t *testing.T) {
	want := []float32{1.5, -2.25, 0.0}
	var raw []byte
	for _, f := range want {
		bits := math.Float32bits(f)
		raw = append(raw, byte(bits), byte(bits>>8), byte(bits>>16), byte(bits>>24))
	}
	b64 := base64.StdEncoding.EncodeToString(raw)

	client, srv := newTestClient(func(w http.ResponseWriter, r *http.Request) {
		body := readBody(t, r)
		// single input -> bare string
		if _, ok := body["input"].(string); !ok {
			t.Errorf("input should be bare string for one element, got %T", body["input"])
		}
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"object":"list","model":"emb","data":[{"object":"embedding","index":0,"embedding":"`+b64+`"}],
		  "usage":{"prompt_tokens":1,"completion_tokens":0,"total_tokens":1}}`)
	})
	defer srv.Close()

	resp, err := client.CreateEmbeddings(context.Background(), &pb.EmbeddingRequest{
		Model:          "emb",
		Input:          []string{"hello"},
		EncodingFormat: ptr("base64"),
	})
	if err != nil {
		t.Fatalf("CreateEmbeddings: %v", err)
	}
	got := resp.GetData()[0].GetEmbedding()
	if len(got) != len(want) {
		t.Fatalf("len = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("embedding[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}

func TestApiErrorEnvelope(t *testing.T) {
	client, srv := newTestClient(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusForbidden)
		io.WriteString(w, `{"error":{"message":"model not allowed"}}`)
	})
	defer srv.Close()

	_, err := client.CreateChatCompletion(context.Background(), &pb.ChatRequest{
		Model:    "m",
		Messages: []*pb.ChatMessage{{Role: pb.Role_USER, Content: &pb.ChatMessage_Text{Text: "hi"}}},
	})
	var apiErr *ApiError
	if !errors.As(err, &apiErr) {
		t.Fatalf("expected *ApiError, got %T: %v", err, err)
	}
	if apiErr.Status != http.StatusForbidden {
		t.Errorf("status = %d", apiErr.Status)
	}
	if apiErr.Message != "model not allowed" {
		t.Errorf("message = %q", apiErr.Message)
	}
}

func TestListModelsQueryAndAdminToken(t *testing.T) {
	client, srv := newTestClient(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Query().Get("type") != "llm" {
			t.Errorf("type = %q", r.URL.Query().Get("type"))
		}
		if r.URL.Query().Get("search") != "gpt" {
			t.Errorf("search = %q", r.URL.Query().Get("search"))
		}
		if r.Header.Get("x-admin-token") != "admin-secret" {
			t.Errorf("x-admin-token = %q", r.Header.Get("x-admin-token"))
		}
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"data":[{"id":"gpt-4o","canonical_slug":"gpt-4o","name":"GPT-4o","created":1,"description":"",
		  "endpoints":[{"provider":"openai","model":"gpt-4o","down":false,"source":"route"}]}]}`)
	}, WithAdminToken("admin-secret"))
	defer srv.Close()

	resp, err := client.ListModels(context.Background(), &ListModelsOptions{Type: "llm", Search: "gpt"})
	if err != nil {
		t.Fatalf("ListModels: %v", err)
	}
	if len(resp.GetData()) != 1 || resp.GetData()[0].GetId() != "gpt-4o" {
		t.Fatalf("data = %+v", resp.GetData())
	}
	if len(resp.GetData()[0].GetEndpoints()) != 1 {
		t.Errorf("admin endpoints not decoded")
	}
}

func TestSpeechBytesAndContentType(t *testing.T) {
	client, srv := newTestClient(func(w http.ResponseWriter, r *http.Request) {
		body := readBody(t, r)
		if body["voice"] != "alloy" {
			t.Errorf("voice = %v", body["voice"])
		}
		w.Header().Set("Content-Type", "audio/mpeg")
		w.Write([]byte{0xFF, 0xFB, 0x10})
	})
	defer srv.Close()

	res, err := client.CreateSpeech(context.Background(), &pb.SpeechRequest{
		Model: "tts-1", Input: "hello", Voice: "alloy", ResponseFormat: ptr("mp3"),
	})
	if err != nil {
		t.Fatalf("CreateSpeech: %v", err)
	}
	if res.ContentType != "audio/mpeg" {
		t.Errorf("content-type = %q", res.ContentType)
	}
	if len(res.Audio) != 3 {
		t.Errorf("audio len = %d", len(res.Audio))
	}
}

func TestTranscriptionMultipartAndText(t *testing.T) {
	client, srv := newTestClient(func(w http.ResponseWriter, r *http.Request) {
		mt, params, err := mime.ParseMediaType(r.Header.Get("Content-Type"))
		if err != nil || mt != "multipart/form-data" {
			t.Fatalf("content-type = %q", r.Header.Get("Content-Type"))
		}
		if err := r.ParseMultipartForm(1 << 20); err != nil {
			t.Fatalf("parse multipart: %v", err)
		}
		_ = params
		f, hdr, err := r.FormFile("file")
		if err != nil {
			t.Fatalf("file part missing: %v", err)
		}
		defer f.Close()
		if hdr.Filename != "speech.mp3" {
			t.Errorf("filename = %q", hdr.Filename)
		}
		if r.FormValue("model") != "whisper-1" {
			t.Errorf("model field = %q", r.FormValue("model"))
		}
		// text format -> plain text body
		w.Header().Set("Content-Type", "text/plain")
		io.WriteString(w, "transcribed text")
	})
	defer srv.Close()

	res, err := client.CreateTranscription(context.Background(),
		TranscriptionFile{Filename: "speech.mp3", Data: strings.NewReader("AUDIOBYTES")},
		&pb.TranscriptionRequest{Model: "whisper-1", ResponseFormat: ptr("text")})
	if err != nil {
		t.Fatalf("CreateTranscription: %v", err)
	}
	if res.Text != "transcribed text" {
		t.Errorf("text = %q", res.Text)
	}
	if res.Response != nil {
		t.Errorf("Response should be nil for text format")
	}
}

func TestTranscriptionJSON(t *testing.T) {
	client, srv := newTestClient(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"text":"hi","language":"en","duration":1.5}`)
	})
	defer srv.Close()

	res, err := client.CreateTranscription(context.Background(),
		TranscriptionFile{Filename: "a.wav", Data: strings.NewReader("x")},
		&pb.TranscriptionRequest{Model: "whisper-1", ResponseFormat: ptr("verbose_json")})
	if err != nil {
		t.Fatalf("CreateTranscription: %v", err)
	}
	if res.Response == nil || res.Response.GetText() != "hi" {
		t.Fatalf("response = %+v", res.Response)
	}
	if res.Response.GetDuration() != 1.5 {
		t.Errorf("duration = %v", res.Response.GetDuration())
	}
}

func TestBatchesCreateAndResults(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/batches", func(w http.ResponseWriter, r *http.Request) {
		body := readBody(t, r)
		reqs := body["requests"].([]any)
		item := reqs[0].(map[string]any)
		if item["custom_id"] != "a" {
			t.Errorf("custom_id = %v", item["custom_id"])
		}
		// nested body is a full ChatRequest object
		inner := item["body"].(map[string]any)
		if inner["model"] != "m" {
			t.Errorf("inner model = %v", inner["model"])
		}
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"id":"batch_1","status":"in_progress","counts":{"total":1,"processing":1,"succeeded":0,"errored":0,"canceled":0,"expired":0}}`)
	})
	mux.HandleFunc("GET /v1/batches/batch_1/results", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/x-ndjson")
		io.WriteString(w, `{"custom_id":"a","response":{"status_code":200,"body":{"id":"r","object":"chat.completion","created":1,"model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}}}`+"\n")
		io.WriteString(w, `{"custom_id":"b","error":{"code":"bad","message":"nope"}}`+"\n")
	})

	srv := httptest.NewServer(mux)
	defer srv.Close()
	client := New(srv.URL, "k")

	handle, err := client.CreateBatch(context.Background(), &pb.BatchCreateRequest{
		Requests: []*pb.BatchRequestItem{{
			CustomId: "a",
			Body: &pb.ChatRequest{
				Model:    "m",
				Messages: []*pb.ChatMessage{{Role: pb.Role_USER, Content: &pb.ChatMessage_Text{Text: "hi"}}},
			},
		}},
	})
	if err != nil {
		t.Fatalf("CreateBatch: %v", err)
	}
	if handle.GetId() != "batch_1" || handle.GetStatus() != pb.BatchStatus_IN_PROGRESS {
		t.Fatalf("handle = %+v", handle)
	}
	if handle.GetCounts().GetTotal() != 1 {
		t.Errorf("counts.total = %d", handle.GetCounts().GetTotal())
	}

	stream, err := client.GetBatchResults(context.Background(), "batch_1")
	if err != nil {
		t.Fatalf("GetBatchResults: %v", err)
	}
	defer stream.Close()
	var lines []*pb.BatchResultLine
	for {
		line, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatalf("Recv: %v", err)
		}
		lines = append(lines, line)
	}
	if len(lines) != 2 {
		t.Fatalf("lines = %d", len(lines))
	}
	if lines[0].GetResponse().GetBody().GetChoices()[0].GetMessage().GetText() != "ok" {
		t.Errorf("first result body wrong: %+v", lines[0])
	}
	if lines[1].GetError().GetMessage() != "nope" {
		t.Errorf("second result error wrong: %+v", lines[1])
	}
}

func TestEnumRoundTrip(t *testing.T) {
	if got := enumToWire(pb.Role_name, int32(pb.Role_ASSISTANT)); got != "assistant" {
		t.Errorf("enumToWire(ASSISTANT) = %q", got)
	}
	if got := enumToWire(pb.FinishReason_name, int32(pb.FinishReason_TOOL_CALLS)); got != "tool_calls" {
		t.Errorf("enumToWire(TOOL_CALLS) = %q", got)
	}
	if got := enumToWire(pb.Role_name, int32(pb.Role_ROLE_UNSPECIFIED)); got != "" {
		t.Errorf("UNSPECIFIED should map to empty, got %q", got)
	}
	if v, ok := enumFromWire(pb.BatchStatus_value, "in_progress"); !ok || pb.BatchStatus(v) != pb.BatchStatus_IN_PROGRESS {
		t.Errorf("enumFromWire(in_progress) = %v, %v", v, ok)
	}
	if _, ok := enumFromWire(pb.Role_value, "bogus"); ok {
		t.Error("unknown token should not map")
	}
}
