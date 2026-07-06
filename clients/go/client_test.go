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

func TestResponsesWireAndDecode(t *testing.T) {
	client, srv := newTestClient(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/responses" {
			t.Errorf("path = %q", r.URL.Path)
		}
		body := readBody(t, r)
		if body["model"] != "gpt-4o" {
			t.Errorf("model = %v", body["model"])
		}
		// store is accepted and echoed on the wire as false here.
		if body["store"] != false {
			t.Errorf("store should be false, got %v", body["store"])
		}

		// Tools are FLAT: type/name at the top level, no nested "function".
		tools := body["tools"].([]any)
		tool0 := tools[0].(map[string]any)
		if tool0["type"] != "function" || tool0["name"] != "get_weather" {
			t.Errorf("flat tool wrong: %v", tool0)
		}
		if _, nested := tool0["function"]; nested {
			t.Errorf("tool must be flat, found nested function: %v", tool0)
		}
		if _, ok := tool0["parameters"].(map[string]any); !ok {
			t.Errorf("tool parameters not spliced as JSON object: %T", tool0["parameters"])
		}

		input := body["input"].([]any)
		if len(input) != 4 {
			t.Fatalf("input len = %d, want 4", len(input))
		}
		// [0] message item: role-keyed, NO "type" token, bare-string content.
		msg := input[0].(map[string]any)
		if _, hasType := msg["type"]; hasType {
			t.Errorf("message item must be role-keyed without a type token: %v", msg)
		}
		if msg["role"] != "user" || msg["content"] != "What's the weather?" {
			t.Errorf("message item wrong: %v", msg)
		}
		// [1] function_call item.
		fc := input[1].(map[string]any)
		if fc["type"] != "function_call" || fc["call_id"] != "call_1" || fc["name"] != "get_weather" {
			t.Errorf("function_call item wrong: %v", fc)
		}
		if fc["arguments"] != `{"city":"Paris"}` {
			t.Errorf("function_call arguments = %v (want raw JSON string)", fc["arguments"])
		}
		// [2] function_call_output item.
		fo := input[2].(map[string]any)
		if fo["type"] != "function_call_output" || fo["call_id"] != "call_1" || fo["output"] != `{"temp":15}` {
			t.Errorf("function_call_output item wrong: %v", fo)
		}
		// [3] reasoning item: summary_text vs reasoning_text tokens by list.
		rs := input[3].(map[string]any)
		if rs["type"] != "reasoning" || rs["encrypted_content"] != "enc-abc" {
			t.Errorf("reasoning item wrong: %v", rs)
		}
		sum := rs["summary"].([]any)[0].(map[string]any)
		if sum["type"] != "summary_text" || sum["text"] != "thinking about weather" {
			t.Errorf("reasoning summary entry wrong: %v", sum)
		}
		cnt := rs["content"].([]any)[0].(map[string]any)
		if cnt["type"] != "reasoning_text" || cnt["text"] != "the user wants weather" {
			t.Errorf("reasoning content entry wrong: %v", cnt)
		}

		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"id":"resp_1","object":"response","created_at":1,"status":"completed","model":"gpt-4o",
		  "output":[{"type":"message","role":"assistant","status":"completed",
		    "content":[{"type":"output_text","text":"It is 15C in Paris.","annotations":[]}]}],
		  "usage":{"input_tokens":20,"output_tokens":8,"total_tokens":28,"input_tokens_details":{"cached_tokens":12}},
		  "store":false}`)
	})
	defer srv.Close()

	resp, err := client.CreateResponse(context.Background(), &pb.ResponsesRequest{
		Model: "gpt-4o",
		Store: ptr(false),
		Input: &pb.ResponsesRequest_Items{Items: &pb.ResponseItemList{Items: []*pb.ResponseItem{
			{Item: &pb.ResponseItem_Message{Message: &pb.ResponseMessageItem{
				Role:    "user",
				Content: &pb.ResponseMessageItem_Text{Text: "What's the weather?"},
			}}},
			{Item: &pb.ResponseItem_FunctionCall{FunctionCall: &pb.ResponseFunctionCallItem{
				CallId: "call_1", Name: "get_weather", Arguments: `{"city":"Paris"}`,
			}}},
			{Item: &pb.ResponseItem_FunctionCallOutput{FunctionCallOutput: &pb.ResponseFunctionCallOutputItem{
				CallId: "call_1", Output: `{"temp":15}`,
			}}},
			{Item: &pb.ResponseItem_Reasoning{Reasoning: &pb.ResponseReasoningItem{
				Summary:          []*pb.ResponseReasoningText{{Text: "thinking about weather"}},
				Content:          []*pb.ResponseReasoningText{{Text: "the user wants weather"}},
				EncryptedContent: ptr("enc-abc"),
			}}},
		}}},
		Tools: []*pb.ResponsesToolDef{{
			Type: "function", Name: "get_weather", Parameters: ptr(`{"type":"object"}`),
		}},
	})
	if err != nil {
		t.Fatalf("CreateResponse: %v", err)
	}
	if resp.GetStatus() != "completed" {
		t.Errorf("status = %q", resp.GetStatus())
	}
	if resp.Store == nil || resp.GetStore() {
		t.Errorf("store should decode as present-and-false, got %v", resp.Store)
	}
	out := resp.GetOutput()
	if len(out) != 1 {
		t.Fatalf("output len = %d", len(out))
	}
	parts := out[0].GetMessage().GetParts().GetItems()
	if len(parts) != 1 || parts[0].GetOutputText().GetText() != "It is 15C in Paris." {
		t.Fatalf("output message parts wrong: %+v", parts)
	}
	u := resp.GetUsage()
	if u.GetInputTokens() != 20 || u.GetOutputTokens() != 8 || u.GetTotalTokens() != 28 {
		t.Errorf("usage tokens wrong: %+v", u)
	}
	if u.GetInputTokensDetails().GetCachedTokens() != 12 {
		t.Errorf("cached_tokens = %d, want 12", u.GetInputTokensDetails().GetCachedTokens())
	}
}

func TestResponsesStreaming(t *testing.T) {
	client, srv := newTestClient(func(w http.ResponseWriter, r *http.Request) {
		body := readBody(t, r)
		if body["stream"] != true {
			t.Errorf("stream should be true, got %v", body["stream"])
		}
		// A bare-string input serialises as a JSON string, not an array.
		if body["input"] != "hi" {
			t.Errorf("input = %v (want bare string)", body["input"])
		}
		w.Header().Set("Content-Type", "text/event-stream")
		fl := w.(http.Flusher)
		// Each frame carries the redundant `event:` line plus the self-describing
		// `data:` JSON. The stream ends on response.completed — there is no [DONE].
		frames := []string{
			`event: response.created
data: {"type":"response.created","sequence_number":0,"response":{"id":"resp_1","object":"response","created_at":1,"status":"in_progress","model":"gpt-4o"}}`,
			`event: response.output_item.added
data: {"type":"response.output_item.added","sequence_number":1,"output_index":0,"item":{"type":"function_call","call_id":"call_1","name":"get_weather","arguments":""}}`,
			`event: response.function_call_arguments.delta
data: {"type":"response.function_call_arguments.delta","sequence_number":2,"item_id":"fc_1","delta":"{\"city\":\"Paris\"}"}`,
			`event: response.output_text.delta
data: {"type":"response.output_text.delta","sequence_number":3,"delta":"Hel"}`,
			`event: response.output_text.delta
data: {"type":"response.output_text.delta","sequence_number":4,"delta":"lo"}`,
			// An event type this SDK does not recognise: MUST be skipped silently.
			`event: response.future_unknown_event
data: {"type":"response.future_unknown_event","sequence_number":5}`,
			`event: response.completed
data: {"type":"response.completed","sequence_number":6,"response":{"id":"resp_1","object":"response","created_at":1,"status":"completed","model":"gpt-4o","output":[{"type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"Hello","annotations":[]}]}],"usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15,"input_tokens_details":{"cached_tokens":4}}}}`,
		}
		for _, f := range frames {
			io.WriteString(w, f+"\n\n")
			fl.Flush()
		}
	})
	defer srv.Close()

	stream, err := client.CreateResponseStream(context.Background(), &pb.ResponsesRequest{
		Model: "gpt-4o",
		Input: &pb.ResponsesRequest_Text{Text: "hi"},
	})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	defer stream.Close()

	var types []string
	var assembled strings.Builder
	var lastSeq uint64
	var terminal *pb.ResponsesStreamEvent
	var sawFunctionCall bool
	for {
		ev, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatalf("Recv: %v", err)
		}
		if len(types) > 0 && ev.GetSequenceNumber() <= lastSeq {
			t.Errorf("sequence_number not strictly increasing: %d after %d", ev.GetSequenceNumber(), lastSeq)
		}
		lastSeq = ev.GetSequenceNumber()
		types = append(types, ev.GetType())
		switch ev.GetType() {
		case "response.output_text.delta":
			assembled.WriteString(ev.GetDelta())
		case "response.output_item.added":
			if fc := ev.GetItem().GetFunctionCall(); fc != nil && fc.GetName() == "get_weather" {
				sawFunctionCall = true
			}
		case "response.completed":
			terminal = ev
		}
	}

	wantTypes := []string{
		"response.created",
		"response.output_item.added",
		"response.function_call_arguments.delta",
		"response.output_text.delta",
		"response.output_text.delta",
		"response.completed",
	}
	if len(types) != len(wantTypes) {
		t.Fatalf("event types = %v (want %v — unknown type must be skipped)", types, wantTypes)
	}
	for i := range wantTypes {
		if types[i] != wantTypes[i] {
			t.Errorf("event[%d] = %q, want %q", i, types[i], wantTypes[i])
		}
	}
	if assembled.String() != "Hello" {
		t.Errorf("assembled = %q (want Hello)", assembled.String())
	}
	if !sawFunctionCall {
		t.Error("function_call item in output_item.added not decoded")
	}
	if terminal == nil {
		t.Fatal("terminal response.completed event not seen")
	}
	tr := terminal.GetResponse()
	if tr.GetStatus() != "completed" {
		t.Errorf("terminal snapshot status = %q", tr.GetStatus())
	}
	if tr.GetUsage().GetInputTokens() != 10 || tr.GetUsage().GetInputTokensDetails().GetCachedTokens() != 4 {
		t.Errorf("terminal snapshot usage wrong: %+v", tr.GetUsage())
	}
	if got := tr.GetOutput()[0].GetMessage().GetParts().GetItems()[0].GetOutputText().GetText(); got != "Hello" {
		t.Errorf("terminal snapshot output text = %q", got)
	}
}

func TestResponsesApiError(t *testing.T) {
	client, srv := newTestClient(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		io.WriteString(w, `{"error":{"message":"previous_response_id is not supported"}}`)
	})
	defer srv.Close()

	_, err := client.CreateResponse(context.Background(), &pb.ResponsesRequest{
		Model: "gpt-4o",
		Input: &pb.ResponsesRequest_Text{Text: "hi"},
	})
	var apiErr *ApiError
	if !errors.As(err, &apiErr) {
		t.Fatalf("expected *ApiError, got %T: %v", err, err)
	}
	if apiErr.Status != http.StatusBadRequest {
		t.Errorf("status = %d", apiErr.Status)
	}
	if apiErr.Message != "previous_response_id is not supported" {
		t.Errorf("message = %q", apiErr.Message)
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
