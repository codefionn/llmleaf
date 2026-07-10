# llmleaf Go client

Go SDK for the [llmleaf](../../README.md) LLM proxy. Speaks llmleaf's OpenAI/OpenRouter-shaped
JSON over HTTP (see [`../SPEC.md`](../SPEC.md)); the typed model is generated from the
[proto](../proto/llmleaf/v1/llmleaf.proto).

- Module `github.com/codefionn/llmleaf/clients/go`, Go 1.23+
- Dependencies: the protobuf runtime + the standard library only

## Install

```sh
go get github.com/codefionn/llmleaf/clients/go@latest
```

The package `llmleaf` is the transport/client; the generated types live in the sub-package
`llmleafpb` (imported as `pb` below).

## Example

```go
import (
    llmleaf "github.com/codefionn/llmleaf/clients/go"
    pb "github.com/codefionn/llmleaf/clients/go/llmleafpb"
)

client := llmleaf.New(baseURL, apiKey, llmleaf.WithTimeout(30*time.Second))

// Non-streaming chat
resp, err := client.CreateChatCompletion(ctx, &pb.ChatRequest{
    Model: "gpt-4o-mini",
    Messages: []*pb.ChatMessage{{
        Role:    pb.Role_USER,
        Content: &pb.ChatMessage_Text{Text: "Say hi."},
    }},
})
fmt.Println(resp.GetChoices()[0].GetMessage().GetText())

// Streaming chat — Recv() until io.EOF; the [DONE] sentinel is handled internally
stream, err := client.CreateChatCompletionStream(ctx, req)
defer stream.Close()
for {
    chunk, err := stream.Recv()
    if errors.Is(err, io.EOF) {
        break
    }
    if err != nil {
        log.Fatal(err)
    }
    fmt.Print(chunk.GetChoices()[0].GetDelta().GetContent())
}

// OpenAI Responses dialect — `input` is a bare string or an array of items
resp, err := client.CreateResponse(ctx, &pb.ResponsesRequest{
    Model: "gpt-4o-mini",
    Input: &pb.ResponsesRequest_Text{Text: "Say hi."},
})

// Streaming Responses — typed SSE events, NO [DONE]; Recv() stops on the terminal
// response.completed / .incomplete / .failed event. Unknown event types are
// skipped; a mid-stream "error" event surfaces as *ApiError.
rs, err := client.CreateResponseStream(ctx, req)
defer rs.Close()
for {
    ev, err := rs.Recv()
    if errors.Is(err, io.EOF) {
        break
    }
    if err != nil {
        log.Fatal(err)
    }
    if ev.GetType() == "response.output_text.delta" {
        fmt.Print(ev.GetDelta())
    }
}
```

Construct with options: `WithTimeout`, `WithAdminToken` (adds the `endpoints` array to
`GET /v1/models`), and `WithHTTPClient` for proxies / TLS / transport tuning. `WithHTTPClient`
wins over `WithTimeout`; for long-lived streams prefer a client with `Timeout: 0` plus a
`context` deadline per call.

## Endpoints

| Method | Endpoint |
|--------|----------|
| `CreateChatCompletion` / `CreateChatCompletionStream` | `POST /v1/chat/completions` |
| `CreateResponse` / `CreateResponseStream` | `POST /v1/responses` (typed SSE, no `[DONE]`) |
| `CreateEmbeddings` | `POST /v1/embeddings` (decodes base64 vectors) |
| `CreateRerank` | `POST /v1/rerank` → `RerankResponse{Results, Usage}` |
| `ListModels` | `GET /v1/models` |
| `CreateSpeech` | `POST /v1/audio/speech` → `SpeechResult{Audio, ContentType}` |
| `ListVoices` | `GET /v1/audio/voices` |
| `CreateTranscription` | `POST /v1/audio/transcriptions` (multipart) |
| `CreateBatch` / `GetBatch` / `CancelBatch` / `GetBatchResults` | `/v1/batches…` (results: `Recv()` until `io.EOF`) |

Non-2xx responses come back as a typed `*ApiError`:

```go
var apiErr *llmleaf.ApiError
if errors.As(err, &apiErr) {
    fmt.Println(apiErr.Status, apiErr.Message) // 403, "model not allowed"
}
```

Free-form fields (`Extra`, `Parameters`, `JsonSchema`, …) are raw-JSON strings spliced
verbatim into the body; `Extra` keys merge at the top level (an explicit field of the same
name wins).

## Run the example

[`examples/basic/main.go`](examples/basic/main.go) — chat, streaming, model list;
[`examples/responses/main.go`](examples/responses/main.go) — the Responses dialect, non-stream + stream:

```sh
export LLMLEAF_BASE_URL=https://gateway.example.com
export LLMLEAF_API_KEY=sk-...
go run ./examples/basic
go run ./examples/responses
```

## Regenerate from the proto

The generated `llmleafpb/llmleaf.pb.go` is committed, so consumers don't need the toolchain.
After editing the proto:

```sh
./scripts/gen.sh        # from clients/go   (or: make gen-go from clients/)
```

Needs `protoc` (libprotoc 35) and `protoc-gen-go`
(`go install google.golang.org/protobuf/cmd/protoc-gen-go@latest`, with `$(go env GOPATH)/bin`
on your `PATH`).

## Notes

Verified against an in-process `httptest` server that mirrors the spec (bearer auth, lowercased
enum tokens, base64 embeddings, chat SSE `[DONE]`, typed Responses SSE with no sentinel + unknown-event
skipping, multipart, NDJSON), not yet against a live gateway. The SSE reader caps a single event
frame at 1 MiB.

## License

Dual-licensed under [Apache-2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT). Copyright (C) 2026
Fionn Langhans.
