# llmleaf Go client

Official Go SDK for the [llmleaf](../../README.md) LLM proxy. It speaks llmleaf's
OpenAI/OpenRouter-shaped JSON over HTTP (see [`../SPEC.md`](../SPEC.md)); the typed
model is generated from the single proto source of truth,
[`../proto/llmleaf/v1/llmleaf.proto`](../proto/llmleaf/v1/llmleaf.proto).

- Module: `github.com/codefionn/llmleaf/clients/go`
- Requires Go 1.23+
- Dependencies: the protobuf runtime (`google.golang.org/protobuf`) and the Go
  standard library only.

## Install

```sh
go get github.com/codefionn/llmleaf/clients/go@latest
```

```go
import (
    llmleaf "github.com/codefionn/llmleaf/clients/go"
    pb "github.com/codefionn/llmleaf/clients/go/llmleafpb"
)
```

The package name is `llmleaf` (the transport/client); the generated typed model
lives in the sub-package `llmleafpb` (imported as `pb` above). The generated
types are the public model; the JSON wire mapping is internal.

## Quickstart

```go
client := llmleaf.New(baseURL, apiKey)

// Non-streaming chat
resp, err := client.CreateChatCompletion(ctx, &pb.ChatRequest{
    Model: "gpt-4o-mini",
    Messages: []*pb.ChatMessage{{
        Role:    pb.Role_USER,
        Content: &pb.ChatMessage_Text{Text: "Hello!"},
    }},
})
fmt.Println(resp.GetChoices()[0].GetMessage().GetText())

// Streaming chat
stream, err := client.CreateChatCompletionStream(ctx, &pb.ChatRequest{ /* ... */ })
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
```

### Constructor options

```go
client := llmleaf.New(baseURL, apiKey,
    llmleaf.WithTimeout(30*time.Second),     // HTTP client timeout
    llmleaf.WithAdminToken("…"),             // adds the `endpoints` array to GET /v1/models
    llmleaf.WithHTTPClient(myCustomClient),  // proxies / TLS / transport tuning
)
```

`WithHTTPClient` takes precedence over `WithTimeout`. For long-lived streams,
prefer a custom client with `Timeout: 0` and bound each call with a
`context.Context` deadline instead.

## Endpoints

| Method | Endpoint |
|--------|----------|
| `CreateChatCompletion` | `POST /v1/chat/completions` (non-streaming) |
| `CreateChatCompletionStream` | `POST /v1/chat/completions` (SSE; `Recv()` until `io.EOF`) |
| `CreateEmbeddings` | `POST /v1/embeddings` (decodes `base64` vectors) |
| `ListModels` | `GET /v1/models` (`Type`/`Search` filters; admin `endpoints`) |
| `CreateSpeech` | `POST /v1/audio/speech` → `SpeechResult{Audio, ContentType}` |
| `ListVoices` | `GET /v1/audio/voices` |
| `CreateTranscription` | `POST /v1/audio/transcriptions` (multipart `file` part) |
| `CreateBatch` / `GetBatch` / `CancelBatch` | `POST/GET /v1/batches[/{id}[/cancel]]` |
| `GetBatchResults` | `GET /v1/batches/{id}/results` (NDJSON; `Recv()` until `io.EOF`) |

Streaming chat and batch results are exposed as stream objects with
`Recv() (*T, error)` that return `io.EOF` when exhausted; both have `Close()`.
The `data: [DONE]` SSE sentinel is handled internally and never JSON-parsed.

### Errors

Any non-2xx response is returned as a typed `*ApiError`:

```go
var apiErr *llmleaf.ApiError
if errors.As(err, &apiErr) {
    fmt.Println(apiErr.Status, apiErr.Message) // e.g. 403, "model not allowed"
}
```

### Free-form JSON fields

`ChatRequest.Extra`, `FunctionDef.Parameters`, `ResponseFormat.JsonSchema`,
`EmbeddingRequest.Extra`, and `SpeechRequest.Extra` are raw-JSON **strings** on
the typed model. The transport splices the parsed JSON value into the request
body verbatim (never double-encoded). `Extra` keys merge at the top level of the
request object; an explicit `ChatRequest` field of the same name wins.

```go
&pb.ChatRequest{
    Model: "gpt-4o-mini",
    Extra: ptr(`{"provider":{"order":["openai","azure"]}}`), // raw JSON string
}
```

## Example

A runnable example (non-streaming chat, streaming chat, list models) lives in
[`examples/basic/main.go`](examples/basic/main.go). It reads its config from the
environment:

```sh
export LLMLEAF_BASE_URL=https://gateway.example.com
export LLMLEAF_API_KEY=sk-...
# optional: export LLMLEAF_MODEL=gpt-4o-mini
go run ./examples/basic
```

## Regenerate the typed model from the proto

The proto is the source of truth. After editing
[`../proto/llmleaf/v1/llmleaf.proto`](../proto/llmleaf/v1/llmleaf.proto),
regenerate `llmleafpb/llmleaf.pb.go`:

```sh
# from clients/go
./scripts/gen.sh

# or from clients/
make gen-go
```

### Toolchain

`scripts/gen.sh` runs:

```sh
protoc \
  --proto_path=../proto \
  --go_out=. \
  --go_opt=module=github.com/codefionn/llmleaf/clients/go \
  ../proto/llmleaf/v1/llmleaf.proto
```

You need:

- **`protoc`** — libprotoc 35 (the schema compiler).
- **`protoc-gen-go`** — install with
  `go install google.golang.org/protobuf/cmd/protoc-gen-go@latest` and ensure
  `$(go env GOPATH)/bin` is on your `PATH`.

The generated `llmleafpb/llmleaf.pb.go` is committed so consumers do not need the
toolchain to build the SDK.

## Verified locally

```sh
go build ./...   # ok
go vet ./...     # ok
go test ./...    # ok (httptest round-trips for every endpoint)
```

The test suite (`client_test.go`) round-trips each endpoint against an
`httptest` server, asserting the exact wire shape per SPEC.md: bearer auth,
lowercased enum tokens, bare-string vs array `content`/`stop`/`input`, top-level
`extra` merge, SSE `[DONE]` handling, base64 embedding decode, the typed error
envelope, multipart transcription, and NDJSON batch results.

## Honest notes / untested corners

- **No live gateway was exercised.** Verification is against an in-process
  `httptest` server that mirrors SPEC.md, not a real llmleaf instance. The
  request bodies and response parsing are byte-checked against the spec, but
  behaviour against a production gateway (real provider quirks, exact streaming
  cadence) has not been observed here.
- **TTS / STT / batches** are fully implemented and unit-tested against the
  spec's wire shapes, but have likewise not been run end-to-end against a live
  TTS/STT/batch backend.
- **Multimodal content parts** (`image_url`) round-trip through the wire DTOs and
  are unit-tested for the text arm; the `image_url` arm is covered by encode/
  decode logic but not asserted against a live multimodal model.
- The SSE reader caps a single event frame at 1 MiB; pathologically large single
  chunks would error. Raise the scanner buffer in `stream.go` if needed.

## License

Copyright (C) 2026 Fionn Langhans. Licensed under either of
[Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT) at
your option (`MIT OR Apache-2.0`).
