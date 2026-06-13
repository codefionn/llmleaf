# llmleaf C# / .NET client

Official C#/.NET client for the [llmleaf](../../README.md) LLM proxy. It speaks llmleaf's
OpenAI/OpenRouter-shaped **JSON over HTTP** surface (see [`../SPEC.md`](../SPEC.md)); the typed
model is generated from the single Protocol Buffers schema in
[`../proto/llmleaf/v1/llmleaf.proto`](../proto/llmleaf/v1/llmleaf.proto).

- **Library**: `src/Llmleaf.Client.csproj` — target `net8.0` (LTS, broadly consumable), nullable
  enabled, `LangVersion latest`.
- **Public API**: hand-written immutable `record` DTOs + `LlmleafClient`, all `System.Text.Json`
  and `HttpClient`. The only third-party dependency is `Google.Protobuf` (for the generated types).
- **Example**: `examples/Basic` — non-streaming chat, streaming chat, list models.
- **Tests**: `tests/Llmleaf.Client.Tests` — the wire mapping is exercised end-to-end against an
  in-process HTTP server (`xUnit`).

## Install

NuGet (once published):

```sh
dotnet add package Llmleaf.Client
```

Or reference the project directly from a sibling project:

```sh
dotnet add reference path/to/clients/csharp/src/Llmleaf.Client.csproj
```

## Quickstart

```csharp
using Llmleaf.Client;

using var client = new LlmleafClient(
    "https://gateway.example.com",
    Environment.GetEnvironmentVariable("LLMLEAF_API_KEY")!,
    new LlmleafClientOptions
    {
        Timeout = TimeSpan.FromSeconds(60),
        // AdminToken = "...",      // optional: widens GET /v1/models with `endpoints`
        // HttpClient = myClient,   // optional: inject your own HttpClient (proxies, pooling, TLS)
    });

// Non-streaming chat.
var resp = await client.CreateChatCompletionAsync(new ChatRequest
{
    Model = "gpt-4o-mini",
    Messages = [ChatMessage.Text(Role.User, "Hello!")],
});
Console.WriteLine(resp.Choices[0].Message.Content?.Text);

// Streaming chat — idiomatic IAsyncEnumerable; the `data: [DONE]` sentinel is handled internally.
await foreach (var chunk in client.CreateChatCompletionStreamAsync(new ChatRequest
{
    Model = "gpt-4o-mini",
    Messages = [ChatMessage.Text(Role.User, "Stream me a haiku.")],
}))
{
    Console.Write(chunk.Choices.Count > 0 ? chunk.Choices[0].Delta.Content : null);
}

// List models.
var models = await client.ListModelsAsync(new ListModelsOptions { Type = ModelType.Llm });
foreach (var m in models.Data) Console.WriteLine(m.Id);
```

Every call takes a final `CancellationToken`. Any non-2xx response is thrown as a typed
`ApiException` with an `int Status` and the message parsed from the `{"error":{"message":...}}`
envelope.

### Covered endpoints

| Method | Endpoint |
|--------|----------|
| `CreateChatCompletionAsync` | `POST /v1/chat/completions` |
| `CreateChatCompletionStreamAsync` | `POST /v1/chat/completions` (SSE → `IAsyncEnumerable<ChatCompletionChunk>`) |
| `CreateEmbeddingAsync` | `POST /v1/embeddings` (base64 `encoding_format` is decoded to floats for you) |
| `ListModelsAsync` | `GET /v1/models` |
| `CreateSpeechAsync` | `POST /v1/audio/speech` (returns bytes + `Content-Type`) |
| `ListVoicesAsync` | `GET /v1/audio/voices` |
| `CreateTranscriptionAsync` | `POST /v1/audio/transcriptions` (`multipart/form-data`) |
| `CreateBatchAsync` / `RetrieveBatchAsync` / `CancelBatchAsync` | `POST /v1/batches`, `GET /v1/batches/{id}`, `POST /v1/batches/{id}/cancel` |
| `GetBatchResultsAsync` | `GET /v1/batches/{id}/results` (NDJSON → `IAsyncEnumerable<BatchResultLine>`) |

### Free-form JSON & passthrough

`ChatRequest.Extra`, `FunctionDef.Parameters`, `ResponseFormat.JsonSchema`,
`EmbeddingRequest.Extra`, and `SpeechRequest.Extra` are **raw JSON object strings**. They are
spliced verbatim into the request body (never double-encoded), and `Extra` keys are merged at the
top level of the request (explicit fields win over `Extra` keys of the same name). On decode,
`ModelEntry.DefaultParameters` comes back as a raw JSON string.

```csharp
new ChatRequest
{
    Model = "gpt-4o-mini",
    Messages = [ChatMessage.Text(Role.User, "hi")],
    Extra = """{"provider":{"order":["openai","azure"]}}""",
};
```

## Run the example

```sh
export LLMLEAF_BASE_URL="https://gateway.example.com"
export LLMLEAF_API_KEY="sk-..."
# export LLMLEAF_MODEL="gpt-4o-mini"   # optional

dotnet run --project examples/Basic
```

## Regenerate the typed model from the proto

The proto is the source of truth. Regenerate after changing it:

```sh
# from clients/csharp
./scripts/gen.sh

# or from clients/
make gen-csharp
```

`scripts/gen.sh` runs:

```sh
protoc --proto_path=../proto --csharp_out=src/Gen ../proto/llmleaf/v1/llmleaf.proto
```

**Toolchain**: `protoc` (libprotoc 35; the native `--csharp_out` backend, no plugin needed) and
the .NET SDK 8.0+ (built/verified here with SDK 10.0.108). The generated `src/Gen/Llmleaf.cs` is
**committed**; because it lives under the project directory it is compiled by default.

### Why generated types AND hand-written DTOs?

`protoc --csharp_out` emits classes that use the **Google.Protobuf** runtime, whose JSON formatter
produces proto *canonical* JSON — camelCase keys and enum *value names* (`"ASSISTANT"`). That does
**not** match the OpenAI/OpenRouter wire (snake_case keys, lowercase tokens like `"assistant"`), and
it has no notion of the string-or-array `content`/`stop`/`input` fields or the raw free-form JSON
passthrough. So:

- The generated `Llmleaf.cs` is the **codegen artifact / schema proof** — it proves the public
  records track the proto, and a unit test asserts the public enums stay in lockstep with the
  generated ones (same members, values, and proto names).
- The **public request/response types and the HTTP transport** are hand-written `System.Text.Json`
  records (`src/Models.cs`, `src/Wire/*.cs`) that produce the exact JSON in `SPEC.md`:
  `[JsonPropertyName("snake_case")]` on every field, and one generic enum⇄wire helper pair
  (`src/Enums.cs`) that lowercases the proto value name to get the wire token.

The generated code proves the schema; System.Text.Json drives the wire.

## Build & test

```sh
./scripts/gen.sh        # (re)generate src/Gen/Llmleaf.cs
dotnet build            # builds the library + example + tests
dotnet test             # 22 wire/enum tests against an in-process HTTP server
```

## Notes / untested corners

- **Honest caveats — not yet verified against a live gateway.** All endpoints are exercised
  end-to-end against an in-process HTTP test server (`tests/`), which proves the request bytes and
  the response decode match `SPEC.md`. They have **not** been run against a production llmleaf
  gateway here; chat (sync + stream), models, and embeddings are the most heavily covered.
- **TTS / STT / batches** are fully implemented and unit-tested (multipart upload with quoted
  RFC 7578 `Content-Disposition`, NDJSON result streaming, base64-vs-array embedding decode), but
  likewise await live-gateway confirmation. The transcription method returns the structured
  `TranscriptionResponse` for `json`/`verbose_json` and surfaces the raw text in
  `TranscriptionResponse.Text` for `text`/`srt`/`vtt`.
- **Runtime note for the test/example in this environment**: only the .NET 10 runtime is installed
  here, so the `net8.0` test project sets `<RollForward>LatestMajor</RollForward>` to execute on it.
  The shipped library still targets `net8.0`; a consumer with the 8.0 runtime needs no roll-forward.
- The realtime WebSocket surface is intentionally out of scope for the client SDKs.
```
