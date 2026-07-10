# llmleaf C# / .NET client

.NET client for the [llmleaf](../../README.md) LLM proxy. Speaks llmleaf's
OpenAI/OpenRouter-shaped JSON over HTTP (see [`../SPEC.md`](../SPEC.md)). Targets `net8.0`;
hand-written immutable `record` DTOs over `System.Text.Json` + `HttpClient`, with
`Google.Protobuf` as the only third-party dependency (for the generated schema-proof types).

## Install

```sh
dotnet add package Llmleaf.Client            # NuGet (once published)
# or reference the project directly:
dotnet add reference path/to/clients/csharp/src/Llmleaf.Client.csproj
```

## Example

```csharp
using Llmleaf.Client;

using var client = new LlmleafClient(
    "https://gateway.example.com",
    Environment.GetEnvironmentVariable("LLMLEAF_API_KEY")!,
    new LlmleafClientOptions { Timeout = TimeSpan.FromSeconds(60) });

// Non-streaming chat
var resp = await client.CreateChatCompletionAsync(new ChatRequest
{
    Model = "gpt-4o-mini",
    Messages = [ChatMessage.Text(Role.User, "Say hi.")],
});
Console.WriteLine(resp.Choices[0].Message.Content?.Text);

// Streaming chat — IAsyncEnumerable; the `data: [DONE]` sentinel is handled for you
await foreach (var chunk in client.CreateChatCompletionStreamAsync(new ChatRequest
{
    Model = "gpt-4o-mini",
    Messages = [ChatMessage.Text(Role.User, "Stream me a haiku.")],
}))
{
    Console.Write(chunk.Choices.Count > 0 ? chunk.Choices[0].Delta.Content : null);
}

// Responses dialect (POST /v1/responses) — `Input` is a bare string or an item array
var response = await client.CreateResponseAsync(new ResponsesRequest
{
    Model = "gpt-4o-mini",
    Input = "Say hi.",
});
// (see the example for assembling the output_text parts)

// Streaming responses — typed events, NO `[DONE]`; stops on the terminal event
await foreach (var evt in client.CreateResponseStreamAsync(new ResponsesRequest
{
    Model = "gpt-4o-mini",
    Input = "Stream me a haiku.",
}))
{
    if (evt.Type == "response.output_text.delta")
    {
        Console.Write(evt.Delta);
    }
}
```

Every call takes a final `CancellationToken`. `LlmleafClientOptions` also accepts `AdminToken`
(widens `GET /v1/models` with `endpoints`) and `HttpClient` (bring your own, for proxies /
pooling / TLS).

## Endpoints

| Method | Endpoint |
|--------|----------|
| `CreateChatCompletionAsync` / `CreateChatCompletionStreamAsync` | `POST /v1/chat/completions` |
| `CreateResponseAsync` / `CreateResponseStreamAsync` | `POST /v1/responses` (OpenAI Responses dialect; streaming is typed events with no `[DONE]`) |
| `CreateEmbeddingAsync` | `POST /v1/embeddings` (base64 decoded to floats) |
| `CreateRerankAsync` | `POST /v1/rerank` (plain JSON; no vector decode) |
| `ListModelsAsync` | `GET /v1/models` |
| `CreateSpeechAsync` | `POST /v1/audio/speech` (bytes + `Content-Type`) |
| `ListVoicesAsync` | `GET /v1/audio/voices` |
| `CreateTranscriptionAsync` | `POST /v1/audio/transcriptions` (multipart) |
| `CreateBatchAsync` / `RetrieveBatchAsync` / `CancelBatchAsync` / `GetBatchResultsAsync` | `/v1/batches…` (results: `IAsyncEnumerable`) |

Non-2xx responses throw a typed `ApiException` (`int Status` + the parsed message). Free-form
fields (`Extra`, `Parameters`, `JsonSchema`, …) are raw JSON object strings spliced verbatim;
`Extra` keys merge at the top level (explicit fields win).

The Responses surface is stateless: `Store` is accepted but the response always reports
`store:false`, and there is no retrieval call (`previous_response_id` / `background:true` are
rejected upstream). To continue an encrypted reasoning turn, echo a `ResponseReasoningItem`'s
`EncryptedContent` back verbatim in the next request's `Input`.

## Run the example

```sh
export LLMLEAF_BASE_URL="https://gateway.example.com"
export LLMLEAF_API_KEY="sk-..."
dotnet run --project examples/Basic
```

## Regenerate from the proto

`src/Gen/Llmleaf.cs` is committed (and compiled by default). After editing the proto:

```sh
./scripts/gen.sh        # from clients/csharp   (or: make gen-csharp)
```

Needs `protoc` (libprotoc 35, native `--csharp_out`, no plugin) and the .NET SDK 8.0+. The
generated types are a schema proof — a unit test keeps the public enums in lockstep with them —
while the hand-written `System.Text.Json` records drive the actual OpenAI-shaped wire.

## Notes

- Exercised end-to-end against an in-process HTTP test server (`tests/`) that asserts the
  request bytes and response decode match the spec; not yet run against a live gateway.
- The realtime WebSocket surface is out of scope.
