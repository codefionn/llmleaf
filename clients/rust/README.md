# llmleaf-client (Rust)

Async Rust SDK for the [llmleaf](https://github.com/codefionn/llmleaf) LLM proxy.

llmleaf speaks OpenAI/OpenRouter-shaped JSON over HTTP, so the public types are plain `serde`
structs and the transport is `reqwest` over `rustls` by default, or native Compio support when
the `compio` feature is selected. The [proto](../proto/llmleaf/v1/llmleaf.proto) is the source
of truth; its generated Rust model is checked in and exposed under `pb` — the
serde types at the crate root are what you actually use.

## Install

This crate is its own standalone workspace inside the monorepo (intentionally not a member of
the root workspace). Depend on it by path or git:

The MSRV is Rust 1.85.

```toml
[dependencies]
llmleaf-client = { git = "https://github.com/codefionn/llmleaf", branch = "main" }
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
futures = "0.3"
```

No build-time `protoc` installation is needed.

## Example

```rust
use futures::StreamExt;
use llmleaf_client::{Client, ChatRequest, ChatMessage};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::new("https://gateway.example.com", "sk-...")?;

    // Non-streaming chat.
    let resp = client
        .chat(ChatRequest::new("gpt-4o-mini", vec![ChatMessage::user("Say hi.")]))
        .await?;
    println!("{}", resp.first_text().unwrap_or_default());

    // Streaming chat — yields chunks, stops on the [DONE] sentinel.
    let mut stream = client
        .chat_stream(ChatRequest::new("gpt-4o-mini", vec![ChatMessage::user("Count to 5.")]))
        .await?;
    while let Some(chunk) = stream.next().await {
        if let Some(delta) = chunk?.first_delta_text() {
            print!("{delta}");
        }
    }
    Ok(())
}
```

Streaming tool calls are exposed as `choice.delta.tool_calls`. Group fragments by the choice index
and each `ToolCallDelta.index`; retain `id` / `kind` / function `name` whenever present, and append every
`function.arguments` fragment in arrival order until `finish_reason == Some(FinishReason::ToolCalls)`.

### OpenAI Responses dialect

`POST /v1/responses` is the same canonical core behind a different edge dialect. `input` is a
bare string (one user message) or an array of items; use `ResponseItem` to build multi-turn
input (messages, `function_call` / `function_call_output` replay, reasoning items). llmleaf is
stateless by default; `store:true` and `previous_response_id` enable upstream-managed continuation.

```rust
use futures::StreamExt;
use llmleaf_client::{ResponsesRequest, ResponseItem};

// Non-streaming — bare-string input.
let resp = client
    .responses(ResponsesRequest::new("gpt-4o-mini", "Say hi."))
    .await?;
println!("{}", resp.output_text());

// Multi-turn item array (replaying a tool call and its result).
let req = ResponsesRequest::new(
    "gpt-4o-mini",
    vec![
        ResponseItem::user("What's the weather in Paris?"),
        ResponseItem::function_call("call_1", "get_weather", r#"{"city":"Paris"}"#),
        ResponseItem::function_call_output("call_1", r#"{"temp_c":21}"#),
    ],
);

// Streaming — typed events, NO [DONE] sentinel: the stream ends on the terminal
// response.completed / .incomplete / .failed event. Accumulate output_text deltas.
let mut events = client.responses_stream(req).await?;
while let Some(event) = events.next().await {
    if let Some(delta) = event?.output_text_delta() {
        print!("{delta}");
    }
}
```

Need a timeout, an admin token, or your own `reqwest::Client`? Use the builder:

```rust
use std::time::Duration;

let client = Client::builder("https://gateway.example.com", "sk-...")
    .timeout(Duration::from_secs(30))
    .admin_token("admin-...")   // adds per-model `endpoints` to GET /v1/models
    .build()?;
```

## Endpoints

| Call | Method | Notes |
|------|--------|-------|
| Chat | `chat` / `chat_stream` | stream yields `ChatCompletionChunk`, stops on `[DONE]` |
| Responses | `responses` / `responses_stream` | stream yields `ResponsesStreamEvent`, ends on the terminal event (no `[DONE]`) |
| Embeddings | `embeddings` | decodes base64 vectors → `Vec<f32>` |
| Models | `list_models` | `type` filter + `search` |
| Speech (TTS) | `speech` | returns `(bytes, content_type)` |
| Voices | `voices` | |
| Transcribe (STT) | `transcribe` | multipart; `Transcription::Json` or `::Text` |
| Batches | `create_batch` / `get_batch` / `cancel_batch` / `batch_results` | results stream `BatchResultLine` |

Gateway errors come back as `Error::Api { status, message }`, parsed from
`{"error":{"message":"..."}}`.

## Run the example

```sh
export LLMLEAF_BASE_URL="https://gateway.example.com"
export LLMLEAF_API_KEY="sk-..."
cargo run --example basic
```

It lists models, does a non-streaming chat, streams one, then does the same pair over the
Responses dialect (printing deltas live).

## Runtimes

Tokio is the default transport and keeps the `ClientBuilder::http_client(reqwest::Client)` API.
For a Compio-only application, disable default features and enable `compio`:

```toml
llmleaf-client = { version = "0.1", default-features = false, features = ["compio"] }
compio = { version = "0.16", features = ["macros"] }
```

Both features can be enabled together in a dependency graph; Tokio remains the selected default
backend.

The `compio` example runs with:

```sh
cargo run --example compio --no-default-features --features compio
```

## Regenerate from the proto

Consumers do not need `protoc`: `src/gen/llmleaf/v1/llmleaf.v1.rs` is checked in. After editing
[`../proto/llmleaf/v1/llmleaf.proto`](../proto/llmleaf/v1/llmleaf.proto), contributors run
`scripts/gen.sh`, which requires `protoc` and `protoc-gen-prost` 0.5.0 on `PATH`.

## Notes

- The `pb` (prost) types are the codegen proof, not the wire types — use the serde types at
  the crate root for anything over HTTP.
- Free-form JSON fields (`extra`, `ResponseFormat.json_schema`, …) are spliced verbatim and
  not validated (intentional passthrough); `ChatRequest.extra` keys merge at the top level.
- No retries/backoff. The realtime WebSocket surface is out of scope.

## License

Dual-licensed under either the [MIT License](LICENSE-MIT) or the
[Apache License 2.0](LICENSE-APACHE), at your option.
Copyright (C) 2026 Fionn Langhans.
