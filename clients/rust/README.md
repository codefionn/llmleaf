# llmleaf-client (Rust)

Async Rust SDK for the [llmleaf](https://github.com/codefionn/llmleaf) LLM proxy.

llmleaf speaks OpenAI/OpenRouter-shaped JSON over HTTP, so the public types are plain `serde`
structs and the transport is `reqwest` over `rustls` (no system OpenSSL). The
[proto](../proto/llmleaf/v1/llmleaf.proto) is the source of truth: `build.rs` compiles it with
`prost-build` on every build and exposes the messages under `pb` as a codegen check — the
serde types at the crate root are what you actually use.

## Install

This crate is its own standalone workspace inside the monorepo (intentionally not a member of
the root workspace). Depend on it by path or git:

```toml
[dependencies]
llmleaf-client = { git = "https://github.com/codefionn/llmleaf", branch = "main" }
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
futures = "0.3"
```

You need `protoc` (libprotoc 35) on `PATH` at build time — prost shells out to it.

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

It lists models, does a non-streaming chat, and streams one (printing deltas live).

## Regenerate from the proto

Codegen is wired into `build.rs`, so a build *is* a regeneration — after editing
[`../proto/llmleaf/v1/llmleaf.proto`](../proto/llmleaf/v1/llmleaf.proto), just rebuild
(`cargo build`, or `make gen-rust` from `clients/`).

## Notes

- The `pb` (prost) types are the codegen proof, not the wire types — use the serde types at
  the crate root for anything over HTTP.
- Free-form JSON fields (`extra`, `ResponseFormat.json_schema`, …) are spliced verbatim and
  not validated (intentional passthrough); `ChatRequest.extra` keys merge at the top level.
- No retries/backoff. The realtime WebSocket surface is out of scope.

## License

Dual-licensed under [Apache-2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT) at your option.
Copyright (C) 2026 Fionn Langhans.
