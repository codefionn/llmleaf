# llmleaf-client (Rust)

Official async Rust SDK for the [llmleaf](https://github.com/codefionn/llmleaf) LLM proxy.

llmleaf speaks **OpenAI/OpenRouter-shaped JSON over HTTP** — the wire is JSON, never
protobuf-binary. This crate's public types and transport are hand-written `serde` structs
that produce that exact JSON (see [`../SPEC.md`](../SPEC.md)). The typed model is also
**genuinely compiled from the proto**: `build.rs` runs `prost-build` on
[`../proto/llmleaf/v1/llmleaf.proto`](../proto/llmleaf/v1/llmleaf.proto) every build and
exposes the generated messages under the `pb` module. So prost compiles the proto (codegen
proof + a canonical-proto mirror) while serde drives the wire.

## Install

This crate is part of the llmleaf monorepo and is **its own standalone workspace** (it is
intentionally *not* a member of the repo-root Cargo workspace). Depend on it by path or git.

Path dependency (vendored / monorepo):

```toml
[dependencies]
llmleaf-client = { path = "../llmleaf/clients/rust" }
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
futures = "0.3"
```

Git dependency:

```toml
[dependencies]
llmleaf-client = { git = "https://github.com/codefionn/llmleaf", branch = "main" }
```

## Quickstart

```rust
use futures::StreamExt;
use llmleaf_client::{Client, ChatRequest, ChatMessage};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::new("https://gateway.example.com", "sk-...")?;

    // Non-streaming chat.
    let resp = client
        .chat(ChatRequest::new("gpt-4o-mini", vec![ChatMessage::user("hi")]))
        .await?;
    println!("{}", resp.first_text().unwrap_or_default());

    // Streaming chat — yields ChatCompletionChunk, stops on the [DONE] sentinel.
    let mut stream = client
        .chat_stream(ChatRequest::new("gpt-4o-mini", vec![ChatMessage::user("count to 3")]))
        .await?;
    while let Some(chunk) = stream.next().await {
        if let Some(delta) = chunk?.first_delta_text() {
            print!("{delta}");
        }
    }
    Ok(())
}
```

### Builder (timeout, admin token, custom HTTP client)

```rust
use std::time::Duration;
use llmleaf_client::Client;

let client = Client::builder("https://gateway.example.com", "sk-...")
    .timeout(Duration::from_secs(30))
    .admin_token("admin-...")           // adds per-model `endpoints` to GET /v1/models
    // .http_client(my_reqwest_client)  // bring your own reqwest::Client
    .build()?;
```

## Endpoints

Every endpoint in [`../SPEC.md`](../SPEC.md) is covered:

| Call | Method on `Client` | Notes |
|------|--------------------|-------|
| Chat (non-streaming) | `chat` | forces `stream:false` |
| Chat (streaming) | `chat_stream` | `impl Stream<Item = Result<ChatCompletionChunk>>`, stops on `[DONE]` |
| Embeddings | `embeddings` | decodes `encoding_format:"base64"` (LE f32) → `Vec<f32>` |
| List models | `list_models` | `type` filter + `search` substring |
| Speech (TTS) | `speech` | returns `(bytes, content_type)` |
| Voices | `voices` | |
| Transcriptions (STT) | `transcribe` | multipart with the `file` part; returns `Transcription::Json` or `::Text` |
| Batch create / get / cancel | `create_batch` / `get_batch` / `cancel_batch` | return `BatchHandle` |
| Batch results | `batch_results` | NDJSON `impl Stream<Item = Result<BatchResultLine>>` |

Errors surface as the typed [`Error`] enum; gateway errors become
`Error::Api { status, message }`, parsed from `{"error":{"message":"..."}}`.

## Run the example

```sh
export LLMLEAF_BASE_URL="https://gateway.example.com"
export LLMLEAF_API_KEY="sk-..."
export LLMLEAF_MODEL="gpt-4o-mini"   # optional
cargo run --example basic
```

It lists models, does a non-streaming chat, and streams a chat (printing deltas live).

## Regenerate the typed model from the proto

The proto is the single source of truth. Codegen is wired into `build.rs`, so there is
nothing to run by hand — **a build *is* a regeneration**:

```sh
cargo build              # build.rs → prost-build → protoc → $OUT_DIR/llmleaf.v1.rs
./scripts/gen.sh         # the same thing, with a protoc presence check
make gen-rust            # from clients/, delegates to scripts/gen.sh
```

After editing `../proto/llmleaf/v1/llmleaf.proto`, just rebuild. The generated messages
land in `$OUT_DIR/llmleaf.v1.rs` and are `include!`d by `src/pb.rs` (exposed as
`llmleaf_client::pb`).

## Required toolchain

- **Rust** stable, edition 2021 (tested with 1.95; MSRV pinned at 1.74 in `Cargo.toml`).
- **`protoc`** on `PATH` — `prost-build` shells out to it at build time. `libprotoc 35`
  matches the rest of `clients/`; most distros ship it as the `protobuf` package.
- TLS uses **rustls** (no system OpenSSL needed).

## Caveats (honest)

- **`protoc` is a build-time requirement.** Without it on `PATH`, `cargo build` fails in
  `build.rs`. This is the cost of the "real codegen" guarantee. If you would rather vendor
  the binary, add `protoc-bin-vendored` as a build-dependency and point `PROTOC` at it; the
  default here relies on the system `protoc` because it is already installed across the
  repo.
- **`pb` (prost) types are not the wire types.** They are the compiled-proto mirror /
  codegen proof. Use the serde types re-exported at the crate root (`ChatRequest`,
  `EmbeddingRequest`, …) for anything that touches HTTP; prost's oneofs and `i32` enums do
  not serialise to OpenAI JSON.
- **Standalone workspace.** `Cargo.toml` carries an empty `[workspace]` table on purpose so
  this crate is not adopted by the repo-root workspace (`members = ["crates/*"]`). Keep it.
- **Realtime WebSocket is out of scope** for the SDKs (per the clients README).
- **`extra` / free-form JSON** is spliced verbatim; `ChatRequest.extra` keys merge at the
  top level of the request body. The SDK does not validate them — that is intentional
  passthrough (P7), and it means a malformed `extra` reaches the gateway as-is.
- The streaming SSE parser only consumes `data:` lines (it ignores comments and other SSE
  fields), which is all the llmleaf chat stream emits.

## License

Copyright (C) 2026 Fionn Langhans.

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option (`MIT OR Apache-2.0`). Unless you
explicitly state otherwise, any contribution intentionally submitted for
inclusion in this client by you shall be dual licensed as above, without any
additional terms or conditions.
