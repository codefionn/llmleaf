//! # llmleaf-client
//!
//! Official async Rust SDK for the [llmleaf](https://github.com/codefionn/llmleaf) LLM
//! proxy. It speaks llmleaf's OpenAI/OpenRouter-shaped JSON over HTTP — the wire is JSON,
//! never protobuf-binary — and covers every endpoint in `clients/SPEC.md`:
//!
//! * chat completions, non-streaming ([`Client::chat`]) and streaming
//!   ([`Client::chat_stream`], an `impl Stream` of [`ChatCompletionChunk`]s that stops on
//!   the `[DONE]` sentinel);
//! * the OpenAI Responses dialect, non-streaming ([`Client::responses`]) and streaming
//!   ([`Client::responses_stream`], an `impl Stream` of [`ResponsesStreamEvent`]s that ends
//!   on the terminal `response.completed`/`incomplete`/`failed` event — there is no
//!   `[DONE]` sentinel);
//! * embeddings ([`Client::embeddings`], with base64 → float decoding);
//! * model catalog ([`Client::list_models`]);
//! * text-to-speech ([`Client::speech`] → `(bytes, content_type)`) and the voice catalog
//!   ([`Client::voices`]);
//! * speech-to-text ([`Client::transcribe`], multipart with the `file` part);
//! * batches ([`Client::create_batch`], [`Client::get_batch`], [`Client::cancel_batch`],
//!   [`Client::batch_results`] → an NDJSON stream of [`BatchResultLine`]s).
//!
//! ## Codegen
//!
//! `build.rs` compiles `proto/llmleaf/v1/llmleaf.proto` with `prost-build` (real codegen,
//! run every build); the generated messages are exposed under [`pb`]. The public types
//! (re-exported at the crate root, e.g. [`ChatRequest`]) are hand-written serde structs
//! that produce the exact OpenAI JSON wire — prost compiles the proto while serde drives
//! the wire.
//!
//! ## Quickstart
//!
//! ```no_run
//! use llmleaf_client::{Client, ChatRequest, ChatMessage};
//!
//! # async fn run() -> Result<(), llmleaf_client::Error> {
//! let client = Client::new("https://gateway.example.com", "sk-...")?;
//! let resp = client
//!     .chat(ChatRequest::new(
//!         "gpt-4o-mini",
//!         vec![ChatMessage::user("hi")],
//!     ))
//!     .await?;
//! println!("{}", resp.first_text().unwrap_or_default());
//! # Ok(())
//! # }
//! ```

// Public types mirror `proto/llmleaf/v1/llmleaf.proto` field-for-field; the proto and
// `clients/SPEC.md` are the authoritative documentation for individual fields, so we warn
// on undocumented top-level items but not on every mirrored struct field.
#![warn(missing_docs)]
#![allow(clippy::doc_markdown)]

pub mod pb;

mod client;
mod error;
mod stream;
mod wire;

// The wire-mirror structs document themselves at the type level; their fields map
// one-to-one onto the proto/SPEC keys, so we don't repeat that prose per field.
#[allow(missing_docs)]
mod types;

pub use client::{Client, ClientBuilder};
pub use error::{Error, Result};
pub use types::*;
