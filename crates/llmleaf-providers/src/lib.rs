//! First-party, compiled-in providers (SOUL.md principle 2: first-party providers are compiled Rust
//! trait implementations, zero overhead).
//!
//! This crate is deliberately **not** a dependency of the core. The binary wires these into a
//! [`ProviderRegistry`] via [`build`], keyed on the config `kind`. Third-party providers will arrive
//! as WASM plugins presenting the same [`llmleaf_provider::Provider`] trait across the sandbox — they
//! register through the same registry, so the core stays oblivious to either flavor.
//!
//! Two shapes of provider live here:
//!   - the **OpenAI-compatible family** ([`compat`]) — one config-driven provider over a quirk table,
//!     covering OpenAI, OpenRouter, Groq, DeepSeek, xAI, Mistral, Together, Fireworks, Perplexity,
//!     Cerebras, Z.AI (GLM), Moonshot (Kimi), and Azure OpenAI;
//!   - **distinct-dialect providers** — Anthropic, Google Gemini, Google Vertex AI, Cohere, Ollama (its
//!     native `/api/*` surface), and LM Studio (its native `/api/v0/*` surface) — each a native mapping
//!     module because its wire format is its own thing. (Vertex reuses Gemini's body mapping but owns its
//!     own transport: OAuth2 bearer, a project/location publisher URL, and the `:predict` embeddings
//!     dialect. LM Studio reuses the OpenAI-wire chat/embeddings mapping but owns its `/api/v0` transport
//!     and rich model catalog.)

use std::sync::Arc;

use llmleaf_provider::Provider;

mod anthropic;
mod batch;
mod cohere;
mod compat;
mod gemini;
mod http;
mod lmstudio;
mod mock;
mod ollama;
mod openai_wire;
mod realtime_ws;
mod thinking;
pub mod transport;
mod vertex;

/// Injectable fakes for unit tests, benches, and the seeded simulation — never compiled into the
/// binary (the `test-support` feature is enabled only by `llmleaf-bench`/`llmleaf-simulation` and the
/// crate's own tests). They satisfy [`transport::HttpTransport`]/[`transport::RealtimeTransport`] with
/// canned responses or seeded errors, so a real provider's build+parse path runs without a network.
#[cfg(any(test, feature = "test-support"))]
pub mod fake;

/// Integration-style round-trip tests that drive the *real* provider implementations through the
/// injectable fake transport: a canned upstream response (or seeded error) goes in, the provider's
/// request-build + response-parse path runs end to end, and the canonical output (or the exact
/// [`llmleaf_model::ModelError`] variant) is asserted. No network, no live keys — see [`crate::fake`].
#[cfg(test)]
mod roundtrip;

pub use anthropic::AnthropicProvider;
pub use cohere::CohereProvider;
pub use compat::{Brand, OpenAiCompatProvider};
pub use gemini::GeminiProvider;
pub use lmstudio::LmStudioProvider;
pub use mock::EchoProvider;
pub use ollama::OllamaProvider;
pub use transport::{
    HttpRequest, HttpResponse, HttpTransport, Method, RealtimeTransport, ReqwestTransport,
    Transports,
};
pub use vertex::VertexProvider;

/// Instantiate a first-party provider by its config `kind`. Returns `None` for unknown kinds so the
/// binary can produce a clear startup error. This is the factory the core must never contain.
pub fn build(kind: &str, transports: &Transports) -> Option<Arc<dyn Provider>> {
    match kind {
        "echo" | "mock" => Some(Arc::new(EchoProvider)),
        "anthropic" | "claude" => Some(Arc::new(AnthropicProvider::new(transports))),
        "gemini" | "google" => Some(Arc::new(GeminiProvider::new(transports))),
        // Vertex AI: the enterprise Gemini surface (OAuth2 bearer, project/location publisher path).
        "vertex" | "vertex-ai" | "google-vertex" => Some(Arc::new(VertexProvider::new(transports))),
        "cohere" => Some(Arc::new(CohereProvider::new(transports))),
        // Local runtimes with their own native APIs (NOT the OpenAI-compat shims): Ollama's `/api/*`
        // (NDJSON streaming, native model management) and LM Studio's `/api/v0/*` (rich catalog).
        "ollama" => Some(Arc::new(OllamaProvider::new(transports))),
        "lmstudio" | "lm-studio" => Some(Arc::new(LmStudioProvider::new(transports))),
        // Everything else falls through to the OpenAI-compatible family table.
        other => OpenAiCompatProvider::for_kind(other, transports)
            .map(|p| Arc::new(p) as Arc<dyn Provider>),
    }
}

/// The set of `kind`s this build knows how to instantiate (for diagnostics).
pub fn known_kinds() -> Vec<&'static str> {
    let mut kinds = vec![
        "echo",
        "anthropic",
        "claude",
        "gemini",
        "google",
        "vertex",
        "vertex-ai",
        "google-vertex",
        "cohere",
        "ollama",
        "lmstudio",
        "lm-studio",
    ];
    kinds.extend_from_slice(Brand::kinds());
    kinds
}
