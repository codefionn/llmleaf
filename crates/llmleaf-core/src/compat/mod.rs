//! Compat surfaces — the dialect mappings at the front edge (SOUL.md principle 3).
//!
//! Consumers speak a familiar dialect; these modules map that dialect *into* the canonical
//! [`llmleaf_model`] on the way in and *out of* it on the way out. No dialect is "native" and none
//! gets a shortcut through the core: OpenAI here is exactly as privileged as any future surface.
//!
//! Fidelity beats convenience (decision filter): match the documented wire behavior even when awkward.
//!
//! One surface per modality, mostly in the OpenAI/OpenRouter dialect: [`openai`] (chat), [`embeddings`],
//! [`rerank`] (query + documents → ordered scores, the Cohere/Jina/OpenRouter dialect), [`speech`]
//! (text-to-speech), [`transcription`] (speech-to-text), and [`batch`] (asynchronous jobs).
//! [`anthropic`] is a second chat dialect on the same canonical core — the Anthropic Messages API — and
//! [`responses`] is a third — the OpenAI Responses API, served statelessly — both mapped at the edge
//! exactly like the rest (principle 3: no dialect is privileged). Routing treats them uniformly — a
//! logical model resolves to the same fallback chain whatever the dialect or modality (batch excepted:
//! it targets one provider and does not fall back — see [`batch`]).
//!
//! [`openapi`] is the odd one out: not a per-request mapping but the static *description* of all the
//! others — the OpenAPI 3.1 contract served at `GET /v1/openapi.json`.

pub mod anthropic;
pub mod batch;
pub mod embeddings;
pub mod openai;
pub mod openapi;
pub mod realtime;
pub mod rerank;
pub mod responses;
pub mod speech;
pub mod transcription;
