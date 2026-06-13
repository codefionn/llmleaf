//! Canonical embeddings model.
//!
//! The OpenAI/OpenRouter `/v1/embeddings` surface maps *into* [`EmbeddingRequest`]; providers that
//! offer embeddings map *out of* it (those that don't return [`crate::ModelError::Unsupported`]).
//!
//! Embeddings have no token-by-token semantics, so — unlike chat — the canonical response is a single
//! batch object, not a stream. Principle 4 ("streaming is the default") is about *incremental
//! generation*; there is nothing to stream here. Usage rides along for the event/pricing path.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::Usage;

/// A canonical embeddings request. Every dialect maps into this; every embedding-capable provider
/// maps out of it. Anything we don't model rides verbatim in [`EmbeddingRequest::extra`] (principle 7).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EmbeddingRequest {
    /// The logical model the consumer asked for. Routing resolves it to provider targets exactly as
    /// for chat; a provider extension may rewrite it to its own upstream id.
    pub model: String,

    /// One or more inputs to embed. A single-string request maps to a one-element vector; order is
    /// preserved end to end so the response indices line up with the consumer's inputs.
    pub input: Vec<String>,

    /// Optional output dimensionality (providers supporting Matryoshka truncation, e.g. OpenAI
    /// `text-embedding-3-*`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<u32>,

    /// The consumer's requested wire encoding (`float` | `base64`). Carried for the *output* edge to
    /// honor; the canonical vector here is always `f32`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encoding_format: Option<String>,

    /// Dialect-/provider-specific fields preserved verbatim through the core.
    #[serde(default, skip_serializing_if = "Map::is_empty")]
    pub extra: Map<String, Value>,
}

/// One embedding vector and the index of the input it corresponds to.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Embedding {
    pub index: u32,
    pub vector: Vec<f32>,
}

/// A canonical embeddings response. A single batch object (see module note); `usage` is what the
/// provider reported, priced at the edge by lookup like every other usage report (principle 5).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EmbeddingResponse {
    /// The model that actually served (a provider may report its own upstream id).
    pub model: String,
    pub embeddings: Vec<Embedding>,
    #[serde(default)]
    pub usage: Usage,
}
