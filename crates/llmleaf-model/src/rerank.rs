//! Canonical rerank model.
//!
//! Rerank is the fifth modality: a query plus a set of candidate documents in, a relevance-ordered
//! list of `(index, score)` out. The Cohere/Jina/Voyage/OpenRouter `/v1/rerank` (and Cohere's
//! `/v2/rerank`) surface maps *into* [`RerankRequest`]; every rerank-capable provider maps *out of* it
//! (those that don't return [`crate::ModelError::Unsupported`]).
//!
//! Like embeddings, rerank has no token-by-token semantics — the canonical response is a single batch
//! object, not a stream (principle 4 is about incremental *generation*; there is nothing to stream
//! here). Usage rides along for the event/pricing path (principle 5: relayed, never computed).

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::Usage;

/// A canonical rerank request. Every dialect maps into this; every rerank-capable provider maps out of
/// it. Anything we don't model rides verbatim in [`RerankRequest::extra`] (principle 7).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RerankRequest {
    /// The logical model the consumer asked for. Routing resolves it to provider targets exactly as
    /// for chat/embeddings; a provider extension may rewrite it to its own upstream id.
    pub model: String,

    /// The search query the documents are ranked against.
    pub query: String,

    /// The candidate documents to rank. Order is preserved end to end so a result's `index` refers to
    /// the document at that position in *this* list. A document is usually plain text, but multimodal
    /// rerankers (and OpenRouter's surface) accept a structured `{ text?, image? }` object — carried
    /// verbatim as [`RerankDocument::Rich`] so nothing is dropped (principle 7).
    pub documents: Vec<RerankDocument>,

    /// Return only the `top_n` highest-scoring documents. `None` ⇒ rank and return all of them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_n: Option<u32>,

    /// Whether the response should echo each ranked document back alongside its score (Jina/Voyage/
    /// OpenRouter honour this; Cohere v2 never returns documents). `None` leaves it to the upstream
    /// default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub return_documents: Option<bool>,

    /// Dialect-/provider-specific fields preserved verbatim through the core (`max_tokens_per_doc`,
    /// `truncation`, …).
    #[serde(default, skip_serializing_if = "Map::is_empty")]
    pub extra: Map<String, Value>,
}

/// One candidate document. Plain text is the overwhelming common case; a [`RerankDocument::Rich`]
/// object (`{ text?, image?, … }`) carries a multimodal or structured document through the core
/// untouched, so a provider whose upstream understands it receives exactly what the consumer sent.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum RerankDocument {
    /// A plain-text document.
    Text(String),
    /// A structured document object (e.g. `{ "text": "...", "image": "..." }`), preserved verbatim.
    Rich(Map<String, Value>),
}

impl RerankDocument {
    /// The document's text, when it *is* text (a bare string, or a `Rich` object carrying a string
    /// `text` field). `None` for a rich object with no text — used by string-only upstreams (Cohere
    /// v2) to decide how to render each document on the wire.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            RerankDocument::Text(s) => Some(s),
            RerankDocument::Rich(obj) => obj.get("text").and_then(Value::as_str),
        }
    }
}

/// One ranked result: the index of the input document and its relevance score. `document` is present
/// only when the consumer asked to echo documents back (and the provider supports it).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RerankResult {
    /// The position of this document in the request's [`RerankRequest::documents`] list.
    pub index: u32,
    /// The upstream relevance score. Higher means more relevant; the exact range is model-defined
    /// (Cohere/Jina normalise to `[0, 1]`) and relayed as reported, never rescaled (principle 5).
    pub relevance_score: f32,
    /// The ranked document echoed back (string or structured object), when requested. Carried as a raw
    /// [`Value`] so a multimodal document round-trips unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub document: Option<Value>,
}

/// A canonical rerank response. A single batch object (see module note); `usage` is what the provider
/// reported, priced at the edge by lookup like every other usage report (principle 5). Token-billed
/// upstreams (Jina/Voyage/OpenRouter) report tokens; Cohere reports *search units*, relayed into
/// [`Usage::total_tokens`] as its billed count.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RerankResponse {
    /// The model that actually served (a provider may report its own upstream id).
    pub model: String,
    /// Results in the provider's ranked order (most relevant first).
    pub results: Vec<RerankResult>,
    #[serde(default)]
    pub usage: Usage,
}
