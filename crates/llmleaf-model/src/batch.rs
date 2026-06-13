//! Canonical batch model.
//!
//! A *batch* is an asynchronous job: a consumer submits a set of independent chat requests, the
//! provider processes them out-of-band (typically within ~24h, at a discount), and the consumer
//! polls for status and later retrieves the results. The OpenAI (`/v1/batches`), Anthropic
//! (`/v1/messages/batches`), and Mistral (`/v1/batch/jobs`) dialects all map *into* and *out of* the
//! types here; every batch-capable provider maps the other way.
//!
//! Two SOUL principles shape this model:
//!   - **Principle 3 (one internal model).** A batch is, by construction, just a list of canonical
//!     chat requests, each tagged with a consumer-chosen `custom_id`. The per-item body reuses
//!     [`ChatRequest`]/[`ChatResponse`] verbatim, so there is no batch-specific request/response
//!     mapping anywhere — only the *envelope* (the job object and its id) is new.
//!   - **Principle 4 (streaming is the default).** A completed batch's result set is large, so
//!     [`BatchResultStream`] yields results incrementally as the provider's result file is read —
//!     never buffered whole.
//!
//! The core holds no batch state (principle 5): the provider's upstream *is* the store. The `id` on a
//! [`BatchHandle`] is the *upstream* id at the provider boundary; the core wraps it into an opaque,
//! provider-encoding token before it reaches a consumer and unwraps it on the way back, so any node
//! can serve a poll/cancel/results call purely from the id (principle 9).

use std::pin::Pin;

use futures::Stream;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::{ChatRequest, ChatResponse, ModelError};

/// A canonical batch submission: an ordered set of independent requests, each tagged with a
/// consumer-chosen `custom_id` that correlates its result back to its input.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BatchSpec {
    pub items: Vec<BatchItem>,
}

/// One request in a batch. `request` is an ordinary [`ChatRequest`] — the same type the chat hot path
/// moves — so a provider re-encodes batch items with the exact mapper it uses for live chat.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BatchItem {
    /// Consumer-chosen correlation id, echoed back on the matching [`BatchResult`]. Opaque to llmleaf.
    pub custom_id: String,
    pub request: ChatRequest,
}

/// The lifecycle state of a batch job — the union of the provider dialects' states. Each provider
/// maps its own vocabulary into this on the way in; each output edge maps back out on the way out.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BatchStatus {
    /// Submitted, inputs being validated before processing (OpenAI `validating`).
    Validating,
    /// Actively processing (OpenAI `in_progress`, Anthropic `in_progress`).
    InProgress,
    /// Processing done, results being assembled (OpenAI `finalizing`).
    Finalizing,
    /// Finished; results are available (OpenAI `completed`, Anthropic `ended`).
    Completed,
    /// The job as a whole failed (OpenAI `failed`).
    Failed,
    /// The job expired before completing (OpenAI `expired`).
    Expired,
    /// A cancellation is in flight (OpenAI `cancelling`, Anthropic `canceling`).
    Canceling,
    /// The job was canceled (OpenAI `cancelled`).
    Canceled,
}

/// Per-request tallies for a batch. A superset of the dialects: OpenAI reports `{total, completed,
/// failed}` (mapped to `total`/`succeeded`/`errored`); Anthropic reports `{processing, succeeded,
/// errored, canceled, expired}` directly. An output edge reads back only the fields its dialect names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct BatchCounts {
    /// Total requests in the batch (0 if the provider does not report it).
    #[serde(default)]
    pub total: u64,
    #[serde(default)]
    pub processing: u64,
    #[serde(default)]
    pub succeeded: u64,
    #[serde(default)]
    pub errored: u64,
    #[serde(default)]
    pub canceled: u64,
    #[serde(default)]
    pub expired: u64,
}

/// A batch job object. The `id` is the *upstream* id at the provider boundary and the opaque,
/// provider-encoding token once the core has wrapped it (see module note). Instants are unix seconds;
/// `None` where the provider does not report a given one. Anything dialect-specific we do not model
/// rides verbatim in `extra` (principle 7).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BatchHandle {
    pub id: String,
    pub status: BatchStatus,
    #[serde(default)]
    pub counts: BatchCounts,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created_at: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ended_at: Option<u64>,
    /// The upstream endpoint the batch targets, when reported (e.g. OpenAI `/v1/chat/completions`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub endpoint: Option<String>,
    /// Dialect-/provider-specific fields preserved verbatim for the output edge.
    #[serde(default, skip_serializing_if = "Map::is_empty")]
    pub extra: Map<String, Value>,
}

/// One element of a batch's result set: the outcome of the request that carried `custom_id`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BatchResult {
    pub custom_id: String,
    pub outcome: BatchOutcome,
}

/// What became of a single batched request. A success carries a fully collected [`ChatResponse`]
/// (principle 4: a collected stream); the other arms mirror the terminal states a provider reports
/// per request.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum BatchOutcome {
    Succeeded(ChatResponse),
    /// The request errored upstream. `status` is the upstream HTTP status when known (0 otherwise).
    Errored {
        status: u16,
        message: String,
    },
    Canceled,
    Expired,
}

/// The canonical streaming result set of a completed batch: a pinned, boxed, `Send` stream of
/// [`BatchResult`]s, parsed line-by-line from the provider's result file (principle 4). A trait object
/// for the same reason as [`crate::ResponseStream`]: providers and edge mappings produce/consume it
/// without the core knowing their concrete stream types.
pub type BatchResultStream = Pin<Box<dyn Stream<Item = Result<BatchResult, ModelError>> + Send>>;
