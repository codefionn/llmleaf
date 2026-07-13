//! The extension boundary (SOUL.md principle 2: "The core knows no provider").
//!
//! The core depends only on this crate's [`Provider`] trait and [`ProviderRegistry`]. It never
//! names a concrete provider. First-party providers are compiled-in implementations of [`Provider`]
//! (zero overhead); third-party providers are WASM plugins that present the same trait across the
//! sandbox boundary. Both register into a [`ProviderRegistry`] that the binary hands to the core.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use llmleaf_model::{
    AudioStream, BatchHandle, BatchResultStream, BatchSpec, ChatRequest, EmbeddingRequest,
    EmbeddingResponse, ModelError, ModelInfo, RerankRequest, RerankResponse, ResponseStream,
    SpeechRequest, TranscriptionRequest, TranscriptionResponse, VoiceInfo,
};
use serde_json::{Map, Value};
use tokio::sync::mpsc;

/// A provider: something that can serve a canonical request for one of llmleaf's modalities. The chat
/// modality is mandatory; embeddings, speech synthesis, and transcription are opt-in — a provider
/// overrides only the methods it actually supports (SOUL "implement this in providers supporting
/// this"). The defaults return [`ModelError::Unsupported`], which routing falls past *without*
/// penalizing health (it is not a failure — the provider simply lacks the modality).
///
/// Implementations live entirely outside the core.
#[async_trait]
pub trait Provider: Send + Sync {
    /// Stable identifier used by routing and config to refer to this provider instance.
    fn name(&self) -> &str;

    /// Serve a chat request. The returned stream is the canonical internal representation; how the
    /// provider talks to its upstream (streaming or not) is its own business (principle 4 holds at
    /// the core boundary regardless).
    async fn chat(&self, req: ChatRequest, cx: &ProviderCx) -> Result<ResponseStream, ModelError>;

    /// Embed one or more inputs. Opt-in: the default declares the modality unsupported.
    async fn embed(
        &self,
        _req: EmbeddingRequest,
        _cx: &ProviderCx,
    ) -> Result<EmbeddingResponse, ModelError> {
        Err(unsupported(self.name(), "embeddings"))
    }

    /// Rank a query's candidate documents by relevance. Opt-in: the default declares the modality
    /// unsupported, which routing falls past *without* penalizing health (the provider is not
    /// degraded — it simply lacks the modality).
    async fn rerank(
        &self,
        _req: RerankRequest,
        _cx: &ProviderCx,
    ) -> Result<RerankResponse, ModelError> {
        Err(unsupported(self.name(), "rerank"))
    }

    /// Synthesize speech (text-to-speech). Opt-in: the default declares the modality unsupported.
    async fn speech(
        &self,
        _req: SpeechRequest,
        _cx: &ProviderCx,
    ) -> Result<AudioStream, ModelError> {
        Err(unsupported(self.name(), "speech synthesis"))
    }

    /// List the voices a speech `model` can synthesize with — what a consumer would put in
    /// [`SpeechRequest::voice`]. Opt-in: the default declares the modality unsupported, which routing
    /// falls past without penalizing health (same fall-through as the other optional modalities). A
    /// static-catalog provider returns a compiled-in list (e.g. OpenAI's documented voices); a provider
    /// whose upstream exposes a live voices API fetches it here. `model` is the upstream model id routing
    /// resolved, so a provider may scope the catalog per model.
    async fn voices(&self, _model: &str, _cx: &ProviderCx) -> Result<Vec<VoiceInfo>, ModelError> {
        Err(unsupported(self.name(), "voice listing"))
    }

    /// Enumerate the models this provider serves — its upstream catalog, for the model-listing surface
    /// (`GET /v1/models`). Opt-in: the default declares it unsupported, which the listing falls past
    /// *without* penalizing health (a provider that cannot enumerate its catalog is not degraded — the
    /// listing simply shows its namespace as non-enumerable). A provider whose upstream exposes a real
    /// list-models API (most do) fetches it here and returns each model with whatever metadata the API
    /// reports; fields the API does not report are left `None` (NEVER guessed — gaps are enhanced
    /// downstream from the bundled dataset). Unlike [`Self::voices`] this takes no model: it lists the
    /// whole catalog, so the listing surface can pass an upstream provider's models through by prefix.
    async fn models(&self, _cx: &ProviderCx) -> Result<Vec<ModelInfo>, ModelError> {
        Err(unsupported(self.name(), "model listing"))
    }

    /// Transcribe audio (speech-to-text). Opt-in: the default declares the modality unsupported.
    async fn transcribe(
        &self,
        _req: TranscriptionRequest,
        _cx: &ProviderCx,
    ) -> Result<TranscriptionResponse, ModelError> {
        Err(unsupported(self.name(), "transcription"))
    }

    /// Submit an asynchronous batch job to this provider's native batch API. Opt-in: the default
    /// declares the modality unsupported, which routing falls past *without* penalizing health (the
    /// same fall-through as the other optional modalities — the provider is not degraded, it simply
    /// offers no batch endpoint). The returned [`BatchHandle::id`] is the provider's *upstream* batch
    /// id; the core wraps it into an opaque, provider-encoding token before it reaches a consumer.
    async fn batch_create(
        &self,
        _req: BatchSpec,
        _cx: &ProviderCx,
    ) -> Result<BatchHandle, ModelError> {
        Err(unsupported(self.name(), "batch"))
    }

    /// Fetch the current status of a previously submitted batch. `upstream_id` is the provider's own
    /// batch id (the core has already decoded it from the opaque token). Opt-in like [`Self::batch_create`].
    async fn batch_retrieve(
        &self,
        _upstream_id: &str,
        _cx: &ProviderCx,
    ) -> Result<BatchHandle, ModelError> {
        Err(unsupported(self.name(), "batch"))
    }

    /// Stream the results of a completed batch, one [`llmleaf_model::BatchResult`] per request, parsed
    /// incrementally from the provider's result file (principle 4 — never buffered whole). `upstream_id`
    /// is the provider's own batch id. Opt-in like [`Self::batch_create`].
    async fn batch_results(
        &self,
        _upstream_id: &str,
        _cx: &ProviderCx,
    ) -> Result<BatchResultStream, ModelError> {
        Err(unsupported(self.name(), "batch"))
    }

    /// Request cancellation of an in-flight batch. `upstream_id` is the provider's own batch id.
    /// Returns the updated handle. Opt-in like [`Self::batch_create`].
    async fn batch_cancel(
        &self,
        _upstream_id: &str,
        _cx: &ProviderCx,
    ) -> Result<BatchHandle, ModelError> {
        Err(unsupported(self.name(), "batch"))
    }

    /// Whether this provider can serve a *native* realtime session ([`Self::realtime`]). The core uses
    /// this as a cheap pre-check to pick the native path over its own protocol bridge, without opening a
    /// peer first. Default `false`; a provider that overrides `realtime` overrides this too.
    fn supports_realtime(&self) -> bool {
        false
    }

    /// Proxy a *native* realtime session (e.g. OpenAI's Realtime WebSocket) to this provider's
    /// upstream. Opt-in: the default declares it unsupported, so the core falls through to its own
    /// protocol bridge for providers without a native realtime transport — the same `Unsupported`
    /// fall-through that routing already uses for the other optional modalities (it is not a failure
    /// and never penalizes health). The provider owns the upstream connection and pumps [`RealtimePeer`]
    /// frames **verbatim** in both directions (principle 7: transparent — the core never inspects a
    /// native provider's frames, it only transports them) until either side closes, then returns.
    async fn realtime(
        &self,
        _params: RealtimeParams,
        _peer: RealtimePeer,
        _cx: &ProviderCx,
    ) -> Result<(), ModelError> {
        Err(unsupported(self.name(), "realtime"))
    }

    /// A cheap liveness signal. Default: assume up. The core uses this only as one input to its own
    /// node-local health decisions (principle 9) — it is never authoritative across nodes.
    async fn health(&self) -> Health {
        Health::Up
    }
}

/// One frame on a realtime WebSocket. Kept opaque on purpose: the native realtime path proxies frames
/// straight through (principle 7), so the boundary type between the core and a native provider carries
/// no dialect — text frames hold the realtime JSON events, binary frames hold things like raw audio.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RealtimeWire {
    /// A text frame (realtime JSON events are text frames).
    Text(String),
    /// A binary frame (e.g. raw audio on the native path).
    Binary(Bytes),
    /// The peer closed the connection.
    Close,
}

/// Parameters for opening a realtime session, derived from the consumer handshake by the core. `model`
/// is the upstream model id routing already resolved from the consumer's logical model.
#[derive(Debug, Clone, Default)]
pub struct RealtimeParams {
    pub model: String,
}

/// A bidirectional, dialect-opaque frame channel between the core's consumer socket and a provider's
/// upstream. The core builds the channel pair and hands one [`RealtimePeer`] to the provider; the
/// provider reads consumer frames from `inbound` and writes upstream frames to `outbound`, pumping in
/// both directions until either side closes. Using plain `mpsc` keeps `llmleaf-provider` free of an
/// async-trait duplex and of any runtime.
pub struct RealtimePeer {
    /// Frames arriving from the consumer (consumer → provider).
    pub inbound: mpsc::Receiver<RealtimeWire>,
    /// Frames to deliver to the consumer (provider → consumer).
    pub outbound: mpsc::Sender<RealtimeWire>,
}

/// Build the canonical "this provider lacks that modality" error, phrased consistently everywhere.
fn unsupported(provider: &str, capability: &str) -> ModelError {
    ModelError::Unsupported(format!(
        "provider '{provider}' does not support {capability}"
    ))
}

/// Per-call context handed to a provider. Carries resolved credentials and instance settings drawn
/// from config — the provider never reads global config or the core's state itself.
#[derive(Debug, Clone, Default)]
pub struct ProviderCx {
    /// The request id, for correlation in logs/events.
    pub request_id: String,
    /// Resolved secret (e.g. a provider API key), already dereferenced from any env indirection.
    pub credential: Option<String>,
    /// Base endpoint / URL override for this provider instance.
    pub endpoint: Option<String>,
    /// Free-form provider-specific settings from config (org id, api version, etc.).
    pub settings: Map<String, Value>,
}

impl ProviderCx {
    pub fn setting_str(&self, key: &str) -> Option<&str> {
        self.settings.get(key).and_then(Value::as_str)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Health {
    Up,
    Degraded,
    Down,
}

/// Builds a provider instance from its `kind` string. The same `kind` → implementation mapping the
/// binary applies to the config base at startup, packaged as a handle so the pulled control plane can
/// instantiate providers at runtime (a pulled topology's providers behave exactly like file-configured
/// ones). Lives on the extension boundary like [`ProviderRegistry`]: the core only ever calls the
/// trait, it never names a kind itself (principle 2).
pub trait ProviderFactory: Send + Sync {
    /// Instantiate the implementation registered for `kind`, or `None` when the kind is unknown.
    fn build(&self, kind: &str) -> Option<Arc<dyn Provider>>;
}

/// The set of provider instances available to a node, keyed by their config name.
///
/// Built once by the binary from config + first-party/WASM factories, then shared read-only with
/// the core. Lookups on the hot path are a plain map get — no arithmetic, no I/O.
#[derive(Default, Clone)]
pub struct ProviderRegistry {
    providers: HashMap<String, Arc<dyn Provider>>,
}

impl ProviderRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a provider instance under `name` (its config name, which may differ from the
    /// implementation's own `name()` when an operator runs two instances of the same kind).
    pub fn register(&mut self, name: impl Into<String>, provider: Arc<dyn Provider>) {
        self.providers.insert(name.into(), provider);
    }

    pub fn get(&self, name: &str) -> Option<Arc<dyn Provider>> {
        self.providers.get(name).cloned()
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.providers.keys().map(String::as_str)
    }
}
