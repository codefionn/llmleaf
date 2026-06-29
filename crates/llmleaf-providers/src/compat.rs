//! One config-driven provider for the entire OpenAI-compatible family.
//!
//! Every brand here speaks the same wire format ([`crate::openai_wire`]); they differ only in
//! endpoint, auth header, URL shape, and a couple of optional headers. So instead of N near-identical
//! structs we have one [`OpenAiCompatProvider`] parameterized by a small [`Brand`] table. Adding a new
//! compatible vendor is a single row — no new mapping code. This is the SOUL "push work to the edges"
//! decision filter in practice: quirks live in the provider layer, the core stays oblivious.

use async_trait::async_trait;
use futures::{stream, StreamExt};
use llmleaf_model::{
    collect_chunks, AudioChunk, AudioStream, BatchCounts, BatchHandle, BatchItem, BatchOutcome,
    BatchResult, BatchResultStream, BatchSpec, BatchStatus, ChatRequest, EmbeddingRequest,
    EmbeddingResponse, ModelError, ModelInfo, ResponseStream, SpeechRequest, TranscriptionRequest,
    TranscriptionResponse, VoiceInfo,
};
use llmleaf_provider::{Provider, ProviderCx, RealtimeParams, RealtimePeer};
use serde_json::{json, Map, Value};

use std::sync::Arc;

use crate::batch::{build_jsonl, jsonl_result_stream};
use crate::http::{post_json, send_checked};
use crate::openai_wire::{
    audio_content_type, decode_speech_envelope, embedding_request_to_openai,
    mistral_voices_to_canonical, openai_speech_model, openai_sse_to_stream, openai_to_chunks,
    openai_to_embeddings, openai_to_transcription, openai_voices, openai_wire_models_to_canonical,
    openrouter_to_transcription, request_to_openai, speech_request_to_openai,
    transcription_request_to_openrouter,
};
use crate::transport::{HttpRequest, HttpTransport, MultipartForm, RealtimeTransport, Transports};

/// How a brand presents its API key.
#[derive(Clone, Copy, Debug)]
pub enum AuthStyle {
    /// `Authorization: Bearer <key>` — the OpenAI default.
    Bearer,
    /// A bespoke header carrying the raw key (e.g. Azure's `api-key`).
    ApiKeyHeader(&'static str),
}

/// Which batch dialect a brand's upstream speaks. The OpenAI-wire family is *not* uniform here: most
/// brands that speak OpenAI chat have no batch API at all, a few mirror OpenAI's `/v1/batches` +
/// `/v1/files` exactly, and Mistral has its own `/v1/batch/jobs` shape. Batch is therefore opt-in per
/// brand (default [`BatchFlavor::Unsupported`]), enabled only where the wire is confirmed to match.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BatchFlavor {
    /// No batch API (the default): [`Provider::batch_create`] et al. return `Unsupported`.
    Unsupported,
    /// OpenAI's reference shape: upload JSONL to `/files` (purpose=batch) → `POST /batches` →
    /// `GET /batches/{id}` → results via the output file. Covers OpenAI, Groq, Moonshot.
    OpenAi,
    /// Mistral's "jobs" shape: upload JSONL to `/files` (purpose=batch) → `POST /batch/jobs` →
    /// `GET /batch/jobs/{id}` → results via the output file.
    MistralJobs,
    /// Together's shape: OpenAI-wire, but the file upload uses `purpose=batch-api`, the create body
    /// omits `completion_window`, the create *response* wraps the batch under a `job` key, and there is
    /// no `request_counts` object. Per-request bodies and result lines are OpenAI-shaped.
    Together,
    /// Azure OpenAI's shape: OpenAI-wire result/line bodies, but batch is resource-scoped at
    /// `/openai/batches` (never deployment-scoped) with an `?api-version=` query, files at
    /// `/openai/files`, and the create `endpoint` is `/chat/completions` (no `/v1`). Per-line
    /// `body.model` is the deployment name. Auth is the `api-key` header (already set by [`AuthStyle`]).
    AzureOpenAi,
}

/// How a brand builds its request URL.
#[derive(Clone, Copy, Debug)]
pub enum UrlStyle {
    /// `<endpoint>/chat/completions` — the OpenAI default.
    Standard,
    /// `<endpoint>/openai/deployments/<model>/chat/completions?api-version=<v>` — Azure OpenAI.
    Azure,
}

/// Cerebras's *public*, unauthenticated model catalog (`GET /public/v1/models`). Far richer than the
/// authed `/v1/models` (which lists ids only): it reports per-token `pricing`, nested `limits`
/// (`max_context_length`, `max_completion_tokens`, rate limits), `capabilities`, `architecture`, and
/// `quantization`. This is the "specific" Cerebras list endpoint — selected via
/// [`Brand::models_url_override`] and read by the shared [`crate::openai_wire::openai_wire_models_to_canonical`].
/// It is on a different host path than the chat base, so it is a fixed URL rather than `<endpoint>/models`.
const CEREBRAS_PUBLIC_MODELS_URL: &str = "https://api.cerebras.ai/public/v1/models";

/// The per-brand quirk table. Endpoints are *defaults*: an operator may override `endpoint` in config.
#[derive(Clone, Copy, Debug)]
pub struct Brand {
    pub name: &'static str,
    pub default_endpoint: &'static str,
    pub auth: AuthStyle,
    pub url_style: UrlStyle,
    /// Azure-only default `api-version` (empty for non-Azure brands).
    pub default_api_version: &'static str,
    /// Output-cap field name. `max_completion_tokens` for OpenAI/Azure/xAI (their o-series/reasoning
    /// models reject the deprecated `max_tokens`); `max_tokens` for the rest.
    pub max_tokens_field: &'static str,
    /// Whether this brand's upstream serves real SSE streaming (`stream:true` +
    /// `stream_options.include_usage`). Default `true`; a couple of local servers that may choke on
    /// `stream_options` keep the collect-then-rechunk path (`false`).
    pub supports_stream: bool,
    /// Whether this brand exposes a native OpenAI-Realtime WebSocket upstream (only OpenAI today). When
    /// `false`, [`Provider::realtime`] returns `Unsupported` and the core bridges over chat streaming.
    pub realtime_native: bool,
    /// Whether `/audio/transcriptions` takes OpenRouter's JSON+base64 body instead of the OpenAI
    /// multipart upload. `true` only for OpenRouter, whose endpoint rejects `multipart/form-data`.
    pub transcription_json_base64: bool,
    /// Whether this brand's upstream exposes a real `GET /audio/voices` listing (Mistral does). When
    /// `true`, [`Provider::voices`] fetches the live catalog; when `false`, voices come from a documented
    /// static catalog (OpenAI) or an operator-declared one, never a guess.
    pub voices_api: bool,
    /// Whether this brand's upstream exposes a usable `GET /models` listing. When `true`,
    /// [`Provider::models`] fetches the live catalog and parses it tolerantly (OpenAI id-only,
    /// OpenRouter context+pricing, Together's bare array, Cerebras's rich public catalog, …); when
    /// `false` it stays `Unsupported` and the listing surface shows the namespace as non-enumerable
    /// rather than guessing a catalog.
    pub models_api: bool,
    /// A fixed, brand-specific list-models URL that overrides the default `<endpoint>/models`. Used for
    /// Cerebras, whose richest catalog is its *unauthenticated* public endpoint
    /// ([`CEREBRAS_PUBLIC_MODELS_URL`]) — pricing, context/output limits, capability flags — on a
    /// different path than its chat base (the authed `/v1/models` reports ids only). `None` (the
    /// default) builds the standard per-endpoint URL ([`OpenAiCompatProvider::models_url`]). Only
    /// consulted when `models_api` is `true`.
    pub models_url_override: Option<&'static str>,
    /// Extra query string appended to the default `<endpoint>/models` URL (no leading `?`/`&` — the
    /// builder picks the right separator). Needed for brands whose `GET /models` hides part of the
    /// catalog behind a filter that defaults to a subset: OpenRouter defaults `output_modalities=text`,
    /// so TTS models (audio output) are omitted unless `output_modalities=all` is sent — without this
    /// the catalog can never enumerate a text-to-speech model. Empty (the default) appends nothing.
    /// Ignored when [`Brand::models_url_override`] is set (that URL is used verbatim).
    pub models_query: &'static str,
    /// Whether this brand's `/models` listing flags decommissioned models with `active: false` and they
    /// should be omitted from the catalog. Groq is the one brand that does this — its listing keeps a
    /// model for a grace period after it is retired, marked `active: false`; advertising it as a live
    /// model would let a consumer route to one that no longer serves. Default `false`: every other
    /// brand's listing carries no such flag, so nothing is dropped and the shared parser's "never drop an
    /// id-bearing model" contract is unaffected. The drop happens in the provider layer (the `active`
    /// flag rides verbatim in `extra` via [`crate::openai_wire::openai_wire_models_to_canonical`]), never
    /// in that shared parser — the quirk stays at the edge. Only consulted when `models_api` is `true`.
    pub filter_inactive_models: bool,
    /// Which batch dialect this brand speaks (default [`BatchFlavor::Unsupported`]).
    pub batch_flavor: BatchFlavor,
}

impl Brand {
    /// Resolve a config `kind` to its quirk table. Returns `None` for kinds this build doesn't know.
    pub fn for_kind(kind: &str) -> Option<Brand> {
        // `b` = legacy `max_tokens`; `bc` = current `max_completion_tokens`.
        let b = |name, default_endpoint, auth| Brand {
            name,
            default_endpoint,
            auth,
            url_style: UrlStyle::Standard,
            default_api_version: "",
            max_tokens_field: "max_tokens",
            supports_stream: true,
            realtime_native: false,
            transcription_json_base64: false,
            voices_api: false,
            models_api: false,
            models_url_override: None,
            models_query: "",
            filter_inactive_models: false,
            batch_flavor: BatchFlavor::Unsupported,
        };
        let bc = |name, default_endpoint, auth| Brand {
            name,
            default_endpoint,
            auth,
            url_style: UrlStyle::Standard,
            default_api_version: "",
            max_tokens_field: "max_completion_tokens",
            supports_stream: true,
            realtime_native: false,
            transcription_json_base64: false,
            voices_api: false,
            models_api: false,
            models_url_override: None,
            models_query: "",
            filter_inactive_models: false,
            batch_flavor: BatchFlavor::Unsupported,
        };
        Some(match kind {
            // OpenAI is the one brand with a native Realtime WebSocket upstream.
            "openai" => Brand {
                realtime_native: true,
                models_api: true,
                batch_flavor: BatchFlavor::OpenAi,
                ..bc("openai", "https://api.openai.com/v1", AuthStyle::Bearer)
            },
            // OpenRouter's `/audio/transcriptions` takes a JSON body with base64 audio, not the OpenAI
            // multipart upload every other brand uses — flag it so `transcribe` sends the right shape.
            "openrouter" => Brand {
                transcription_json_base64: true,
                models_api: true,
                // OpenRouter's `/models` defaults to `output_modalities=text`; without `=all` the
                // catalog omits TTS models (audio output), so a speech model could never be listed.
                models_query: "output_modalities=all",
                ..b(
                    "openrouter",
                    "https://openrouter.ai/api/v1",
                    AuthStyle::Bearer,
                )
            },
            // Requesty is a multi-provider gateway like OpenRouter: OpenAI wire, `provider/model`
            // ids (e.g. `openai/gpt-4o`), and the same HTTP-Referer/X-Title attribution headers
            // (carried by the passthrough settings). Its router normalizes `max_tokens`.
            "requesty" => b(
                "requesty",
                "https://router.requesty.ai/v1",
                AuthStyle::Bearer,
            ),
            // Groq lists models at `<base>/models` (its base already carries `/openai/v1`, so the
            // standard `<endpoint>/models` shape resolves to `https://api.groq.com/openai/v1/models`) and
            // mirrors OpenAI's batch shape. Its listing is the one that flags retired-but-still-listed
            // models with `active: false` — drop those so the catalog only advertises live models.
            "groq" => Brand {
                models_api: true,
                filter_inactive_models: true,
                batch_flavor: BatchFlavor::OpenAi,
                ..b("groq", "https://api.groq.com/openai/v1", AuthStyle::Bearer)
            },
            // DeepSeek's base has no `/v1` segment (verified against official docs).
            "deepseek" => b("deepseek", "https://api.deepseek.com", AuthStyle::Bearer),
            // xAI deprecates max_tokens in favor of max_completion_tokens.
            "xai" | "grok" => bc("xai", "https://api.x.ai/v1", AuthStyle::Bearer),
            // Mistral exposes a real `GET /v1/audio/voices` listing — fetch it live. Its batch API is
            // its own "jobs" dialect (`/v1/batch/jobs`), not OpenAI's `/v1/batches`.
            "mistral" => Brand {
                voices_api: true,
                models_api: true,
                batch_flavor: BatchFlavor::MistralJobs,
                ..b("mistral", "https://api.mistral.ai/v1", AuthStyle::Bearer)
            },
            // api.together.ai is canonical; api.together.xyz is the SDK-default alias (same API).
            "together" => Brand {
                models_api: true,
                batch_flavor: BatchFlavor::Together,
                ..b("together", "https://api.together.ai/v1", AuthStyle::Bearer)
            },
            "fireworks" => b(
                "fireworks",
                "https://api.fireworks.ai/inference/v1",
                AuthStyle::Bearer,
            ),
            "perplexity" => b("perplexity", "https://api.perplexity.ai", AuthStyle::Bearer),
            // NOTE: Ollama and LM Studio are NOT here — they are first-class native providers (their
            // own `/api/*` and `/api/v0/*` dialects), built directly in `lib.rs::build`, not OpenAI-wire
            // shims over this table.
            // Cerebras documents max_completion_tokens (its reasoning models count thinking tokens). Its
            // richest model catalog is the *public*, unauthenticated `/public/v1/models` — pricing,
            // context/output limits, and capability flags — far more than the id-only authed
            // `/v1/models`; point list-models at that specific endpoint (read by the shared wire parser).
            "cerebras" => Brand {
                models_api: true,
                models_url_override: Some(CEREBRAS_PUBLIC_MODELS_URL),
                ..bc("cerebras", "https://api.cerebras.ai/v1", AuthStyle::Bearer)
            },
            // Z.AI (Zhipu GLM), international host. The /api/paas/v4 base already carries the version
            // segment — unlike the OpenAI default there is no trailing /v1 to append.
            "zai" | "z.ai" | "glm" => b("zai", "https://api.z.ai/api/paas/v4", AuthStyle::Bearer),
            // Moonshot (Kimi, incl. Kimi K2), international host. Its API deprecates max_tokens in
            // favor of max_completion_tokens.
            "moonshot" | "kimi" | "kimi-k2" => Brand {
                models_api: true,
                batch_flavor: BatchFlavor::OpenAi,
                ..bc("moonshot", "https://api.moonshot.ai/v1", AuthStyle::Bearer)
            },
            "azure-openai" | "azure" => Brand {
                name: "azure-openai",
                default_endpoint: "",
                auth: AuthStyle::ApiKeyHeader("api-key"),
                url_style: UrlStyle::Azure,
                default_api_version: "2024-10-21",
                max_tokens_field: "max_completion_tokens",
                supports_stream: true,
                // Azure's realtime surface uses a different URL/auth shape — bridge it for now.
                realtime_native: false,
                transcription_json_base64: false,
                voices_api: false,
                // Azure lists models at the resource root (`/openai/models?api-version=`), not under a
                // deployment — `models_url` builds that, NOT `url_for`.
                models_api: true,
                models_url_override: None,
                models_query: "",
                // Azure's listing carries no `active` flag — nothing to filter.
                filter_inactive_models: false,
                // Azure batch is resource-scoped (`/openai/batches?api-version=`), not under the
                // deployment URL its chat uses — handled by the AzureOpenAi batch flavor.
                batch_flavor: BatchFlavor::AzureOpenAi,
            },
            _ => return None,
        })
    }

    /// The kinds this generic provider can serve.
    pub fn kinds() -> &'static [&'static str] {
        &[
            "openai",
            "openrouter",
            "requesty",
            "groq",
            "deepseek",
            "xai",
            "grok",
            "mistral",
            "together",
            "fireworks",
            "perplexity",
            "cerebras",
            "zai",
            "z.ai",
            "glm",
            "moonshot",
            "kimi",
            "kimi-k2",
            "azure-openai",
            "azure",
        ]
    }
}

/// A ready-to-use OpenAI-compatible provider for one [`Brand`].
#[derive(Clone)]
pub struct OpenAiCompatProvider {
    http: Arc<dyn HttpTransport>,
    realtime: Arc<dyn RealtimeTransport>,
    brand: Brand,
}

impl OpenAiCompatProvider {
    pub fn new(brand: Brand, transports: &Transports) -> Self {
        OpenAiCompatProvider {
            http: transports.http.clone(),
            realtime: transports.realtime.clone(),
            brand,
        }
    }

    /// Construct directly from a config `kind`, or `None` if unknown.
    pub fn for_kind(kind: &str, transports: &Transports) -> Option<Self> {
        Brand::for_kind(kind).map(|brand| Self::new(brand, transports))
    }

    /// Build the URL for an OpenAI-wire operation (`chat/completions`, `embeddings`, `audio/speech`,
    /// `audio/transcriptions`). Standard brands append the path; Azure addresses it under a per-model
    /// deployment. One shape serves every modality (decision filter: quirks in one place).
    fn url_for(&self, cx: &ProviderCx, model: &str, op: &str) -> String {
        let endpoint = cx
            .endpoint
            .as_deref()
            .unwrap_or(self.brand.default_endpoint)
            .trim_end_matches('/');
        match self.brand.url_style {
            UrlStyle::Standard => format!("{endpoint}/{op}"),
            UrlStyle::Azure => {
                let api_version = cx
                    .setting_str("api_version")
                    .unwrap_or(self.brand.default_api_version);
                format!("{endpoint}/openai/deployments/{model}/{op}?api-version={api_version}")
            }
        }
    }

    fn build_url(&self, cx: &ProviderCx, model: &str) -> String {
        self.url_for(cx, model, "chat/completions")
    }

    /// Apply the brand's auth header and the brand-agnostic passthrough headers (org id, OpenRouter
    /// attribution). Shared by every modality so they authenticate identically.
    fn apply_auth(&self, mut req: HttpRequest, cx: &ProviderCx) -> HttpRequest {
        req = match self.brand.auth {
            AuthStyle::Bearer => match &cx.credential {
                Some(c) => req.bearer(c),
                None => req,
            },
            AuthStyle::ApiKeyHeader(h) => match &cx.credential {
                Some(c) => req.header(h, c),
                None => req,
            },
        };
        if let Some(org) = cx.setting_str("organization") {
            req = req.header("OpenAI-Organization", org);
        }
        if let Some(referer) = cx.setting_str("http_referer") {
            req = req.header("HTTP-Referer", referer); // OpenRouter attribution
        }
        if let Some(title) = cx.setting_str("x_title") {
            req = req.header("X-Title", title); // OpenRouter attribution
        }
        req
    }

    /// Fetch a brand's live voice catalog from its `GET /audio/voices` endpoint (Mistral). The listing
    /// is account-wide (no model parameter) and paginated; we ask for a large page so a single request
    /// returns the whole catalog. Reuses the shared auth + failure taxonomy, so a transport error is
    /// fallback-eligible exactly like any other call.
    async fn fetch_voices(&self, cx: &ProviderCx) -> Result<Vec<VoiceInfo>, ModelError> {
        let url = format!("{}?limit=1000", self.url_for(cx, "", "audio/voices"));
        let req = self.apply_auth(HttpRequest::get(&url), cx);
        let value = post_json(&*self.http, req).await?;
        Ok(mistral_voices_to_canonical(value))
    }

    /// The list-models URL: a fixed brand-specific override when set (Cerebras's public catalog,
    /// [`CEREBRAS_PUBLIC_MODELS_URL`], which lives on a different host path than the chat base and so
    /// ignores any `endpoint` override); otherwise `<endpoint>/models` for standard brands, or — for
    /// Azure — the resource root (`<endpoint>/openai/models?api-version=`), NOT under a deployment like
    /// `url_for` would (the same resource-scoped shape `batch_collection` uses).
    fn models_url(&self, cx: &ProviderCx) -> String {
        if let Some(url) = self.brand.models_url_override {
            return url.to_string();
        }
        let e = self.batch_endpoint(cx);
        let mut url = match self.brand.url_style {
            UrlStyle::Standard => format!("{e}/models"),
            UrlStyle::Azure => {
                let api_version = cx
                    .setting_str("api_version")
                    .unwrap_or(self.brand.default_api_version);
                format!("{e}/openai/models?api-version={api_version}")
            }
        };
        // A brand-specific catalog filter (e.g. OpenRouter's `output_modalities=all`); pick `&` when the
        // base already carries a query (Azure's `api-version`), `?` otherwise.
        if !self.brand.models_query.is_empty() {
            url.push(if url.contains('?') { '&' } else { '?' });
            url.push_str(self.brand.models_query);
        }
        url
    }

    /// Fetch and parse the brand's upstream `GET /models` catalog. Reuses the shared auth + failure
    /// taxonomy (a transport error stays fallback-eligible); the tolerant parser reads whatever fields
    /// the brand reports and leaves the rest `None`.
    async fn fetch_models(&self, cx: &ProviderCx) -> Result<Vec<ModelInfo>, ModelError> {
        let url = self.models_url(cx);
        // A `models_url_override` is, by its documented contract, the brand's *unauthenticated* public
        // catalog (Cerebras's `/public/v1/models`) — fetch it anonymously; sending a stray bearer to a
        // documented no-auth endpoint is pointless at best. The standard `<endpoint>/models` still
        // authenticates like every other call.
        let request = HttpRequest::get(&url);
        let request = if self.brand.models_url_override.is_some() {
            request
        } else {
            self.apply_auth(request, cx)
        };
        let value = post_json(&*self.http, request).await?;
        Ok(drop_inactive(
            &self.brand,
            openai_wire_models_to_canonical(value),
        ))
    }

    /// `Unsupported` for a brand with no batch API — the guard every batch method runs first so an
    /// override of the trait default still falls through to the next target without a health penalty.
    fn ensure_batch(&self) -> Result<(), ModelError> {
        if self.brand.batch_flavor == BatchFlavor::Unsupported {
            return Err(ModelError::Unsupported(format!(
                "provider '{}' does not support batch",
                self.brand.name
            )));
        }
        Ok(())
    }

    /// This instance's base endpoint (config override or brand default), trailing slash trimmed. Batch
    /// builds its own URLs from this rather than via [`Self::url_for`], because Azure batch is
    /// resource-scoped (`/openai/batches`), not under the deployment URL its chat uses.
    fn batch_endpoint(&self, cx: &ProviderCx) -> String {
        cx.endpoint
            .as_deref()
            .unwrap_or(self.brand.default_endpoint)
            .trim_end_matches('/')
            .to_string()
    }

    /// The `?api-version=…` query Azure batch requires on every call (empty for every other flavor).
    fn batch_query(&self, cx: &ProviderCx) -> String {
        match self.brand.batch_flavor {
            BatchFlavor::AzureOpenAi => {
                let v = cx
                    .setting_str("api_version")
                    .unwrap_or(self.brand.default_api_version);
                format!("?api-version={v}")
            }
            _ => String::new(),
        }
    }

    /// The batch-jobs collection base (no id, no query): `/batch/jobs` for Mistral, `/openai/batches`
    /// for Azure, `/batches` for the OpenAI-wire flavors.
    fn batch_collection(&self, cx: &ProviderCx) -> String {
        let e = self.batch_endpoint(cx);
        match self.brand.batch_flavor {
            BatchFlavor::MistralJobs => format!("{e}/batch/jobs"),
            BatchFlavor::AzureOpenAi => format!("{e}/openai/batches"),
            _ => format!("{e}/batches"),
        }
    }

    /// The files base (no id, no query): `/openai/files` for Azure, `/files` otherwise.
    fn batch_files(&self, cx: &ProviderCx) -> String {
        let e = self.batch_endpoint(cx);
        match self.brand.batch_flavor {
            BatchFlavor::AzureOpenAi => format!("{e}/openai/files"),
            _ => format!("{e}/files"),
        }
    }

    /// The multipart `purpose` for the input file upload (`batch-api` for Together, `batch` otherwise).
    fn batch_purpose(&self) -> &'static str {
        match self.brand.batch_flavor {
            BatchFlavor::Together => "batch-api",
            _ => "batch",
        }
    }

    /// One canonical item → this flavor's JSONL input line. The request body is built by the very same
    /// [`request_to_openai`] the live chat path uses; only the line envelope differs (OpenAI/Azure wrap
    /// it with `method`/`url`; Mistral/Together carry just `custom_id` + `body`).
    fn batch_line(&self, item: &BatchItem) -> Value {
        let body = request_to_openai(&item.request, self.brand.max_tokens_field, false);
        match self.brand.batch_flavor {
            BatchFlavor::OpenAi | BatchFlavor::AzureOpenAi => json!({
                "custom_id": item.custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }),
            _ => json!({ "custom_id": item.custom_id, "body": body }),
        }
    }

    /// Upload a JSONL batch input file and return the upstream file id.
    async fn upload_batch_file(
        &self,
        cx: &ProviderCx,
        jsonl: Vec<u8>,
    ) -> Result<String, ModelError> {
        let url = format!("{}{}", self.batch_files(cx), self.batch_query(cx));
        let form = MultipartForm::new()
            .text("purpose", self.batch_purpose())
            .bytes(
                "file",
                Some("batch.jsonl".to_string()),
                Some("application/jsonl".to_string()),
                jsonl.into(),
            );
        let req = self.apply_auth(HttpRequest::post(&url).multipart(form), cx);
        let value = post_json(&*self.http, req).await?;
        value
            .get("id")
            .and_then(Value::as_str)
            .map(str::to_string)
            .ok_or_else(|| ModelError::Mapping("file upload response had no id".into()))
    }

    /// Fetch the raw upstream batch object (shared by retrieve and by results, which reads its output
    /// file id).
    async fn fetch_batch_value(
        &self,
        cx: &ProviderCx,
        upstream_id: &str,
    ) -> Result<Value, ModelError> {
        let url = format!(
            "{}/{upstream_id}{}",
            self.batch_collection(cx),
            self.batch_query(cx)
        );
        let req = self.apply_auth(HttpRequest::get(&url), cx);
        post_json(&*self.http, req).await
    }
}

/// Operator-declared voices for an upstream `model`, read from the provider's `settings.voices` table.
/// The config fallback for an upstream with no documented catalog and no live voices API — e.g. a TTS
/// model reached through OpenRouter. The table maps an upstream model id to either a list of voice-id
/// strings or a list of `{ id, name?, languages? }` tables:
///
/// ```toml
/// [providers.settings.voices]
/// "mistralai/voxtral-mini-tts-2603" = ["aurora", "basalt"]
/// "some/other-tts" = [{ id = "nia", name = "Nia", languages = ["en", "sw"] }]
/// ```
///
/// Returns `None` when nothing usable is declared for `model`, so the caller reports the modality
/// unsupported rather than an empty catalog.
fn declared_voices(cx: &ProviderCx, model: &str) -> Option<Vec<VoiceInfo>> {
    let entry = cx
        .settings
        .get("voices")?
        .as_object()?
        .get(model)?
        .as_array()?;
    let voices: Vec<VoiceInfo> = entry.iter().filter_map(voice_from_value).collect();
    (!voices.is_empty()).then_some(voices)
}

/// One declared voice → [`VoiceInfo`]. Accepts a bare id string or a `{ id, name?, languages? }` table;
/// anything else (or a table missing a string `id`) is skipped.
fn voice_from_value(v: &Value) -> Option<VoiceInfo> {
    match v {
        Value::String(id) => Some(VoiceInfo {
            id: id.clone(),
            name: None,
            languages: Vec::new(),
            extra: Map::new(),
        }),
        Value::Object(obj) => {
            let id = obj.get("id")?.as_str()?.to_string();
            let name = obj.get("name").and_then(Value::as_str).map(str::to_string);
            let languages = obj
                .get("languages")
                .and_then(Value::as_array)
                .map(|a| {
                    a.iter()
                        .filter_map(Value::as_str)
                        .map(str::to_string)
                        .collect()
                })
                .unwrap_or_default();
            Some(VoiceInfo {
                id,
                name,
                languages,
                extra: Map::new(),
            })
        }
        _ => None,
    }
}

/// An OpenAI/Mistral batch object → canonical [`BatchHandle`]. Both flavors report unix-second instants
/// and an `id`/`status`; counts and the output-file key differ and are normalized below. Upstream file
/// ids are deliberately *not* carried in `extra` — the consumer fetches results via llmleaf's own
/// `/results` route, so an opaque-free upstream id must never leak.
fn batch_value_to_handle(value: &Value) -> BatchHandle {
    let id = value
        .get("id")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let status = value
        .get("status")
        .or_else(|| value.get("processing_status"))
        .and_then(Value::as_str)
        .map(map_batch_status)
        .unwrap_or(BatchStatus::InProgress);
    let as_unix = |k: &str| value.get(k).and_then(Value::as_u64);
    BatchHandle {
        id,
        status,
        counts: batch_counts(value),
        created_at: as_unix("created_at"),
        expires_at: as_unix("expires_at"),
        ended_at: as_unix("completed_at").or_else(|| as_unix("ended_at")),
        endpoint: value
            .get("endpoint")
            .and_then(Value::as_str)
            .map(str::to_string),
        extra: Map::new(),
    }
}

/// Normalize a batch status string from either dialect onto the canonical ladder. The two vocabularies
/// (OpenAI's lowercase set, Mistral's uppercase `RUNNING`/`SUCCESS`/…) do not collide.
fn map_batch_status(s: &str) -> BatchStatus {
    match s.to_ascii_lowercase().as_str() {
        "validating" => BatchStatus::Validating,
        "in_progress" | "running" | "queued" => BatchStatus::InProgress,
        "finalizing" => BatchStatus::Finalizing,
        "completed" | "success" => BatchStatus::Completed,
        "failed" => BatchStatus::Failed,
        "expired" | "timeout_exceeded" => BatchStatus::Expired,
        "cancelling" | "cancellation_requested" => BatchStatus::Canceling,
        "cancelled" | "canceled" => BatchStatus::Canceled,
        _ => BatchStatus::InProgress,
    }
}

/// Per-request counts from either dialect. OpenAI nests `request_counts { total, completed, failed }`;
/// Mistral reports flat `*_requests` fields. `processing` is inferred as the remainder.
fn batch_counts(value: &Value) -> BatchCounts {
    let (total, succeeded, errored) = if let Some(rc) = value.get("request_counts") {
        let g = |k: &str| rc.get(k).and_then(Value::as_u64).unwrap_or(0);
        (g("total"), g("completed"), g("failed"))
    } else {
        let g = |k: &str| value.get(k).and_then(Value::as_u64).unwrap_or(0);
        (
            g("total_requests"),
            g("succeeded_requests"),
            g("failed_requests"),
        )
    };
    BatchCounts {
        total,
        succeeded,
        errored,
        processing: total.saturating_sub(succeeded + errored),
        canceled: 0,
        expired: 0,
    }
}

/// The result files to stream for a completed batch, in emit order: the *output* file (succeeded
/// requests) then the *error* file (per-request failures). Each file-based flavor names them with one of
/// these key pairs (OpenAI/Azure/Together use `*_file_id`, Mistral uses `*_file`); an absent file is
/// skipped. Reading *both* is what keeps a failed request from disappearing — its line lives only in the
/// error file, never in the output file.
fn batch_result_file_ids(batch: &Value) -> Vec<String> {
    [
        ["output_file_id", "output_file"],
        ["error_file_id", "error_file"],
    ]
    .iter()
    .filter_map(|keys| {
        keys.iter()
            .find_map(|k| batch.get(*k).and_then(Value::as_str))
            .map(str::to_string)
    })
    .collect()
}

/// One line of an OpenAI/Mistral batch output file → canonical [`BatchResult`]. A success carries the
/// response body (an OpenAI-wire chat completion) mapped through the same chunk path the chat surface
/// uses, then folded; a top-level `error` or a ≥400 status code becomes an `Errored` outcome.
fn openai_batch_result_line(value: Value) -> Option<BatchResult> {
    let custom_id = value.get("custom_id")?.as_str()?.to_string();
    if let Some(err) = value.get("error") {
        if !err.is_null() {
            let message = err
                .get("message")
                .and_then(Value::as_str)
                .map(str::to_string)
                .unwrap_or_else(|| err.to_string());
            return Some(BatchResult {
                custom_id,
                outcome: BatchOutcome::Errored { status: 0, message },
            });
        }
    }
    let response = value.get("response")?;
    let status = response
        .get("status_code")
        .and_then(Value::as_u64)
        .unwrap_or(200) as u16;
    let body = response.get("body")?;
    if status >= 400 {
        let message = body
            .get("error")
            .and_then(|e| e.get("message"))
            .and_then(Value::as_str)
            .map(str::to_string)
            .unwrap_or_else(|| body.to_string());
        return Some(BatchResult {
            custom_id,
            outcome: BatchOutcome::Errored { status, message },
        });
    }
    Some(BatchResult {
        custom_id,
        outcome: BatchOutcome::Succeeded(collect_chunks(openai_to_chunks(body.clone(), ""))),
    })
}

/// Drop the models a brand's listing marks inactive ([`Brand::filter_inactive_models`]). Groq keeps a
/// decommissioned model in its `/models` response for a grace period, flagged `active: false`; this omits
/// exactly those so the catalog never advertises a model that no longer serves. The flag rides verbatim in
/// `extra` (the shared parser preserves it; the drop is this provider-layer quirk, not the parser's). Only
/// an explicit `active: false` drops a model — an absent or `true` flag keeps it, so this is a no-op for
/// every brand whose listing has no such field. Pure (no I/O) so the policy is unit-testable.
fn drop_inactive(brand: &Brand, mut models: Vec<ModelInfo>) -> Vec<ModelInfo> {
    if brand.filter_inactive_models {
        models.retain(|m| m.extra.get("active").and_then(Value::as_bool) != Some(false));
    }
    models
}

#[async_trait]
impl Provider for OpenAiCompatProvider {
    fn name(&self) -> &str {
        self.brand.name
    }

    async fn chat(&self, req: ChatRequest, cx: &ProviderCx) -> Result<ResponseStream, ModelError> {
        let url = self.build_url(cx, &req.model);

        // Brands that don't stream cleanly: collect the whole response and re-chunk it (the original
        // path). The canonical boundary is a stream either way (principle 4).
        if !self.brand.supports_stream {
            let body = request_to_openai(&req, self.brand.max_tokens_field, false);
            let http_req = self.apply_auth(HttpRequest::post(&url).json(body), cx);
            let value = post_json(&*self.http, http_req).await?;
            let chunks = openai_to_chunks(value, &req.model);
            return Ok(Box::pin(stream::iter(chunks.into_iter().map(Ok))));
        }

        // Streaming brands: ask upstream for SSE and parse it incrementally into canonical chunks, so
        // tokens flow as they arrive (principle 4). Both the realtime bridge and the SSE consumer
        // surface get genuinely live output instead of a collected-then-replayed response.
        let body = request_to_openai(&req, self.brand.max_tokens_field, true);
        let http_req = self.apply_auth(HttpRequest::post(&url).json(body), cx);
        let resp = send_checked(&*self.http, http_req).await?;
        Ok(openai_sse_to_stream(resp.body, req.model.clone()))
    }

    async fn embed(
        &self,
        req: EmbeddingRequest,
        cx: &ProviderCx,
    ) -> Result<EmbeddingResponse, ModelError> {
        let url = self.url_for(cx, &req.model, "embeddings");
        let body = embedding_request_to_openai(&req);
        let http_req = self.apply_auth(HttpRequest::post(&url).json(body), cx);
        let value = post_json(&*self.http, http_req).await?;
        Ok(openai_to_embeddings(value, &req.model))
    }

    async fn speech(&self, req: SpeechRequest, cx: &ProviderCx) -> Result<AudioStream, ModelError> {
        let url = self.url_for(cx, &req.model, "audio/speech");
        let body = speech_request_to_openai(&req);
        let http_req = self.apply_auth(HttpRequest::post(&url).json(body), cx);
        let resp = send_checked(&*self.http, http_req).await?;

        let upstream_ct = resp.header("content-type").map(str::to_owned);

        // Most OpenAI-wire brands stream the audio file as the raw response body. A few (Mistral)
        // instead return a JSON envelope with the audio base64-encoded under `audio_data` — a body
        // marked `application/json`, not a streamable audio file. Detect that and decode it into a
        // single audio frame; the content type comes from the requested format, since the JSON header
        // describes the envelope, not the audio inside it.
        if upstream_ct
            .as_deref()
            .is_some_and(|ct| ct.contains("application/json"))
        {
            let bytes = resp.collect_body().await?;
            let value: Value =
                serde_json::from_slice(&bytes).map_err(|e| ModelError::Mapping(e.to_string()))?;
            let audio = decode_speech_envelope(&value).ok_or_else(|| {
                ModelError::Mapping(
                    "speech response was JSON without a decodable `audio_data` field".into(),
                )
            })?;
            let content_type = audio_content_type(req.response_format.as_deref()).to_string();
            let opened = stream::once(async move { Ok(AudioChunk::Start { content_type }) });
            let data = stream::once(async move { Ok(AudioChunk::Data(audio.into())) });
            let closed = stream::once(async { Ok(AudioChunk::Finish) });
            return Ok(Box::pin(opened.chain(data).chain(closed)));
        }

        // Prefer the upstream's own Content-Type; fall back to the requested format's MIME.
        let content_type = upstream_ct
            .unwrap_or_else(|| audio_content_type(req.response_format.as_deref()).to_string());

        // Frame the byte stream as a canonical audio stream: Start → Data* → Finish. The transport body
        // already yields canonical `ModelError`s, so the frames flow through unchanged.
        let data = resp.body.map(|chunk| chunk.map(AudioChunk::Data));
        let opened = stream::once(async move { Ok(AudioChunk::Start { content_type }) });
        let closed = stream::once(async { Ok(AudioChunk::Finish) });
        Ok(Box::pin(opened.chain(data).chain(closed)))
    }

    async fn voices(&self, model: &str, cx: &ProviderCx) -> Result<Vec<VoiceInfo>, ModelError> {
        // Resolution order (the operator's chosen policy): a real catalog first — documented or live —
        // then an operator-declared catalog as the fallback.
        //
        // 1. A *documented* static catalog. Today that is OpenAI's published voice set — applied when
        //    this brand IS OpenAI, when it proxies OpenAI under a namespace (OpenRouter's
        //    `openai/<model>`), or for Azure OpenAI deployments (which serve the same voices).
        if let Some(model) = openai_speech_model(self.brand.name, model) {
            return Ok(openai_voices(model));
        }
        // 2. A *live* catalog, for a brand whose upstream exposes a real `GET /audio/voices` (Mistral).
        //    The fetched list is authoritative; only if the call fails do we fall back to a declared one
        //    (else surface the upstream error) — config is a genuine fallback, never a silent mask.
        if self.brand.voices_api {
            return match self.fetch_voices(cx).await {
                Ok(voices) => Ok(voices),
                Err(e) => declared_voices(cx, model).ok_or(e),
            };
        }
        // 3. Config-declared fallback. For an upstream llmleaf cannot enumerate itself and that has no
        //    fetchable catalog (e.g. a TTS model proxied through OpenRouter), the operator — who knows
        //    their deployment's ground truth — declares the voices under `[providers.settings.voices]`.
        //    We return exactly what they declared; we never guess a voice id (a wrong one would only
        //    fail at synthesis).
        if let Some(voices) = declared_voices(cx, model) {
            return Ok(voices);
        }
        Err(ModelError::Unsupported(format!(
            "provider '{}' does not list voices for '{model}' \
             (no documented or live catalog; declare them under [providers.settings.voices])",
            self.brand.name
        )))
    }

    async fn models(&self, cx: &ProviderCx) -> Result<Vec<ModelInfo>, ModelError> {
        // Only brands with a confirmed `GET /models` enumerate; others stay Unsupported so the listing
        // surface shows their namespace as non-enumerable rather than guessing a catalog.
        if !self.brand.models_api {
            return Err(ModelError::Unsupported(format!(
                "provider '{}' does not list models",
                self.brand.name
            )));
        }
        self.fetch_models(cx).await
    }

    async fn transcribe(
        &self,
        req: TranscriptionRequest,
        cx: &ProviderCx,
    ) -> Result<TranscriptionResponse, ModelError> {
        let url = self.url_for(cx, &req.model, "audio/transcriptions");

        // OpenRouter diverges here: its `/audio/transcriptions` rejects the OpenAI multipart upload and
        // wants a JSON body with the audio base64-encoded under `input_audio`. Honor that exactly
        // (decision filter: dialect mapping fidelity beats convenience) and parse its `{text, usage}`
        // reply; every other brand keeps the OpenAI multipart shape below. The subtitle/plain
        // `response_format`s (`srt`/`vtt`) have no equivalent on this endpoint, so it only yields JSON.
        if self.brand.transcription_json_base64 {
            let body = transcription_request_to_openrouter(&req, audio_format_token(&req.filename));
            let http_req = self.apply_auth(HttpRequest::post(&url).json(body), cx);
            let value = post_json(&*self.http, http_req).await?;
            return Ok(openrouter_to_transcription(value));
        }

        // The subtitle/plain formats (`text`, `srt`, `vtt`) return a non-JSON body; ask the upstream for
        // exactly what the consumer wants and pass that body through verbatim — that is the *faithful*
        // mapping (the surface returns it as-is). JSON shapes (`json`/`verbose_json`) parse into the
        // canonical transcript; an unset format defaults to `json`.
        let text_body = matches!(req.response_format.as_deref(), Some("text" | "srt" | "vtt"));
        let upstream_format = req.response_format.as_deref().unwrap_or("json");

        let mut form = MultipartForm::new()
            .text("model", req.model.clone())
            .text("response_format", upstream_format)
            .bytes(
                "file",
                Some(req.filename.clone()),
                Some(guess_audio_mime(&req.filename).to_string()),
                req.audio.clone(),
            );
        if let Some(language) = &req.language {
            form = form.text("language", language);
        }
        if let Some(prompt) = &req.prompt {
            form = form.text("prompt", prompt);
        }
        if let Some(temp) = req.temperature {
            form = form.text("temperature", temp.to_string());
        }
        for (k, v) in &req.extra {
            if let Some(s) = v.as_str() {
                form = form.text(k, s);
            }
        }

        let http_req = self.apply_auth(HttpRequest::post(&url).multipart(form), cx);
        if text_body {
            // Verbatim subtitle/plain body — carried in the canonical transcript's `text`, re-emitted
            // unchanged by the consumer surface for `text`/`srt`/`vtt`.
            let resp = send_checked(&*self.http, http_req).await?;
            let bytes = resp.collect_body().await?;
            let text = String::from_utf8_lossy(&bytes).into_owned();
            Ok(TranscriptionResponse {
                text,
                language: None,
                duration: None,
                usage: Default::default(),
            })
        } else {
            let value = post_json(&*self.http, http_req).await?;
            Ok(openai_to_transcription(value))
        }
    }

    /// Submit a batch. The file-based dialects (OpenAI, Mistral) both upload a JSONL file then create a
    /// job referencing it — entirely within this one call, holding nothing afterward (principle 5). The
    /// per-line bodies reuse [`request_to_openai`]; only the line envelope and the create body differ by
    /// flavor.
    async fn batch_create(
        &self,
        req: BatchSpec,
        cx: &ProviderCx,
    ) -> Result<BatchHandle, ModelError> {
        self.ensure_batch()?;
        let jsonl = build_jsonl(req.items.iter().map(|item| self.batch_line(item)));
        let file_id = self.upload_batch_file(cx, jsonl).await?;
        let url = format!("{}{}", self.batch_collection(cx), self.batch_query(cx));
        let body = match self.brand.batch_flavor {
            BatchFlavor::OpenAi => json!({
                "input_file_id": file_id,
                "endpoint": "/v1/chat/completions",
                "completion_window": "24h",
            }),
            // Azure's create `endpoint` drops the `/v1` segment; the deployment is per-line in the file.
            BatchFlavor::AzureOpenAi => json!({
                "input_file_id": file_id,
                "endpoint": "/chat/completions",
                "completion_window": "24h",
            }),
            // Together omits completion_window (fixed 24h upstream).
            BatchFlavor::Together => json!({
                "input_file_id": file_id,
                "endpoint": "/v1/chat/completions",
            }),
            // Mistral's job takes the file in an array and a single model for the whole batch (the core
            // guarantees every item routed to this one provider).
            BatchFlavor::MistralJobs => json!({
                "input_files": [file_id],
                "endpoint": "/v1/chat/completions",
                "model": req.items.first().map(|i| i.request.model.as_str()).unwrap_or_default(),
            }),
            BatchFlavor::Unsupported => unreachable!("guarded by ensure_batch"),
        };
        let http_req = self.apply_auth(HttpRequest::post(&url).json(body), cx);
        let value = post_json(&*self.http, http_req).await?;
        // Together wraps the created batch under a `job` key (`{job:{…}, warning}`); unwrap it. Every
        // other flavor returns the batch object directly.
        let value = match self.brand.batch_flavor {
            BatchFlavor::Together => value.get("job").cloned().unwrap_or(value),
            _ => value,
        };
        Ok(batch_value_to_handle(&value))
    }

    async fn batch_retrieve(
        &self,
        upstream_id: &str,
        cx: &ProviderCx,
    ) -> Result<BatchHandle, ModelError> {
        self.ensure_batch()?;
        let value = self.fetch_batch_value(cx, upstream_id).await?;
        Ok(batch_value_to_handle(&value))
    }

    async fn batch_cancel(
        &self,
        upstream_id: &str,
        cx: &ProviderCx,
    ) -> Result<BatchHandle, ModelError> {
        self.ensure_batch()?;
        let url = format!(
            "{}/{upstream_id}/cancel{}",
            self.batch_collection(cx),
            self.batch_query(cx)
        );
        let req = self.apply_auth(HttpRequest::post(&url), cx);
        let value = post_json(&*self.http, req).await?;
        Ok(batch_value_to_handle(&value))
    }

    /// Stream a batch's results. Every file-based flavor splits results across two files: an *output*
    /// file holding the succeeded requests and an *error* file holding the per-request failures
    /// (OpenAI/Azure/Together/Mistral all do this). We fetch the batch object, then stream both files —
    /// output first, then errors — concatenated, so a request that failed still surfaces as an `Errored`
    /// outcome instead of silently vanishing from the result set. Each line maps via the shared
    /// [`openai_batch_result_line`]; the error file's lines carry a non-null `error`, which that mapper
    /// already turns into an `Errored` outcome. Files are fetched lazily as the stream is consumed
    /// (principle 4 — the error file's connection isn't even opened until the output file is drained).
    async fn batch_results(
        &self,
        upstream_id: &str,
        cx: &ProviderCx,
    ) -> Result<BatchResultStream, ModelError> {
        self.ensure_batch()?;
        let batch = self.fetch_batch_value(cx, upstream_id).await?;
        let file_ids = batch_result_file_ids(&batch);
        if file_ids.is_empty() {
            return Err(ModelError::Mapping("batch has no output file yet".into()));
        }
        // Build one authed GET per file now (sync, no I/O); each is sent lazily inside the stream below.
        let files = self.batch_files(cx);
        let query = self.batch_query(cx);
        let requests: Vec<HttpRequest> = file_ids
            .into_iter()
            .map(|id| self.apply_auth(HttpRequest::get(format!("{files}/{id}/content{query}")), cx))
            .collect();
        let http = self.http.clone();
        Ok(Box::pin(async_stream::stream! {
            for req in requests {
                let resp = match send_checked(&*http, req).await {
                    Ok(r) => r,
                    Err(e) => {
                        yield Err(e);
                        return;
                    }
                };
                let mut lines = jsonl_result_stream(resp.body, openai_batch_result_line);
                while let Some(item) = lines.next().await {
                    yield item;
                }
            }
        }))
    }

    fn supports_realtime(&self) -> bool {
        self.brand.realtime_native
    }

    async fn realtime(
        &self,
        params: RealtimeParams,
        peer: RealtimePeer,
        cx: &ProviderCx,
    ) -> Result<(), ModelError> {
        // Only the OpenAI brand has a native Realtime WS upstream; every other compatible brand
        // declares it unsupported so the core bridges over chat streaming (no health penalty).
        if !self.brand.realtime_native {
            return Err(ModelError::Unsupported(format!(
                "provider '{}' has no native realtime transport",
                self.brand.name
            )));
        }
        let endpoint = cx
            .endpoint
            .as_deref()
            .unwrap_or(self.brand.default_endpoint);
        let credential = cx
            .credential
            .as_deref()
            .ok_or_else(|| ModelError::Unavailable("realtime requires a credential".to_string()))?;
        // The OpenAI-specific header knowledge lives here in the provider; the transport is
        // dialect-neutral — it just pumps frames over the dialed socket (principle 7).
        let url = crate::realtime_ws::build_ws_url(endpoint, &params.model);
        let mut headers = vec![("Authorization".to_string(), format!("Bearer {credential}"))];
        if let Some(sid) = cx.setting_str("safety_identifier") {
            headers.push(("openai-safety-identifier".to_string(), sid.to_string()));
        }
        self.realtime.run(url, headers, peer).await
    }
}

/// Best-effort MIME from a filename extension — providers detect the container by extension, but a
/// sensible `Content-Type` on the multipart part keeps strict gateways happy.
fn guess_audio_mime(filename: &str) -> &'static str {
    match filename
        .rsplit('.')
        .next()
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("mp3") => "audio/mpeg",
        Some("wav") => "audio/wav",
        Some("flac") => "audio/flac",
        Some("ogg") | Some("oga") => "audio/ogg",
        Some("webm") => "audio/webm",
        Some("m4a") | Some("mp4") => "audio/mp4",
        Some("mpeg") | Some("mpga") => "audio/mpeg",
        _ => "application/octet-stream",
    }
}

/// The OpenRouter `input_audio.format` token for an upload, by filename extension. OpenRouter accepts
/// `wav | mp3 | flac | m4a | ogg | webm | aac`; anything unrecognized falls back to `mp3`.
fn audio_format_token(filename: &str) -> &'static str {
    match filename
        .rsplit('.')
        .next()
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("wav") => "wav",
        Some("flac") => "flac",
        Some("m4a") | Some("mp4") => "m4a",
        Some("ogg") | Some("oga") => "ogg",
        Some("webm") => "webm",
        Some("aac") => "aac",
        Some("mp3") | Some("mpeg") | Some("mpga") => "mp3",
        _ => "mp3",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openrouter_models_url_requests_all_output_modalities() {
        // OpenRouter's `/models` defaults to `output_modalities=text`, hiding TTS (audio-output)
        // models; the brand must ask for `=all` so a speech model can be enumerated at all.
        let p = OpenAiCompatProvider::for_kind("openrouter", &crate::transport::Transports::fake())
            .unwrap();
        assert_eq!(
            p.models_url(&ProviderCx::default()),
            "https://openrouter.ai/api/v1/models?output_modalities=all"
        );
        // A brand without the quirk keeps the bare `<endpoint>/models` URL — no stray filter.
        let p = OpenAiCompatProvider::for_kind("openai", &crate::transport::Transports::fake())
            .unwrap();
        assert!(!p.models_url(&ProviderCx::default()).contains("output_modalities"));
    }

    #[tokio::test]
    async fn voices_fall_back_to_config_declared_for_unknown_upstream() {
        // OpenRouter proxying a Mistral TTS model: no documented catalog, no live API — the operator
        // declares the voices under settings.voices, keyed by the upstream model id.
        let p = OpenAiCompatProvider::for_kind("openrouter", &crate::transport::Transports::fake())
            .unwrap();
        let cx = ProviderCx {
            settings: serde_json::from_value(serde_json::json!({
                "voices": {
                    "mistralai/voxtral-mini-tts-2603": [
                        "aurora",
                        { "id": "basalt", "name": "Basalt", "languages": ["en", "fr"] }
                    ]
                }
            }))
            .unwrap(),
            ..Default::default()
        };

        let voices = p
            .voices("mistralai/voxtral-mini-tts-2603", &cx)
            .await
            .unwrap();
        assert_eq!(voices.len(), 2);
        assert_eq!(voices[0].id, "aurora");
        assert_eq!(voices[0].name, None);
        assert_eq!(voices[1].id, "basalt");
        assert_eq!(voices[1].name.as_deref(), Some("Basalt"));
        assert_eq!(voices[1].languages, vec!["en", "fr"]);

        // A model with nothing declared still reports unsupported (not an empty catalog).
        assert!(matches!(
            p.voices("mistralai/other", &cx).await,
            Err(ModelError::Unsupported(_))
        ));
    }

    #[tokio::test]
    async fn voices_documented_catalog_wins_over_config() {
        // The OpenAI namespace resolves to the documented catalog even if config also declares voices —
        // the live/documented source takes precedence; config is only a fallback.
        let p = OpenAiCompatProvider::for_kind("openrouter", &crate::transport::Transports::fake())
            .unwrap();
        let cx = ProviderCx {
            settings: serde_json::from_value(serde_json::json!({
                "voices": { "openai/tts-1": ["should-not-be-used"] }
            }))
            .unwrap(),
            ..Default::default()
        };
        let voices = p.voices("openai/tts-1", &cx).await.unwrap();
        assert_eq!(voices.len(), 6); // OpenAI classic catalog, not the single declared id
        assert!(voices.iter().any(|v| v.id == "alloy"));
    }

    #[test]
    fn only_mistral_advertises_a_live_voices_api() {
        assert!(Brand::for_kind("mistral").unwrap().voices_api);
        // Brands with no real list-voices endpoint must not claim one (they'd fetch a 404).
        for kind in ["openai", "openrouter", "groq", "azure-openai", "together"] {
            assert!(
                !Brand::for_kind(kind).unwrap().voices_api,
                "{kind} should not advertise a voices API"
            );
        }
    }

    #[test]
    fn known_kinds_resolve() {
        for kind in Brand::kinds() {
            assert!(
                Brand::for_kind(kind).is_some(),
                "kind {kind} should resolve"
            );
        }
        assert!(Brand::for_kind("not-a-provider").is_none());
    }

    #[test]
    fn azure_url_embeds_deployment_and_version() {
        let p =
            OpenAiCompatProvider::for_kind("azure", &crate::transport::Transports::fake()).unwrap();
        let cx = ProviderCx {
            endpoint: Some("https://my-res.openai.azure.com".into()),
            ..Default::default()
        };
        let url = p.build_url(&cx, "gpt-4o-deploy");
        assert_eq!(
            url,
            "https://my-res.openai.azure.com/openai/deployments/gpt-4o-deploy/chat/completions?api-version=2024-10-21"
        );
    }

    #[test]
    fn standard_url_appends_chat_completions() {
        let p =
            OpenAiCompatProvider::for_kind("groq", &crate::transport::Transports::fake()).unwrap();
        let url = p.build_url(&ProviderCx::default(), "llama-3.3-70b");
        assert_eq!(url, "https://api.groq.com/openai/v1/chat/completions");
    }

    #[test]
    fn cerebras_lists_models_from_its_public_endpoint() {
        // Cerebras's richest catalog is the unauthenticated public endpoint, not the id-only
        // authed `/v1/models` — the brand advertises listing and points at that fixed URL.
        let brand = Brand::for_kind("cerebras").unwrap();
        assert!(brand.models_api);
        assert_eq!(
            brand.models_url_override,
            Some("https://api.cerebras.ai/public/v1/models")
        );
        let p = OpenAiCompatProvider::for_kind("cerebras", &crate::transport::Transports::fake())
            .unwrap();
        assert_eq!(
            p.models_url(&ProviderCx::default()),
            "https://api.cerebras.ai/public/v1/models"
        );
        // The public catalog is a fixed host path; an operator `endpoint` override (which redirects
        // chat) must NOT redirect it.
        let cx = ProviderCx {
            endpoint: Some("https://proxy.example/v1".into()),
            ..Default::default()
        };
        assert_eq!(
            p.models_url(&cx),
            "https://api.cerebras.ai/public/v1/models"
        );
    }

    #[test]
    fn standard_brands_list_models_under_their_endpoint() {
        // No override → the default `<endpoint>/models` shape (and Azure's resource-scoped variant).
        let oa = OpenAiCompatProvider::for_kind("openai", &crate::transport::Transports::fake())
            .unwrap();
        assert!(Brand::for_kind("openai")
            .unwrap()
            .models_url_override
            .is_none());
        assert_eq!(
            oa.models_url(&ProviderCx::default()),
            "https://api.openai.com/v1/models"
        );
        let az =
            OpenAiCompatProvider::for_kind("azure", &crate::transport::Transports::fake()).unwrap();
        let az_cx = ProviderCx {
            endpoint: Some("https://my-res.openai.azure.com".into()),
            ..Default::default()
        };
        assert_eq!(
            az.models_url(&az_cx),
            "https://my-res.openai.azure.com/openai/models?api-version=2024-10-21"
        );
    }

    #[test]
    fn groq_lists_models_from_its_openai_v1_endpoint() {
        // Groq enumerates models at the standard `<endpoint>/models`; because its base already carries
        // `/openai/v1`, that resolves to Groq's documented list-models URL. No fixed override (unlike
        // Cerebras): an operator `endpoint` override redirects listing along with chat.
        let brand = Brand::for_kind("groq").unwrap();
        assert!(brand.models_api);
        assert!(brand.models_url_override.is_none());
        let p =
            OpenAiCompatProvider::for_kind("groq", &crate::transport::Transports::fake()).unwrap();
        assert_eq!(
            p.models_url(&ProviderCx::default()),
            "https://api.groq.com/openai/v1/models"
        );
        let cx = ProviderCx {
            endpoint: Some("https://proxy.example/openai/v1".into()),
            ..Default::default()
        };
        assert_eq!(p.models_url(&cx), "https://proxy.example/openai/v1/models");
    }

    #[test]
    fn only_groq_filters_inactive_models() {
        // Groq is the one brand whose listing flags retired models with `active:false`; it alone filters.
        assert!(Brand::for_kind("groq").unwrap().filter_inactive_models);
        for kind in [
            "openai",
            "openrouter",
            "mistral",
            "together",
            "cerebras",
            "azure",
        ] {
            assert!(
                !Brand::for_kind(kind).unwrap().filter_inactive_models,
                "{kind} must not drop models"
            );
        }
    }

    #[test]
    fn drop_inactive_omits_only_explicit_false_for_filtering_brands() {
        // Build three parsed entries: live (active:true), retired (active:false), and one with no flag.
        let entry = |id: &str, active: Option<bool>| {
            let mut m = ModelInfo::new(id);
            if let Some(a) = active {
                m.extra.insert("active".into(), json!(a));
            }
            m
        };
        let models = || {
            vec![
                entry("live", Some(true)),
                entry("retired", Some(false)),
                entry("no-flag", None),
            ]
        };

        // Groq filters: the retired model is dropped; live and flag-less ones are kept.
        let groq = Brand::for_kind("groq").unwrap();
        let kept: Vec<String> = drop_inactive(&groq, models())
            .into_iter()
            .map(|m| m.id)
            .collect();
        assert_eq!(kept, ["live", "no-flag"]);

        // A non-filtering brand keeps everything — even an `active:false` entry rides through untouched.
        let openai = Brand::for_kind("openai").unwrap();
        assert_eq!(drop_inactive(&openai, models()).len(), 3);
    }

    #[test]
    fn reasoning_brands_select_max_completion_tokens() {
        assert_eq!(
            Brand::for_kind("openai").unwrap().max_tokens_field,
            "max_completion_tokens"
        );
        assert_eq!(
            Brand::for_kind("xai").unwrap().max_tokens_field,
            "max_completion_tokens"
        );
        assert_eq!(
            Brand::for_kind("azure").unwrap().max_tokens_field,
            "max_completion_tokens"
        );
        assert_eq!(
            Brand::for_kind("cerebras").unwrap().max_tokens_field,
            "max_completion_tokens"
        );
        assert_eq!(
            Brand::for_kind("moonshot").unwrap().max_tokens_field,
            "max_completion_tokens"
        );
        // Broadest-compatibility brands keep legacy max_tokens.
        assert_eq!(
            Brand::for_kind("groq").unwrap().max_tokens_field,
            "max_tokens"
        );
        assert_eq!(
            Brand::for_kind("zai").unwrap().max_tokens_field,
            "max_tokens"
        );
    }

    #[test]
    fn aliases_resolve_to_canonical_brand() {
        assert_eq!(Brand::for_kind("z.ai").unwrap().name, "zai");
        assert_eq!(Brand::for_kind("glm").unwrap().name, "zai");
        assert_eq!(Brand::for_kind("kimi").unwrap().name, "moonshot");
        assert_eq!(Brand::for_kind("kimi-k2").unwrap().name, "moonshot");
    }

    #[test]
    fn requesty_routes_through_the_gateway() {
        let brand = Brand::for_kind("requesty").unwrap();
        assert!(matches!(brand.auth, AuthStyle::Bearer));
        // Gateway, like OpenRouter: normalizes the legacy max_tokens field, no native batch/voices.
        assert_eq!(brand.max_tokens_field, "max_tokens");
        assert!(!brand.voices_api);
        assert_eq!(brand.batch_flavor, BatchFlavor::Unsupported);
        let p = OpenAiCompatProvider::for_kind("requesty", &crate::transport::Transports::fake())
            .unwrap();
        let url = p.build_url(&ProviderCx::default(), "openai/gpt-4o");
        assert_eq!(url, "https://router.requesty.ai/v1/chat/completions");
    }

    #[test]
    fn zai_base_already_carries_version_segment() {
        // The /api/paas/v4 base must NOT gain an extra /v1 — chat/completions appends directly.
        let p =
            OpenAiCompatProvider::for_kind("zai", &crate::transport::Transports::fake()).unwrap();
        let url = p.build_url(&ProviderCx::default(), "glm-4.6");
        assert_eq!(url, "https://api.z.ai/api/paas/v4/chat/completions");
    }

    #[test]
    fn only_openrouter_uses_json_base64_transcription() {
        // OpenRouter's transcription endpoint takes JSON+base64; every other brand stays multipart.
        assert!(
            Brand::for_kind("openrouter")
                .unwrap()
                .transcription_json_base64
        );
        for kind in ["openai", "groq", "azure", "mistral", "together"] {
            assert!(
                !Brand::for_kind(kind).unwrap().transcription_json_base64,
                "{kind} must keep the OpenAI multipart transcription shape"
            );
        }
    }

    #[test]
    fn audio_format_token_maps_by_extension() {
        assert_eq!(audio_format_token("clip.mp3"), "mp3");
        assert_eq!(audio_format_token("clip.WAV"), "wav"); // case-insensitive
        assert_eq!(audio_format_token("clip.m4a"), "m4a");
        assert_eq!(audio_format_token("clip.oga"), "ogg");
        assert_eq!(audio_format_token("noext"), "mp3"); // unknown → mp3
    }

    #[test]
    fn batch_flavor_assignment() {
        use BatchFlavor::*;
        let f = |k: &str| Brand::for_kind(k).unwrap().batch_flavor;
        assert_eq!(f("openai"), OpenAi);
        assert_eq!(f("groq"), OpenAi);
        assert_eq!(f("moonshot"), OpenAi);
        assert_eq!(f("mistral"), MistralJobs);
        assert_eq!(f("together"), Together);
        assert_eq!(f("azure"), AzureOpenAi);
        // Brands with no batch API stay unsupported.
        assert_eq!(f("deepseek"), Unsupported);
        assert_eq!(f("openrouter"), Unsupported);
        assert_eq!(f("perplexity"), Unsupported);
    }

    #[test]
    fn batch_urls_and_purpose_by_flavor() {
        // OpenAI-wire: /batches + /files, no query, purpose "batch", line carries method/url.
        let oa = OpenAiCompatProvider::for_kind("openai", &crate::transport::Transports::fake())
            .unwrap();
        let cx = ProviderCx::default();
        assert_eq!(
            oa.batch_collection(&cx),
            "https://api.openai.com/v1/batches"
        );
        assert_eq!(oa.batch_files(&cx), "https://api.openai.com/v1/files");
        assert_eq!(oa.batch_query(&cx), "");
        assert_eq!(oa.batch_purpose(), "batch");

        // Together: own purpose, OpenAI-wire URLs.
        let tg = OpenAiCompatProvider::for_kind("together", &crate::transport::Transports::fake())
            .unwrap();
        assert_eq!(tg.batch_purpose(), "batch-api");
        assert_eq!(
            tg.batch_collection(&cx),
            "https://api.together.ai/v1/batches"
        );

        // Mistral: jobs path.
        let ms = OpenAiCompatProvider::for_kind("mistral", &crate::transport::Transports::fake())
            .unwrap();
        assert_eq!(
            ms.batch_collection(&cx),
            "https://api.mistral.ai/v1/batch/jobs"
        );

        // Azure: resource-scoped /openai/batches with ?api-version, operator endpoint.
        let az =
            OpenAiCompatProvider::for_kind("azure", &crate::transport::Transports::fake()).unwrap();
        let az_cx = ProviderCx {
            endpoint: Some("https://my-res.openai.azure.com".into()),
            ..Default::default()
        };
        assert_eq!(
            az.batch_collection(&az_cx),
            "https://my-res.openai.azure.com/openai/batches"
        );
        assert_eq!(
            az.batch_files(&az_cx),
            "https://my-res.openai.azure.com/openai/files"
        );
        assert_eq!(az.batch_query(&az_cx), "?api-version=2024-10-21");
    }

    #[test]
    fn batch_line_envelope_by_flavor() {
        use llmleaf_model::{BatchItem, ChatRequest, Message, Role};
        let item = BatchItem {
            custom_id: "x".into(),
            request: ChatRequest {
                model: "m".into(),
                messages: vec![Message::text(Role::User, "hi")],
                max_tokens: None,
                temperature: None,
                top_p: None,
                stop: vec![],
                stream: false,
                tools: vec![],
                tool_choice: None,
                thinking: None,
                extra: Default::default(),
            },
        };
        // OpenAI/Azure wrap with method + url.
        let oa = OpenAiCompatProvider::for_kind("openai", &crate::transport::Transports::fake())
            .unwrap()
            .batch_line(&item);
        assert_eq!(oa["method"], "POST");
        assert_eq!(oa["url"], "/v1/chat/completions");
        assert_eq!(oa["custom_id"], "x");
        assert!(oa["body"]["model"] == "m");
        // Together/Mistral carry just custom_id + body.
        let tg = OpenAiCompatProvider::for_kind("together", &crate::transport::Transports::fake())
            .unwrap()
            .batch_line(&item);
        assert!(tg.get("method").is_none());
        assert_eq!(tg["custom_id"], "x");
        assert_eq!(tg["body"]["model"], "m");
    }

    #[test]
    fn batch_results_read_output_then_error_file() {
        // Both files present → output first, error second (so failures never go missing).
        let v = json!({ "output_file_id": "out", "error_file_id": "err" });
        assert_eq!(batch_result_file_ids(&v), ["out", "err"]);
        // Mistral's key names (`*_file`) work via the fallback.
        let v = json!({ "output_file": "o", "error_file": "e" });
        assert_eq!(batch_result_file_ids(&v), ["o", "e"]);
        // A fully-failed batch may have only an error file — it must still be streamed.
        let v = json!({ "error_file_id": "err" });
        assert_eq!(batch_result_file_ids(&v), ["err"]);
        // Output only (the common success case) is unchanged.
        let v = json!({ "output_file_id": "out" });
        assert_eq!(batch_result_file_ids(&v), ["out"]);
        // No files yet → empty, which the caller turns into "no output file yet".
        assert!(batch_result_file_ids(&json!({})).is_empty());
    }
}
