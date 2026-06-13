//! LM Studio local-server provider — its native `/api/v0/` REST surface.
//!
//! LM Studio exposes three HTTP surfaces: the pure OpenAI-compat `/v1/`, the OpenAI-*like* `/api/v0/`
//! used here, and a newer event-based `/api/v1/chat`. The `/api/v0/` surface speaks the OpenAI chat and
//! embeddings wire, so request mapping and response parsing reuse [`crate::openai_wire`] verbatim — the
//! decision filter in practice: where the dialect already matches, share the one correct mapping and
//! contribute only what differs. What differs, and why this is a first-class provider rather than a
//! compat-table row, is the catalog: `GET /api/v0/models` reports real per-model metadata — `type`
//! (`llm`/`vlm`/`embeddings` → modality), `state` (`loaded`/`not-loaded`), `arch`, `quantization`,
//! `compatibility_type`, `max_context_length` — which the bare OpenAI `/v1/models` id-only listing
//! cannot give, and which lets the listing surface report modality and context without guessing (SOUL).
//!
//! The `/api/v0/` chat/completions response additionally carries native `stats` (tokens/sec, time to
//! first token), `model_info`, and `runtime` telemetry blocks. The canonical model accounts in tokens,
//! not tok/s, so those blocks have no home and are not invented into canonical fields — the standard
//! OpenAI parts (content, tool calls, usage, finish) map through exactly as for any OpenAI-wire brand.
//!
//! Auth: none for a local server (the default); an optional API token (LM Studio's Developer page) is
//! sent as `Authorization: Bearer` so a token-protected instance works through the same path.

use async_trait::async_trait;
use llmleaf_model::{
    ChatRequest, EmbeddingRequest, EmbeddingResponse, Modality, ModelError, ModelInfo,
    ResponseStream,
};
use llmleaf_provider::{Provider, ProviderCx};
use serde_json::Value;

use std::sync::Arc;

use crate::http::{post_json, send_checked};
use crate::openai_wire::{
    embedding_request_to_openai, openai_sse_to_stream, openai_to_embeddings, request_to_openai,
};
use crate::transport::{HttpRequest, HttpTransport, Transports};

const DEFAULT_ENDPOINT: &str = "http://localhost:1234/api/v0";

pub struct LmStudioProvider {
    http: Arc<dyn HttpTransport>,
}

impl LmStudioProvider {
    pub fn new(transports: &Transports) -> Self {
        LmStudioProvider {
            http: transports.http.clone(),
        }
    }

    /// This instance's base endpoint (config override or the local `/api/v0` default), trailing slash
    /// trimmed. Operations append their path (`chat/completions`, `embeddings`, `models`).
    fn endpoint(&self, cx: &ProviderCx) -> String {
        cx.endpoint
            .as_deref()
            .unwrap_or(DEFAULT_ENDPOINT)
            .trim_end_matches('/')
            .to_string()
    }

    /// Apply optional bearer auth. A local server needs none; a token-protected one takes the LM Studio
    /// API token as `Authorization: Bearer`. Shared by every method so they authenticate identically.
    fn auth(&self, req: HttpRequest, cx: &ProviderCx) -> HttpRequest {
        match &cx.credential {
            Some(cred) => req.bearer(cred),
            None => req,
        }
    }
}

#[async_trait]
impl Provider for LmStudioProvider {
    fn name(&self) -> &str {
        "lmstudio"
    }

    async fn chat(&self, req: ChatRequest, cx: &ProviderCx) -> Result<ResponseStream, ModelError> {
        // `/api/v0/chat/completions` is OpenAI-wire and streams SSE; reuse the shared request mapping
        // and SSE streaming core so tokens flow live (principle 4). `max_tokens` is the cap field
        // (LM Studio accepts it; `-1` would mean unlimited, but the canonical cap is always a count).
        let url = format!("{}/chat/completions", self.endpoint(cx));
        let body = request_to_openai(&req, "max_tokens", true);
        let http_req = self.auth(HttpRequest::post(&url).json(body), cx);
        let resp = send_checked(&*self.http, http_req).await?;
        Ok(openai_sse_to_stream(resp.body, req.model.clone()))
    }

    async fn embed(
        &self,
        req: EmbeddingRequest,
        cx: &ProviderCx,
    ) -> Result<EmbeddingResponse, ModelError> {
        let url = format!("{}/embeddings", self.endpoint(cx));
        let body = embedding_request_to_openai(&req);
        let http_req = self.auth(HttpRequest::post(&url).json(body), cx);
        let value = post_json(&*self.http, http_req).await?;
        Ok(openai_to_embeddings(value, &req.model))
    }

    /// Enumerate the catalog via `GET /api/v0/models` — the rich listing (modality, context, load
    /// state, arch, quant) the OpenAI-compat `/v1/models` cannot provide.
    async fn models(&self, cx: &ProviderCx) -> Result<Vec<ModelInfo>, ModelError> {
        let url = format!("{}/models", self.endpoint(cx));
        let http_req = self.auth(HttpRequest::get(&url), cx);
        let value = post_json(&*self.http, http_req).await?;
        Ok(lmstudio_models_to_canonical(value))
    }
}

/// LM Studio `GET /api/v0/models` reply → canonical [`ModelInfo`]s. The body wraps the list as
/// `{ object: "list", data: [ … ] }`; a bare array is tolerated defensively.
fn lmstudio_models_to_canonical(value: Value) -> Vec<ModelInfo> {
    let items = match value {
        Value::Array(arr) => arr,
        Value::Object(mut obj) => match obj.remove("data") {
            Some(Value::Array(a)) => a,
            _ => Vec::new(),
        },
        _ => Vec::new(),
    };
    items.iter().filter_map(lmstudio_model_to_info).collect()
}

/// One `/api/v0/models` entry → [`ModelInfo`]. `None` (skipped) when it has no string `id`. Modality
/// comes from the explicit `type` field (never the id — SOUL); `max_context_length` maps to the context
/// window; the load state and GGUF/MLX descriptors ride through verbatim in `extra` (principle 7). There
/// is no pricing on a local server, so it stays `None` (enhanced downstream if the dataset knows it).
fn lmstudio_model_to_info(item: &Value) -> Option<ModelInfo> {
    let id = item.get("id").and_then(Value::as_str)?;
    let mut info = ModelInfo::new(id);
    info.modality = item
        .get("type")
        .and_then(Value::as_str)
        .and_then(lmstudio_modality);
    info.max_context = item
        .get("max_context_length")
        .and_then(Value::as_u64)
        .map(|n| n.min(u32::MAX as u64) as u32);
    for key in [
        "type",
        "publisher",
        "arch",
        "compatibility_type",
        "quantization",
        "state",
        "loaded_context_length",
    ] {
        if let Some(v) = item.get(key) {
            if !v.is_null() {
                info.extra.insert(key.to_string(), v.clone());
            }
        }
    }
    Some(info)
}

/// LM Studio's `type` → canonical modality. `llm`/`vlm` are both text-out language models; `embeddings`
/// is an embedder; anything else (a future type) yields `None` rather than a guess.
fn lmstudio_modality(t: &str) -> Option<Modality> {
    match t {
        "llm" | "vlm" => Some(Modality::Llm),
        "embeddings" => Some(Modality::Embedding),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn models_listing_maps_type_context_and_state() {
        // The official `/api/v0/models` example: a vlm, an llm, and an embeddings model.
        let body = json!({
            "object": "list",
            "data": [
                {
                    "id": "qwen2-vl-7b-instruct", "object": "model", "type": "vlm",
                    "publisher": "mlx-community", "arch": "qwen2_vl",
                    "compatibility_type": "mlx", "quantization": "4bit",
                    "state": "not-loaded", "max_context_length": 32768
                },
                {
                    "id": "meta-llama-3.1-8b-instruct", "object": "model", "type": "llm",
                    "publisher": "lmstudio-community", "arch": "llama",
                    "compatibility_type": "gguf", "quantization": "Q4_K_M",
                    "state": "loaded", "max_context_length": 131072,
                    "loaded_context_length": 4096
                },
                {
                    "id": "text-embedding-nomic-embed-text-v1.5", "object": "model",
                    "type": "embeddings", "publisher": "nomic-ai", "arch": "nomic-bert",
                    "compatibility_type": "gguf", "quantization": "Q4_0",
                    "state": "not-loaded", "max_context_length": 2048
                }
            ]
        });
        let out = lmstudio_models_to_canonical(body);
        assert_eq!(out.len(), 3);

        // vlm → Llm; context + descriptors lifted; modality never guessed from the id.
        assert_eq!(out[0].id, "qwen2-vl-7b-instruct");
        assert_eq!(out[0].modality, Some(Modality::Llm));
        assert_eq!(out[0].max_context, Some(32768));
        assert_eq!(out[0].extra.get("state").unwrap(), "not-loaded");
        assert_eq!(out[0].extra.get("quantization").unwrap(), "4bit");
        assert_eq!(out[0].extra.get("arch").unwrap(), "qwen2_vl");
        // Pricing is unknown on a local server — never fabricated.
        assert_eq!(out[0].input_per_mtok, None);

        // llm, loaded → loaded_context_length rides through in extra.
        assert_eq!(out[1].modality, Some(Modality::Llm));
        assert_eq!(out[1].max_context, Some(131072));
        assert_eq!(out[1].extra.get("loaded_context_length").unwrap(), 4096);

        // embeddings → Embedding.
        assert_eq!(out[2].modality, Some(Modality::Embedding));
        assert_eq!(out[2].max_context, Some(2048));
    }

    #[test]
    fn models_tolerates_bare_array_and_skips_idless() {
        let bare = json!([{ "id": "solo", "type": "llm", "max_context_length": 8192 }]);
        let out = lmstudio_models_to_canonical(bare);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].id, "solo");

        // An entry without an id is dropped, not fabricated.
        let junk = json!({ "data": [{ "object": "model", "type": "llm" }] });
        assert!(lmstudio_models_to_canonical(junk).is_empty());
        // Neither list-wrapper nor array → empty, never a panic.
        assert!(lmstudio_models_to_canonical(json!("nonsense")).is_empty());
    }

    #[test]
    fn modality_from_type_only() {
        assert_eq!(lmstudio_modality("llm"), Some(Modality::Llm));
        assert_eq!(lmstudio_modality("vlm"), Some(Modality::Llm));
        assert_eq!(lmstudio_modality("embeddings"), Some(Modality::Embedding));
        // An unknown/future type yields no modality rather than a guess.
        assert_eq!(lmstudio_modality("diffusion"), None);
    }
}
