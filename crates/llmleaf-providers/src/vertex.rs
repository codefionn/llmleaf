//! Google Vertex AI provider (the enterprise Gemini surface on `*-aiplatform.googleapis.com`).
//!
//! Vertex speaks the **same `generateContent` dialect** as the AI Studio / Generative Language API
//! that [`crate::gemini`] already maps — the request and response bodies are byte-for-byte the same
//! schema. What differs is the transport, and only the transport:
//!   - **URL** — a project/location-scoped publisher path,
//!     `{host}/v1/projects/{project}/locations/{location}/publishers/google/models/{model}:generateContent`,
//!     where `host` is the regional `https://{location}-aiplatform.googleapis.com` (or the bare
//!     `https://aiplatform.googleapis.com` when `location = "global"`, which still keeps `locations/global`
//!     in the path).
//!   - **auth** — a Google OAuth2 bearer (`Authorization: Bearer <token>`, `cloud-platform` scope), not
//!     AI Studio's `x-goog-api-key`. The operator supplies an access token as the instance `credential`
//!     (e.g. `env:` from `gcloud auth print-access-token` or an ADC sidecar) and keeps it fresh; the
//!     proxy mints nothing (no service-account private key handled in-process).
//!   - **embeddings** — Vertex uses the `:predict` shape (`instances[].content` + `predictions[].
//!     embeddings.values`), a different dialect than AI Studio's `batchEmbedContents`, mapped here.
//!   - **model listing** — the `v1beta1` publisher-models catalog, which reports no generation methods,
//!     so modality stays unknown (enhanced downstream) rather than guessed.
//!
//! So the chat path reuses [`crate::gemini::request_to_gemini`] / [`crate::gemini::gemini_to_chunks`]
//! verbatim, and this module owns only the URL, the bearer auth, the `:predict` embedding mapping, and
//! the publisher-models listing.

use async_trait::async_trait;
use futures::stream;
use llmleaf_model::{
    ChatRequest, Embedding, EmbeddingRequest, EmbeddingResponse, ModelError, ModelInfo,
    ResponseStream, Usage,
};
use llmleaf_provider::{Provider, ProviderCx};
use serde_json::{json, Map, Value};

use std::sync::Arc;

use crate::gemini::{gemini_to_chunks, request_to_gemini};
use crate::http::post_json;
use crate::transport::{HttpRequest, HttpTransport, Transports};

/// The global (region-less) host. Used when `location = "global"` or no location is configured. Regional
/// deployments derive `https://{location}-aiplatform.googleapis.com` instead.
const GLOBAL_HOST: &str = "https://aiplatform.googleapis.com";

pub struct VertexProvider {
    http: Arc<dyn HttpTransport>,
}

impl VertexProvider {
    pub fn new(transports: &Transports) -> Self {
        VertexProvider {
            http: transports.http.clone(),
        }
    }

    /// The API host for this instance. An explicit `endpoint` override wins (host base, no version or
    /// path — those are appended per operation); otherwise it is derived from `settings.location`:
    /// a regional location gives `https://{location}-aiplatform.googleapis.com`, while `global` (or no
    /// location) gives the bare [`GLOBAL_HOST`]. Trailing slash trimmed.
    fn host(&self, cx: &ProviderCx) -> String {
        if let Some(ep) = cx.endpoint.as_deref() {
            return ep.trim_end_matches('/').to_string();
        }
        match cx.setting_str("location") {
            Some(loc) if !loc.is_empty() && loc != "global" => {
                format!("https://{loc}-aiplatform.googleapis.com")
            }
            _ => GLOBAL_HOST.to_string(),
        }
    }

    /// The project/location-scoped publisher-model resource URL (without the `:method` suffix). Both
    /// `project` and `location` are mandatory on the standard Vertex path — there is no project-implicit
    /// form — so a missing one is a [`ModelError::Mapping`] (the extension cannot build a valid request),
    /// surfaced before any network call rather than as an opaque upstream 404.
    fn resource_base(&self, cx: &ProviderCx, model: &str) -> Result<String, ModelError> {
        let project = cx
            .setting_str("project")
            .filter(|s| !s.is_empty())
            .ok_or_else(|| {
                ModelError::Mapping("vertex provider requires `settings.project`".into())
            })?;
        let location = cx
            .setting_str("location")
            .filter(|s| !s.is_empty())
            .ok_or_else(|| {
                ModelError::Mapping("vertex provider requires `settings.location`".into())
            })?;
        Ok(format!(
            "{}/v1/projects/{project}/locations/{location}/publishers/google/models/{model}",
            self.host(cx)
        ))
    }

    /// Apply Vertex's OAuth2 bearer auth. The operator supplies the access token as the instance
    /// credential and refreshes it out-of-band; the proxy sends it verbatim and mints nothing.
    fn auth(&self, req: HttpRequest, cx: &ProviderCx) -> HttpRequest {
        match &cx.credential {
            Some(token) => req.bearer(token),
            None => req,
        }
    }
}

#[async_trait]
impl Provider for VertexProvider {
    fn name(&self) -> &str {
        "vertex"
    }

    /// Enumerate Google's foundation (publisher) models via the Model Garden listing
    /// `GET /v1beta1/publishers/{publisher}/models` (note: this list lives only under `v1beta1`, and
    /// carries no project/location in its path). Paginated on `nextPageToken`. The catalog reports no
    /// generation methods or token limits, so every field beyond the id is left `None` and enhanced
    /// downstream — see [`vertex_publisher_to_info`].
    async fn models(&self, cx: &ProviderCx) -> Result<Vec<ModelInfo>, ModelError> {
        let host = self.host(cx);
        // Defaults to Google's catalog; an operator may point at another Model Garden publisher.
        let publisher = cx
            .setting_str("publisher")
            .filter(|s| !s.is_empty())
            .unwrap_or("google");
        let mut out: Vec<ModelInfo> = Vec::new();
        let mut page_token: Option<String> = None;
        // Bound the loop so a misbehaving/repeating token cannot spin forever (mirrors the Gemini list).
        for _ in 0..100 {
            let mut url = format!("{host}/v1beta1/publishers/{publisher}/models?pageSize=200");
            if let Some(tok) = &page_token {
                url.push_str(&format!("&pageToken={tok}"));
            }
            let req = self.auth(HttpRequest::get(&url), cx);
            let value = post_json(&*self.http, req).await?;
            if let Some(items) = value.get("publisherModels").and_then(Value::as_array) {
                out.extend(items.iter().filter_map(vertex_publisher_to_info));
            }
            match value.get("nextPageToken").and_then(Value::as_str) {
                Some(tok) if !tok.is_empty() => page_token = Some(tok.to_string()),
                _ => break,
            }
        }
        Ok(out)
    }

    async fn chat(&self, req: ChatRequest, cx: &ProviderCx) -> Result<ResponseStream, ModelError> {
        let url = format!("{}:generateContent", self.resource_base(cx, &req.model)?);
        // Same body the live Gemini chat path builds — Vertex's generateContent schema is identical.
        let body = request_to_gemini(&req);
        let http_req = self.auth(HttpRequest::post(&url).json(body), cx);
        let value = post_json(&*self.http, http_req).await?;
        let chunks = gemini_to_chunks(value, &req.model);
        Ok(Box::pin(stream::iter(chunks.into_iter().map(Ok))))
    }

    async fn embed(
        &self,
        req: EmbeddingRequest,
        cx: &ProviderCx,
    ) -> Result<EmbeddingResponse, ModelError> {
        // Vertex embeddings use the `:predict` custom method (the model lives only in the URL — there is
        // no per-instance/top-level `model` field, unlike AI Studio's batchEmbedContents).
        let url = format!("{}:predict", self.resource_base(cx, &req.model)?);
        // `task_type` (RETRIEVAL_DOCUMENT, RETRIEVAL_QUERY, …) materially changes embedding quality and
        // is operator-configurable, exactly as for Gemini; `auto_truncate` toggles upstream's
        // truncate-on-overflow (default true). Both omitted when unset so Vertex applies its defaults.
        let task_type = cx.setting_str("task_type");
        let auto_truncate = cx.settings.get("auto_truncate").and_then(Value::as_bool);
        let body = vertex_embed_request(&req, task_type, auto_truncate);
        let http_req = self.auth(HttpRequest::post(&url).json(body), cx);
        let value = post_json(&*self.http, http_req).await?;
        Ok(vertex_to_embeddings(value, &req.model))
    }
}

/// Build the Vertex `:predict` embeddings body. Each input becomes one `instances` entry whose text is a
/// bare `content` string; `task_type` is snake_case with an UPPERCASE enum value (the `:predict` dialect,
/// distinct from embedContent's camelCase `taskType`). The `parameters` object — Matryoshka
/// `outputDimensionality` and `autoTruncate` — is camelCase and emitted only when something is set.
fn vertex_embed_request(
    req: &EmbeddingRequest,
    task_type: Option<&str>,
    auto_truncate: Option<bool>,
) -> Value {
    let instances: Vec<Value> = req
        .input
        .iter()
        .map(|text| {
            let mut obj = Map::new();
            obj.insert("content".into(), json!(text));
            if let Some(task) = task_type {
                obj.insert("task_type".into(), json!(task));
            }
            Value::Object(obj)
        })
        .collect();

    let mut params = Map::new();
    if let Some(dims) = req.dimensions {
        params.insert("outputDimensionality".into(), json!(dims));
    }
    if let Some(trunc) = auto_truncate {
        params.insert("autoTruncate".into(), json!(trunc));
    }

    let mut body = Map::new();
    body.insert("instances".into(), Value::Array(instances));
    if !params.is_empty() {
        body.insert("parameters".into(), Value::Object(params));
    }
    Value::Object(body)
}

/// Parse a Vertex `:predict` embeddings response. Vectors live at `predictions[i].embeddings.values`, in
/// input order. Per-instance `statistics.token_count` is the one token signal Vertex reports here, so it
/// is summed into the canonical usage (prompt tokens; embeddings have no completion) — relayed, never
/// invented (principle 5). Google may serialize that int64 as a JSON string, so read either.
fn vertex_to_embeddings(value: Value, model: &str) -> EmbeddingResponse {
    let mut tokens: u64 = 0;
    let embeddings = value
        .get("predictions")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .enumerate()
                .map(|(i, item)| {
                    let emb = item.get("embeddings");
                    let vector = emb
                        .and_then(|e| e.get("values"))
                        .and_then(Value::as_array)
                        .map(|vs| {
                            vs.iter()
                                .filter_map(Value::as_f64)
                                .map(|f| f as f32)
                                .collect()
                        })
                        .unwrap_or_default();
                    if let Some(tc) = emb
                        .and_then(|e| e.get("statistics"))
                        .and_then(|s| s.get("token_count"))
                        .and_then(|v| {
                            v.as_u64()
                                .or_else(|| v.as_str().and_then(|s| s.parse::<u64>().ok()))
                        })
                    {
                        tokens += tc;
                    }
                    Embedding {
                        index: i as u32,
                        vector,
                    }
                })
                .collect()
        })
        .unwrap_or_default();

    // Only surface usage Vertex actually reported; an absent count stays at the canonical default.
    let usage = if tokens > 0 {
        Usage {
            prompt_tokens: tokens,
            completion_tokens: 0,
            total_tokens: tokens,
            cost_usd: None,
        }
    } else {
        Usage::default()
    };

    EmbeddingResponse {
        model: model.to_string(),
        embeddings,
        usage,
    }
}

/// Map one `publisherModels` entry → canonical [`ModelInfo`]. `None` (skipped) when it has no `name`. The
/// id is `name` with the `publishers/{publisher}/models/` prefix stripped (what the generate/predict path
/// takes). The list reports no generation methods or token limits, so modality, context, and output
/// windows stay `None` — never guessed from the id (SOUL forbids id-substring guessing); the gaps are
/// enhanced downstream from the bundled dataset. The descriptive fields are stashed verbatim (principle 7).
fn vertex_publisher_to_info(m: &Value) -> Option<ModelInfo> {
    let raw = m.get("name").and_then(Value::as_str)?;
    let id = raw
        .rsplit_once("/models/")
        .map(|(_, model)| model)
        .unwrap_or(raw);
    let mut info = ModelInfo::new(id);
    for key in [
        "versionId",
        "openSourceCategory",
        "launchStage",
        "versionState",
        "publisherModelTemplate",
    ] {
        if let Some(v) = m.get(key) {
            if !v.is_null() {
                info.extra.insert(key.to_string(), v.clone());
            }
        }
    }
    Some(info)
}

#[cfg(test)]
mod tests {
    use super::*;
    use llmleaf_model::{Message, Role};

    fn cx_with(settings: Value, endpoint: Option<&str>) -> ProviderCx {
        ProviderCx {
            settings: settings.as_object().cloned().unwrap_or_default(),
            endpoint: endpoint.map(str::to_string),
            ..Default::default()
        }
    }

    #[test]
    fn regional_host_and_resource_path() {
        let p = VertexProvider::new(&crate::transport::Transports::fake());
        let cx = cx_with(
            json!({ "project": "proj", "location": "us-central1" }),
            None,
        );
        assert_eq!(p.host(&cx), "https://us-central1-aiplatform.googleapis.com");
        assert_eq!(
            p.resource_base(&cx, "gemini-2.5-flash").unwrap(),
            "https://us-central1-aiplatform.googleapis.com/v1/projects/proj/locations/us-central1/publishers/google/models/gemini-2.5-flash"
        );
    }

    #[test]
    fn global_location_drops_region_prefix_but_keeps_path_segment() {
        let p = VertexProvider::new(&crate::transport::Transports::fake());
        let cx = cx_with(json!({ "project": "proj", "location": "global" }), None);
        // Bare host, no `global-` prefix...
        assert_eq!(p.host(&cx), "https://aiplatform.googleapis.com");
        // ...but the path still pins `locations/global`.
        let base = p.resource_base(&cx, "gemini-2.5-pro").unwrap();
        assert_eq!(
            base,
            "https://aiplatform.googleapis.com/v1/projects/proj/locations/global/publishers/google/models/gemini-2.5-pro"
        );
    }

    #[test]
    fn endpoint_override_replaces_host_only() {
        let p = VertexProvider::new(&crate::transport::Transports::fake());
        let cx = cx_with(
            json!({ "project": "proj", "location": "us-central1" }),
            Some("https://vertex.proxy.test/"),
        );
        // Override wins for the host; the version + publisher path are still appended.
        assert_eq!(p.host(&cx), "https://vertex.proxy.test");
        assert_eq!(
            p.resource_base(&cx, "m").unwrap(),
            "https://vertex.proxy.test/v1/projects/proj/locations/us-central1/publishers/google/models/m"
        );
    }

    #[test]
    fn missing_project_or_location_is_a_mapping_error() {
        let p = VertexProvider::new(&crate::transport::Transports::fake());
        let no_project = cx_with(json!({ "location": "us-central1" }), None);
        assert!(matches!(
            p.resource_base(&no_project, "m"),
            Err(ModelError::Mapping(_))
        ));
        let no_location = cx_with(json!({ "project": "proj" }), None);
        assert!(matches!(
            p.resource_base(&no_location, "m"),
            Err(ModelError::Mapping(_))
        ));
    }

    #[test]
    fn chat_body_is_the_gemini_body_verbatim() {
        // Vertex's generateContent body is byte-for-byte the Gemini mapping; assert we reuse it as-is.
        let req = ChatRequest {
            model: "gemini-2.5-flash".into(),
            messages: vec![
                Message::text(Role::System, "be brief"),
                Message::text(Role::User, "hi"),
            ],
            max_tokens: Some(128),
            temperature: Some(0.3),
            top_p: None,
            stop: vec![],
            stream: false,
            tools: vec![],
            tool_choice: None,
            thinking: None,
            extra: Default::default(),
        };
        let wire = request_to_gemini(&req);
        assert_eq!(wire["systemInstruction"]["parts"][0]["text"], "be brief");
        assert_eq!(wire["contents"][0]["role"], "user");
        assert_eq!(wire["generationConfig"]["maxOutputTokens"], 128);
    }

    #[test]
    fn predict_request_uses_content_string_and_snake_case_task_type() {
        let req = EmbeddingRequest {
            model: "text-embedding-005".into(),
            input: vec!["hello".into(), "world".into()],
            dimensions: Some(256),
            encoding_format: None,
            extra: Default::default(),
        };
        let wire = vertex_embed_request(&req, Some("RETRIEVAL_DOCUMENT"), Some(false));
        // One instance per input, in order; text is a bare `content` string (not nested parts).
        assert_eq!(wire["instances"][0]["content"], "hello");
        assert_eq!(wire["instances"][1]["content"], "world");
        // The model id is NOT in the body for `:predict` (it rides only in the URL).
        assert!(wire["instances"][0].get("model").is_none());
        assert!(wire.get("model").is_none());
        // Instance field is snake_case `task_type` with an UPPERCASE value.
        assert_eq!(wire["instances"][0]["task_type"], "RETRIEVAL_DOCUMENT");
        // `parameters` are camelCase.
        assert_eq!(wire["parameters"]["outputDimensionality"], 256);
        assert_eq!(wire["parameters"]["autoTruncate"], false);
        // No task_type / parameters bleed when unset.
        let bare = vertex_embed_request(
            &EmbeddingRequest {
                dimensions: None,
                ..req.clone()
            },
            None,
            None,
        );
        assert!(bare["instances"][0].get("task_type").is_none());
        assert!(bare.get("parameters").is_none());
    }

    #[test]
    fn predict_response_parses_values_and_sums_token_count() {
        let resp = json!({
            "predictions": [
                { "embeddings": { "values": [0.1, 0.2, 0.3], "statistics": { "truncated": false, "token_count": 6 } } },
                // token_count may arrive as a JSON string; tolerate it.
                { "embeddings": { "values": [0.4, 0.5], "statistics": { "token_count": "11" } } }
            ],
            "metadata": { "billableCharacterCount": 50 }
        });
        let out = vertex_to_embeddings(resp, "text-embedding-005");
        assert_eq!(out.model, "text-embedding-005");
        assert_eq!(out.embeddings[0].index, 0);
        assert_eq!(out.embeddings[1].index, 1);
        assert_eq!(out.embeddings[0].vector.len(), 3);
        assert!((out.embeddings[1].vector[0] - 0.4).abs() < 1e-6);
        // 6 + 11 summed into prompt/total; embeddings have no completion.
        assert_eq!(out.usage.prompt_tokens, 17);
        assert_eq!(out.usage.total_tokens, 17);
        assert_eq!(out.usage.completion_tokens, 0);
    }

    #[test]
    fn predict_response_without_statistics_keeps_default_usage() {
        let resp = json!({ "predictions": [ { "embeddings": { "values": [0.1] } } ] });
        let out = vertex_to_embeddings(resp, "m");
        assert_eq!(out.usage, Usage::default());
    }

    #[test]
    fn publisher_model_id_strips_prefix_and_leaves_modality_unknown() {
        let entry = json!({
            "name": "publishers/google/models/gemini-2.0-flash",
            "versionId": "001",
            "openSourceCategory": "PROPRIETARY",
            "launchStage": "GA"
        });
        let info = vertex_publisher_to_info(&entry).unwrap();
        assert_eq!(info.id, "gemini-2.0-flash");
        // The list reports no generation methods → never guess modality from the id.
        assert!(info.modality.is_none());
        assert!(info.max_context.is_none());
        // Descriptive fields stashed verbatim.
        assert_eq!(info.extra["versionId"], "001");
        assert_eq!(info.extra["openSourceCategory"], "PROPRIETARY");
        // No `name` → skipped.
        assert!(vertex_publisher_to_info(&json!({ "versionId": "1" })).is_none());
    }
}
