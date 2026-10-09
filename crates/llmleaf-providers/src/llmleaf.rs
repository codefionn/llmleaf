//! Another llmleaf node as an upstream (`kind = "llmleaf"`).
//!
//! An llmleaf node's consumer surface is the OpenAI wire, so chat, embeddings, rerank, speech,
//! transcription, the model catalog, and realtime ride the shared `llmleaf` brand row in
//! [`crate::compat`]. Three operations use llmleaf's own paths and shapes, and this wrapper owns them:
//!   - decisions: `POST /decisions` (OpenRouter's body shape, but not its `/alpha` sibling path);
//!   - voices: `GET /audio/voices?model=` returning `{ model, voices: [VoiceInfo] }`;
//!   - batch: inline `POST /batches { requests: [{ custom_id, body }] }`, `GET /batches/{id}`,
//!     `POST /batches/{id}/cancel`, and JSONL from `GET /batches/{id}/results`. There is no file
//!     upload step, and request counts carry llmleaf's processing/canceled/expired superset.
//!
//! Realtime dials the upstream node's `/realtime` WebSocket directly, so a voice or text session stays
//! one socket end to end. If that dial fails before the first frame (the hop strips upgrades, or the
//! upstream is an older node), the core falls back to its chat bridge, which then reaches the
//! upstream over plain HTTP chat.
//!
//! Nothing stops an operator from pointing a node at itself. That loops until a timeout, so the
//! endpoint must name a different node.

use async_trait::async_trait;
use llmleaf_model::{
    AudioStream, BatchHandle, BatchOutcome, BatchResult, BatchResultStream, BatchSpec, ChatRequest,
    DecisionsRequest, DecisionsResponse, EmbeddingRequest, EmbeddingResponse, ModelError,
    ModelInfo, RerankRequest, RerankResponse, ResponseStream, SpeechRequest, TranscriptionRequest,
    TranscriptionResponse, VoiceInfo,
};
use llmleaf_provider::{Provider, ProviderCx, RealtimeParams, RealtimePeer};
use serde_json::{json, Value};

use std::sync::Arc;

use crate::batch::jsonl_result_stream;
use crate::compat::{batch_value_to_handle, openai_batch_result_line, Brand, OpenAiCompatProvider};
use crate::decisions::{decisions_request_to_wire, decisions_response_from_wire};
use crate::http::{post_json, send_checked};
use crate::openai_wire::request_to_openai;
use crate::transport::{HttpRequest, HttpTransport, Transports};

pub struct LlmleafProvider {
    compat: OpenAiCompatProvider,
    http: Arc<dyn HttpTransport>,
    default_endpoint: &'static str,
}

impl LlmleafProvider {
    pub fn new(transports: &Transports) -> Self {
        let brand = Brand::for_kind("llmleaf").expect("llmleaf brand is registered");
        Self {
            compat: OpenAiCompatProvider::new(brand, transports),
            http: transports.http.clone(),
            default_endpoint: brand.default_endpoint,
        }
    }

    /// The upstream node's `/v1` base: the config endpoint, else the brand default.
    fn base(&self, cx: &ProviderCx) -> String {
        cx.endpoint
            .as_deref()
            .unwrap_or(self.default_endpoint)
            .trim_end_matches('/')
            .to_string()
    }

    fn authed(&self, req: HttpRequest, cx: &ProviderCx) -> HttpRequest {
        match &cx.credential {
            Some(credential) => req.bearer(credential),
            None => req,
        }
    }

    fn batch_url(&self, cx: &ProviderCx, upstream_id: &str, suffix: &str) -> String {
        format!(
            "{}/batches/{}{suffix}",
            self.base(cx),
            encode_component(upstream_id)
        )
    }
}

#[async_trait]
impl Provider for LlmleafProvider {
    fn name(&self) -> &str {
        "llmleaf"
    }

    async fn chat(&self, req: ChatRequest, cx: &ProviderCx) -> Result<ResponseStream, ModelError> {
        self.compat.chat(req, cx).await.map_err(reclassify)
    }

    async fn embed(
        &self,
        req: EmbeddingRequest,
        cx: &ProviderCx,
    ) -> Result<EmbeddingResponse, ModelError> {
        self.compat.embed(req, cx).await.map_err(reclassify)
    }

    async fn rerank(
        &self,
        req: RerankRequest,
        cx: &ProviderCx,
    ) -> Result<RerankResponse, ModelError> {
        self.compat.rerank(req, cx).await.map_err(reclassify)
    }

    async fn decisions(
        &self,
        req: DecisionsRequest,
        cx: &ProviderCx,
    ) -> Result<DecisionsResponse, ModelError> {
        let url = format!("{}/decisions", self.base(cx));
        let request = self.authed(
            HttpRequest::post(url).json(decisions_request_to_wire(&req)),
            cx,
        );
        let response = post_json(&*self.http, request).await.map_err(reclassify)?;
        decisions_response_from_wire(response, &req.model)
    }

    async fn speech(&self, req: SpeechRequest, cx: &ProviderCx) -> Result<AudioStream, ModelError> {
        self.compat.speech(req, cx).await.map_err(reclassify)
    }

    /// The upstream node answers per model with the voices its own provider resolved, already in the
    /// canonical shape.
    async fn voices(&self, model: &str, cx: &ProviderCx) -> Result<Vec<VoiceInfo>, ModelError> {
        let url = format!(
            "{}/audio/voices?model={}",
            self.base(cx),
            encode_component(model)
        );
        let mut value = post_json(&*self.http, self.authed(HttpRequest::get(url), cx))
            .await
            .map_err(reclassify)?;
        let voices = value
            .get_mut("voices")
            .map(Value::take)
            .ok_or_else(|| ModelError::Mapping("voices response had no `voices` array".into()))?;
        serde_json::from_value(voices).map_err(|e| ModelError::Mapping(e.to_string()))
    }

    async fn models(&self, cx: &ProviderCx) -> Result<Vec<ModelInfo>, ModelError> {
        let mut models = self.compat.models(cx).await.map_err(reclassify)?;
        // A prefix route's non-enumerable namespace is listed as `<prefix>/*`. It is not a model id.
        models.retain(|m| !m.id.ends_with("/*"));
        for info in &mut models {
            lift_llmleaf_catalog_fields(info);
        }
        Ok(models)
    }

    fn supports_compaction(&self, model: &str, cx: &ProviderCx) -> Option<bool> {
        self.compat.supports_compaction(model, cx)
    }

    async fn transcribe(
        &self,
        req: TranscriptionRequest,
        cx: &ProviderCx,
    ) -> Result<TranscriptionResponse, ModelError> {
        self.compat.transcribe(req, cx).await.map_err(reclassify)
    }

    /// Batch items go inline as chat-completions bodies, built by the same mapper live chat uses.
    async fn batch_create(
        &self,
        req: BatchSpec,
        cx: &ProviderCx,
    ) -> Result<BatchHandle, ModelError> {
        let requests: Vec<Value> = req
            .items
            .iter()
            .map(|item| {
                json!({
                    "custom_id": item.custom_id,
                    "body": request_to_openai(&item.request, "max_completion_tokens", false),
                })
            })
            .collect();
        let url = format!("{}/batches", self.base(cx));
        let request = self.authed(
            HttpRequest::post(url).json(json!({ "requests": requests })),
            cx,
        );
        let value = post_json(&*self.http, request).await.map_err(reclassify)?;
        Ok(llmleaf_batch_handle(&value))
    }

    async fn batch_retrieve(
        &self,
        upstream_id: &str,
        cx: &ProviderCx,
    ) -> Result<BatchHandle, ModelError> {
        let url = self.batch_url(cx, upstream_id, "");
        let value = post_json(&*self.http, self.authed(HttpRequest::get(url), cx))
            .await
            .map_err(reclassify)?;
        Ok(llmleaf_batch_handle(&value))
    }

    async fn batch_results(
        &self,
        upstream_id: &str,
        cx: &ProviderCx,
    ) -> Result<BatchResultStream, ModelError> {
        let url = self.batch_url(cx, upstream_id, "/results");
        let resp = send_checked(&*self.http, self.authed(HttpRequest::get(url), cx))
            .await
            .map_err(reclassify)?;
        Ok(jsonl_result_stream(resp.body, llmleaf_batch_result_line))
    }

    async fn batch_cancel(
        &self,
        upstream_id: &str,
        cx: &ProviderCx,
    ) -> Result<BatchHandle, ModelError> {
        let url = self.batch_url(cx, upstream_id, "/cancel");
        let value = post_json(&*self.http, self.authed(HttpRequest::post(url), cx))
            .await
            .map_err(reclassify)?;
        Ok(llmleaf_batch_handle(&value))
    }

    fn supports_realtime(&self) -> bool {
        self.compat.supports_realtime()
    }

    async fn realtime(
        &self,
        params: RealtimeParams,
        peer: RealtimePeer,
        cx: &ProviderCx,
    ) -> Result<(), ModelError> {
        self.compat
            .realtime(params, peer, cx)
            .await
            .map_err(reclassify)
    }
}

/// When every target on the upstream node lacks a capability, that node answers 502 with an
/// `unsupported: …` message. Turn it back into [`ModelError::Unsupported`] so routing here falls past
/// this provider without a health penalty, the same as for any provider without the modality.
fn reclassify(err: ModelError) -> ModelError {
    let ModelError::Upstream {
        status: 502,
        message,
    } = &err
    else {
        return err;
    };
    let reason = serde_json::from_str::<Value>(message)
        .ok()
        .and_then(|body| {
            body.pointer("/error/message")
                .and_then(Value::as_str)
                .and_then(|m| m.strip_prefix("unsupported: "))
                .map(str::to_string)
        });
    match reason {
        Some(reason) => ModelError::Unsupported(reason),
        None => err,
    }
}

/// llmleaf's `/models` adds a few fields to the OpenRouter shape. The shared parser keeps them in
/// `extra`; move the ones with typed homes there.
fn lift_llmleaf_catalog_fields(info: &mut ModelInfo) {
    if info.max_thinking.is_none() {
        info.max_thinking = info
            .extra
            .get("top_provider")
            .and_then(|tp| tp.get("max_thinking_tokens"))
            .and_then(Value::as_u64)
            .map(|n| n.min(u32::MAX as u64) as u32);
    }
    if let Some(Value::Array(list)) = info.extra.remove("unsupported_parameters") {
        info.unsupported_parameters = list
            .into_iter()
            .filter_map(|v| match v {
                Value::String(s) => Some(s),
                _ => None,
            })
            .collect();
    }
}

/// The OpenAI-shaped batch object, plus llmleaf's processing/canceled/expired counts when present.
fn llmleaf_batch_handle(value: &Value) -> BatchHandle {
    let mut handle = batch_value_to_handle(value);
    if let Some(rc) = value.get("request_counts") {
        let count = |k: &str| rc.get(k).and_then(Value::as_u64);
        if let Some(n) = count("processing") {
            handle.counts.processing = n;
        }
        if let Some(n) = count("canceled") {
            handle.counts.canceled = n;
        }
        if let Some(n) = count("expired") {
            handle.counts.expired = n;
        }
    }
    handle
}

/// One llmleaf result line. Successes are OpenAI-shaped. Failures carry the upstream HTTP status as a
/// string `code`, or the literal `canceled`/`expired`, which map back to their own outcomes.
fn llmleaf_batch_result_line(value: Value) -> Option<BatchResult> {
    let Some(err) = value.get("error").filter(|e| !e.is_null()) else {
        return openai_batch_result_line(value);
    };
    let custom_id = value.get("custom_id")?.as_str()?.to_string();
    let outcome = match err.get("code").and_then(Value::as_str) {
        Some("canceled") => BatchOutcome::Canceled,
        Some("expired") => BatchOutcome::Expired,
        code => BatchOutcome::Errored {
            status: code.and_then(|c| c.parse().ok()).unwrap_or(0),
            message: err
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
        },
    };
    Some(BatchResult { custom_id, outcome })
}

/// Percent-encode a path segment or query value. Model ids carry `/` and `:`, batch ids are opaque.
fn encode_component(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~') {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use futures::StreamExt;
    use llmleaf_model::{collect, BatchItem, BatchStatus, ContentPart, Message, Role};
    use serde_json::{json, Map};

    use super::*;
    use crate::fake::{FakeHttpTransport, FakeRealtimeTransport, FakeResponse};
    use crate::transport::{HttpBody, Method};

    fn transports(http: FakeHttpTransport) -> Transports {
        Transports {
            http: Arc::new(http),
            realtime: Arc::new(FakeRealtimeTransport::scripted(Vec::new())),
        }
    }

    fn cx() -> ProviderCx {
        ProviderCx {
            credential: Some("leaf-key".into()),
            endpoint: Some("http://upstream:9000/v1/".into()),
            settings: serde_json::from_value(json!({ "upstream_streaming": "never" })).unwrap(),
            ..Default::default()
        }
    }

    fn has_bearer(req: &HttpRequest) -> bool {
        req.headers
            .iter()
            .any(|(name, value)| name == "Authorization" && value == "Bearer leaf-key")
    }

    fn json_body(req: &HttpRequest) -> &Value {
        let HttpBody::Json(body) = &req.body else {
            panic!("expected a JSON body")
        };
        body
    }

    fn chat_request(messages: Vec<Message>) -> ChatRequest {
        ChatRequest {
            model: "claude-sonnet".into(),
            messages,
            max_tokens: Some(64),
            temperature: None,
            top_p: None,
            stop: vec![],
            stream: false,
            tools: vec![],
            tool_choice: None,
            thinking: None,
            extra: Map::new(),
        }
    }

    #[test]
    fn factory_registers_the_llmleaf_kind() {
        let provider = crate::build("llmleaf", &Transports::fake()).expect("known kind");
        assert_eq!(provider.name(), "llmleaf");
        assert!(provider.supports_realtime());
        assert!(crate::known_kinds().contains(&"llmleaf"));
    }

    #[tokio::test]
    async fn chat_uses_responses_and_replays_signed_reasoning() {
        let http = FakeHttpTransport::new(|req| {
            assert_eq!(req.method, Method::Post);
            assert_eq!(req.url, "http://upstream:9000/v1/responses");
            assert!(has_bearer(req));
            let body = json_body(req);
            let reasoning = body["input"]
                .as_array()
                .unwrap()
                .iter()
                .find(|item| item["type"] == "reasoning")
                .expect("signed thinking replays as a reasoning item");
            assert_eq!(reasoning["signature"], "SIG");
            Ok(FakeResponse::ok_json(&json!({
                "id": "resp_1",
                "object": "response",
                "model": "claude-sonnet",
                "status": "completed",
                "output": [
                    {
                        "type": "reasoning",
                        "summary": [],
                        "content": [{ "type": "reasoning_text", "text": "think" }],
                        "signature": "SIG2"
                    },
                    {
                        "type": "message",
                        "content": [{ "type": "output_text", "text": "done" }]
                    }
                ],
                "usage": { "input_tokens": 3, "output_tokens": 2, "total_tokens": 5 }
            })))
        });
        let provider = LlmleafProvider::new(&transports(http));
        let mut assistant = Message::text(Role::Assistant, "earlier answer");
        assistant.content.insert(
            0,
            ContentPart::Thinking {
                thinking: "earlier thought".into(),
                signature: Some("SIG".into()),
            },
        );
        let req = chat_request(vec![
            Message::text(Role::User, "hi"),
            assistant,
            Message::text(Role::User, "again"),
        ]);

        let response = collect(provider.chat(req, &cx()).await.unwrap())
            .await
            .unwrap();
        assert_eq!(response.choices[0].text, "done");
        assert!(response.choices[0].thinking.iter().any(|part| matches!(
            part,
            ContentPart::Thinking { signature: Some(sig), .. } if sig == "SIG2"
        )));
    }

    #[tokio::test]
    async fn stop_sequences_fall_back_to_chat_completions() {
        let http = FakeHttpTransport::new(|req| {
            assert_eq!(req.url, "http://upstream:9000/v1/chat/completions");
            assert_eq!(json_body(req)["max_completion_tokens"], 64);
            Ok(FakeResponse::ok_json(&json!({
                "id": "c1",
                "model": "claude-sonnet",
                "choices": [{
                    "index": 0,
                    "message": { "role": "assistant", "content": "ok" },
                    "finish_reason": "stop"
                }],
                "usage": { "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2 }
            })))
        });
        let provider = LlmleafProvider::new(&transports(http));
        let mut req = chat_request(vec![Message::text(Role::User, "hi")]);
        req.stop = vec!["END".into()];
        let response = collect(provider.chat(req, &cx()).await.unwrap())
            .await
            .unwrap();
        assert_eq!(response.choices[0].text, "ok");
    }

    #[tokio::test]
    async fn decisions_post_to_the_native_path() {
        let http = FakeHttpTransport::new(|req| {
            assert_eq!(req.url, "http://upstream:9000/v1/decisions");
            assert!(has_bearer(req));
            assert_eq!(json_body(req)["model"], "jev-latest");
            Ok(FakeResponse::ok_json(&json!({
                "model": "jev-latest",
                "answers": { "q": true },
                "usage": { "input_tokens": 2, "output_tokens": 1 }
            })))
        });
        let provider = LlmleafProvider::new(&transports(http));
        let req = DecisionsRequest {
            model: "jev-latest".into(),
            state: json!("state"),
            questions: serde_json::from_value(json!({ "q": { "type": "boolean" } })).unwrap(),
            extra: Map::new(),
        };
        let response = provider.decisions(req, &cx()).await.unwrap();
        assert_eq!(response.answers["q"], true);
    }

    #[tokio::test]
    async fn voices_query_the_model_and_parse_canonical_entries() {
        let http = FakeHttpTransport::new(|req| {
            assert_eq!(req.method, Method::Get);
            assert_eq!(
                req.url,
                "http://upstream:9000/v1/audio/voices?model=openai%2Fgpt-4o-mini-tts"
            );
            assert!(has_bearer(req));
            Ok(FakeResponse::ok_json(&json!({
                "model": "openai/gpt-4o-mini-tts",
                "voices": [
                    { "id": "alloy" },
                    { "id": "nia", "name": "Nia", "languages": ["en", "sw"] }
                ]
            })))
        });
        let provider = LlmleafProvider::new(&transports(http));
        let voices = provider
            .voices("openai/gpt-4o-mini-tts", &cx())
            .await
            .unwrap();
        assert_eq!(voices.len(), 2);
        assert_eq!(voices[1].name.as_deref(), Some("Nia"));
        assert_eq!(voices[1].languages, ["en", "sw"]);
    }

    #[tokio::test]
    async fn models_drop_namespace_markers_and_lift_llmleaf_fields() {
        let http = FakeHttpTransport::new(|req| {
            assert_eq!(req.url, "http://upstream:9000/v1/models");
            assert!(has_bearer(req));
            Ok(FakeResponse::ok_json(&json!({ "data": [
                {
                    "id": "smart",
                    "name": "smart",
                    "context_length": 200000,
                    "architecture": {
                        "input_modalities": ["text"],
                        "output_modalities": ["text"],
                        "modality": "text->text"
                    },
                    "pricing": { "prompt": "0.000003", "completion": "0.000015" },
                    "top_provider": {
                        "context_length": 200000,
                        "max_completion_tokens": 64000,
                        "max_thinking_tokens": 32000
                    },
                    "unsupported_parameters": ["temperature"]
                },
                { "id": "openrouter/*", "architecture": { "modality": null } }
            ] })))
        });
        let provider = LlmleafProvider::new(&transports(http));
        let models = provider.models(&cx()).await.unwrap();
        assert_eq!(models.len(), 1);
        let smart = &models[0];
        assert_eq!(smart.id, "smart");
        assert_eq!(smart.max_context, Some(200000));
        assert_eq!(smart.max_output, Some(64000));
        assert_eq!(smart.max_thinking, Some(32000));
        assert_eq!(smart.unsupported_parameters, ["temperature"]);
        assert_eq!(smart.input_per_mtok, Some(3.0));
    }

    #[tokio::test]
    async fn batch_create_sends_inline_requests() {
        let http = FakeHttpTransport::new(|req| {
            assert_eq!(req.method, Method::Post);
            assert_eq!(req.url, "http://upstream:9000/v1/batches");
            assert!(has_bearer(req));
            let body = json_body(req);
            assert_eq!(body["requests"][0]["custom_id"], "a");
            assert_eq!(body["requests"][0]["body"]["model"], "claude-sonnet");
            assert_eq!(body["requests"][0]["body"]["stream"], false);
            Ok(FakeResponse::ok_json(&json!({
                "id": "lb_upstream",
                "object": "batch",
                "status": "in_progress",
                "request_counts": {
                    "total": 3, "completed": 1, "failed": 0,
                    "processing": 1, "canceled": 1, "expired": 0
                },
                "created_at": 10
            })))
        });
        let provider = LlmleafProvider::new(&transports(http));
        let spec = BatchSpec {
            items: vec![BatchItem {
                custom_id: "a".into(),
                request: chat_request(vec![Message::text(Role::User, "hi")]),
            }],
        };
        let handle = provider.batch_create(spec, &cx()).await.unwrap();
        assert_eq!(handle.id, "lb_upstream");
        assert_eq!(handle.status, BatchStatus::InProgress);
        assert_eq!(handle.counts.total, 3);
        assert_eq!(handle.counts.succeeded, 1);
        assert_eq!(handle.counts.processing, 1);
        assert_eq!(handle.counts.canceled, 1);
        assert_eq!(handle.created_at, Some(10));
    }

    #[tokio::test]
    async fn batch_id_is_encoded_into_retrieve_and_cancel_paths() {
        let http = FakeHttpTransport::new(|req| {
            let expected = match req.method {
                Method::Get => "http://upstream:9000/v1/batches/a%2Bb%3D",
                _ => "http://upstream:9000/v1/batches/a%2Bb%3D/cancel",
            };
            assert_eq!(req.url, expected);
            Ok(FakeResponse::ok_json(&json!({
                "id": "a+b=",
                "status": "cancelled",
                "request_counts": { "total": 1, "completed": 0, "failed": 0 }
            })))
        });
        let provider = LlmleafProvider::new(&transports(http));
        let retrieved = provider.batch_retrieve("a+b=", &cx()).await.unwrap();
        assert_eq!(retrieved.status, BatchStatus::Canceled);
        let canceled = provider.batch_cancel("a+b=", &cx()).await.unwrap();
        assert_eq!(canceled.id, "a+b=");
    }

    #[tokio::test]
    async fn batch_results_map_every_llmleaf_outcome() {
        let lines = [
            json!({
                "custom_id": "ok",
                "error": null,
                "response": {
                    "status_code": 200,
                    "body": {
                        "id": "c1",
                        "model": "m",
                        "choices": [{
                            "index": 0,
                            "message": { "role": "assistant", "content": "hi" },
                            "finish_reason": "stop"
                        }]
                    }
                }
            }),
            json!({ "custom_id": "limited", "error": { "code": "429", "message": "slow down" }, "response": null }),
            json!({ "custom_id": "stopped", "error": { "code": "canceled", "message": "x" }, "response": null }),
            json!({ "custom_id": "late", "error": { "code": "expired", "message": "x" }, "response": null }),
        ]
        .iter()
        .map(|line| format!("{line}\n"))
        .collect::<String>();
        let http = FakeHttpTransport::new(move |req| {
            assert_eq!(req.url, "http://upstream:9000/v1/batches/lb_1/results");
            Ok(FakeResponse::ok_bytes("application/jsonl", lines.clone()))
        });
        let provider = LlmleafProvider::new(&transports(http));
        let results: Vec<BatchResult> = provider
            .batch_results("lb_1", &cx())
            .await
            .unwrap()
            .map(Result::unwrap)
            .collect()
            .await;
        assert_eq!(results.len(), 4);
        assert!(
            matches!(&results[0].outcome, BatchOutcome::Succeeded(r) if r.choices[0].text == "hi")
        );
        assert!(matches!(
            &results[1].outcome,
            BatchOutcome::Errored { status: 429, message } if message == "slow down"
        ));
        assert_eq!(results[2].outcome, BatchOutcome::Canceled);
        assert_eq!(results[3].outcome, BatchOutcome::Expired);
    }

    #[tokio::test]
    async fn upstream_unsupported_is_reclassified_and_other_502s_are_not() {
        let unsupported = LlmleafProvider::new(&transports(FakeHttpTransport::status(
            502,
            r#"{"error":{"message":"unsupported: provider 'echo' does not support rerank"}}"#,
        )));
        let req = RerankRequest {
            model: "r".into(),
            query: "q".into(),
            documents: vec![],
            top_n: None,
            return_documents: None,
            extra: Map::new(),
        };
        assert!(matches!(
            unsupported.rerank(req.clone(), &cx()).await,
            Err(ModelError::Unsupported(m)) if m == "provider 'echo' does not support rerank"
        ));

        let failed = LlmleafProvider::new(&transports(FakeHttpTransport::status(
            502,
            r#"{"error":{"message":"upstream 500: boom"}}"#,
        )));
        assert!(matches!(
            failed.rerank(req, &cx()).await,
            Err(ModelError::Upstream { status: 502, .. })
        ));
    }

    /// The URL and headers of one dial.
    type Dial = (String, Vec<(String, String)>);

    /// Records where the provider dials instead of opening a socket.
    struct RecordingRealtime(std::sync::Mutex<Option<Dial>>);

    #[async_trait]
    impl crate::transport::RealtimeTransport for RecordingRealtime {
        async fn run(
            &self,
            url: String,
            headers: Vec<(String, String)>,
            _peer: RealtimePeer,
        ) -> Result<(), ModelError> {
            *self.0.lock().unwrap() = Some((url, headers));
            Ok(())
        }
    }

    #[tokio::test]
    async fn realtime_dials_the_upstream_node_socket_with_the_bearer() {
        let realtime = Arc::new(RecordingRealtime(std::sync::Mutex::new(None)));
        let provider = LlmleafProvider::new(&Transports {
            http: Arc::new(FakeHttpTransport::status(500, "unused")),
            realtime: realtime.clone(),
        });
        let (_in_tx, inbound) = tokio::sync::mpsc::channel(1);
        let (outbound, _out_rx) = tokio::sync::mpsc::channel(1);
        provider
            .realtime(
                RealtimeParams {
                    model: "gpt-realtime".into(),
                },
                RealtimePeer { inbound, outbound },
                &cx(),
            )
            .await
            .unwrap();
        let (url, headers) = realtime.0.lock().unwrap().take().expect("dialed");
        assert_eq!(url, "ws://upstream:9000/v1/realtime?model=gpt-realtime");
        assert!(headers
            .iter()
            .any(|(name, value)| name == "Authorization" && value == "Bearer leaf-key"));
    }
}
