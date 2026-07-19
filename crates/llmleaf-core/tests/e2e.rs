//! End-to-end hot-path test: a request travels authenticate → map in → route → stream → map out →
//! emit events, through a real (inline) provider and the real axum router. Exercises principles 1, 3,
//! 4, 5, verdict enforcement via the pulled overlay, the read-only admin surface, and the absence of
//! any inbound mutation surface — all offline.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use async_trait::async_trait;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use bytes::Bytes;
use http_body_util::BodyExt;
use llmleaf_core::{build_router, build_state, Config, Event, EventBus, KeyStore, Verdict};
use llmleaf_model::{
    AudioChunk, AudioStream, ChatRequest, Embedding, EmbeddingRequest, EmbeddingResponse,
    FinishReason, Modality, ModelError, ModelInfo, RerankRequest, RerankResponse, RerankResult,
    ResponseStream, SpeechRequest, StreamChunk, TranscriptionRequest, TranscriptionResponse, Usage,
    VoiceInfo,
};
use llmleaf_provider::{Provider, ProviderCx, ProviderRegistry};
use serde_json::{json, Value};
use tower::ServiceExt;

/// A minimal inline provider so the test depends only on the extension boundary, not on any provider
/// crate (mirrors how the core itself never names a provider). It supports every modality so the four
/// hot paths can be exercised offline.
struct MockProvider;

#[async_trait]
impl Provider for MockProvider {
    fn name(&self) -> &str {
        "mock"
    }
    async fn chat(&self, req: ChatRequest, _cx: &ProviderCx) -> Result<ResponseStream, ModelError> {
        let reply = format!("reply to: {}", req.messages.last().unwrap().text_content());
        let chunks: Vec<Result<StreamChunk, ModelError>> = vec![
            Ok(StreamChunk::Start {
                id: "mock-1".into(),
                model: req.model.clone(),
            }),
            Ok(StreamChunk::Content {
                index: 0,
                delta: reply,
            }),
            Ok(StreamChunk::Finish {
                index: 0,
                reason: FinishReason::Stop,
            }),
            Ok(StreamChunk::Usage(Usage {
                prompt_tokens: 4,
                completion_tokens: 3,
                total_tokens: 7,
                cost_usd: None,
                cache_read_tokens: 0,
                cache_creation_tokens: 0,
            })),
        ];
        Ok(Box::pin(futures::stream::iter(chunks)))
    }

    async fn embed(
        &self,
        req: EmbeddingRequest,
        _cx: &ProviderCx,
    ) -> Result<EmbeddingResponse, ModelError> {
        let embeddings = req
            .input
            .iter()
            .enumerate()
            .map(|(i, _)| Embedding {
                index: i as u32,
                vector: vec![0.1, 0.2, 0.3],
            })
            .collect();
        Ok(EmbeddingResponse {
            model: req.model,
            embeddings,
            usage: Usage {
                prompt_tokens: 5,
                total_tokens: 5,
                ..Default::default()
            },
        })
    }

    async fn rerank(
        &self,
        req: RerankRequest,
        _cx: &ProviderCx,
    ) -> Result<RerankResponse, ModelError> {
        // A trivial deterministic ranking (mirrors `embed`'s canned-response shape): score each
        // document by its input position — later documents rank higher — then return them most-relevant
        // first. For a 2-document request this yields [{index:1, 0.9}, {index:0, 0.1}], enough to assert
        // order and score at the HTTP edge. Documents are echoed back only when the consumer asked.
        let n = req.documents.len();
        let echo = req.return_documents == Some(true);
        let mut results: Vec<RerankResult> = req
            .documents
            .iter()
            .enumerate()
            .map(|(i, doc)| RerankResult {
                index: i as u32,
                relevance_score: if n > 1 {
                    0.1 + 0.8 * (i as f32 / (n - 1) as f32)
                } else {
                    0.9
                },
                document: echo.then(|| serde_json::to_value(doc).unwrap()),
            })
            .collect();
        // Most relevant first.
        results.sort_by(|a, b| b.relevance_score.partial_cmp(&a.relevance_score).unwrap());
        Ok(RerankResponse {
            model: req.model,
            results,
            usage: Usage {
                total_tokens: 4,
                ..Default::default()
            },
        })
    }

    async fn speech(
        &self,
        req: SpeechRequest,
        _cx: &ProviderCx,
    ) -> Result<AudioStream, ModelError> {
        // Echo the input text as the "audio" payload so the test can assert byte fidelity.
        let chunks: Vec<Result<AudioChunk, ModelError>> = vec![
            Ok(AudioChunk::Start {
                content_type: "audio/mpeg".into(),
            }),
            Ok(AudioChunk::Data(Bytes::from(req.input.into_bytes()))),
            Ok(AudioChunk::Usage(Usage {
                prompt_tokens: 2,
                total_tokens: 2,
                ..Default::default()
            })),
            Ok(AudioChunk::Finish),
        ];
        Ok(Box::pin(futures::stream::iter(chunks)))
    }

    async fn transcribe(
        &self,
        req: TranscriptionRequest,
        _cx: &ProviderCx,
    ) -> Result<TranscriptionResponse, ModelError> {
        Ok(TranscriptionResponse {
            text: format!("transcribed {} bytes", req.audio.len()),
            language: Some("en".into()),
            duration: Some(1.0),
            usage: Usage {
                prompt_tokens: 6,
                total_tokens: 6,
                ..Default::default()
            },
        })
    }

    async fn models(&self, _cx: &ProviderCx) -> Result<Vec<ModelInfo>, ModelError> {
        // A synthetic catalog covering every modality + an id-only entry (to exercise enrichment from
        // the bundled dataset). The listing surface passes these through under the provider's prefix.
        let mut alpha = ModelInfo::new("alpha");
        alpha.name = Some("Alpha".into());
        alpha.modality = Some(Modality::Llm);
        alpha.max_context = Some(64000);
        alpha.max_output = Some(8000);
        // Explicit reasoning support without a published numeric thinking budget (Moonshot shape).
        alpha.supports_reasoning = Some(true);
        alpha.input_per_mtok = Some(1.0);
        alpha.output_per_mtok = Some(2.0);
        let mut beta = ModelInfo::new("beta-embed");
        beta.modality = Some(Modality::Embedding);
        beta.input_per_mtok = Some(0.01);
        beta.output_per_mtok = Some(0.0);
        let mut epsilon = ModelInfo::new("beta-rerank");
        epsilon.modality = Some(Modality::Rerank);
        let mut gamma = ModelInfo::new("gamma-tts");
        gamma.modality = Some(Modality::Tts);
        let mut delta = ModelInfo::new("delta-stt");
        delta.modality = Some(Modality::Stt);
        // id-only — the provider reports nothing; gaps are enhanced from the bundled dataset by bare id.
        let gpt = ModelInfo::new("gpt-4o");
        // id-only reasoning model — the dataset records the sampling params it REJECTS (a reasoning
        // model 400s on temperature) and its pinned defaults; the catalog must surface both.
        let reasoning = ModelInfo::new("gpt-5");
        Ok(vec![alpha, beta, epsilon, gamma, delta, gpt, reasoning])
    }

    async fn voices(&self, _model: &str, _cx: &ProviderCx) -> Result<Vec<VoiceInfo>, ModelError> {
        Ok(vec![
            VoiceInfo {
                id: "alloy".into(),
                name: Some("Alloy".into()),
                languages: vec!["en".into()],
                extra: Default::default(),
            },
            // `name` unset so the test can assert it is omitted from the envelope.
            VoiceInfo {
                id: "echo".into(),
                name: None,
                languages: vec![],
                extra: Default::default(),
            },
        ])
    }
}

/// A provider that only supports chat. Used to prove that a non-chat request falls *through* a
/// chat-only target to the next one in the chain, with no health penalty (it is not a failure).
struct ChatOnlyProvider;

#[async_trait]
impl Provider for ChatOnlyProvider {
    fn name(&self) -> &str {
        "chatonly"
    }
    async fn chat(&self, req: ChatRequest, _cx: &ProviderCx) -> Result<ResponseStream, ModelError> {
        let chunks: Vec<Result<StreamChunk, ModelError>> = vec![
            Ok(StreamChunk::Start {
                id: "co-1".into(),
                model: req.model.clone(),
            }),
            Ok(StreamChunk::Finish {
                index: 0,
                reason: FinishReason::Stop,
            }),
        ];
        Ok(Box::pin(futures::stream::iter(chunks)))
    }
    // embed/speech/transcribe intentionally left as the default `Unsupported`.
}

/// A provider that enumerates a catalog but never tags a modality — i.e. it does NOT support model
/// types. Used to prove the listing surface IGNORES a `?type=` filter for such a provider (its models
/// pass through unfiltered) instead of hiding its whole catalog. Ids are synthetic so the bundled
/// dataset cannot enrich a modality back in.
struct TypelessProvider;

#[async_trait]
impl Provider for TypelessProvider {
    fn name(&self) -> &str {
        "typeless"
    }
    async fn chat(&self, req: ChatRequest, _cx: &ProviderCx) -> Result<ResponseStream, ModelError> {
        let chunks: Vec<Result<StreamChunk, ModelError>> = vec![
            Ok(StreamChunk::Start {
                id: "tl-1".into(),
                model: req.model.clone(),
            }),
            Ok(StreamChunk::Finish {
                index: 0,
                reason: FinishReason::Stop,
            }),
        ];
        Ok(Box::pin(futures::stream::iter(chunks)))
    }
    async fn models(&self, _cx: &ProviderCx) -> Result<Vec<ModelInfo>, ModelError> {
        // No `modality` set on either entry, and the ids are not in the bundled dataset, so they stay
        // `None` — the provider reports no types at all.
        Ok(vec![ModelInfo::new("synth-a"), ModelInfo::new("synth-b")])
    }
}

const CONFIG: &str = r#"
[server]
listen = "127.0.0.1:0"
admin_token = "test-admin"
include_payloads = true

[[providers]]
name = "mock"
kind = "mock"
prefix = "m"

[[providers]]
name = "chatonly"
kind = "chatonly"
prefix = "co"

[[providers]]
name = "typeless"
kind = "typeless"
prefix = "tl"

[[routes]]
model = "demo"
targets = [{ provider = "mock" }]

# A chat-only provider ahead of the full provider: non-chat requests must fall through to `mock`.
[[routes]]
model = "fallback"
targets = [{ provider = "chatonly" }, { provider = "mock" }]

# Consumers present `Bearer base64(id:password)`; config stores only the password hash. This is
# bcrypt of "s3cret" at cost 4 (`htpasswd -bnBC 4 local s3cret`). The token below is base64("local:s3cret").
[[keys]]
id = "local"
pw_hash = "$2y$04$IcVq6nhz5Tf85lBpWclgKeDjWxWMHlIXLE696.T7m9Eg12HekWFJO"
name = "local"
"#;

/// The bearer token a consumer presents for the `local` key: base64("local:s3cret").
const LOCAL_TOKEN: &str = "bG9jYWw6czNjcmV0";

fn app_bus_keys() -> (axum::Router, EventBus, Arc<KeyStore>) {
    let config = Config::from_toml_str(CONFIG).unwrap();
    let mut registry = ProviderRegistry::new();
    registry.register("mock", Arc::new(MockProvider));
    registry.register("chatonly", Arc::new(ChatOnlyProvider));
    registry.register("typeless", Arc::new(TypelessProvider));
    let state = build_state(&config, Arc::new(registry)).unwrap();
    let bus = state.events.clone();
    let keys = state.keys.clone();
    (build_router(state), bus, keys)
}

fn app_and_bus() -> (axum::Router, EventBus) {
    let (app, bus, _keys) = app_bus_keys();
    (app, bus)
}

async fn body_json(resp: axum::response::Response) -> Value {
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    serde_json::from_slice(&bytes).unwrap()
}

async fn body_text(resp: axum::response::Response) -> String {
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    String::from_utf8(bytes.to_vec()).unwrap()
}

async fn body_bytes(resp: axum::response::Response) -> Vec<u8> {
    resp.into_body()
        .collect()
        .await
        .unwrap()
        .to_bytes()
        .to_vec()
}

fn chat_request(model: &str, stream: bool) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("authorization", format!("Bearer {LOCAL_TOKEN}"))
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&json!({
                "model": model,
                "stream": stream,
                "messages": [{ "role": "user", "content": "hello" }]
            }))
            .unwrap(),
        ))
        .unwrap()
}

#[tokio::test]
async fn non_streaming_round_trip() {
    let (app, bus) = app_and_bus();
    let mut events = bus.subscribe();

    let resp = app
        .clone()
        .oneshot(chat_request("demo", false))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let v = body_json(resp).await;

    assert_eq!(v["object"], "chat.completion");
    assert_eq!(v["model"], "demo");
    assert_eq!(v["choices"][0]["message"]["content"], "reply to: hello");
    assert_eq!(v["choices"][0]["finish_reason"], "stop");
    assert_eq!(v["usage"]["total_tokens"], 7);

    // Events flowed out the bus, first being RequestStarted (with payload, since include_payloads).
    let first = events.recv().await.unwrap();
    match &first.event {
        Event::RequestStarted { model, request, .. } => {
            assert_eq!(model, "demo");
            assert!(
                request.is_some(),
                "payload should be present when include_payloads = true"
            );
        }
        other => panic!("expected RequestStarted, got {other:?}"),
    }
}

#[tokio::test]
async fn streaming_emits_sse_frames() {
    let (app, _bus) = app_and_bus();
    let resp = app.oneshot(chat_request("demo", true)).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let ct = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    assert!(ct.starts_with("text/event-stream"), "got {ct}");

    let text = body_text(resp).await;
    assert!(
        text.contains("chat.completion.chunk"),
        "missing chunk frames: {text}"
    );
    assert!(text.contains("reply to: hello"));
    assert!(text.contains("[DONE]"));
}

#[tokio::test]
async fn missing_bearer_is_unauthorized() {
    let (app, _bus) = app_and_bus();
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(r#"{"model":"demo","messages":[]}"#))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn unrouted_model_is_not_found() {
    let (app, _bus) = app_and_bus();
    let resp = app
        .oneshot(chat_request("does-not-exist", false))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

// ---------------------------------------------------------------------------------------------
// Anthropic Messages surface (`POST /v1/messages`) — a second chat dialect on the same core
// ---------------------------------------------------------------------------------------------

/// Build a `POST /v1/messages` request authenticated with the Anthropic `x-api-key` header (the same
/// `local` virtual key, just in a different header).
fn anthropic_request(model: &str, stream: bool) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/v1/messages")
        .header("x-api-key", LOCAL_TOKEN)
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&json!({
                "model": model,
                "max_tokens": 1024,
                "stream": stream,
                "messages": [{ "role": "user", "content": "hello" }]
            }))
            .unwrap(),
        ))
        .unwrap()
}

#[tokio::test]
async fn anthropic_non_streaming_round_trip() {
    let (app, _bus) = app_and_bus();
    let resp = app.oneshot(anthropic_request("demo", false)).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let v = body_json(resp).await;

    assert_eq!(v["type"], "message");
    assert_eq!(v["role"], "assistant");
    assert_eq!(v["model"], "demo");
    assert!(v["id"].as_str().unwrap().starts_with("msg_"));
    assert_eq!(v["content"][0]["type"], "text");
    assert_eq!(v["content"][0]["text"], "reply to: hello");
    assert_eq!(v["stop_reason"], "end_turn");
    assert!(v["stop_sequence"].is_null());
    // Anthropic-shaped usage (input/output tokens), relayed from the provider's 4/3 report.
    assert_eq!(v["usage"]["input_tokens"], 4);
    assert_eq!(v["usage"]["output_tokens"], 3);
}

#[tokio::test]
async fn anthropic_streaming_emits_event_sequence() {
    let (app, _bus) = app_and_bus();
    let resp = app.oneshot(anthropic_request("demo", true)).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let ct = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    assert!(ct.starts_with("text/event-stream"), "got {ct}");

    let text = body_text(resp).await;
    // The Anthropic event lifecycle, in order, carried as named SSE events.
    assert!(
        text.contains("event: message_start"),
        "missing message_start: {text}"
    );
    assert!(text.contains("event: content_block_start"));
    assert!(text.contains("content_block_delta"));
    assert!(text.contains("reply to: hello"));
    assert!(text.contains("event: message_delta"));
    assert!(text.contains("\"stop_reason\":\"end_turn\""));
    assert!(text.contains("event: message_stop"));
    // No OpenAI `[DONE]` sentinel on the Anthropic dialect.
    assert!(!text.contains("[DONE]"));
}

#[tokio::test]
async fn anthropic_accepts_bearer_credential_too() {
    // The surface prefers `x-api-key` but falls back to `Authorization: Bearer` for the same key.
    let (app, _bus) = app_and_bus();
    let req = Request::builder()
        .method("POST")
        .uri("/v1/messages")
        .header("authorization", format!("Bearer {LOCAL_TOKEN}"))
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&json!({
                "model": "demo",
                "max_tokens": 16,
                "messages": [{ "role": "user", "content": "hello" }]
            }))
            .unwrap(),
        ))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(body_json(resp).await["type"], "message");
}

#[tokio::test]
async fn anthropic_missing_credential_is_unauthorized() {
    let (app, _bus) = app_and_bus();
    let req = Request::builder()
        .method("POST")
        .uri("/v1/messages")
        .header("content-type", "application/json")
        .body(Body::from(
            r#"{"model":"demo","max_tokens":16,"messages":[]}"#,
        ))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    // Errors on this surface use the Anthropic envelope, not the OpenAI one.
    let v = body_json(resp).await;
    assert_eq!(v["type"], "error");
    assert_eq!(v["error"]["type"], "authentication_error");
}

#[tokio::test]
async fn anthropic_missing_max_tokens_is_bad_request() {
    // Anthropic requires `max_tokens`; the surface enforces that (fidelity beats convenience).
    let (app, _bus) = app_and_bus();
    let req = Request::builder()
        .method("POST")
        .uri("/v1/messages")
        .header("x-api-key", LOCAL_TOKEN)
        .header("content-type", "application/json")
        .body(Body::from(
            r#"{"model":"demo","messages":[{"role":"user","content":"hi"}]}"#,
        ))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let v = body_json(resp).await;
    assert_eq!(v["error"]["type"], "invalid_request_error");
}

#[tokio::test]
async fn anthropic_unrouted_model_is_not_found() {
    let (app, _bus) = app_and_bus();
    let resp = app
        .oneshot(anthropic_request("does-not-exist", false))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    assert_eq!(body_json(resp).await["error"]["type"], "not_found_error");
}

#[tokio::test]
async fn read_only_admin_requires_token_and_pulled_verdict_blocks_a_key() {
    let (app, _bus, keys) = app_bus_keys();

    // The read-only admin surface still self-guards on the token.
    let unauth = Request::builder()
        .method("GET")
        .uri("/admin/keys")
        .body(Body::empty())
        .unwrap();
    assert_eq!(
        app.clone().oneshot(unauth).await.unwrap().status(),
        StatusCode::UNAUTHORIZED
    );

    // With the token, list the keys (read-only).
    let listed = Request::builder()
        .method("GET")
        .uri("/admin/keys")
        .header("x-admin-token", "test-admin")
        .body(Body::empty())
        .unwrap();
    assert_eq!(
        app.clone().oneshot(listed).await.unwrap().status(),
        StatusCode::OK
    );

    // Install a blocking verdict the way the control plane does — a pulled overlay swap, no inbound
    // HTTP. The hot path then rejects the key by lookup (no arithmetic, just the current verdict).
    keys.install_verdicts(HashMap::from([(
        "local".to_string(),
        Verdict {
            blocked: true,
            ..Default::default()
        },
    )]));
    let blocked = app.oneshot(chat_request("demo", false)).await.unwrap();
    assert_eq!(blocked.status(), StatusCode::FORBIDDEN);
}

#[tokio::test]
async fn inbound_mutation_surfaces_are_gone() {
    let (app, _bus) = app_and_bus();

    // The verdict-push and event-stream surfaces no longer exist — the control plane is outbound only.
    let verdict = Request::builder()
        .method("POST")
        .uri("/admin/keys/verdict")
        .header("x-admin-token", "test-admin")
        .header("content-type", "application/json")
        .body(Body::from(r#"{"id":"local","verdict":{"blocked":true}}"#))
        .unwrap();
    assert_eq!(
        app.clone().oneshot(verdict).await.unwrap().status(),
        StatusCode::NOT_FOUND
    );

    let events = Request::builder()
        .method("GET")
        .uri("/events")
        .header("x-admin-token", "test-admin")
        .body(Body::empty())
        .unwrap();
    assert_eq!(
        app.oneshot(events).await.unwrap().status(),
        StatusCode::NOT_FOUND
    );
}

// ---------------------------------------------------------------------------------------------
// OpenAI Responses surface (`POST /v1/responses`) — a third chat dialect on the same core
// ---------------------------------------------------------------------------------------------

/// Build a `POST /v1/responses` request with the `local` bearer.
fn responses_request(model: &str, stream: bool) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/v1/responses")
        .header("authorization", format!("Bearer {LOCAL_TOKEN}"))
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&json!({
                "model": model,
                "stream": stream,
                "input": "hello",
                "store": true
            }))
            .unwrap(),
        ))
        .unwrap()
}

#[tokio::test]
async fn responses_non_streaming_round_trip() {
    let (app, _bus) = app_and_bus();
    let resp = app.oneshot(responses_request("demo", false)).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let v = body_json(resp).await;

    assert_eq!(v["object"], "response");
    assert_eq!(v["model"], "demo");
    assert_eq!(v["status"], "completed");
    assert!(v["id"].as_str().unwrap().starts_with("resp_"));
    // Stateless: `store: true` in the request is answered `false`.
    assert_eq!(v["store"], false);
    // The visible answer is a `message` output item.
    let output = v["output"].as_array().unwrap();
    let msg = output.iter().find(|i| i["type"] == "message").unwrap();
    assert_eq!(msg["role"], "assistant");
    assert_eq!(msg["content"][0]["type"], "output_text");
    assert_eq!(msg["content"][0]["text"], "reply to: hello");
    // Usage in the Responses shape, relayed from the provider's 4/3/7 report.
    assert_eq!(v["usage"]["input_tokens"], 4);
    assert_eq!(v["usage"]["output_tokens"], 3);
    assert_eq!(v["usage"]["total_tokens"], 7);
}

#[tokio::test]
async fn responses_streaming_emits_event_sequence() {
    let (app, _bus) = app_and_bus();
    let resp = app.oneshot(responses_request("demo", true)).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let ct = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    assert!(ct.starts_with("text/event-stream"), "got {ct}");

    let text = body_text(resp).await;
    // The Responses event lifecycle, in order, carried as named SSE events.
    let order = [
        "event: response.created",
        "event: response.in_progress",
        "event: response.output_item.added",
        "event: response.content_part.added",
        "event: response.output_text.delta",
        "event: response.output_text.done",
        "event: response.content_part.done",
        "event: response.output_item.done",
        "event: response.completed",
    ];
    let mut cursor = 0usize;
    for needle in order {
        let at = text[cursor..]
            .find(needle)
            .unwrap_or_else(|| panic!("missing `{needle}` after position {cursor} in:\n{text}"));
        cursor += at + needle.len();
    }
    // The streamed text and the terminal snapshot's usage both landed.
    assert!(text.contains("reply to: hello"));
    assert!(text.contains("\"output_tokens\":3"));
    // No OpenAI `[DONE]` sentinel on the Responses dialect.
    assert!(!text.contains("[DONE]"));
}

#[tokio::test]
async fn responses_previous_response_id_is_bad_request() {
    // Stateless: continuing from a stored response is unsupported and rejected at the map-in edge.
    let (app, _bus) = app_and_bus();
    let req = Request::builder()
        .method("POST")
        .uri("/v1/responses")
        .header("authorization", format!("Bearer {LOCAL_TOKEN}"))
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&json!({
                "model": "demo",
                "input": "hi",
                "previous_response_id": "resp_abc"
            }))
            .unwrap(),
        ))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let v = body_json(resp).await;
    assert!(v["error"]["message"]
        .as_str()
        .unwrap()
        .contains("previous_response_id"));
}

#[tokio::test]
async fn responses_missing_bearer_is_unauthorized() {
    let (app, _bus) = app_and_bus();
    let req = Request::builder()
        .method("POST")
        .uri("/v1/responses")
        .header("content-type", "application/json")
        .body(Body::from(r#"{"model":"demo","input":"hi"}"#))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn responses_retrieval_is_404_stateless() {
    // Retrieval is unsupported by design; the client is told exactly why (P7).
    let (app, _bus) = app_and_bus();
    let req = Request::builder()
        .method("GET")
        .uri("/v1/responses/resp_xyz")
        .header("authorization", format!("Bearer {LOCAL_TOKEN}"))
        .body(Body::empty())
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    let v = body_json(resp).await;
    let msg = v["error"]["message"].as_str().unwrap();
    assert!(msg.contains("stateless"));
    assert!(msg.contains("store"));

    // Even the always-404 stub authenticates first — the unauthenticated surface is enumerated in
    // SOUL.md (`/healthz`, `/v1/openapi.json`) and this endpoint is not on it.
    let req = Request::builder()
        .method("GET")
        .uri("/v1/responses/resp_xyz")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

// ---------------------------------------------------------------------------------------------
// The new modalities: embeddings, speech (TTS), transcription (STT) — through the real HTTP edge.
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn embeddings_round_trip() {
    let (app, _bus) = app_and_bus();
    let req = Request::builder()
        .method("POST")
        .uri("/v1/embeddings")
        .header("authorization", format!("Bearer {LOCAL_TOKEN}"))
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&json!({ "model": "demo", "input": ["alpha", "beta"] })).unwrap(),
        ))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let v = body_json(resp).await;

    assert_eq!(v["object"], "list");
    assert_eq!(v["model"], "demo");
    assert_eq!(v["data"].as_array().unwrap().len(), 2);
    assert_eq!(v["data"][1]["index"], 1);
    assert!((v["data"][0]["embedding"][0].as_f64().unwrap() - 0.1).abs() < 1e-6);
    assert_eq!(v["usage"]["prompt_tokens"], 5);
}

#[tokio::test]
async fn embeddings_base64_encoding() {
    let (app, _bus) = app_and_bus();
    let req = Request::builder()
        .method("POST")
        .uri("/v1/embeddings")
        .header("authorization", format!("Bearer {LOCAL_TOKEN}"))
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(
                &json!({ "model": "demo", "input": "x", "encoding_format": "base64" }),
            )
            .unwrap(),
        ))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let v = body_json(resp).await;
    // The vector is returned as a base64 string rather than a float array.
    assert!(
        v["data"][0]["embedding"].is_string(),
        "expected base64 string"
    );
}

#[tokio::test]
async fn rerank_round_trip() {
    let (app, _bus) = app_and_bus();
    let req = Request::builder()
        .method("POST")
        .uri("/v1/rerank")
        .header("authorization", format!("Bearer {LOCAL_TOKEN}"))
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&json!({
                "model": "demo",
                "query": "capital of france?",
                "documents": ["berlin is in germany", "paris is the capital"],
            }))
            .unwrap(),
        ))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let v = body_json(resp).await;

    assert_eq!(v["object"], "list");
    assert_eq!(v["model"], "demo");
    let results = v["results"].as_array().unwrap();
    assert_eq!(results.len(), 2);
    // Ranked most-relevant first: the second input document (index 1) tops the list.
    assert_eq!(results[0]["index"], 1);
    assert!((results[0]["relevance_score"].as_f64().unwrap() - 0.9).abs() < 1e-6);
    assert_eq!(results[1]["index"], 0);
    assert!((results[1]["relevance_score"].as_f64().unwrap() - 0.1).abs() < 1e-6);
    // Order is by descending score.
    assert!(
        results[0]["relevance_score"].as_f64().unwrap()
            >= results[1]["relevance_score"].as_f64().unwrap()
    );
    // Documents are not echoed unless requested.
    assert!(results[0].get("document").is_none());
    // Usage is relayed, not computed (principle 5).
    assert_eq!(v["usage"]["total_tokens"], 4);
}

#[tokio::test]
async fn rerank_return_documents() {
    let (app, _bus) = app_and_bus();
    let req = Request::builder()
        .method("POST")
        .uri("/v1/rerank")
        .header("authorization", format!("Bearer {LOCAL_TOKEN}"))
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&json!({
                "model": "demo",
                "query": "capital of france?",
                "documents": ["berlin is in germany", "paris is the capital"],
                "return_documents": true,
            }))
            .unwrap(),
        ))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let v = body_json(resp).await;

    let results = v["results"].as_array().unwrap();
    assert_eq!(results.len(), 2);
    // With `return_documents: true`, every result echoes its document, matched to its input index.
    for r in results {
        assert!(r.get("document").is_some(), "document echoed: {r}");
    }
    assert_eq!(results[0]["index"], 1);
    assert_eq!(results[0]["document"], "paris is the capital");
    assert_eq!(results[1]["index"], 0);
    assert_eq!(results[1]["document"], "berlin is in germany");
}

#[tokio::test]
async fn speech_returns_audio_bytes() {
    let (app, _bus) = app_and_bus();
    let req = Request::builder()
        .method("POST")
        .uri("/v1/audio/speech")
        .header("authorization", format!("Bearer {LOCAL_TOKEN}"))
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(
                &json!({ "model": "demo", "input": "hello audio", "voice": "alloy" }),
            )
            .unwrap(),
        ))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let ct = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert_eq!(ct, "audio/mpeg");
    assert_eq!(body_bytes(resp).await, b"hello audio");
}

#[tokio::test]
async fn voices_lists_for_model() {
    let (app, _bus) = app_and_bus();
    let req = Request::builder()
        .method("GET")
        .uri("/v1/audio/voices?model=demo")
        .header("authorization", format!("Bearer {LOCAL_TOKEN}"))
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let v = body_json(resp).await;
    assert_eq!(v["model"], "demo");
    assert_eq!(v["voices"][0]["id"], "alloy");
    assert_eq!(v["voices"][0]["name"], "Alloy");
    assert_eq!(v["voices"][1]["id"], "echo");
    // `name` was None → omitted from the envelope (skip_serializing_if), not serialized as null.
    assert!(v["voices"][1].get("name").is_none());
}

#[tokio::test]
async fn voices_unrouted_model_is_not_found() {
    let (app, _bus) = app_and_bus();
    let req = Request::builder()
        .method("GET")
        .uri("/v1/audio/voices?model=nope")
        .header("authorization", format!("Bearer {LOCAL_TOKEN}"))
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn transcription_multipart_round_trip() {
    let (app, _bus) = app_and_bus();
    let boundary = "XLLMLEAFBOUNDARY";
    let audio: &[u8] = b"RIFFxxxxWAVE"; // 12 bytes
    let mut body = Vec::new();
    body.extend_from_slice(
        format!("--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\ndemo\r\n")
            .as_bytes(),
    );
    body.extend_from_slice(
        format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\nContent-Type: audio/wav\r\n\r\n"
        )
        .as_bytes(),
    );
    body.extend_from_slice(audio);
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());

    let req = Request::builder()
        .method("POST")
        .uri("/v1/audio/transcriptions")
        .header("authorization", format!("Bearer {LOCAL_TOKEN}"))
        .header(
            "content-type",
            format!("multipart/form-data; boundary={boundary}"),
        )
        .body(Body::from(body))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let v = body_json(resp).await;
    assert_eq!(v["text"], "transcribed 12 bytes");
}

#[tokio::test]
async fn non_chat_falls_through_chat_only_without_penalty() {
    let (app, _bus) = app_and_bus();

    // The `fallback` route starts with a chat-only provider; an embeddings request must skip it
    // (Unsupported) and land on `mock` — no error, a real result.
    let req = Request::builder()
        .method("POST")
        .uri("/v1/embeddings")
        .header("authorization", format!("Bearer {LOCAL_TOKEN}"))
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&json!({ "model": "fallback", "input": "x" })).unwrap(),
        ))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let v = body_json(resp).await;
    assert_eq!(v["data"].as_array().unwrap().len(), 1);

    // Crucially, the chat-only provider must NOT have been penalized — Unsupported is not a failure.
    let health = Request::builder()
        .method("GET")
        .uri("/admin/health")
        .header("x-admin-token", "test-admin")
        .body(Body::empty())
        .unwrap();
    let hv = body_json(app.oneshot(health).await.unwrap()).await;
    for p in hv["providers"].as_array().unwrap() {
        assert_eq!(p["down"], false, "no provider should be penalized: {p}");
    }
}

#[tokio::test]
async fn rerank_falls_through_chat_only_without_penalty() {
    let (app, _bus) = app_and_bus();

    // The `fallback` route starts with a chat-only provider; a rerank request must skip it
    // (Unsupported) and land on `mock` — no error, a real ranked result.
    let req = Request::builder()
        .method("POST")
        .uri("/v1/rerank")
        .header("authorization", format!("Bearer {LOCAL_TOKEN}"))
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&json!({
                "model": "fallback",
                "query": "q",
                "documents": ["a", "b"],
            }))
            .unwrap(),
        ))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let v = body_json(resp).await;
    assert_eq!(v["results"].as_array().unwrap().len(), 2);

    // Crucially, the chat-only provider must NOT have been penalized — Unsupported is not a failure.
    let health = Request::builder()
        .method("GET")
        .uri("/admin/health")
        .header("x-admin-token", "test-admin")
        .body(Body::empty())
        .unwrap();
    let hv = body_json(app.oneshot(health).await.unwrap()).await;
    for p in hv["providers"].as_array().unwrap() {
        assert_eq!(p["down"], false, "no provider should be penalized: {p}");
    }
}

// ---------------------------------------------------------------------------------------------
// Model catalog surface (`GET /v1/models`) — served-only union, OpenRouter shape, views, filters
// ---------------------------------------------------------------------------------------------
//
// The CONFIG above routes `demo`(mock) and `fallback`(chatonly,mock), and gives `mock` the prefix `m`
// and `chatonly` the prefix `co`. So the served set is: the two routes, plus mock's enumerated catalog
// passed through as `m/<id>` (mock implements models()), plus a `co/*` marker (chatonly does NOT
// implement models() → non-enumerable). The bundled pricing dataset is NOT listed — it only enriches.

/// Build a `GET /v1/models<query>` request with the `local` bearer and an optional `x-admin-token`.
fn models_request(query: &str, admin: Option<&str>) -> Request<Body> {
    let mut b = Request::builder()
        .method("GET")
        .uri(format!("/v1/models{query}"))
        .header("authorization", format!("Bearer {LOCAL_TOKEN}"));
    if let Some(t) = admin {
        b = b.header("x-admin-token", t);
    }
    b.body(Body::empty()).unwrap()
}

/// Find the model object with the given id in a `{ "data": [...] }` body.
fn find_model<'a>(body: &'a Value, id: &str) -> Option<&'a Value> {
    body["data"]
        .as_array()?
        .iter()
        .find(|m| m["id"] == json!(id))
}

#[tokio::test]
async fn models_requires_bearer() {
    let (app, _bus) = app_and_bus();
    let req = Request::builder()
        .method("GET")
        .uri("/v1/models")
        .body(Body::empty())
        .unwrap();
    assert_eq!(
        app.oneshot(req).await.unwrap().status(),
        StatusCode::UNAUTHORIZED
    );
}

#[tokio::test]
async fn models_lists_served_not_bundled_catalog() {
    // The key regression: only SERVED models appear — routed ids + prefix passthrough — NOT the bundled
    // pricing dataset. `gpt-4o` is in the dataset but is only reachable here via the `m/` prefix.
    let (app, _bus) = app_and_bus();
    let v = body_json(app.oneshot(models_request("", None)).await.unwrap()).await;
    assert!(v["data"].is_array());
    assert!(v.get("object").is_none()); // OpenRouter envelope, no `object` wrapper
    assert!(find_model(&v, "demo").is_some(), "routed model listed");
    assert!(
        find_model(&v, "m/alpha").is_some(),
        "prefix passthrough listed"
    );
    // A bare bundled-catalog id is NOT served and must be absent; only the prefixed form exists.
    assert!(
        find_model(&v, "gpt-4o").is_none(),
        "bundled catalog must not be an availability source"
    );
    assert!(
        find_model(&v, "m/gpt-4o").is_some(),
        "but it is reachable via the prefix"
    );
}

#[tokio::test]
async fn models_passthrough_openrouter_shape() {
    let (app, _bus) = app_and_bus();
    let v = body_json(app.oneshot(models_request("", None)).await.unwrap()).await;
    let m = find_model(&v, "m/alpha").expect("m/alpha");
    assert!(m.get("object").is_none());
    // Prefix passthroughs tag the display name with the namespace: `[<prefix>] <name>`.
    assert_eq!(m["name"], "[m] Alpha");
    assert_eq!(m["created"], 0);
    assert_eq!(m["context_length"], 64000);
    assert_eq!(m["architecture"]["modality"], "text->text");
    assert_eq!(m["top_provider"]["max_completion_tokens"], 8000);
    // per-Mtok 1.0/2.0 → per-token decimal strings.
    assert_eq!(m["pricing"]["prompt"], "0.000001");
    assert_eq!(m["pricing"]["completion"], "0.000002");
}

#[tokio::test]
async fn models_enriches_sparse_passthrough_from_bundled() {
    // mock reports `gpt-4o` id-only; the gaps fill from the bundled dataset (the provider-specific
    // enhancement for upstreams whose list-models API is sparse).
    let (app, _bus) = app_and_bus();
    let v = body_json(app.oneshot(models_request("", None)).await.unwrap()).await;
    let m = find_model(&v, "m/gpt-4o").expect("m/gpt-4o");
    assert_eq!(m["context_length"], 128000, "filled from bundled dataset");
    assert_eq!(m["architecture"]["modality"], "text->text");
    assert_eq!(m["pricing"]["prompt"], "0.0000025");
}

#[tokio::test]
async fn models_pricing_strings_and_null() {
    let (app, _bus) = app_and_bus();
    let v = body_json(app.oneshot(models_request("", None)).await.unwrap()).await;
    // Embedding: zero output rate renders the literal "0", not null.
    let emb = find_model(&v, "m/beta-embed").unwrap();
    assert_eq!(emb["pricing"]["completion"], "0");
    // A rate-less model (mock gamma-tts has no rates) prices to null, never a "0" object.
    assert_eq!(
        find_model(&v, "m/gamma-tts").unwrap()["pricing"],
        Value::Null
    );
}

#[tokio::test]
async fn models_architecture_modality_mapping() {
    let (app, _bus) = app_and_bus();
    let v = body_json(app.oneshot(models_request("", None)).await.unwrap()).await;
    assert_eq!(
        find_model(&v, "m/alpha").unwrap()["architecture"]["modality"],
        "text->text"
    );
    assert_eq!(
        find_model(&v, "m/beta-embed").unwrap()["architecture"]["modality"],
        "text->embeddings"
    );
    assert_eq!(
        find_model(&v, "m/gamma-tts").unwrap()["architecture"]["modality"],
        "text->audio"
    );
    assert_eq!(
        find_model(&v, "m/delta-stt").unwrap()["architecture"]["modality"],
        "audio->text"
    );
}

#[tokio::test]
async fn models_public_view_hides_provider_and_endpoints() {
    let (app, _bus) = app_and_bus();
    let v = body_json(app.oneshot(models_request("", None)).await.unwrap()).await;
    for m in v["data"].as_array().unwrap() {
        assert!(m.get("endpoints").is_none(), "public leaked endpoints: {m}");
        assert!(m.get("owned_by").is_none());
    }
    // Provider instance names ("mock"/"chatonly") must not appear in the public body (the prefixes are
    // "m"/"co"; ids never contain the provider name).
    let s = v.to_string();
    assert!(!s.contains("mock"), "public leaked a provider name");
    assert!(!s.contains("chatonly"));
}

#[tokio::test]
async fn models_admin_view_adds_endpoints() {
    let (app, _bus) = app_and_bus();
    let v = body_json(
        app.oneshot(models_request("", Some("test-admin")))
            .await
            .unwrap(),
    )
    .await;
    // Routed model: the fallback chain.
    let demo = find_model(&v, "demo").unwrap();
    assert_eq!(demo["endpoints"][0]["provider"], "mock");
    assert_eq!(demo["endpoints"][0]["model"], "demo");
    assert_eq!(demo["endpoints"][0]["source"], "route");
    assert_eq!(demo["endpoints"][0]["down"], false);
    // Prefix passthrough: a single upstream endpoint with the bare model id.
    let alpha = find_model(&v, "m/alpha").unwrap();
    assert_eq!(alpha["endpoints"][0]["provider"], "mock");
    assert_eq!(alpha["endpoints"][0]["model"], "alpha");
    assert_eq!(alpha["endpoints"][0]["source"], "prefix");
}

#[tokio::test]
async fn models_wrong_admin_token_is_public_not_error() {
    let (app, _bus) = app_and_bus();
    let resp = app
        .oneshot(models_request("", Some("wrong")))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let v = body_json(resp).await;
    for m in v["data"].as_array().unwrap() {
        assert!(m.get("endpoints").is_none());
    }
}

#[tokio::test]
async fn models_prefix_marker_admin_only() {
    // chatonly does NOT implement models() → its namespace is non-enumerable → a `co/*` marker, and
    // only in the admin view (markers are pruned from the public catalog).
    let (app, _bus) = app_and_bus();
    let pubv = body_json(app.clone().oneshot(models_request("", None)).await.unwrap()).await;
    assert!(
        find_model(&pubv, "co/*").is_none(),
        "marker hidden from public view"
    );
    let adminv = body_json(
        app.oneshot(models_request("", Some("test-admin")))
            .await
            .unwrap(),
    )
    .await;
    let marker = find_model(&adminv, "co/*").expect("marker in admin view");
    assert_eq!(marker["endpoints"][0]["provider"], "chatonly");
    assert_eq!(marker["endpoints"][0]["source"], "prefix");
    assert_eq!(marker["endpoints"][0]["model"], "*");
}

#[tokio::test]
async fn models_supported_parameters_by_modality() {
    let (app, _bus) = app_and_bus();
    let v = body_json(app.oneshot(models_request("", None)).await.unwrap()).await;
    // A plain LLM advertises the full chat baseline and carries no restriction fields.
    let alpha = find_model(&v, "m/alpha").unwrap();
    let sp = alpha["supported_parameters"].as_array().unwrap();
    assert!(sp.iter().any(|p| p == "temperature"));
    assert!(sp.iter().any(|p| p == "tools"));
    assert!(
        sp.iter().any(|p| p == "reasoning_effort"),
        "an explicit reasoning capability must not require a numeric thinking budget"
    );
    assert!(
        alpha["top_provider"].get("max_thinking_tokens").is_none(),
        "unknown budget remains absent"
    );
    assert!(
        alpha.get("unsupported_parameters").is_none(),
        "no restriction → field omitted"
    );
    assert!(alpha.get("default_parameters").is_none());
    // An embedding model advertises the embedding baseline, never chat params.
    let emb = find_model(&v, "m/beta-embed").unwrap();
    let esp = emb["supported_parameters"].as_array().unwrap();
    assert!(esp.iter().any(|p| p == "dimensions"));
    assert!(!esp.iter().any(|p| p == "temperature"));
}

#[tokio::test]
async fn models_reasoning_model_drops_unsupported_params() {
    // mock reports `gpt-5` id-only; the bundled dataset records it as a reasoning model that REJECTS
    // temperature/top_p/penalties. The catalog must subtract them from supported_parameters, list them
    // under unsupported_parameters, and surface the pinned defaults — so a client never sends gpt-5 a
    // temperature it would 400 on.
    let (app, _bus) = app_and_bus();
    let v = body_json(app.oneshot(models_request("", None)).await.unwrap()).await;
    let m = find_model(&v, "m/gpt-5").expect("m/gpt-5");
    let sp: Vec<&str> = m["supported_parameters"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| p.as_str().unwrap())
        .collect();
    assert!(
        !sp.contains(&"temperature"),
        "temperature subtracted: {sp:?}"
    );
    assert!(!sp.contains(&"top_p"));
    assert!(!sp.contains(&"frequency_penalty"));
    assert!(sp.contains(&"max_tokens"), "non-rejected params remain");
    assert!(sp.contains(&"tools"));
    let unsup: Vec<&str> = m["unsupported_parameters"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| p.as_str().unwrap())
        .collect();
    assert!(unsup.contains(&"temperature") && unsup.contains(&"top_p"));
    assert!(unsup.contains(&"frequency_penalty") && unsup.contains(&"presence_penalty"));
    // supported and unsupported are exact complements within the baseline → never overlap.
    for u in &unsup {
        assert!(
            !sp.contains(u),
            "{u} appears in both supported and unsupported"
        );
    }
    assert_eq!(m["default_parameters"]["temperature"], 1);
    assert_eq!(m["default_parameters"]["top_p"], 1);
}

#[tokio::test]
async fn models_type_filter() {
    let (app, _bus) = app_and_bus();
    let v = body_json(
        app.clone()
            .oneshot(models_request("?type=embedding", None))
            .await
            .unwrap(),
    )
    .await;
    // `mock` classifies by type, so the filter keeps only its embedding model and drops the rest.
    assert!(find_model(&v, "m/beta-embed").is_some());
    assert!(
        find_model(&v, "m/alpha").is_none(),
        "llm excluded by embedding filter"
    );
    assert!(find_model(&v, "m/gamma-tts").is_none());
    let v = body_json(
        app.oneshot(models_request("?type=llm", None))
            .await
            .unwrap(),
    )
    .await;
    assert!(find_model(&v, "m/alpha").is_some());
    assert!(find_model(&v, "m/beta-embed").is_none());
}

#[tokio::test]
async fn models_type_filter_rerank() {
    let (app, _bus) = app_and_bus();
    let v = body_json(
        app.oneshot(models_request("?type=rerank", None))
            .await
            .unwrap(),
    )
    .await;
    // The filter keeps only the rerank model and surfaces its `text->scores` modality; the llm and
    // embedding models are dropped.
    let rr = find_model(&v, "m/beta-rerank").expect("m/beta-rerank");
    assert_eq!(rr["architecture"]["modality"], "text->scores");
    assert!(
        find_model(&v, "m/alpha").is_none(),
        "llm excluded by rerank filter"
    );
    assert!(
        find_model(&v, "m/beta-embed").is_none(),
        "embedding excluded by rerank filter"
    );
}

/// A provider that reports no modality for any model does not support model types, so a `?type=` filter
/// is ignored for it: its whole catalog passes through under every filter rather than being hidden.
#[tokio::test]
async fn models_type_filter_ignored_for_untyped_provider() {
    let (app, _bus) = app_and_bus();
    for kind in ["embedding", "llm", "tts", "stt"] {
        let v = body_json(
            app.clone()
                .oneshot(models_request(&format!("?type={kind}"), None))
                .await
                .unwrap(),
        )
        .await;
        assert!(
            find_model(&v, "tl/synth-a").is_some(),
            "untyped provider's model dropped by ?type={kind}"
        );
        assert!(
            find_model(&v, "tl/synth-b").is_some(),
            "untyped provider's model dropped by ?type={kind}"
        );
    }
}

#[tokio::test]
async fn models_unknown_type_is_400() {
    let (app, _bus) = app_and_bus();
    let resp = app
        .oneshot(models_request("?type=bogus", None))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let v = body_json(resp).await;
    assert!(v["error"]["message"].as_str().unwrap().contains("bogus"));
}

#[tokio::test]
async fn models_search_is_case_insensitive_and_empty_is_200() {
    let (app, _bus) = app_and_bus();
    let v = body_json(
        app.clone()
            .oneshot(models_request("?search=ALPHA", None))
            .await
            .unwrap(),
    )
    .await;
    assert!(find_model(&v, "m/alpha").is_some());
    for m in v["data"].as_array().unwrap() {
        assert!(m["id"]
            .as_str()
            .unwrap()
            .to_ascii_lowercase()
            .contains("alpha"));
    }
    let resp = app
        .oneshot(models_request("?search=zzz-nope", None))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(body_json(resp).await["data"].as_array().unwrap().len(), 0);
}

#[tokio::test]
async fn models_per_key_scoping() {
    let (app, _bus, keys) = app_bus_keys();
    let v = body_json(app.clone().oneshot(models_request("", None)).await.unwrap()).await;
    assert!(v["data"].as_array().unwrap().len() > 1);

    keys.install_verdicts(HashMap::from([(
        "local".to_string(),
        Verdict {
            allowed_models: Some(HashSet::from(["demo".to_string()])),
            ..Default::default()
        },
    )]));
    let v = body_json(app.oneshot(models_request("", None)).await.unwrap()).await;
    let ids: Vec<&str> = v["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["id"].as_str().unwrap())
        .collect();
    assert_eq!(ids, vec!["demo"]);
}

#[tokio::test]
async fn models_deterministic_order() {
    let (app, _bus) = app_and_bus();
    let a = body_text(app.clone().oneshot(models_request("", None)).await.unwrap()).await;
    let b = body_text(app.oneshot(models_request("", None)).await.unwrap()).await;
    assert_eq!(a, b, "id-sorted output must be byte-stable");
}

// ---------------------------------------------------------------------------------------------
// Self-description (`GET /v1/openapi.json`) — the static, public OpenAPI 3.1 contract
// ---------------------------------------------------------------------------------------------

fn openapi_request() -> Request<Body> {
    Request::builder()
        .method("GET")
        .uri("/v1/openapi.json")
        .body(Body::empty())
        .unwrap()
}

#[tokio::test]
async fn openapi_is_public_json() {
    // No bearer: the contract is served credential-free (like /healthz), and as application/json.
    let (app, _bus) = app_and_bus();
    let resp = app.oneshot(openapi_request()).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let ct = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert_eq!(ct, "application/json");
    let v = body_json(resp).await;
    assert_eq!(v["openapi"], "3.1.0");
    assert_eq!(v["info"]["title"], "llmleaf");
    assert!(v["info"]["version"].is_string());
}

#[tokio::test]
async fn openapi_documents_every_served_consumer_path() {
    // Cross-check the document against the routes the core actually mounts: every consumer-facing
    // path the router serves must appear in the spec (and vice versa), so the contract cannot drift
    // silently from `build_router`.
    let (app, _bus) = app_and_bus();
    let v = body_json(app.oneshot(openapi_request()).await.unwrap()).await;
    let documented: HashSet<&str> = v["paths"]
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();

    // The served, publicly-documented surface (the read-only `/admin/*` endpoints are intentionally
    // omitted — they are observability, not part of the consumer contract).
    let expected: HashSet<&str> = HashSet::from([
        "/healthz",
        "/v1/openapi.json",
        "/v1/chat/completions",
        "/v1/messages",
        "/v1/responses",
        "/v1/responses/{id}",
        "/v1/embeddings",
        "/v1/rerank",
        "/v1/audio/speech",
        "/v1/audio/voices",
        "/v1/audio/transcriptions",
        "/v1/models",
        "/v1/realtime",
        "/v1/batches",
        "/v1/batches/{id}",
        "/v1/batches/{id}/cancel",
        "/v1/batches/{id}/results",
    ]);
    assert_eq!(
        documented, expected,
        "openapi paths drifted from the router"
    );
}

#[tokio::test]
async fn openapi_security_and_components_are_sound() {
    let (app, _bus) = app_and_bus();
    let v = body_json(app.oneshot(openapi_request()).await.unwrap()).await;

    // Bearer is the default scheme; the two public endpoints opt out with an empty requirement.
    assert_eq!(v["security"][0]["bearerAuth"], json!([]));
    assert!(v["components"]["securitySchemes"]["bearerAuth"].is_object());
    assert_eq!(v["paths"]["/healthz"]["get"]["security"], json!([]));
    assert_eq!(v["paths"]["/v1/openapi.json"]["get"]["security"], json!([]));
    assert!(v["paths"]["/v1/chat/completions"]["post"]
        .get("security")
        .is_none());

    // Every `$ref` resolves to a defined component schema (no dangling references).
    let schemas = v["components"]["schemas"].as_object().unwrap();
    let text = v.to_string();
    for cap in text.split("#/components/schemas/").skip(1) {
        let name: String = cap
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric())
            .collect();
        assert!(
            schemas.contains_key(&name),
            "dangling $ref to schema '{name}'"
        );
    }
}
