//! End-to-end batch test: submit → poll → results → cancel through the real axum router and engine,
//! with an inline provider (no provider crate). Proves the batch-specific properties:
//!   - the id handed back is opaque and provider-encoding (principle 9 — see the multi-node test),
//!   - retrieve/cancel/results route purely by that id, with no node-local batch state,
//!   - a batch whose models span providers is rejected (one upstream job),
//!   - per-result usage is relayed out the event bus (principle 5),
//!   - the no-model surfaces authenticate identity only, while create gates every model.

use std::sync::Arc;

use async_trait::async_trait;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use llmleaf_core::{build_router, build_state, AppState, Config, Event, EventBus};
use llmleaf_model::{
    BatchCounts, BatchHandle, BatchOutcome, BatchResult, BatchResultStream, BatchSpec, BatchStatus,
    ChatRequest, ChatResponse, Choice, FinishReason, ModelError, ResponseStream, StreamChunk,
    Usage,
};
use llmleaf_provider::{Provider, ProviderCx, ProviderRegistry};
use serde_json::{json, Value};
use tower::ServiceExt;

/// An inline provider whose batch methods are *stateless* — every call returns a fixed shape derived
/// only from its arguments. That is exactly what proves the design: if a batch can be retrieved with no
/// memory of having created it, then any node can serve it from the id alone (principle 9).
struct MockBatchProvider {
    name: &'static str,
}

#[async_trait]
impl Provider for MockBatchProvider {
    fn name(&self) -> &str {
        self.name
    }

    async fn chat(&self, req: ChatRequest, _cx: &ProviderCx) -> Result<ResponseStream, ModelError> {
        let chunks: Vec<Result<StreamChunk, ModelError>> = vec![
            Ok(StreamChunk::Start {
                id: "c-1".into(),
                model: req.model,
            }),
            Ok(StreamChunk::Finish {
                index: 0,
                reason: FinishReason::Stop,
            }),
        ];
        Ok(Box::pin(futures::stream::iter(chunks)))
    }

    async fn batch_create(
        &self,
        req: BatchSpec,
        _cx: &ProviderCx,
    ) -> Result<BatchHandle, ModelError> {
        Ok(BatchHandle {
            id: "up-batch".into(),
            status: BatchStatus::InProgress,
            counts: BatchCounts {
                total: req.items.len() as u64,
                processing: req.items.len() as u64,
                ..Default::default()
            },
            created_at: Some(1000),
            expires_at: Some(2000),
            ended_at: None,
            endpoint: Some("/v1/chat/completions".into()),
            extra: Default::default(),
        })
    }

    async fn batch_retrieve(
        &self,
        upstream_id: &str,
        _cx: &ProviderCx,
    ) -> Result<BatchHandle, ModelError> {
        Ok(BatchHandle {
            id: upstream_id.to_string(),
            status: BatchStatus::Completed,
            counts: BatchCounts {
                total: 2,
                succeeded: 2,
                ..Default::default()
            },
            created_at: Some(1000),
            expires_at: Some(2000),
            ended_at: Some(1500),
            endpoint: Some("/v1/chat/completions".into()),
            extra: Default::default(),
        })
    }

    async fn batch_cancel(
        &self,
        upstream_id: &str,
        _cx: &ProviderCx,
    ) -> Result<BatchHandle, ModelError> {
        Ok(BatchHandle {
            id: upstream_id.to_string(),
            status: BatchStatus::Canceled,
            counts: BatchCounts::default(),
            created_at: Some(1000),
            expires_at: Some(2000),
            ended_at: None,
            endpoint: Some("/v1/chat/completions".into()),
            extra: Default::default(),
        })
    }

    async fn batch_results(
        &self,
        _upstream_id: &str,
        _cx: &ProviderCx,
    ) -> Result<BatchResultStream, ModelError> {
        let results: Vec<Result<BatchResult, ModelError>> = vec![
            Ok(BatchResult {
                custom_id: "a".into(),
                outcome: BatchOutcome::Succeeded(ChatResponse {
                    id: "r-a".into(),
                    model: "demo".into(),
                    choices: vec![Choice {
                        index: 0,
                        text: "hi a".into(),
                        thinking: vec![],
                        tool_calls: vec![],
                        finish_reason: Some(FinishReason::Stop),
                    }],
                    usage: Usage {
                        prompt_tokens: 1,
                        completion_tokens: 2,
                        total_tokens: 3,
                        cost_usd: None,
                        cache_read_tokens: 0,
                        cache_creation_tokens: 0,
                    },
                }),
            }),
            Ok(BatchResult {
                custom_id: "b".into(),
                outcome: BatchOutcome::Errored {
                    status: 429,
                    message: "rate limited".into(),
                },
            }),
        ];
        Ok(Box::pin(futures::stream::iter(results)))
    }
}

const CONFIG: &str = r#"
[server]
listen = "127.0.0.1:0"
admin_token = "test-admin"

[[providers]]
name = "mocka"
kind = "mocka"

[[providers]]
name = "mockb"
kind = "mockb"

[[routes]]
model = "demo"
targets = [{ provider = "mocka" }]

[[routes]]
model = "other"
targets = [{ provider = "mockb" }]

# bcrypt of "s3cret" at cost 4 (same hash for both keys — bcrypt verifies the password, not the id).
[[keys]]
id = "local"
pw_hash = "$2y$04$IcVq6nhz5Tf85lBpWclgKeDjWxWMHlIXLE696.T7m9Eg12HekWFJO"
name = "local"

[[keys]]
id = "restricted"
pw_hash = "$2y$04$IcVq6nhz5Tf85lBpWclgKeDjWxWMHlIXLE696.T7m9Eg12HekWFJO"
allowed_models = ["demo"]
"#;

/// base64("local:s3cret") and base64("restricted:s3cret").
const LOCAL_TOKEN: &str = "bG9jYWw6czNjcmV0";
const RESTRICTED_TOKEN: &str = "cmVzdHJpY3RlZDpzM2NyZXQ=";

fn build() -> AppState {
    let config = Config::from_toml_str(CONFIG).unwrap();
    let mut registry = ProviderRegistry::new();
    registry.register("mocka", Arc::new(MockBatchProvider { name: "mocka" }));
    registry.register("mockb", Arc::new(MockBatchProvider { name: "mockb" }));
    build_state(&config, Arc::new(registry)).unwrap()
}

fn app() -> axum::Router {
    build_router(build())
}

async fn body_json(resp: axum::response::Response) -> Value {
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    serde_json::from_slice(&bytes).unwrap()
}

async fn body_text(resp: axum::response::Response) -> String {
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    String::from_utf8(bytes.to_vec()).unwrap()
}

fn create_req(token: &str, models: &[&str]) -> Request<Body> {
    let requests: Vec<Value> = models
        .iter()
        .enumerate()
        .map(|(i, m)| {
            json!({
                "custom_id": format!("req-{i}"),
                "body": { "model": m, "messages": [{ "role": "user", "content": "hi" }] }
            })
        })
        .collect();
    Request::builder()
        .method("POST")
        .uri("/v1/batches")
        .header("authorization", format!("Bearer {token}"))
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&json!({ "requests": requests })).unwrap(),
        ))
        .unwrap()
}

fn get_req(token: &str, uri: &str) -> Request<Body> {
    Request::builder()
        .method("GET")
        .uri(uri)
        .header("authorization", format!("Bearer {token}"))
        .body(Body::empty())
        .unwrap()
}

#[tokio::test]
async fn create_returns_opaque_provider_encoding_id() {
    let resp = app()
        .oneshot(create_req(LOCAL_TOKEN, &["demo"]))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let v = body_json(resp).await;

    assert_eq!(v["object"], "batch");
    assert_eq!(v["status"], "in_progress");
    assert_eq!(v["request_counts"]["total"], 1);
    let id = v["id"].as_str().unwrap();
    assert!(
        id.starts_with("batch_"),
        "id should be an opaque token, got {id}"
    );

    // The id decodes to (provider instance, upstream id) — the routing the later calls rely on lives
    // entirely inside the id, not in any node's memory.
    let (provider, upstream) = llmleaf_core::batch_id::decode_batch(id).unwrap();
    assert_eq!(provider, "mocka");
    assert_eq!(upstream, "up-batch");
}

#[tokio::test]
async fn full_lifecycle_create_retrieve_results_cancel() {
    let app = app();

    // Create.
    let v = body_json(
        app.clone()
            .oneshot(create_req(LOCAL_TOKEN, &["demo"]))
            .await
            .unwrap(),
    )
    .await;
    let id = v["id"].as_str().unwrap().to_string();

    // Retrieve.
    let v = body_json(
        app.clone()
            .oneshot(get_req(LOCAL_TOKEN, &format!("/v1/batches/{id}")))
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(v["status"], "completed");
    assert_eq!(v["request_counts"]["completed"], 2);
    // The id stays the opaque token through retrieve.
    assert_eq!(v["id"], id);

    // Results — newline-delimited JSON, one line per result.
    let resp = app
        .clone()
        .oneshot(get_req(LOCAL_TOKEN, &format!("/v1/batches/{id}/results")))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(
        resp.headers().get("content-type").unwrap(),
        "application/x-ndjson"
    );
    let text = body_text(resp).await;
    let lines: Vec<Value> = text
        .lines()
        .filter(|l| !l.is_empty())
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(lines.len(), 2);
    assert_eq!(lines[0]["custom_id"], "a");
    assert_eq!(lines[0]["response"]["status_code"], 200);
    assert_eq!(
        lines[0]["response"]["body"]["choices"][0]["message"]["content"],
        "hi a"
    );
    assert_eq!(lines[1]["custom_id"], "b");
    assert_eq!(lines[1]["error"]["code"], "429");

    // Cancel.
    let v = body_json(
        app.clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(format!("/v1/batches/{id}/cancel"))
                    .header("authorization", format!("Bearer {LOCAL_TOKEN}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(v["status"], "cancelled");
}

#[tokio::test]
async fn batch_created_on_one_node_is_served_by_another() {
    // Two independent nodes from the same config — no shared state (principle 9).
    let node_a = build_router(build());
    let node_b = build_router(build());

    let id = body_json(
        node_a
            .oneshot(create_req(LOCAL_TOKEN, &["demo"]))
            .await
            .unwrap(),
    )
    .await["id"]
        .as_str()
        .unwrap()
        .to_string();

    // Node B, which never saw the create, serves the poll purely from the opaque id.
    let resp = node_b
        .oneshot(get_req(LOCAL_TOKEN, &format!("/v1/batches/{id}")))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(body_json(resp).await["status"], "completed");
}

#[tokio::test]
async fn results_relay_usage_events() {
    let state = build();
    let bus: EventBus = state.events.clone();
    let mut events = bus.subscribe();
    let app = build_router(state);

    let id = body_json(
        app.clone()
            .oneshot(create_req(LOCAL_TOKEN, &["demo"]))
            .await
            .unwrap(),
    )
    .await["id"]
        .as_str()
        .unwrap()
        .to_string();

    // Drain the results stream so the instrumented stream emits its per-line usage.
    let resp = app
        .oneshot(get_req(LOCAL_TOKEN, &format!("/v1/batches/{id}/results")))
        .await
        .unwrap();
    let _ = body_text(resp).await;

    // A Usage event for the succeeded line surfaced on the bus, keyed by "<batch-id>:<custom-id>".
    let mut saw_usage = false;
    while let Ok(env) = events.try_recv() {
        if let Event::Usage {
            id: ev_id,
            model,
            usage,
            ..
        } = &env.event
        {
            if ev_id.ends_with(":a") {
                assert!(ev_id.starts_with("batch_"));
                assert_eq!(model, "demo");
                assert_eq!(usage.total_tokens, 3);
                saw_usage = true;
            }
        }
    }
    assert!(
        saw_usage,
        "expected a relayed Usage event for the succeeded result"
    );
}

#[tokio::test]
async fn mixed_provider_batch_is_rejected() {
    // `demo` routes to mocka, `other` to mockb — one batch cannot span two upstream jobs.
    let resp = app()
        .oneshot(create_req(LOCAL_TOKEN, &["demo", "other"]))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn unknown_or_foreign_batch_id_is_not_found() {
    let app = app();
    for id in ["batch_bogus", "not-a-token", "batch_dXAtYmF0Y2g"] {
        let resp = app
            .clone()
            .oneshot(get_req(LOCAL_TOKEN, &format!("/v1/batches/{id}")))
            .await
            .unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::NOT_FOUND,
            "id {id} should be 404"
        );
    }
}

#[tokio::test]
async fn create_gates_every_model_against_the_key() {
    // The restricted key allows only `demo`; a batch referencing `other` is forbidden even though the
    // key is otherwise valid.
    let resp = app()
        .oneshot(create_req(RESTRICTED_TOKEN, &["demo", "other"]))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);

    // A batch entirely within the allow-list succeeds.
    let resp = app()
        .oneshot(create_req(RESTRICTED_TOKEN, &["demo"]))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn batch_surfaces_require_a_bearer() {
    let app = app();
    let no_auth = Request::builder()
        .method("GET")
        .uri("/v1/batches/batch_whatever")
        .body(Body::empty())
        .unwrap();
    assert_eq!(
        app.oneshot(no_auth).await.unwrap().status(),
        StatusCode::UNAUTHORIZED
    );
}
