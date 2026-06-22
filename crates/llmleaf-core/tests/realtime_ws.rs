//! End-to-end test of the `/v1/realtime` WebSocket surface against a bridged (non-native) provider.
//!
//! A tiny in-test echo provider stands in for any chat provider; because it does not implement native
//! realtime, the core serves the session via its bridge. A real `tokio-tungstenite` client drives the
//! GA Realtime client events and asserts the synthesized server-event sequence — exercising the whole
//! path: handshake auth (both styles), `session.update`, `conversation.item.create`, `response.create`,
//! and the audio-on-text-bridge error.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures::{SinkExt, StreamExt};
use llmleaf_core::{build_router, build_state, Config, Event, EventBus};
use llmleaf_model::{
    ChatRequest, FinishReason, ModelError, ResponseStream, Role, StreamChunk, Usage,
};
use llmleaf_provider::{Provider, ProviderCx, ProviderRegistry};
use serde_json::{json, Value};
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::Message;

const CONFIG: &str = r#"
[[providers]]
name = "echo"
kind = "echo"

[[routes]]
model = "echo-rt"
targets = [{ provider = "echo" }]

[[keys]]
id = "k"
pw_hash = "$2y$04$IcVq6nhz5Tf85lBpWclgKeDjWxWMHlIXLE696.T7m9Eg12HekWFJO"
name = "test"
"#;

/// A minimal echo provider: it returns a real canonical stream (Start → Content → Finish → Usage) and
/// does NOT serve native realtime, so the core uses its bridge. Mirrors the first-party EchoProvider
/// without pulling `llmleaf-providers` into the core's test deps (the core depends on no provider).
struct TestEcho;

#[async_trait]
impl Provider for TestEcho {
    fn name(&self) -> &str {
        "echo"
    }

    async fn chat(&self, req: ChatRequest, _cx: &ProviderCx) -> Result<ResponseStream, ModelError> {
        let last = req
            .messages
            .iter()
            .rev()
            .find(|m| m.role == Role::User)
            .map(|m| m.text_content())
            .unwrap_or_default();
        let reply = format!("echo: {last}");
        let chunks = vec![
            Ok(StreamChunk::Start {
                id: "echo-1".into(),
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
                prompt_tokens: 1,
                completion_tokens: 2,
                total_tokens: 3,
                cost_usd: None,
                cache_read_tokens: 0,
                cache_creation_tokens: 0,
            })),
        ];
        Ok(Box::pin(futures::stream::iter(chunks)))
    }
}

/// Standard-alphabet base64 (test-only) — builds the `base64(id:password)` consumer token.
fn b64(s: &str) -> String {
    const A: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();
    for chunk in s.as_bytes().chunks(3) {
        let b = [
            chunk[0],
            *chunk.get(1).unwrap_or(&0),
            *chunk.get(2).unwrap_or(&0),
        ];
        let n = chunk.len();
        out.push(A[(b[0] >> 2) as usize] as char);
        out.push(A[(((b[0] & 0x3) << 4) | (b[1] >> 4)) as usize] as char);
        out.push(if n > 1 {
            A[(((b[1] & 0xf) << 2) | (b[2] >> 6)) as usize] as char
        } else {
            '='
        });
        out.push(if n > 2 {
            A[(b[2] & 0x3f) as usize] as char
        } else {
            '='
        });
    }
    out
}

fn token() -> String {
    b64("k:s3cret")
}

/// Boot the full router on an ephemeral port; returns its address and a handle to the event bus.
async fn start_server() -> (SocketAddr, EventBus) {
    let mut registry = ProviderRegistry::new();
    registry.register("echo", Arc::new(TestEcho));
    let config = Config::from_toml_str(CONFIG).unwrap();
    let state = build_state(&config, Arc::new(registry)).unwrap();
    let events = state.events.clone();
    let app = build_router(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    (addr, events)
}

/// Receive the next JSON frame, failing on timeout / non-text / socket close.
async fn recv_json<S>(ws: &mut S) -> Value
where
    S: futures::Stream<Item = Result<Message, tokio_tungstenite::tungstenite::Error>> + Unpin,
{
    loop {
        let item = tokio::time::timeout(Duration::from_secs(5), ws.next())
            .await
            .expect("frame timed out")
            .expect("stream ended")
            .expect("ws error");
        match item {
            Message::Text(t) => return serde_json::from_str(t.as_str()).expect("valid json frame"),
            Message::Ping(_) | Message::Pong(_) => continue,
            other => panic!("unexpected frame: {other:?}"),
        }
    }
}

/// Drain frames until one of `kind` is seen; returns every frame collected (inclusive).
async fn drain_until<S>(ws: &mut S, kind: &str) -> Vec<Value>
where
    S: futures::Stream<Item = Result<Message, tokio_tungstenite::tungstenite::Error>> + Unpin,
{
    let mut frames = Vec::new();
    loop {
        let f = recv_json(ws).await;
        let ty = f["type"].as_str().unwrap_or("").to_string();
        frames.push(f);
        if ty == kind {
            return frames;
        }
    }
}

async fn send_json<S>(ws: &mut S, v: Value)
where
    S: SinkExt<Message> + Unpin,
    <S as futures::Sink<Message>>::Error: std::fmt::Debug,
{
    ws.send(Message::text(v.to_string()))
        .await
        .expect("send frame");
}

#[tokio::test]
async fn text_turn_over_bridge_with_header_auth() {
    let (addr, _events) = start_server().await;
    let url = format!("ws://{addr}/v1/realtime?model=echo-rt");
    let mut req = url.as_str().into_client_request().unwrap();
    req.headers_mut().insert(
        "authorization",
        format!("Bearer {}", token()).parse().unwrap(),
    );
    let (mut ws, _resp) = connect_async(req).await.expect("connect");

    // The bridge greets with session.created immediately.
    assert_eq!(recv_json(&mut ws).await["type"], "session.created");

    // Configure text-only, then expect session.updated echoing the effective config.
    send_json(
        &mut ws,
        json!({ "type": "session.update", "session": { "type": "realtime", "output_modalities": ["text"] } }),
    )
    .await;
    let updated = recv_json(&mut ws).await;
    assert_eq!(updated["type"], "session.updated");
    assert_eq!(updated["session"]["output_modalities"][0], "text");

    // Add a user message; the bridge echoes it as conversation.item.added.
    send_json(
        &mut ws,
        json!({ "type": "conversation.item.create",
                "item": { "type": "message", "role": "user",
                          "content": [{ "type": "input_text", "text": "hi" }] } }),
    )
    .await;
    assert_eq!(recv_json(&mut ws).await["type"], "conversation.item.added");

    // Trigger generation; assert the full GA scaffold and the terminal response.done.
    send_json(
        &mut ws,
        json!({ "type": "response.create", "response": { "output_modalities": ["text"] } }),
    )
    .await;
    let frames = drain_until(&mut ws, "response.done").await;
    let types: Vec<&str> = frames.iter().map(|f| f["type"].as_str().unwrap()).collect();
    assert_eq!(
        types,
        vec![
            "response.created",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "conversation.item.done",
            "response.done",
        ]
    );
    let done = frames.last().unwrap();
    assert_eq!(done["response"]["status"], "completed");
    assert_eq!(
        done["response"]["output"][0]["content"][0]["text"],
        "echo: hi"
    );
    assert_eq!(done["response"]["usage"]["output_tokens"], 2);
    assert!(done["response"]["usage"].get("cost_usd").is_none());

    // The accumulated text delta reconstructs the reply.
    let delta = frames
        .iter()
        .find(|f| f["type"] == "response.output_text.delta")
        .unwrap();
    assert_eq!(delta["delta"], "echo: hi");
    assert_eq!(delta["sequence_number"], 1);
}

#[tokio::test]
async fn audio_output_request_is_rejected_on_bridge() {
    let (addr, _events) = start_server().await;
    let url = format!("ws://{addr}/v1/realtime?model=echo-rt");
    let mut req = url.as_str().into_client_request().unwrap();
    req.headers_mut().insert(
        "authorization",
        format!("Bearer {}", token()).parse().unwrap(),
    );
    let (mut ws, _resp) = connect_async(req).await.expect("connect");

    assert_eq!(recv_json(&mut ws).await["type"], "session.created");
    send_json(
        &mut ws,
        json!({ "type": "session.update", "session": { "type": "realtime", "output_modalities": ["audio"] } }),
    )
    .await;
    assert_eq!(recv_json(&mut ws).await["type"], "session.updated");

    send_json(&mut ws, json!({ "type": "response.create" })).await;
    let err = recv_json(&mut ws).await;
    assert_eq!(err["type"], "error");
    assert_eq!(err["error"]["code"], "unsupported_modality");
}

#[tokio::test]
async fn subprotocol_auth_is_accepted() {
    let (addr, _events) = start_server().await;
    let url = format!("ws://{addr}/v1/realtime?model=echo-rt");
    let mut req = url.as_str().into_client_request().unwrap();
    // The browser pattern: the token rides in the Sec-WebSocket-Protocol list, no Authorization header.
    req.headers_mut().insert(
        "sec-websocket-protocol",
        format!("realtime, openai-insecure-api-key.{}", token())
            .parse()
            .unwrap(),
    );
    let (mut ws, resp) = connect_async(req).await.expect("connect via subprotocol");
    // The server echoes the GA `realtime` subprotocol.
    assert_eq!(
        resp.headers()
            .get("sec-websocket-protocol")
            .and_then(|v| v.to_str().ok()),
        Some("realtime")
    );
    assert_eq!(recv_json(&mut ws).await["type"], "session.created");
}

#[tokio::test]
async fn missing_credential_is_rejected() {
    let (addr, _events) = start_server().await;
    let url = format!("ws://{addr}/v1/realtime?model=echo-rt");
    // No Authorization header and no subprotocol token ⇒ the upgrade is refused.
    let req = url.as_str().into_client_request().unwrap();
    assert!(connect_async(req).await.is_err());
}

#[tokio::test]
async fn bridge_session_brackets_lifecycle_on_event_bus() {
    let (addr, events) = start_server().await;
    let mut rx = events.subscribe(); // subscribe BEFORE connecting so no event is missed

    let url = format!("ws://{addr}/v1/realtime?model=echo-rt");
    let mut req = url.as_str().into_client_request().unwrap();
    req.headers_mut().insert(
        "authorization",
        format!("Bearer {}", token()).parse().unwrap(),
    );
    let (mut ws, _resp) = connect_async(req).await.expect("connect");
    assert_eq!(recv_json(&mut ws).await["type"], "session.created");

    send_json(
        &mut ws,
        json!({ "type": "conversation.item.create",
                "item": { "type": "message", "role": "user",
                          "content": [{ "type": "input_text", "text": "hi" }] } }),
    )
    .await;
    assert_eq!(recv_json(&mut ws).await["type"], "conversation.item.added");
    send_json(&mut ws, json!({ "type": "response.create" })).await;
    let _ = drain_until(&mut ws, "response.done").await;

    // Closing the socket ends the bridge session, which emits the session-level RequestCompleted.
    drop(ws);

    // Drain the bus: we must see a SESSION-level RequestStarted + RequestCompleted (ids with no `-t`
    // turn suffix), plus a per-turn Usage (id `…-t1`). This is the principle-5 lifecycle the bridge
    // must expose, symmetric with the native path.
    let mut session_started = false;
    let mut session_completed = false;
    let mut turn_usage = false;
    loop {
        let env = match tokio::time::timeout(Duration::from_secs(3), rx.recv()).await {
            Ok(Ok(env)) => env,
            _ => break,
        };
        match &env.event {
            Event::RequestStarted { id, .. } if !id.contains("-t") => session_started = true,
            Event::RequestCompleted { id, .. } if !id.contains("-t") => {
                session_completed = true;
                break;
            }
            Event::Usage { id, .. } if id.contains("-t") => turn_usage = true,
            _ => {}
        }
    }
    assert!(session_started, "expected a session-level RequestStarted");
    assert!(turn_usage, "expected per-turn Usage from engine.run");
    assert!(
        session_completed,
        "expected a session-level RequestCompleted on client close"
    );
}
