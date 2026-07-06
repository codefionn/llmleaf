//! End-to-end test of the `/v1/realtime` WebSocket surface against a bridged (non-native) provider.
//!
//! A tiny in-test echo provider stands in for any chat provider; because it does not implement native
//! realtime, the core serves the session via its bridge. A real `tokio-tungstenite` client drives the
//! GA Realtime client events and asserts the synthesized server-event sequence — exercising the whole
//! path: handshake auth (both styles), `session.update`, `conversation.item.create`, `response.create`,
//! and the audio-on-text-bridge error.

use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use futures::{SinkExt, StreamExt};
use llmleaf_core::{build_router, build_state, Config, Event, EventBus};
use llmleaf_model::{
    ChatRequest, FinishReason, ModelError, ResponseStream, Role, StreamChunk, Usage,
};
use llmleaf_provider::{
    Provider, ProviderCx, ProviderRegistry, RealtimeParams, RealtimePeer, RealtimeWire,
};
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

// ---------------------------------------------------------------------------------------------
// Native realtime → chat-bridge fallback. A provider that DOES advertise native realtime but whose
// session fails BEFORE producing any output must fall back to the bridge, not error the consumer.
// ---------------------------------------------------------------------------------------------

const NATIVE_CONFIG: &str = r#"
[[providers]]
name = "mockrt"
kind = "mockrt"

[[routes]]
model = "mock-rt"
targets = [{ provider = "mockrt" }]

[[keys]]
id = "k"
pw_hash = "$2y$04$IcVq6nhz5Tf85lBpWclgKeDjWxWMHlIXLE696.T7m9Eg12HekWFJO"
name = "test"
"#;

/// How the mock's native `realtime()` behaves.
#[derive(Clone, Copy)]
enum RtMode {
    /// Fail immediately with no frames — a pre-output failure that must fall back to the bridge.
    FailImmediately,
    /// Sleep briefly, then fail — leaving a window for the client to send early frames the fallback
    /// must replay.
    SleepThenFail,
    /// Emit one server frame, THEN fail — a committed (mid-session) failure that must NOT fall back.
    FrameThenFail,
}

/// A mock provider that advertises native realtime with configurable behavior, and a normal `chat()`
/// that records every request it served (so a test can prove the buffered frames were replayed into the
/// bridged turn). Registered under the same config name it reports, exactly like `TestEcho`.
struct MockRealtime {
    mode: RtMode,
    chats: Arc<Mutex<Vec<ChatRequest>>>,
}

#[async_trait]
impl Provider for MockRealtime {
    fn name(&self) -> &str {
        "mockrt"
    }

    fn supports_realtime(&self) -> bool {
        true
    }

    async fn realtime(
        &self,
        _params: RealtimeParams,
        peer: RealtimePeer,
        _cx: &ProviderCx,
    ) -> Result<(), ModelError> {
        match self.mode {
            RtMode::FailImmediately => Err(ModelError::Unavailable("native realtime down".into())),
            RtMode::SleepThenFail => {
                tokio::time::sleep(Duration::from_millis(150)).await;
                Err(ModelError::Unavailable("native realtime down".into()))
            }
            RtMode::FrameThenFail => {
                // One server frame commits the session; the error that follows is therefore MID-session.
                let frame = json!({ "type": "response.output_text.delta", "delta": "native-hi" })
                    .to_string();
                let _ = peer.outbound.send(RealtimeWire::Text(frame)).await;
                Err(ModelError::Unavailable(
                    "native realtime died mid-session".into(),
                ))
            }
        }
    }

    async fn chat(&self, req: ChatRequest, _cx: &ProviderCx) -> Result<ResponseStream, ModelError> {
        self.chats.lock().unwrap().push(req.clone());
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

/// Boot the router with a native-realtime mock in the given mode; returns its address, the event bus,
/// and the shared capture of the chat requests the bridge served.
async fn start_native(mode: RtMode) -> (SocketAddr, EventBus, Arc<Mutex<Vec<ChatRequest>>>) {
    let chats = Arc::new(Mutex::new(Vec::new()));
    let mut registry = ProviderRegistry::new();
    registry.register(
        "mockrt",
        Arc::new(MockRealtime {
            mode,
            chats: chats.clone(),
        }),
    );
    let config = Config::from_toml_str(NATIVE_CONFIG).unwrap();
    let state = build_state(&config, Arc::new(registry)).unwrap();
    let events = state.events.clone();
    let app = build_router(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    (addr, events, chats)
}

#[tokio::test]
async fn native_failure_before_output_falls_back_to_bridge() {
    let (addr, events, _chats) = start_native(RtMode::FailImmediately).await;
    let mut rx = events.subscribe(); // subscribe BEFORE connecting so no event is missed

    let url = format!("ws://{addr}/v1/realtime?model=mock-rt");
    let mut req = url.as_str().into_client_request().unwrap();
    req.headers_mut().insert(
        "authorization",
        format!("Bearer {}", token()).parse().unwrap(),
    );
    let (mut ws, _resp) = connect_async(req).await.expect("connect");

    // The native realtime attempt fails immediately with no output, so the core falls back to the
    // bridge — the consumer still gets a working session, greeted with session.created.
    assert_eq!(recv_json(&mut ws).await["type"], "session.created");

    // A normal user message + response.create produces a bridged turn served by chat().
    send_json(
        &mut ws,
        json!({ "type": "conversation.item.create",
                "item": { "type": "message", "role": "user",
                          "content": [{ "type": "input_text", "text": "hi" }] } }),
    )
    .await;
    assert_eq!(recv_json(&mut ws).await["type"], "conversation.item.added");
    send_json(
        &mut ws,
        json!({ "type": "response.create", "response": { "output_modalities": ["text"] } }),
    )
    .await;
    let frames = drain_until(&mut ws, "response.done").await;
    let done = frames.last().unwrap();
    assert_eq!(done["response"]["status"], "completed");
    assert_eq!(
        done["response"]["output"][0]["content"][0]["text"],
        "echo: hi"
    );

    drop(ws);

    // The fell-back session must still emit EXACTLY ONE session-level RequestStarted (the native path's;
    // the bridge must not emit a second) and terminate with a session-level RequestCompleted. Per-turn
    // events carry a `-t` suffix and are excluded.
    let mut started = 0;
    let mut completed = false;
    loop {
        let env = match tokio::time::timeout(Duration::from_secs(3), rx.recv()).await {
            Ok(Ok(env)) => env,
            _ => break,
        };
        match &env.event {
            Event::RequestStarted { id, .. } if !id.contains("-t") => started += 1,
            Event::RequestCompleted { id, .. } if !id.contains("-t") => {
                completed = true;
                break;
            }
            _ => {}
        }
    }
    assert_eq!(started, 1, "exactly one session-level RequestStarted");
    assert!(completed, "the fell-back session still completes");
}

#[tokio::test]
async fn early_consumer_frames_are_replayed_after_fallback() {
    let (addr, _events, chats) = start_native(RtMode::SleepThenFail).await;
    let url = format!("ws://{addr}/v1/realtime?model=mock-rt");
    let mut req = url.as_str().into_client_request().unwrap();
    req.headers_mut().insert(
        "authorization",
        format!("Bearer {}", token()).parse().unwrap(),
    );
    let (mut ws, _resp) = connect_async(req).await.expect("connect");

    // Send session config + the user message IMMEDIATELY — before the native attempt resolves (it sleeps
    // ~150ms) — so they land in the probe's replay buffer, not the live bridge loop.
    send_json(
        &mut ws,
        json!({ "type": "session.update",
                "session": { "type": "realtime", "instructions": "be terse", "output_modalities": ["text"] } }),
    )
    .await;
    send_json(
        &mut ws,
        json!({ "type": "conversation.item.create",
                "item": { "type": "message", "role": "user",
                          "content": [{ "type": "input_text", "text": "hello" }] } }),
    )
    .await;

    // The native path fails ~150ms later and the bridge takes over: session.created, then the REPLAYED
    // session.updated + conversation.item.added (proving the buffered frames were not dropped).
    assert_eq!(recv_json(&mut ws).await["type"], "session.created");
    let replayed = drain_until(&mut ws, "conversation.item.added").await;
    assert!(
        replayed.iter().any(|f| f["type"] == "session.updated"),
        "the replayed session.update must produce session.updated"
    );

    // Now trigger generation; the turn must carry the replayed instructions + user message.
    send_json(
        &mut ws,
        json!({ "type": "response.create", "response": { "output_modalities": ["text"] } }),
    )
    .await;
    let frames = drain_until(&mut ws, "response.done").await;
    let done = frames.last().unwrap();
    assert_eq!(
        done["response"]["output"][0]["content"][0]["text"],
        "echo: hello"
    );

    // The captured chat request is the proof: a leading system message with the replayed instructions,
    // and the replayed user message — neither was dropped during the native failure.
    let reqs = chats.lock().unwrap();
    let last = reqs.last().expect("chat() ran for the bridged turn");
    assert_eq!(last.messages[0].role, Role::System);
    assert_eq!(last.messages[0].text_content(), "be terse");
    assert!(
        last.messages
            .iter()
            .any(|m| m.role == Role::User && m.text_content() == "hello"),
        "the replayed user message must be in the bridged request"
    );
}

#[tokio::test]
async fn native_frame_then_error_does_not_fall_back() {
    let (addr, _events, chats) = start_native(RtMode::FrameThenFail).await;
    let url = format!("ws://{addr}/v1/realtime?model=mock-rt");
    let mut req = url.as_str().into_client_request().unwrap();
    req.headers_mut().insert(
        "authorization",
        format!("Bearer {}", token()).parse().unwrap(),
    );
    let (mut ws, _resp) = connect_async(req).await.expect("connect");

    // The native session emitted a frame first, so it is COMMITTED — the frame is proxied verbatim.
    let first = recv_json(&mut ws).await;
    assert_eq!(first["type"], "response.output_text.delta");
    assert_eq!(first["delta"], "native-hi");

    // The mid-session error is surfaced as a Realtime error frame, never a silent restart onto the
    // bridge (which would fabricate a fresh context the client never asked for).
    let err = recv_json(&mut ws).await;
    assert_eq!(err["type"], "error");
    assert_eq!(err["error"]["code"], "upstream_error");

    // The bridge never ran after commit, so chat() was never called.
    assert!(
        chats.lock().unwrap().is_empty(),
        "a committed native session must not fall back to the chat bridge"
    );
}
