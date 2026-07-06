//! The realtime **bridge** — a stateful Realtime session in front of the stateless chat pipeline.
//!
//! When the routed provider has no native realtime transport, the core serves the WS surface itself:
//! it accumulates the Realtime session (a [`BridgeSession`]) and, on each `response.create`, builds a
//! canonical [`ChatRequest`] and runs it through [`Engine::run`] — inheriting routing, fallback,
//! pricing, and usage/lifecycle events for free (principles 4, 5, 8) — then re-frames the resulting
//! [`StreamChunk`] stream into Realtime server events via [`ResponseScaffold`]. Text + tools only:
//! an audio request is answered with a Realtime `error`, never a silent downgrade (principle 7).
//!
//! Turns are processed one at a time: while a turn streams, the loop is not reading further client
//! events, so `response.cancel` mid-turn is observed only after the turn completes (a documented v1
//! simplification — text clients send `response.create` and await `response.done`).

use std::ops::ControlFlow;
use std::sync::Arc;

use axum::extract::ws::{Message as WsMessage, WebSocket};
use futures::StreamExt;
use llmleaf_model::{
    ChatRequest, FinishReason, Message, Role, StreamChunk, ToolChoice, ToolDef, Usage,
};
use llmleaf_provider::RealtimeWire;
use serde_json::{json, Value};

use super::wire::{
    error_frame, parse_client_event, ClientEvent, MaxOut, ResponseOverride, ResponseScaffold,
    SessionPatch,
};
use crate::engine::Engine;
use crate::events::{Event, EventBus};

/// The accumulated state of one bridged Realtime session.
pub struct BridgeSession {
    model: String,
    instructions: Option<String>,
    /// Conversation history (user / assistant / tool messages). The system prompt is `instructions`,
    /// prepended at request-build time, not stored here.
    messages: Vec<Message>,
    tools: Vec<ToolDef>,
    tool_choice: Option<ToolChoice>,
    max_tokens: Option<u32>,
    /// `true` ⇒ text output. Defaults to text for the bridge (a chat provider cannot emit audio); set
    /// `false` only when the client explicitly asks for audio, which then errors on the next turn.
    text_only: bool,
    /// Unrecognized session fields (e.g. a smuggled `temperature`) — passed through verbatim into
    /// `ChatRequest.extra` (principle 7).
    extra: serde_json::Map<String, Value>,
    /// Monotonic counter for minting `item_…` ids on echoed conversation items.
    item_seq: u64,
}

impl BridgeSession {
    fn new(model: String) -> Self {
        BridgeSession {
            model,
            instructions: None,
            messages: Vec::new(),
            tools: Vec::new(),
            tool_choice: None,
            max_tokens: None,
            text_only: true,
            extra: serde_json::Map::new(),
            item_seq: 0,
        }
    }

    fn apply(&mut self, patch: SessionPatch) {
        if let Some(m) = patch.model {
            self.model = m;
        }
        if let Some(i) = patch.instructions {
            self.instructions = Some(i);
        }
        if let Some(m) = patch.output_modalities {
            self.text_only = !m.iter().any(|x| x == "audio");
        }
        if let Some(mt) = patch.max_output_tokens {
            self.max_tokens = match mt {
                MaxOut::Num(n) => Some(n),
                MaxOut::Inf => None,
            };
        }
        if let Some(t) = patch.tools {
            self.tools = t;
        }
        if let Some(tc) = patch.tool_choice {
            self.tool_choice = Some(tc);
        }
        for (k, v) in patch.extra {
            self.extra.insert(k, v);
        }
    }

    /// Does this turn ask for audio output the text bridge cannot produce?
    fn wants_audio(&self, over: &ResponseOverride) -> bool {
        match &over.output_modalities {
            Some(m) => m.iter().any(|x| x == "audio"),
            None => !self.text_only,
        }
    }

    /// Build the per-turn [`ChatRequest`] from accumulated state plus the response override.
    fn build_chat_request(&self, over: &ResponseOverride) -> ChatRequest {
        let mut messages = Vec::with_capacity(self.messages.len() + 1);
        if let Some(sys) = over.instructions.as_ref().or(self.instructions.as_ref()) {
            messages.push(Message::text(Role::System, sys.clone()));
        }
        messages.extend(self.messages.iter().cloned());

        let max_tokens = match over.max_output_tokens {
            Some(MaxOut::Num(n)) => Some(n),
            Some(MaxOut::Inf) => None,
            None => self.max_tokens,
        };

        ChatRequest {
            model: self.model.clone(),
            messages,
            max_tokens,
            // GA Realtime has no temperature; a smuggled one rides in `extra`.
            temperature: None,
            top_p: None,
            stop: Vec::new(),
            stream: true,
            tools: over.tools.clone().unwrap_or_else(|| self.tools.clone()),
            tool_choice: over
                .tool_choice
                .clone()
                .or_else(|| self.tool_choice.clone()),
            thinking: None,
            extra: self.extra.clone(),
        }
    }

    /// The effective session object echoed in `session.created` / `session.updated`.
    fn session_object(&self) -> Value {
        json!({
            "object": "realtime.session",
            "type": "realtime",
            "model": self.model,
            "output_modalities": if self.text_only { json!(["text"]) } else { json!(["audio"]) },
            "instructions": self.instructions.clone().unwrap_or_default(),
            "tools": tools_to_realtime(&self.tools),
            "tool_choice": tool_choice_to_realtime(self.tool_choice.as_ref()),
            "max_output_tokens": self.max_tokens.map(|n| json!(n)).unwrap_or(json!("inf")),
        })
    }

    fn session_frame(&self, ty: &str) -> Value {
        json!({ "type": ty, "session": self.session_object() })
    }

    /// Echo a created conversation item back to the client (`conversation.item.added`), minting an id.
    fn item_added(&mut self, item: Value) -> Value {
        self.item_seq += 1;
        json!({ "type": "conversation.item.added", "item": item })
    }
}

/// Realtime tools are FLAT objects (the inverse of [`super::wire`]'s parse).
fn tools_to_realtime(tools: &[ToolDef]) -> Vec<Value> {
    tools
        .iter()
        .map(|t| {
            json!({
                "type": "function",
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            })
        })
        .collect()
}

fn tool_choice_to_realtime(tc: Option<&ToolChoice>) -> Value {
    match tc {
        Some(ToolChoice::Auto) | None => json!("auto"),
        Some(ToolChoice::None) => json!("none"),
        Some(ToolChoice::Required) => json!("required"),
        Some(ToolChoice::Named(name)) => json!({ "type": "function", "name": name }),
    }
}

/// Drive a bridged Realtime session over `socket` until the client closes. Each `response.create`
/// runs one [`Engine::run`] turn (correlated as `request_id-tN`, which emits its own per-turn
/// usage/lifecycle events). The session itself is bracketed with `RequestStarted`/`RequestCompleted`
/// on the bare `request_id` — symmetric with the native path — so an observer sees the whole session
/// on the event bus, not just its individual turns (principle 5).
///
/// This is the entry point used when routing found no native realtime target at all: it owns the
/// lifecycle bracket and delegates the session loop to [`run_bridge_replaying`] with nothing to replay.
pub async fn run_bridge(
    socket: WebSocket,
    engine: Arc<Engine>,
    events: EventBus,
    key: String,
    model: String,
    request_id: String,
) {
    events.emit(Event::RequestStarted {
        id: request_id.clone(),
        key: key.clone(),
        model: model.clone(),
        request: None,
    });
    run_bridge_replaying(socket, engine, key, model, &request_id, Vec::new()).await;
    events.emit(Event::RequestCompleted {
        id: request_id,
        finish: None,
    });
}

/// Run the bridge session loop over `socket`: first replay any frames the caller already received from
/// the consumer (`replay`), then serve live client events until the socket closes.
///
/// This is the shared body behind both the plain [`run_bridge`] entry and the native path's pre-output
/// fallback (in [`crate::server`]). When a native realtime session fails before its first frame, the
/// consumer socket has only ever carried consumer→core traffic, so the core can still serve the whole
/// session over the chat bridge (principle 8) — it hands the consumer frames buffered during the probe
/// here to be replayed through the *exact same* handler as live ones, so nothing the client already
/// sent is dropped (principle 7). The caller owns the `RequestStarted`/`RequestCompleted` bracket on
/// `request_id` (on the fallback path the native side already emitted `RequestStarted`), so this
/// function emits no session-level lifecycle events itself — only the per-turn events that
/// [`run_turn`]'s [`Engine::run`] produces.
pub async fn run_bridge_replaying(
    mut socket: WebSocket,
    engine: Arc<Engine>,
    key: String,
    model: String,
    request_id: &str,
    replay: Vec<RealtimeWire>,
) {
    let mut session = BridgeSession::new(model);
    if send(&mut socket, &session.session_frame("session.created"))
        .await
        .is_err()
    {
        return;
    }

    let mut turn: u64 = 0;

    // Replay the consumer frames buffered during the (failed) native probe, through the SAME handler as
    // a live frame — but only after `session.created`, so the client still sees a clean session opening.
    // A replayed binary frame gets the same `unsupported_modality` error a live one would.
    for wire in replay {
        let frame = match wire {
            RealtimeWire::Text(t) => IncomingFrame::Text(t),
            RealtimeWire::Binary(_) => IncomingFrame::Binary,
            // The probe only ever buffers Text/Binary; a stray Close is nothing to replay.
            RealtimeWire::Close => continue,
        };
        if handle_frame(
            &mut socket,
            &engine,
            &mut session,
            &key,
            request_id,
            &mut turn,
            frame,
        )
        .await
        .is_break()
        {
            return;
        }
    }

    while let Some(incoming) = socket.recv().await {
        let Ok(msg) = incoming else { break };
        let frame = match msg {
            WsMessage::Text(t) => IncomingFrame::Text(t.as_str().to_owned()),
            // Binary frames on the bridge mean audio input — unsupported.
            WsMessage::Binary(_) => IncomingFrame::Binary,
            WsMessage::Close(_) => break,
            WsMessage::Ping(_) | WsMessage::Pong(_) => continue,
        };
        if handle_frame(
            &mut socket,
            &engine,
            &mut session,
            &key,
            request_id,
            &mut turn,
            frame,
        )
        .await
        .is_break()
        {
            break;
        }
    }
}

/// One incoming consumer frame, normalized so the live loop and the replay pass share a handler. A
/// binary frame carries no payload the text bridge can use (it means audio input, which it cannot
/// serve), so it is represented without its bytes.
enum IncomingFrame {
    Text(String),
    Binary,
}

/// Handle one incoming consumer frame. `ControlFlow::Break` means the socket has died and the caller
/// should stop the session; `Continue` means keep serving. This is the exact per-message body the
/// bridge loop runs, factored out so the fallback can replay buffered frames through it verbatim.
async fn handle_frame(
    socket: &mut WebSocket,
    engine: &Arc<Engine>,
    session: &mut BridgeSession,
    key: &str,
    request_id: &str,
    turn: &mut u64,
    frame: IncomingFrame,
) -> ControlFlow<()> {
    let text = match frame {
        IncomingFrame::Text(t) => t,
        IncomingFrame::Binary => {
            let _ = send(
                socket,
                &error_frame(
                    "unsupported_modality",
                    "audio input is not available for the routed provider",
                ),
            )
            .await;
            return ControlFlow::Continue(());
        }
    };

    let Ok(value) = serde_json::from_str::<Value>(&text) else {
        let _ = send(
            socket,
            &error_frame("invalid_request_error", "frame is not valid JSON"),
        )
        .await;
        return ControlFlow::Continue(());
    };

    match parse_client_event(value) {
        ClientEvent::SessionUpdate(patch) => {
            session.apply(patch);
            if send(socket, &session.session_frame("session.updated"))
                .await
                .is_err()
            {
                return ControlFlow::Break(());
            }
        }
        ClientEvent::AddUserMessage(text) => {
            let item = json!({
                "id": format!("item_{}", session.item_seq + 1),
                "object": "realtime.item", "type": "message", "role": "user",
                "content": [{ "type": "input_text", "text": text }]
            });
            session.messages.push(Message::text(Role::User, text));
            let frame = session.item_added(item);
            if send(socket, &frame).await.is_err() {
                return ControlFlow::Break(());
            }
        }
        ClientEvent::AddFunctionOutput { call_id, output } => {
            let item = json!({
                "id": format!("item_{}", session.item_seq + 1),
                "object": "realtime.item", "type": "function_call_output",
                "call_id": call_id, "output": output
            });
            session.messages.push(Message {
                role: Role::Tool,
                content: vec![llmleaf_model::ContentPart::Text { text: output }],
                tool_calls: Vec::new(),
                tool_call_id: Some(call_id),
                name: None,
            });
            let frame = session.item_added(item);
            if send(socket, &frame).await.is_err() {
                return ControlFlow::Break(());
            }
        }
        // v1: item deletion is not mapped back onto accumulated history (clients overwhelmingly
        // append). Tolerated as a no-op rather than silently corrupting state.
        ClientEvent::DeleteItem(_) | ClientEvent::Cancel | ClientEvent::Other => {}
        ClientEvent::AudioUnsupported => {
            let _ = send(
                socket,
                &error_frame(
                    "unsupported_modality",
                    "audio is not available for the routed provider",
                ),
            )
            .await;
        }
        ClientEvent::Create(over) => {
            if session.wants_audio(&over) {
                let _ = send(
                    socket,
                    &error_frame(
                        "unsupported_modality",
                        "audio output is not available for the routed provider; set output_modalities to [\"text\"]",
                    ),
                )
                .await;
                return ControlFlow::Continue(());
            }
            *turn += 1;
            if run_turn(socket, engine, session, &over, key, request_id, *turn)
                .await
                .is_err()
            {
                return ControlFlow::Break(());
            }
        }
    }
    ControlFlow::Continue(())
}

/// Run one generation turn: issue the chat request, stream the synthesized scaffold, persist the
/// assistant turn. Returns `Err(())` only when the socket has died (caller should stop).
async fn run_turn(
    socket: &mut WebSocket,
    engine: &Arc<Engine>,
    session: &mut BridgeSession,
    over: &ResponseOverride,
    key: &str,
    request_id: &str,
    turn: u64,
) -> Result<(), ()> {
    let rid = format!("{request_id}-t{turn}");
    let req = session.build_chat_request(over);
    let mut scaffold = ResponseScaffold::new(format!("resp_{rid}"), format!("msg_{rid}"));

    send(socket, &scaffold.created()).await?;

    let now = crate::server::now_secs();
    let stream = match engine.run(req, key.to_string(), rid, now).await {
        Ok(s) => s,
        Err(e) => {
            // Routing/connection failed before any byte: surface it as a Realtime error frame.
            let _ = send(socket, &error_frame("upstream_error", &e.to_string())).await;
            return Ok(());
        }
    };

    let mut stream = stream;
    let mut reason = FinishReason::Stop;
    let mut usage = Usage::default();
    let mut frames: Vec<Value> = Vec::new();
    while let Some(item) = stream.next().await {
        match item {
            Ok(chunk) => {
                if let StreamChunk::Usage(u) = &chunk {
                    usage = *u;
                }
                if let StreamChunk::Finish { reason: r, .. } = &chunk {
                    reason = *r;
                }
                frames.clear();
                scaffold.on_chunk(&chunk, &mut frames);
                for f in &frames {
                    send(socket, f).await?;
                }
            }
            Err(e) => {
                let _ = send(socket, &error_frame("upstream_error", &e.to_string())).await;
                reason = FinishReason::Error;
                break;
            }
        }
    }

    // Persist the assistant turn into history (unless this was an out-of-band response).
    if over.persist {
        session.messages.push(scaffold.assistant_message());
    }

    frames.clear();
    scaffold.finish(reason, &usage, &mut frames);
    for f in &frames {
        send(socket, f).await?;
    }
    Ok(())
}

/// Serialize a frame and send it as a WS text message; `Err(())` if the socket is gone.
async fn send(socket: &mut WebSocket, frame: &Value) -> Result<(), ()> {
    socket
        .send(WsMessage::Text(frame.to_string().into()))
        .await
        .map_err(|_| ())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accumulates_history_and_builds_request() {
        let mut s = BridgeSession::new("gpt-4o".into());
        s.apply(SessionPatch {
            instructions: Some("be terse".into()),
            output_modalities: Some(vec!["text".into()]),
            max_output_tokens: Some(MaxOut::Num(256)),
            ..Default::default()
        });
        s.messages.push(Message::text(Role::User, "hi"));
        let req = s.build_chat_request(&ResponseOverride::default());
        assert!(req.stream);
        assert_eq!(req.model, "gpt-4o");
        assert_eq!(req.max_tokens, Some(256));
        assert_eq!(req.messages[0].role, Role::System);
        assert_eq!(req.messages[0].text_content(), "be terse");
        assert_eq!(req.messages[1].role, Role::User);
        assert!(s.text_only);
    }

    #[test]
    fn inf_max_tokens_becomes_none() {
        let mut s = BridgeSession::new("m".into());
        s.apply(SessionPatch {
            max_output_tokens: Some(MaxOut::Inf),
            ..Default::default()
        });
        assert_eq!(
            s.build_chat_request(&ResponseOverride::default())
                .max_tokens,
            None
        );
    }

    #[test]
    fn audio_modality_is_detected() {
        let mut s = BridgeSession::new("m".into());
        s.apply(SessionPatch {
            output_modalities: Some(vec!["audio".into()]),
            ..Default::default()
        });
        assert!(!s.text_only);
        assert!(s.wants_audio(&ResponseOverride::default()));
        // A per-turn override can re-assert text.
        let over = ResponseOverride {
            output_modalities: Some(vec!["text".into()]),
            ..Default::default()
        };
        assert!(!s.wants_audio(&over));
    }
}
