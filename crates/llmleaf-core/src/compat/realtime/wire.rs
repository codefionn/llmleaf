//! The OpenAI Realtime dialect ⇄ canonical mapping (consumer edge).
//!
//! This is the realtime sibling of [`crate::compat::openai`]: inbound OpenAI Realtime *client* events
//! are parsed into a small [`ClientEvent`] the bridge acts on, and a [`ResponseScaffold`] synthesizes
//! the full OpenAI Realtime *server* event sequence around the canonical [`StreamChunk`]s a turn
//! produces. All OpenAI-specific event names and JSON shapes live here, at the edge — the engine and
//! the bridge state stay dialect-free (principle 3). Unknown fields ride through verbatim (principle 7).
//!
//! The names below are the **GA** Realtime protocol (verified): `output_modalities` (with the beta
//! `modalities` accepted as an alias), `response.output_text.delta`/`.done`, flat function tools,
//! `function_call_output`, `max_output_tokens: "inf"`, and no `temperature` (parked in `extra`).

use llmleaf_model::{
    ContentPart, FinishReason, Message, Role, StreamChunk, ToolCall, ToolCallDelta, ToolChoice,
    ToolDef, Usage,
};
use serde_json::{json, Map, Value};

// ---------------------------------------------------------------------------------------------
// Inbound: OpenAI Realtime client JSON -> reduced ClientEvent
// ---------------------------------------------------------------------------------------------

/// A parsed inbound client event, reduced to what the bridge needs to act on. Everything the bridge
/// does not model (audio buffer ops aside) collapses to [`ClientEvent::Other`] and is ignored — never
/// an error, just transparently dropped from the *text* bridge's point of view.
#[derive(Debug, Clone, PartialEq)]
pub enum ClientEvent {
    /// `session.update` — a partial patch of session config (only present fields change).
    SessionUpdate(SessionPatch),
    /// `conversation.item.create` with a user message — the flattened `input_text`.
    AddUserMessage(String),
    /// `conversation.item.create` with a `function_call_output` — a tool result.
    AddFunctionOutput { call_id: String, output: String },
    /// `conversation.item.delete` — drop an item from accumulated session state.
    DeleteItem(String),
    /// `response.create` — trigger a generation turn, with optional per-turn overrides.
    Create(ResponseOverride),
    /// `response.cancel` — abort the in-flight turn.
    Cancel,
    /// An audio-buffer op or an audio message part: unsupported on the text bridge (emit an error).
    AudioUnsupported,
    /// Anything else (retrieve, truncate, unknown types) — tolerated and ignored.
    Other,
}

/// `max_output_tokens` is three-state on the wire: absent (don't change), a number, or `"inf"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaxOut {
    /// A concrete cap.
    Num(u32),
    /// `"inf"` — no cap (maps to `ChatRequest.max_tokens = None`).
    Inf,
}

/// The fields a `session.update` may carry. Each is `Option` so a partial update changes only what it
/// names; unknown fields land in `extra` (principle 7).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SessionPatch {
    pub model: Option<String>,
    pub instructions: Option<String>,
    /// Normalized from `output_modalities` (or the beta `modalities` alias).
    pub output_modalities: Option<Vec<String>>,
    pub max_output_tokens: Option<MaxOut>,
    pub tools: Option<Vec<ToolDef>>,
    pub tool_choice: Option<ToolChoice>,
    /// Unrecognized session fields (incl. `temperature`, which GA removed) — preserved verbatim.
    pub extra: Map<String, Value>,
}

/// The per-turn overlay from `response.create.response`. `persist` is `false` when
/// `conversation == "none"` (an out-of-band response not added to session history).
#[derive(Debug, Clone, PartialEq)]
pub struct ResponseOverride {
    pub instructions: Option<String>,
    pub output_modalities: Option<Vec<String>>,
    pub max_output_tokens: Option<MaxOut>,
    pub tools: Option<Vec<ToolDef>>,
    pub tool_choice: Option<ToolChoice>,
    pub persist: bool,
}

impl Default for ResponseOverride {
    fn default() -> Self {
        ResponseOverride {
            instructions: None,
            output_modalities: None,
            max_output_tokens: None,
            tools: None,
            tool_choice: None,
            persist: true,
        }
    }
}

/// Parse one inbound Realtime client event JSON into a [`ClientEvent`].
pub fn parse_client_event(value: Value) -> ClientEvent {
    let Some(kind) = value.get("type").and_then(Value::as_str) else {
        return ClientEvent::Other;
    };
    match kind {
        "session.update" => {
            let session = value.get("session").cloned().unwrap_or(Value::Null);
            ClientEvent::SessionUpdate(parse_session_patch(session))
        }
        "conversation.item.create" => parse_item_create(value.get("item")),
        "response.create" => {
            let resp = value.get("response").cloned().unwrap_or(Value::Null);
            ClientEvent::Create(parse_response_override(resp))
        }
        "response.cancel" => ClientEvent::Cancel,
        "conversation.item.delete" => match value.get("item_id").and_then(Value::as_str) {
            Some(id) => ClientEvent::DeleteItem(id.to_string()),
            None => ClientEvent::Other,
        },
        "input_audio_buffer.append" | "input_audio_buffer.commit" | "input_audio_buffer.clear" => {
            ClientEvent::AudioUnsupported
        }
        _ => ClientEvent::Other,
    }
}

fn parse_session_patch(session: Value) -> SessionPatch {
    let Value::Object(obj) = session else {
        return SessionPatch::default();
    };
    let mut patch = SessionPatch::default();
    for (k, v) in obj {
        match k.as_str() {
            "type" => {} // the constant discriminator ("realtime") — nothing to carry.
            "model" => patch.model = v.as_str().map(str::to_owned),
            "instructions" => patch.instructions = v.as_str().map(str::to_owned),
            "output_modalities" | "modalities" => {
                if let Some(m) = parse_modalities(&v) {
                    patch.output_modalities = Some(m);
                }
            }
            "max_output_tokens" => patch.max_output_tokens = parse_max_out(&v),
            "tools" => patch.tools = Some(parse_tools(&v)),
            "tool_choice" => patch.tool_choice = parse_tool_choice(&v),
            // Everything else (temperature, audio, voice, turn_detection, prompt, …) rides verbatim.
            _ => {
                patch.extra.insert(k, v);
            }
        }
    }
    patch
}

fn parse_response_override(resp: Value) -> ResponseOverride {
    let mut over = ResponseOverride::default();
    let Value::Object(obj) = resp else {
        return over;
    };
    for (k, v) in obj {
        match k.as_str() {
            "instructions" => over.instructions = v.as_str().map(str::to_owned),
            "output_modalities" | "modalities" => over.output_modalities = parse_modalities(&v),
            "max_output_tokens" => over.max_output_tokens = parse_max_out(&v),
            "tools" => over.tools = Some(parse_tools(&v)),
            "tool_choice" => over.tool_choice = parse_tool_choice(&v),
            // `conversation: "none"` ⇒ out-of-band, don't persist the assistant turn into history.
            "conversation" => over.persist = v.as_str() != Some("none"),
            _ => {}
        }
    }
    over
}

fn parse_item_create(item: Option<&Value>) -> ClientEvent {
    let Some(item) = item else {
        return ClientEvent::Other;
    };
    match item.get("type").and_then(Value::as_str) {
        Some("message") => {
            // Only user input drives the bridge; assistant/system items are tolerated/ignored.
            if item.get("role").and_then(Value::as_str) != Some("user") {
                return ClientEvent::Other;
            }
            let Some(parts) = item.get("content").and_then(Value::as_array) else {
                return ClientEvent::AddUserMessage(String::new());
            };
            let mut text = String::new();
            for part in parts {
                match part.get("type").and_then(Value::as_str) {
                    Some("input_text") | Some("text") => {
                        if let Some(t) = part.get("text").and_then(Value::as_str) {
                            text.push_str(t);
                        }
                    }
                    // Any audio part means this turn needs audio the text bridge can't handle.
                    Some("input_audio") | Some("audio") => return ClientEvent::AudioUnsupported,
                    _ => {}
                }
            }
            ClientEvent::AddUserMessage(text)
        }
        Some("function_call_output") => {
            let call_id = item
                .get("call_id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            let output = item
                .get("output")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            ClientEvent::AddFunctionOutput { call_id, output }
        }
        _ => ClientEvent::Other,
    }
}

fn parse_modalities(v: &Value) -> Option<Vec<String>> {
    v.as_array().map(|arr| {
        arr.iter()
            .filter_map(|m| m.as_str().map(str::to_owned))
            .collect()
    })
}

fn parse_max_out(v: &Value) -> Option<MaxOut> {
    if v.as_str() == Some("inf") {
        return Some(MaxOut::Inf);
    }
    v.as_u64().map(|n| MaxOut::Num(n as u32))
}

/// Realtime tools are FLAT: `{type:"function", name, description, parameters}` (unlike Chat
/// Completions, which nests under `function`). Map each to a canonical [`ToolDef`].
fn parse_tools(v: &Value) -> Vec<ToolDef> {
    let Some(arr) = v.as_array() else {
        return Vec::new();
    };
    arr.iter()
        .filter(|t| {
            t.get("type").and_then(Value::as_str) != Some("function") || t.get("name").is_some()
        })
        .filter_map(|t| {
            let name = t.get("name").and_then(Value::as_str)?.to_string();
            Some(ToolDef {
                name,
                description: t
                    .get("description")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                parameters: t.get("parameters").cloned().unwrap_or(Value::Null),
            })
        })
        .collect()
}

fn parse_tool_choice(v: &Value) -> Option<ToolChoice> {
    match v {
        Value::String(s) => match s.as_str() {
            "auto" => Some(ToolChoice::Auto),
            "none" => Some(ToolChoice::None),
            "required" => Some(ToolChoice::Required),
            _ => None,
        },
        Value::Object(o) => o
            .get("name")
            .and_then(Value::as_str)
            .map(|n| ToolChoice::Named(n.to_string())),
        _ => None,
    }
}

// ---------------------------------------------------------------------------------------------
// Outbound: canonical StreamChunks -> OpenAI Realtime server event scaffold
// ---------------------------------------------------------------------------------------------

/// Synthesizes the GA Realtime *server* event sequence for one generation turn from the canonical
/// [`StreamChunk`]s the bridge feeds it. It owns all id/sequence bookkeeping (a plain text stream
/// carries none), so a drop-in OpenAI Realtime client sees a consistent
/// `response.created → output_item.added → content_part.added → output_text.delta* →
/// output_text.done → content_part.done → output_item.done → conversation.item.done → response.done`
/// sequence (function-call items interleave their own added/delta/done frames).
///
/// Frames are built as `serde_json::Value`s: realtime is a session-scoped path, not the
/// allocation-ceilinged chat hot path, so clarity wins over the borrowed-buffer trick `ChunkEncoder`
/// uses for the SSE surface.
pub struct ResponseScaffold {
    response_id: String,
    item_id: String,
    seq: u64,
    text_opened: bool,
    text: String,
    tools: std::collections::BTreeMap<u32, ToolAcc>,
}

#[derive(Default)]
struct ToolAcc {
    call_id: String,
    name: String,
    arguments: String,
    item_id: String,
    output_index: u32,
    opened: bool,
}

impl ResponseScaffold {
    /// Start a turn. `response_id` (`resp_…`) and `item_id` (`msg_…`) are minted by the session so they
    /// are unique and stable across every frame of the turn.
    pub fn new(response_id: String, item_id: String) -> Self {
        ResponseScaffold {
            response_id,
            item_id,
            seq: 0,
            text_opened: false,
            text: String::new(),
            tools: std::collections::BTreeMap::new(),
        }
    }

    /// The assistant turn as a canonical [`Message`], for persisting into the bridge's history so the
    /// next turn replays it (Realtime is stateful; chat completions is not). Call after the stream ends.
    pub fn assistant_message(&self) -> Message {
        let mut content = Vec::new();
        if !self.text.is_empty() {
            content.push(ContentPart::Text {
                text: self.text.clone(),
            });
        }
        let tool_calls = self
            .tools
            .values()
            .map(|t| ToolCall {
                id: t.call_id.clone(),
                name: t.name.clone(),
                arguments: t.arguments.clone(),
            })
            .collect();
        Message {
            role: Role::Assistant,
            content,
            tool_calls,
            tool_call_id: None,
            name: None,
        }
    }

    /// The opening `response.created` frame, sent before any chunk arrives.
    pub fn created(&self) -> Value {
        json!({
            "type": "response.created",
            "response": { "id": self.response_id, "object": "realtime.response", "status": "in_progress" }
        })
    }

    /// Expand one canonical chunk into zero or more server frames, appended to `out`.
    pub fn on_chunk(&mut self, chunk: &StreamChunk, out: &mut Vec<Value>) {
        match chunk {
            // The opening id/model is already captured; the message item is opened lazily on content.
            StreamChunk::Start { .. } => {}
            StreamChunk::Content { delta, .. } => self.on_content(delta, out),
            StreamChunk::ToolCall { call, .. } => self.on_tool_call(call, out),
            // Usage/Finish are folded into the terminal frame by `finish`.
            StreamChunk::Usage(_) | StreamChunk::Finish { .. } => {}
        }
    }

    fn on_content(&mut self, delta: &str, out: &mut Vec<Value>) {
        if !self.text_opened {
            out.push(json!({
                "type": "response.output_item.added",
                "response_id": self.response_id,
                "output_index": 0,
                "item": { "id": self.item_id, "object": "realtime.item", "type": "message",
                          "role": "assistant", "content": [] }
            }));
            out.push(json!({
                "type": "response.content_part.added",
                "response_id": self.response_id,
                "item_id": self.item_id,
                "output_index": 0,
                "content_index": 0,
                "part": { "type": "output_text", "text": "" }
            }));
            self.text_opened = true;
        }
        self.text.push_str(delta);
        self.seq += 1;
        out.push(json!({
            "type": "response.output_text.delta",
            "response_id": self.response_id,
            "item_id": self.item_id,
            "output_index": 0,
            "content_index": 0,
            "delta": delta,
            "sequence_number": self.seq
        }));
    }

    fn on_tool_call(&mut self, call: &ToolCallDelta, out: &mut Vec<Value>) {
        let item_id = format!("{}_call_{}", self.item_id, call.index);
        let entry = self.tools.entry(call.index).or_insert_with(|| ToolAcc {
            item_id,
            // The message item is output_index 0; tool items follow it.
            output_index: 1 + call.index,
            ..ToolAcc::default()
        });
        if let Some(id) = &call.id {
            entry.call_id = id.clone();
        }
        if let Some(name) = &call.name {
            entry.name = name.clone();
        }
        if !entry.opened {
            out.push(json!({
                "type": "response.output_item.added",
                "response_id": self.response_id,
                "output_index": entry.output_index,
                "item": { "id": entry.item_id, "object": "realtime.item", "type": "function_call",
                          "call_id": entry.call_id, "name": entry.name, "arguments": "" }
            }));
            entry.opened = true;
        }
        if let Some(args) = &call.arguments {
            entry.arguments.push_str(args);
            self.seq += 1;
            out.push(json!({
                "type": "response.function_call_arguments.delta",
                "response_id": self.response_id,
                "item_id": entry.item_id,
                "output_index": entry.output_index,
                "call_id": entry.call_id,
                "delta": args,
                "sequence_number": self.seq
            }));
        }
    }

    /// Close the turn: emit the part/item `.done` frames, then the terminal `response.done` carrying
    /// the assembled `output[]` and mapped `usage`.
    pub fn finish(&mut self, reason: FinishReason, usage: &Usage, out: &mut Vec<Value>) {
        let mut output: Vec<Value> = Vec::new();

        if self.text_opened {
            let item = json!({
                "id": self.item_id, "object": "realtime.item", "type": "message", "role": "assistant",
                "content": [{ "type": "output_text", "text": self.text }], "status": "completed"
            });
            out.push(json!({
                "type": "response.output_text.done",
                "response_id": self.response_id,
                "item_id": self.item_id, "output_index": 0, "content_index": 0,
                "text": self.text
            }));
            out.push(json!({
                "type": "response.content_part.done",
                "response_id": self.response_id,
                "item_id": self.item_id, "output_index": 0, "content_index": 0,
                "part": { "type": "output_text", "text": self.text }
            }));
            out.push(json!({
                "type": "response.output_item.done",
                "response_id": self.response_id, "output_index": 0, "item": item
            }));
            out.push(json!({ "type": "conversation.item.done", "item": json!({
                "id": self.item_id, "object": "realtime.item", "type": "message", "role": "assistant",
                "content": [{ "type": "output_text", "text": self.text }], "status": "completed"
            }) }));
            output.push(json!({
                "id": self.item_id, "type": "message", "role": "assistant", "status": "completed",
                "content": [{ "type": "output_text", "text": self.text }]
            }));
        }

        for tool in self.tools.values() {
            let item = json!({
                "id": tool.item_id, "object": "realtime.item", "type": "function_call",
                "call_id": tool.call_id, "name": tool.name, "arguments": tool.arguments, "status": "completed"
            });
            out.push(json!({
                "type": "response.function_call_arguments.done",
                "response_id": self.response_id,
                "item_id": tool.item_id, "output_index": tool.output_index, "call_id": tool.call_id,
                "name": tool.name, "arguments": tool.arguments
            }));
            out.push(json!({
                "type": "response.output_item.done",
                "response_id": self.response_id, "output_index": tool.output_index, "item": item
            }));
            output.push(json!({
                "id": tool.item_id, "type": "function_call", "call_id": tool.call_id,
                "name": tool.name, "arguments": tool.arguments, "status": "completed"
            }));
        }

        out.push(json!({
            "type": "response.done",
            "response": {
                "id": self.response_id,
                "object": "realtime.response",
                "status": status_for(reason),
                "output": output,
                "usage": usage_json(usage)
            }
        }));
    }
}

fn status_for(reason: FinishReason) -> &'static str {
    match reason {
        FinishReason::Stop | FinishReason::ToolCalls => "completed",
        FinishReason::Length | FinishReason::ContentFilter => "incomplete",
        FinishReason::Error => "failed",
    }
}

/// Map canonical [`Usage`] into the Realtime `response.done.usage` object. `cost_usd` is an llmleaf
/// extension and is deliberately NOT surfaced here (it is not part of OpenAI's schema); it still flows
/// on the internal event bus. The text bridge has only text tokens, so audio/cached details are zero.
pub fn usage_json(u: &Usage) -> Value {
    json!({
        "total_tokens": u.total_tokens,
        "input_tokens": u.prompt_tokens,
        "output_tokens": u.completion_tokens,
        "input_token_details": { "cached_tokens": 0, "text_tokens": u.prompt_tokens, "audio_tokens": 0 },
        "output_token_details": { "text_tokens": u.completion_tokens, "audio_tokens": 0 }
    })
}

/// Extract the consumer token from the browser-pattern handshake: the OpenAI Realtime SDK encodes it
/// as the `openai-insecure-api-key.<token>` element of the `Sec-WebSocket-Protocol` list. This
/// provider-SDK-specific subprotocol name is dialect knowledge and so lives here at the edge, never in
/// the core's generic auth path (principle 2).
pub fn token_from_subprotocols(protocols: &str) -> Option<String> {
    protocols.split(',').map(str::trim).find_map(|p| {
        p.strip_prefix("openai-insecure-api-key.")
            .map(str::to_string)
    })
}

/// A top-level Realtime `error` frame.
pub fn error_frame(code: &str, message: &str) -> Value {
    json!({ "type": "error", "error": { "type": "invalid_request_error", "code": code, "message": message } })
}

/// Extract usage from a native provider's outbound `response.done` text frame, for the event bus
/// (principle 5). `None` for any other frame. This is the core reading its *own* consumer dialect to
/// account a native session — it never interprets a provider's private wire.
pub fn usage_from_server_frame(text: &str) -> Option<Usage> {
    let v: Value = serde_json::from_str(text).ok()?;
    if v.get("type").and_then(Value::as_str) != Some("response.done") {
        return None;
    }
    let usage = v.get("response")?.get("usage")?;
    Some(Usage {
        prompt_tokens: usage
            .get("input_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0),
        completion_tokens: usage
            .get("output_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0),
        total_tokens: usage
            .get("total_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0),
        cost_usd: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_text_session_update() {
        let ev = parse_client_event(json!({
            "type": "session.update",
            "session": {
                "type": "realtime",
                "model": "gpt-realtime",
                "modalities": ["text"],                 // beta alias accepted
                "instructions": "be terse",
                "max_output_tokens": "inf",
                "temperature": 0.7,                      // GA removed it — must be parked, not surfaced
                "tools": [{ "type": "function", "name": "get_weather", "description": "w",
                            "parameters": { "type": "object" } }],
                "tool_choice": "required"
            }
        }));
        let ClientEvent::SessionUpdate(p) = ev else {
            panic!("expected SessionUpdate")
        };
        assert_eq!(p.model.as_deref(), Some("gpt-realtime"));
        assert_eq!(p.instructions.as_deref(), Some("be terse"));
        assert_eq!(
            p.output_modalities.as_deref(),
            Some(&["text".to_string()][..])
        );
        assert_eq!(p.max_output_tokens, Some(MaxOut::Inf));
        assert_eq!(p.tools.as_ref().unwrap()[0].name, "get_weather");
        assert_eq!(p.tool_choice, Some(ToolChoice::Required));
        // temperature is preserved in extra, never promoted to a canonical field.
        assert!(p.extra.contains_key("temperature"));
    }

    #[test]
    fn parses_user_message_and_function_output() {
        let m = parse_client_event(json!({
            "type": "conversation.item.create",
            "item": { "type": "message", "role": "user",
                      "content": [{ "type": "input_text", "text": "hi " }, { "type": "input_text", "text": "there" }] }
        }));
        assert_eq!(m, ClientEvent::AddUserMessage("hi there".to_string()));

        let f = parse_client_event(json!({
            "type": "conversation.item.create",
            "item": { "type": "function_call_output", "call_id": "call_1", "output": "{\"ok\":true}" }
        }));
        assert_eq!(
            f,
            ClientEvent::AddFunctionOutput {
                call_id: "call_1".into(),
                output: "{\"ok\":true}".into()
            }
        );
    }

    #[test]
    fn audio_input_is_flagged_unsupported() {
        assert_eq!(
            parse_client_event(json!({ "type": "input_audio_buffer.append", "audio": "AA==" })),
            ClientEvent::AudioUnsupported
        );
        let audio_msg = parse_client_event(json!({
            "type": "conversation.item.create",
            "item": { "type": "message", "role": "user", "content": [{ "type": "input_audio", "audio": "AA==" }] }
        }));
        assert_eq!(audio_msg, ClientEvent::AudioUnsupported);
    }

    #[test]
    fn response_create_conversation_none_does_not_persist() {
        let ev = parse_client_event(
            json!({ "type": "response.create", "response": { "conversation": "none" } }),
        );
        let ClientEvent::Create(o) = ev else {
            panic!("expected Create")
        };
        assert!(!o.persist);
        // The default (no conversation field) persists.
        let ev = parse_client_event(json!({ "type": "response.create" }));
        let ClientEvent::Create(o) = ev else {
            panic!("expected Create")
        };
        assert!(o.persist);
    }

    #[test]
    fn scaffold_text_turn_golden_sequence() {
        let mut s = ResponseScaffold::new("resp_1".into(), "msg_1".into());
        let mut out = vec![s.created()];
        s.on_chunk(
            &StreamChunk::Start {
                id: "x".into(),
                model: "m".into(),
            },
            &mut out,
        );
        s.on_chunk(
            &StreamChunk::Content {
                index: 0,
                delta: "Hel".into(),
            },
            &mut out,
        );
        s.on_chunk(
            &StreamChunk::Content {
                index: 0,
                delta: "lo".into(),
            },
            &mut out,
        );
        s.finish(
            FinishReason::Stop,
            &Usage {
                prompt_tokens: 3,
                completion_tokens: 2,
                total_tokens: 5,
                cost_usd: Some(0.01),
            },
            &mut out,
        );

        let types: Vec<&str> = out.iter().map(|f| f["type"].as_str().unwrap()).collect();
        assert_eq!(
            types,
            vec![
                "response.created",
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.delta",
                "response.output_text.delta",
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "conversation.item.done",
                "response.done",
            ]
        );
        // Monotonic sequence numbers on the two deltas.
        assert_eq!(out[3]["sequence_number"], 1);
        assert_eq!(out[4]["sequence_number"], 2);
        // Consistent ids across the turn.
        assert!(out
            .iter()
            .all(|f| f.get("response_id").is_none() || f["response_id"] == "resp_1"));
        assert_eq!(out[5]["text"], "Hello");
        // response.done carries mapped usage and the assembled output; never the cost extension.
        let done = out.last().unwrap();
        assert_eq!(done["response"]["status"], "completed");
        assert_eq!(done["response"]["usage"]["input_tokens"], 3);
        assert_eq!(done["response"]["usage"]["output_tokens"], 2);
        assert!(done["response"]["usage"].get("cost_usd").is_none());
        assert_eq!(done["response"]["output"][0]["content"][0]["text"], "Hello");
    }

    #[test]
    fn scaffold_tool_call_sequence() {
        let mut s = ResponseScaffold::new("resp_2".into(), "msg_2".into());
        let mut out = Vec::new();
        s.on_chunk(
            &StreamChunk::ToolCall {
                index: 0,
                call: ToolCallDelta {
                    index: 0,
                    id: Some("call_9".into()),
                    name: Some("get_weather".into()),
                    arguments: Some("{\"city\":".into()),
                },
            },
            &mut out,
        );
        s.on_chunk(
            &StreamChunk::ToolCall {
                index: 0,
                call: ToolCallDelta {
                    index: 0,
                    id: None,
                    name: None,
                    arguments: Some("\"NYC\"}".into()),
                },
            },
            &mut out,
        );
        s.finish(FinishReason::ToolCalls, &Usage::default(), &mut out);

        let types: Vec<&str> = out.iter().map(|f| f["type"].as_str().unwrap()).collect();
        assert_eq!(
            types,
            vec![
                "response.output_item.added",
                "response.function_call_arguments.delta",
                "response.function_call_arguments.delta",
                "response.function_call_arguments.done",
                "response.output_item.done",
                "response.done",
            ]
        );
        assert_eq!(out[0]["item"]["call_id"], "call_9");
        assert_eq!(out[0]["item"]["name"], "get_weather");
        assert_eq!(out[3]["arguments"], "{\"city\":\"NYC\"}");
        assert_eq!(
            out.last().unwrap()["response"]["output"][0]["type"],
            "function_call"
        );
    }

    #[test]
    fn extracts_usage_from_native_response_done() {
        let frame = json!({
            "type": "response.done",
            "response": { "usage": { "input_tokens": 11, "output_tokens": 4, "total_tokens": 15 } }
        })
        .to_string();
        let u = usage_from_server_frame(&frame).unwrap();
        assert_eq!(u.prompt_tokens, 11);
        assert_eq!(u.completion_tokens, 4);
        assert_eq!(u.total_tokens, 15);
        // A non-done frame yields nothing.
        assert!(usage_from_server_frame(
            &json!({ "type": "response.output_text.delta" }).to_string()
        )
        .is_none());
    }
}
