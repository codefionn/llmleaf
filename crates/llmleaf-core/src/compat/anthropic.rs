//! The Anthropic Messages API compat surface.
//!
//! In: an Anthropic `POST /v1/messages` request JSON → canonical [`ChatRequest`]. Anthropic's wire
//! shape is its own (decision filter: match the documented dialect even when awkward): `system` is a
//! top-level field rather than a message, message content is a list of typed blocks, tool results ride
//! inside a user turn, and `max_tokens` is mandatory. All of that is mapped here so the core stays
//! dialect-free (principle 3). Unknown fields (`top_k`, `metadata`, `thinking`, …) are preserved
//! verbatim in [`ChatRequest::extra`] (principle 7: transparent — we never silently drop). The
//! canonical [`Thinking`](llmleaf_model::Thinking) ladder is deliberately left `None`: the consumer's
//! own `thinking` object rides through `extra` untouched, so a request routed to an Anthropic upstream
//! reaches it verbatim — exactly as the OpenAI surface leaves `reasoning_effort` alone.
//!
//! Out: a canonical [`ResponseStream`](llmleaf_model::ResponseStream)/[`ChatResponse`] → either a
//! single Anthropic `message` object ([`response_to_anthropic`], a collected stream — principle 4) or
//! the Anthropic streaming-event sequence ([`EventEncoder`]): `message_start`, then per content block a
//! `content_block_start` / `content_block_delta` / `content_block_stop` bracket, then a single
//! `message_delta` carrying the stop reason and final usage, then `message_stop`.
//!
//! This is the inverse of the Anthropic *provider* (`llmleaf-providers`): that maps canonical → Anthropic
//! to *call* Anthropic; this maps Anthropic → canonical to *serve* an Anthropic-shaped consumer.

use llmleaf_model::{
    ChatRequest, ChatResponse, ContentPart, FinishReason, Message, ModelError, Role, StreamChunk,
    ToolCall, ToolChoice, ToolDef, Usage,
};
use serde::Serialize;
use serde_json::{json, Map, Value};

fn mapping(msg: impl Into<String>) -> ModelError {
    ModelError::Mapping(msg.into())
}

// ---------------------------------------------------------------------------------------------
// Inbound: Anthropic Messages request JSON -> canonical ChatRequest
// ---------------------------------------------------------------------------------------------

/// Map an Anthropic `messages` request into the canonical [`ChatRequest`].
///
/// `max_tokens` is required, matching the Anthropic API exactly (fidelity beats convenience): a request
/// without it is rejected rather than silently defaulted. `system` (string or text-block array) is
/// hoisted to a leading canonical [`Role::System`] message; the conversation's typed content blocks are
/// expanded into canonical messages (tool results become [`Role::Tool`] messages, as in the OpenAI
/// dialect). Everything we don't model rides through in [`ChatRequest::extra`].
pub fn parse_messages_request(value: Value) -> Result<ChatRequest, ModelError> {
    let Value::Object(mut obj) = value else {
        return Err(mapping("request body must be a JSON object"));
    };

    let model = match obj.remove("model") {
        Some(Value::String(s)) => s,
        _ => return Err(mapping("`model` is required and must be a string")),
    };

    // Anthropic requires `max_tokens`; honor that exactly (principle: dialect fidelity over convenience).
    let max_tokens = match obj.remove("max_tokens") {
        Some(v) => v
            .as_u64()
            .map(|n| n as u32)
            .ok_or_else(|| mapping("`max_tokens` must be an integer"))?,
        None => return Err(mapping("`max_tokens` is required")),
    };

    let mut messages = Vec::new();
    // `system` is a top-level field in this dialect; hoist it to a leading System message so the rest of
    // the core (and any provider) sees it uniformly. Empty system text contributes no message.
    if let Some(system) = obj.remove("system") {
        let text = system_text(system);
        if !text.is_empty() {
            messages.push(Message::text(Role::System, text));
        }
    }

    let convo = match obj.remove("messages") {
        Some(Value::Array(items)) => items,
        _ => return Err(mapping("`messages` is required and must be an array")),
    };
    for item in convo {
        parse_message(item, &mut messages)?;
    }

    let temperature = obj
        .remove("temperature")
        .and_then(|v| v.as_f64())
        .map(|n| n as f32);
    let top_p = obj
        .remove("top_p")
        .and_then(|v| v.as_f64())
        .map(|n| n as f32);
    let stop = parse_stop_sequences(obj.remove("stop_sequences"));
    let stream = obj
        .remove("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let tools = parse_tools(obj.remove("tools"))?;
    let tool_choice = parse_tool_choice(obj.remove("tool_choice"));

    // `top_k`, `thinking`, `metadata`, `service_tier`, and anything else we don't model ride through
    // untouched (principle 7). Leaving `thinking` here means a request routed to an Anthropic upstream
    // reaches it verbatim, while the canonical `thinking` ladder stays `None`.
    let extra = obj;

    Ok(ChatRequest {
        model,
        messages,
        max_tokens: Some(max_tokens),
        temperature,
        top_p,
        stop,
        stream,
        tools,
        tool_choice,
        thinking: None,
        extra,
    })
}

/// Flatten the top-level `system` field (a string, or an array of text blocks) into one string. Blocks
/// are joined with a blank line, mirroring how the Anthropic provider concatenates hoisted system text.
fn system_text(value: Value) -> String {
    match value {
        Value::String(s) => s,
        Value::Array(parts) => {
            let mut out = String::new();
            for part in parts {
                if let Some(text) = part.get("text").and_then(Value::as_str) {
                    if !out.is_empty() {
                        out.push_str("\n\n");
                    }
                    out.push_str(text);
                }
            }
            out
        }
        _ => String::new(),
    }
}

/// Parse one Anthropic message (`user` or `assistant`) into one or more canonical messages, appended to
/// `out`. A user turn may expand into several canonical messages: each `tool_result` block becomes its
/// own [`Role::Tool`] message (emitted first, as Anthropic places results at the head of the turn),
/// followed by a [`Role::User`] message carrying any remaining text/image parts.
fn parse_message(value: Value, out: &mut Vec<Message>) -> Result<(), ModelError> {
    let Value::Object(mut obj) = value else {
        return Err(mapping("each message must be an object"));
    };
    let role = match obj.remove("role") {
        Some(Value::String(r)) => r,
        _ => return Err(mapping("message is missing `role`")),
    };
    let content = obj.remove("content");
    match role.as_str() {
        "user" => parse_user_content(content, out)?,
        "assistant" => out.push(parse_assistant_message(content)?),
        // Anthropic carries the system prompt top-level; only user/assistant appear in `messages`.
        other => return Err(mapping(format!("unknown message role `{other}`"))),
    }
    Ok(())
}

fn parse_user_content(content: Option<Value>, out: &mut Vec<Message>) -> Result<(), ModelError> {
    match content {
        None | Some(Value::Null) => out.push(user_message(Vec::new())),
        Some(Value::String(text)) => out.push(Message::text(Role::User, text)),
        Some(Value::Array(blocks)) => {
            let mut tool_msgs: Vec<Message> = Vec::new();
            let mut parts: Vec<ContentPart> = Vec::new();
            for block in blocks {
                let Value::Object(obj) = &block else {
                    return Err(mapping("content block must be an object"));
                };
                match obj.get("type").and_then(Value::as_str) {
                    Some("text") => parts.push(ContentPart::Text {
                        text: obj
                            .get("text")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string(),
                    }),
                    Some("image") => parts.push(image_part(obj)?),
                    Some("tool_result") => {
                        let id = obj
                            .get("tool_use_id")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string();
                        let text = obj.get("content").map(block_text).unwrap_or_default();
                        tool_msgs.push(Message {
                            role: Role::Tool,
                            content: vec![ContentPart::Text { text }],
                            tool_calls: Vec::new(),
                            tool_call_id: Some(id),
                            name: None,
                        });
                    }
                    other => {
                        return Err(mapping(format!("unsupported user content block {other:?}")))
                    }
                }
            }
            let had_tool = !tool_msgs.is_empty();
            out.append(&mut tool_msgs);
            // Emit the user turn for the leftover parts. Skip it only when the turn was *purely* tool
            // results (the canonical Tool messages already represent it); an otherwise-empty turn is
            // preserved as an empty user message.
            if !parts.is_empty() || !had_tool {
                out.push(user_message(parts));
            }
        }
        Some(_) => {
            return Err(mapping(
                "message `content` must be a string, array, or null",
            ))
        }
    }
    Ok(())
}

fn parse_assistant_message(content: Option<Value>) -> Result<Message, ModelError> {
    let mut parts: Vec<ContentPart> = Vec::new();
    let mut tool_calls: Vec<ToolCall> = Vec::new();
    match content {
        None | Some(Value::Null) => {}
        Some(Value::String(text)) => parts.push(ContentPart::Text { text }),
        Some(Value::Array(blocks)) => {
            for block in blocks {
                let Value::Object(obj) = &block else {
                    return Err(mapping("content block must be an object"));
                };
                match obj.get("type").and_then(Value::as_str) {
                    Some("text") => parts.push(ContentPart::Text {
                        text: obj
                            .get("text")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string(),
                    }),
                    Some("tool_use") => {
                        let id = obj
                            .get("id")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string();
                        let name = obj
                            .get("name")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string();
                        // Canonical tool arguments are the raw JSON string the model emitted; serialize
                        // the structured `input` object back to one (compact). Absent input → `{}`.
                        let arguments = obj
                            .get("input")
                            .map(Value::to_string)
                            .unwrap_or_else(|| "{}".to_string());
                        tool_calls.push(ToolCall {
                            id,
                            name,
                            arguments,
                        });
                    }
                    // Reasoning blocks are preserved verbatim so a client echoing the prior turn back
                    // (the stateless multi-turn pattern) reaches the upstream intact — the `signature`
                    // especially, which Anthropic rejects the turn without. They lead the turn, so
                    // pushing them in arrival order keeps them ahead of any `tool_use`.
                    Some("thinking") => parts.push(ContentPart::Thinking {
                        thinking: obj
                            .get("thinking")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string(),
                        signature: obj
                            .get("signature")
                            .and_then(Value::as_str)
                            .map(str::to_string),
                    }),
                    Some("redacted_thinking") => parts.push(ContentPart::RedactedThinking {
                        data: obj
                            .get("data")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string(),
                    }),
                    other => {
                        return Err(mapping(format!(
                            "unsupported assistant content block {other:?}"
                        )))
                    }
                }
            }
        }
        Some(_) => {
            return Err(mapping(
                "assistant `content` must be a string, array, or null",
            ))
        }
    }
    Ok(Message {
        role: Role::Assistant,
        content: parts,
        tool_calls,
        tool_call_id: None,
        name: None,
    })
}

fn user_message(content: Vec<ContentPart>) -> Message {
    Message {
        role: Role::User,
        content,
        tool_calls: Vec::new(),
        tool_call_id: None,
        name: None,
    }
}

/// Map an Anthropic `image` block to a canonical [`ContentPart::ImageUrl`]. A `url` source passes its
/// URL through; a `base64` source is folded into a `data:` URL so the single canonical image shape
/// carries both Anthropic source kinds.
fn image_part(block: &Map<String, Value>) -> Result<ContentPart, ModelError> {
    let source = block
        .get("source")
        .ok_or_else(|| mapping("image block missing `source`"))?;
    match source.get("type").and_then(Value::as_str) {
        Some("url") => {
            let url = source
                .get("url")
                .and_then(Value::as_str)
                .ok_or_else(|| mapping("image url source missing `url`"))?
                .to_string();
            Ok(ContentPart::ImageUrl { url, detail: None })
        }
        Some("base64") => {
            let media_type = source
                .get("media_type")
                .and_then(Value::as_str)
                .unwrap_or("application/octet-stream");
            let data = source
                .get("data")
                .and_then(Value::as_str)
                .ok_or_else(|| mapping("image base64 source missing `data`"))?;
            Ok(ContentPart::ImageUrl {
                url: format!("data:{media_type};base64,{data}"),
                detail: None,
            })
        }
        other => Err(mapping(format!("unsupported image source {other:?}"))),
    }
}

/// Flatten a `tool_result` block's `content` (a string, or an array of text blocks) to a plain string.
fn block_text(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Array(parts) => {
            let mut out = String::new();
            for part in parts {
                if let Some(text) = part.get("text").and_then(Value::as_str) {
                    out.push_str(text);
                }
            }
            out
        }
        _ => String::new(),
    }
}

fn parse_tools(value: Option<Value>) -> Result<Vec<ToolDef>, ModelError> {
    let Some(Value::Array(items)) = value else {
        return Ok(Vec::new());
    };
    items
        .into_iter()
        .map(|t| {
            let name = t
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| mapping("tool missing `name`"))?
                .to_string();
            Ok(ToolDef {
                name,
                description: t
                    .get("description")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                // Anthropic names the parameter schema `input_schema`.
                parameters: t.get("input_schema").cloned().unwrap_or(Value::Null),
            })
        })
        .collect()
}

fn parse_tool_choice(value: Option<Value>) -> Option<ToolChoice> {
    let Some(Value::Object(o)) = value else {
        return None;
    };
    match o.get("type").and_then(Value::as_str) {
        Some("auto") => Some(ToolChoice::Auto),
        Some("any") => Some(ToolChoice::Required),
        Some("tool") => o
            .get("name")
            .and_then(Value::as_str)
            .map(|n| ToolChoice::Named(n.to_string())),
        Some("none") => Some(ToolChoice::None),
        _ => None,
    }
}

fn parse_stop_sequences(value: Option<Value>) -> Vec<String> {
    match value {
        Some(Value::String(s)) => vec![s],
        Some(Value::Array(arr)) => arr
            .into_iter()
            .filter_map(|v| v.as_str().map(str::to_owned))
            .collect(),
        _ => Vec::new(),
    }
}

// ---------------------------------------------------------------------------------------------
// Outbound: canonical -> Anthropic wire format
// ---------------------------------------------------------------------------------------------

/// Map a canonical [`FinishReason`] to Anthropic's `stop_reason` vocabulary (the inverse of the
/// Anthropic provider's `map_stop_reason`). `Error` has no Anthropic equivalent on a `200` message and
/// reads as a normal end.
fn finish_to_anthropic(reason: FinishReason) -> &'static str {
    match reason {
        FinishReason::Stop => "end_turn",
        FinishReason::Length => "max_tokens",
        FinishReason::ToolCalls => "tool_use",
        FinishReason::ContentFilter => "refusal",
        FinishReason::Error => "end_turn",
    }
}

/// Anthropic usage. `input_tokens`/`output_tokens` are the relayed provider counts; `cost_usd` is an
/// llmleaf enrichment (omitted when unknown), additive and ignored by Anthropic clients.
#[derive(Serialize)]
struct UsageOut {
    input_tokens: u64,
    output_tokens: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    cost_usd: Option<f64>,
}

impl From<&Usage> for UsageOut {
    fn from(u: &Usage) -> Self {
        UsageOut {
            input_tokens: u.prompt_tokens,
            output_tokens: u.completion_tokens,
            cost_usd: u.cost_usd,
        }
    }
}

/// One content block of an Anthropic `message`. `text` borrows straight from the response; `tool_use`
/// carries the structured `input` parsed from the canonical argument string.
#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ContentBlockOut<'a> {
    Text {
        text: &'a str,
    },
    Thinking {
        thinking: &'a str,
        #[serde(skip_serializing_if = "Option::is_none")]
        signature: Option<&'a str>,
    },
    RedactedThinking {
        data: &'a str,
    },
    ToolUse {
        id: &'a str,
        name: &'a str,
        input: Value,
    },
}

/// The Anthropic `message` object — the collected (non-streaming) response shape. Borrows string leaves
/// from the [`ChatResponse`] and the supplied `id`; the only owned allocations are the per-tool-call
/// parsed `input` values (the canonical arguments string is JSON we must re-inflate).
#[derive(Serialize)]
pub struct MessageView<'a> {
    id: &'a str,
    #[serde(rename = "type")]
    kind: &'static str,
    role: &'static str,
    model: &'a str,
    content: Vec<ContentBlockOut<'a>>,
    stop_reason: &'static str,
    // Always emitted as `null`: the canonical model does not track which stop sequence matched.
    stop_sequence: Option<&'static str>,
    usage: UsageOut,
}

/// Collect-mode mapping: a canonical [`ChatResponse`] to an Anthropic `message` object. Anthropic has no
/// multi-choice concept, so the single assistant turn is built from choice 0.
pub fn response_to_anthropic<'a>(resp: &'a ChatResponse, id: &'a str) -> MessageView<'a> {
    let choice = resp.choices.first();
    let mut content: Vec<ContentBlockOut> = Vec::new();
    if let Some(c) = choice {
        // Reasoning leads the turn (ahead of text and tool_use), carrying its signature verbatim.
        for part in &c.thinking {
            match part {
                ContentPart::Thinking {
                    thinking,
                    signature,
                } => content.push(ContentBlockOut::Thinking {
                    thinking,
                    signature: signature.as_deref(),
                }),
                ContentPart::RedactedThinking { data } => {
                    content.push(ContentBlockOut::RedactedThinking { data })
                }
                // `thinking` only ever holds reasoning parts; ignore anything else defensively.
                _ => {}
            }
        }
        if !c.text.is_empty() {
            content.push(ContentBlockOut::Text { text: &c.text });
        }
        for call in &c.tool_calls {
            content.push(ContentBlockOut::ToolUse {
                id: &call.id,
                name: &call.name,
                input: serde_json::from_str(&call.arguments).unwrap_or_else(|_| json!({})),
            });
        }
    }
    let stop_reason = choice
        .and_then(|c| c.finish_reason)
        .map(finish_to_anthropic)
        .unwrap_or("end_turn");

    MessageView {
        id,
        kind: "message",
        role: "assistant",
        model: &resp.model,
        content,
        stop_reason,
        stop_sequence: None,
        usage: UsageOut::from(&resp.usage),
    }
}

// ---------------------------------------------------------------------------------------------
// Outbound streaming: canonical stream -> Anthropic SSE event sequence
// ---------------------------------------------------------------------------------------------

/// One Anthropic SSE frame: a named event plus its serialized `data:` payload. The HTTP edge wraps each
/// in an `axum` SSE event (`event: <event>\ndata: <data>`); this module stays transport-free.
pub struct Frame {
    pub event: &'static str,
    pub data: String,
}

fn push<T: Serialize>(out: &mut Vec<Frame>, event: &'static str, payload: &T) {
    out.push(Frame {
        event,
        data: serde_json::to_string(payload).expect("anthropic event serialises"),
    });
}

enum OpenBlock {
    Text { index: u32 },
    Thinking { index: u32 },
    Tool { index: u32, canon: u32 },
}

impl OpenBlock {
    fn index(&self) -> u32 {
        match self {
            OpenBlock::Text { index }
            | OpenBlock::Thinking { index }
            | OpenBlock::Tool { index, .. } => *index,
        }
    }
}

/// Maps the canonical stream to Anthropic's streaming-event sequence. Stateful by necessity: Anthropic
/// brackets every content block with `content_block_start`/`content_block_stop` and emits a single
/// trailing `message_delta` that carries *both* the stop reason and the final usage — so the encoder
/// holds the open-block state and defers the closing frames to [`EventEncoder::finish`].
pub struct EventEncoder {
    id: String,
    model: String,
    started: bool,
    next_index: u32,
    open: Option<OpenBlock>,
    usage: Usage,
    stop_reason: Option<FinishReason>,
}

impl EventEncoder {
    pub fn new(id: impl Into<String>, model: impl Into<String>) -> Self {
        EventEncoder {
            id: id.into(),
            model: model.into(),
            started: false,
            next_index: 0,
            open: None,
            usage: Usage::default(),
            stop_reason: None,
        }
    }

    /// Emit `message_start` once. Defensive against a stream that begins with content (providers send
    /// [`StreamChunk::Start`] first, but the message envelope must precede any block regardless).
    fn ensure_started(&mut self, out: &mut Vec<Frame>) {
        if self.started {
            return;
        }
        self.started = true;
        push(
            out,
            "message_start",
            &MessageStartEvent {
                kind: "message_start",
                message: MessageEnvelope {
                    id: &self.id,
                    kind: "message",
                    role: "assistant",
                    model: &self.model,
                    content: [],
                    stop_reason: None,
                    stop_sequence: None,
                    // Input tokens are not known until the upstream usage arrives at the end; they are
                    // reported in the final `message_delta`. Anthropic clients accumulate usage fields
                    // across the stream, so the real counts still land.
                    usage: UsageStart {
                        input_tokens: 0,
                        output_tokens: 0,
                    },
                },
            },
        );
    }

    fn close_open(&mut self, out: &mut Vec<Frame>) {
        if let Some(block) = self.open.take() {
            push(
                out,
                "content_block_stop",
                &BlockStop {
                    kind: "content_block_stop",
                    index: block.index(),
                },
            );
        }
    }

    /// Ensure a `thinking` block is open and return its index. Reuses the current block when it is
    /// already a thinking block, so a run of thinking deltas and the trailing signature share one
    /// block; otherwise it closes whatever is open and starts a fresh thinking block.
    fn open_thinking(&mut self, out: &mut Vec<Frame>) -> u32 {
        if let Some(OpenBlock::Thinking { index }) = &self.open {
            return *index;
        }
        self.close_open(out);
        let index = self.next_index;
        self.next_index += 1;
        push(
            out,
            "content_block_start",
            &ThinkingBlockStart {
                kind: "content_block_start",
                index,
                content_block: ThinkingBlock {
                    kind: "thinking",
                    thinking: "",
                },
            },
        );
        self.open = Some(OpenBlock::Thinking { index });
        index
    }

    /// Translate one canonical chunk into zero or more Anthropic events, appended to `out` (cleared by
    /// the caller per chunk). `Usage` and `Finish` are buffered — they surface together in the trailing
    /// `message_delta` produced by [`finish`](Self::finish).
    pub fn encode(&mut self, chunk: &StreamChunk, out: &mut Vec<Frame>) {
        match chunk {
            StreamChunk::Start { .. } => self.ensure_started(out),
            StreamChunk::Content { delta, .. } => {
                self.ensure_started(out);
                let reuse = matches!(self.open, Some(OpenBlock::Text { .. }));
                let index = if reuse {
                    self.open.as_ref().expect("text block open").index()
                } else {
                    self.close_open(out);
                    let index = self.next_index;
                    self.next_index += 1;
                    push(
                        out,
                        "content_block_start",
                        &TextBlockStart {
                            kind: "content_block_start",
                            index,
                            content_block: TextBlock {
                                kind: "text",
                                text: "",
                            },
                        },
                    );
                    self.open = Some(OpenBlock::Text { index });
                    index
                };
                if !delta.is_empty() {
                    push(
                        out,
                        "content_block_delta",
                        &TextDeltaEvent {
                            kind: "content_block_delta",
                            index,
                            delta: TextDelta {
                                kind: "text_delta",
                                text: delta,
                            },
                        },
                    );
                }
            }
            StreamChunk::Thinking { delta, .. } => {
                self.ensure_started(out);
                let index = self.open_thinking(out);
                if !delta.is_empty() {
                    push(
                        out,
                        "content_block_delta",
                        &ThinkingDeltaEvent {
                            kind: "content_block_delta",
                            index,
                            delta: ThinkingDelta {
                                kind: "thinking_delta",
                                thinking: delta,
                            },
                        },
                    );
                }
            }
            StreamChunk::ThinkingSignature { signature, .. } => {
                self.ensure_started(out);
                // The signature attaches to the thinking block the deltas opened (consecutive on the
                // wire); reuse it, or open one defensively if the provider sent a bare signature.
                let index = self.open_thinking(out);
                push(
                    out,
                    "content_block_delta",
                    &SignatureDeltaEvent {
                        kind: "content_block_delta",
                        index,
                        delta: SignatureDelta {
                            kind: "signature_delta",
                            signature,
                        },
                    },
                );
            }
            StreamChunk::RedactedThinking { data, .. } => {
                self.ensure_started(out);
                self.close_open(out);
                // Redacted blocks arrive whole — a start carrying the data, then an immediate stop;
                // they have no deltas, so nothing stays open.
                let index = self.next_index;
                self.next_index += 1;
                push(
                    out,
                    "content_block_start",
                    &RedactedBlockStart {
                        kind: "content_block_start",
                        index,
                        content_block: RedactedBlock {
                            kind: "redacted_thinking",
                            data,
                        },
                    },
                );
                push(
                    out,
                    "content_block_stop",
                    &BlockStop {
                        kind: "content_block_stop",
                        index,
                    },
                );
            }
            StreamChunk::ToolCall { call, .. } => {
                self.ensure_started(out);
                let reuse =
                    matches!(self.open, Some(OpenBlock::Tool { canon, .. }) if canon == call.index);
                let index = if reuse {
                    self.open.as_ref().expect("tool block open").index()
                } else {
                    self.close_open(out);
                    let index = self.next_index;
                    self.next_index += 1;
                    push(
                        out,
                        "content_block_start",
                        &ToolBlockStart {
                            kind: "content_block_start",
                            index,
                            content_block: ToolBlock {
                                kind: "tool_use",
                                id: call.id.as_deref().unwrap_or(""),
                                name: call.name.as_deref().unwrap_or(""),
                                input: EmptyObject {},
                            },
                        },
                    );
                    self.open = Some(OpenBlock::Tool {
                        index,
                        canon: call.index,
                    });
                    index
                };
                if let Some(args) = &call.arguments {
                    if !args.is_empty() {
                        push(
                            out,
                            "content_block_delta",
                            &JsonDeltaEvent {
                                kind: "content_block_delta",
                                index,
                                delta: JsonDelta {
                                    kind: "input_json_delta",
                                    partial_json: args,
                                },
                            },
                        );
                    }
                }
            }
            StreamChunk::Usage(u) => self.usage = *u,
            StreamChunk::Finish { reason, .. } => self.stop_reason = Some(*reason),
        }
    }

    /// Emit the closing frames: stop any open block, then the single `message_delta` (stop reason +
    /// final usage), then `message_stop`. Always emits `message_start` first if the stream was empty, so
    /// the wire is a well-formed message even for a no-content response.
    pub fn finish(&mut self, out: &mut Vec<Frame>) {
        self.ensure_started(out);
        self.close_open(out);
        let stop_reason = self
            .stop_reason
            .map(finish_to_anthropic)
            .unwrap_or("end_turn");
        push(
            out,
            "message_delta",
            &MessageDeltaEvent {
                kind: "message_delta",
                delta: MessageDeltaBody {
                    stop_reason,
                    stop_sequence: None,
                },
                usage: UsageOut::from(&self.usage),
            },
        );
        push(
            out,
            "message_stop",
            &MessageStop {
                kind: "message_stop",
            },
        );
    }
}

/// The `data:` payload for a mid-stream `error` event. Anthropic surfaces an upstream failure as an
/// `event: error` frame rather than a `message_stop`.
pub fn error_data(message: &str) -> String {
    serde_json::to_string(&ErrorEvent {
        kind: "error",
        error: ErrorBody {
            kind: "api_error",
            message,
        },
    })
    .expect("anthropic error event serialises")
}

// --- Borrowed wire shapes for the streaming events (serialized straight to each frame's `data`). ---

#[derive(Serialize)]
struct EmptyObject {}

#[derive(Serialize)]
struct MessageStartEvent<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    message: MessageEnvelope<'a>,
}

#[derive(Serialize)]
struct MessageEnvelope<'a> {
    id: &'a str,
    #[serde(rename = "type")]
    kind: &'static str,
    role: &'static str,
    model: &'a str,
    content: [(); 0],
    stop_reason: Option<&'static str>,
    stop_sequence: Option<&'static str>,
    usage: UsageStart,
}

#[derive(Serialize)]
struct UsageStart {
    input_tokens: u64,
    output_tokens: u64,
}

#[derive(Serialize)]
struct TextBlockStart {
    #[serde(rename = "type")]
    kind: &'static str,
    index: u32,
    content_block: TextBlock,
}

#[derive(Serialize)]
struct TextBlock {
    #[serde(rename = "type")]
    kind: &'static str,
    text: &'static str,
}

#[derive(Serialize)]
struct ToolBlockStart<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    index: u32,
    content_block: ToolBlock<'a>,
}

#[derive(Serialize)]
struct ToolBlock<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    id: &'a str,
    name: &'a str,
    input: EmptyObject,
}

#[derive(Serialize)]
struct TextDeltaEvent<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    index: u32,
    delta: TextDelta<'a>,
}

#[derive(Serialize)]
struct TextDelta<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    text: &'a str,
}

#[derive(Serialize)]
struct JsonDeltaEvent<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    index: u32,
    delta: JsonDelta<'a>,
}

#[derive(Serialize)]
struct JsonDelta<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    partial_json: &'a str,
}

#[derive(Serialize)]
struct ThinkingBlockStart {
    #[serde(rename = "type")]
    kind: &'static str,
    index: u32,
    content_block: ThinkingBlock,
}

#[derive(Serialize)]
struct ThinkingBlock {
    #[serde(rename = "type")]
    kind: &'static str,
    thinking: &'static str,
}

#[derive(Serialize)]
struct ThinkingDeltaEvent<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    index: u32,
    delta: ThinkingDelta<'a>,
}

#[derive(Serialize)]
struct ThinkingDelta<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    thinking: &'a str,
}

#[derive(Serialize)]
struct SignatureDeltaEvent<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    index: u32,
    delta: SignatureDelta<'a>,
}

#[derive(Serialize)]
struct SignatureDelta<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    signature: &'a str,
}

#[derive(Serialize)]
struct RedactedBlockStart<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    index: u32,
    content_block: RedactedBlock<'a>,
}

#[derive(Serialize)]
struct RedactedBlock<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    data: &'a str,
}

#[derive(Serialize)]
struct BlockStop {
    #[serde(rename = "type")]
    kind: &'static str,
    index: u32,
}

#[derive(Serialize)]
struct MessageDeltaEvent {
    #[serde(rename = "type")]
    kind: &'static str,
    delta: MessageDeltaBody,
    usage: UsageOut,
}

#[derive(Serialize)]
struct MessageDeltaBody {
    stop_reason: &'static str,
    stop_sequence: Option<&'static str>,
}

#[derive(Serialize)]
struct MessageStop {
    #[serde(rename = "type")]
    kind: &'static str,
}

#[derive(Serialize)]
struct ErrorEvent<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    error: ErrorBody<'a>,
}

#[derive(Serialize)]
struct ErrorBody<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    message: &'a str,
}

#[cfg(test)]
mod tests {
    use super::*;
    use llmleaf_model::{Choice, ToolCallDelta};

    #[test]
    fn parse_minimal_request() {
        let v = json!({
            "model": "claude-opus-4-8",
            "max_tokens": 1024,
            "messages": [{ "role": "user", "content": "hi" }]
        });
        let req = parse_messages_request(v).unwrap();
        assert_eq!(req.model, "claude-opus-4-8");
        assert_eq!(req.max_tokens, Some(1024));
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.messages[0].role, Role::User);
        assert_eq!(req.messages[0].text_content(), "hi");
        assert!(!req.stream);
    }

    #[test]
    fn max_tokens_is_required() {
        let v = json!({
            "model": "claude-opus-4-8",
            "messages": [{ "role": "user", "content": "hi" }]
        });
        assert!(parse_messages_request(v).is_err());
    }

    #[test]
    fn system_hoists_to_leading_message() {
        // String form.
        let v = json!({
            "model": "m", "max_tokens": 10,
            "system": "be terse",
            "messages": [{ "role": "user", "content": "hi" }]
        });
        let req = parse_messages_request(v).unwrap();
        assert_eq!(req.messages[0].role, Role::System);
        assert_eq!(req.messages[0].text_content(), "be terse");
        assert_eq!(req.messages[1].role, Role::User);

        // Array-of-text-blocks form (blocks joined with a blank line).
        let v = json!({
            "model": "m", "max_tokens": 10,
            "system": [
                { "type": "text", "text": "line one" },
                { "type": "text", "text": "line two" }
            ],
            "messages": [{ "role": "user", "content": "hi" }]
        });
        let req = parse_messages_request(v).unwrap();
        assert_eq!(req.messages[0].text_content(), "line one\n\nline two");
    }

    #[test]
    fn parses_blocks_tools_and_tool_choice() {
        let v = json!({
            "model": "m", "max_tokens": 10, "stream": true,
            "stop_sequences": ["STOP"],
            "top_k": 40,
            "thinking": { "type": "enabled", "budget_tokens": 1024 },
            "messages": [{
                "role": "user",
                "content": [
                    { "type": "text", "text": "look" },
                    { "type": "image", "source": { "type": "url", "url": "http://x/y.png" } }
                ]
            }],
            "tools": [{ "name": "get_weather", "description": "w", "input_schema": { "type": "object" } }],
            "tool_choice": { "type": "any" }
        });
        let req = parse_messages_request(v).unwrap();
        assert!(req.stream);
        assert_eq!(req.stop, vec!["STOP".to_string()]);
        assert_eq!(req.messages[0].content.len(), 2);
        assert_eq!(req.tools[0].name, "get_weather");
        assert_eq!(req.tool_choice, Some(ToolChoice::Required));
        // `top_k` and `thinking` are not modeled canonically; they ride through verbatim.
        assert_eq!(req.extra.get("top_k"), Some(&json!(40)));
        assert!(req.extra.contains_key("thinking"));
        assert_eq!(req.thinking, None);
    }

    #[test]
    fn base64_image_becomes_data_url() {
        let v = json!({
            "model": "m", "max_tokens": 10,
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "image",
                    "source": { "type": "base64", "media_type": "image/png", "data": "QUJD" }
                }]
            }]
        });
        let req = parse_messages_request(v).unwrap();
        match &req.messages[0].content[0] {
            ContentPart::ImageUrl { url, .. } => {
                assert_eq!(url, "data:image/png;base64,QUJD");
            }
            other => panic!("expected ImageUrl, got {other:?}"),
        }
    }

    #[test]
    fn tool_result_expands_to_tool_message_then_user_text() {
        let v = json!({
            "model": "m", "max_tokens": 10,
            "messages": [{
                "role": "user",
                "content": [
                    { "type": "tool_result", "tool_use_id": "tu_1", "content": "72F" },
                    { "type": "text", "text": "and now?" }
                ]
            }]
        });
        let req = parse_messages_request(v).unwrap();
        // Tool result first (as Anthropic places it), then the user text.
        assert_eq!(req.messages[0].role, Role::Tool);
        assert_eq!(req.messages[0].tool_call_id.as_deref(), Some("tu_1"));
        assert_eq!(req.messages[0].text_content(), "72F");
        assert_eq!(req.messages[1].role, Role::User);
        assert_eq!(req.messages[1].text_content(), "and now?");
    }

    #[test]
    fn assistant_tool_use_and_thinking_preserved() {
        let v = json!({
            "model": "m", "max_tokens": 10,
            "messages": [{
                "role": "assistant",
                "content": [
                    { "type": "thinking", "thinking": "hmm", "signature": "sig" },
                    { "type": "text", "text": "let me check" },
                    { "type": "tool_use", "id": "tu_9", "name": "get_weather", "input": { "city": "NYC" } }
                ]
            }]
        });
        let req = parse_messages_request(v).unwrap();
        let m = &req.messages[0];
        assert_eq!(m.role, Role::Assistant);
        // Reasoning is preserved verbatim (not dropped): the thinking block — with its signature —
        // leads, ahead of the visible text, so a client echoing the turn back round-trips it.
        assert_eq!(m.content.len(), 2);
        match &m.content[0] {
            ContentPart::Thinking {
                thinking,
                signature,
            } => {
                assert_eq!(thinking, "hmm");
                assert_eq!(signature.as_deref(), Some("sig"));
            }
            other => panic!("expected a leading thinking block, got {other:?}"),
        }
        assert_eq!(m.text_content(), "let me check");
        assert_eq!(m.tool_calls.len(), 1);
        assert_eq!(m.tool_calls[0].id, "tu_9");
        assert_eq!(m.tool_calls[0].name, "get_weather");
        assert!(m.tool_calls[0].arguments.contains("NYC"));
    }

    #[test]
    fn tool_choice_variants() {
        let choice = |t: Value| {
            parse_messages_request(json!({
                "model": "m", "max_tokens": 1,
                "messages": [{ "role": "user", "content": "x" }],
                "tool_choice": t
            }))
            .unwrap()
            .tool_choice
        };
        assert_eq!(choice(json!({ "type": "auto" })), Some(ToolChoice::Auto));
        assert_eq!(choice(json!({ "type": "any" })), Some(ToolChoice::Required));
        assert_eq!(choice(json!({ "type": "none" })), Some(ToolChoice::None));
        assert_eq!(
            choice(json!({ "type": "tool", "name": "f" })),
            Some(ToolChoice::Named("f".to_string()))
        );
    }

    #[test]
    fn collected_response_shape() {
        let resp = ChatResponse {
            id: "ignored".into(),
            model: "claude-opus-4-8".into(),
            choices: vec![Choice {
                index: 0,
                text: "hi there".into(),
                thinking: vec![],
                tool_calls: vec![ToolCall {
                    id: "tu_1".into(),
                    name: "get_weather".into(),
                    arguments: "{\"city\":\"NYC\"}".into(),
                }],
                finish_reason: Some(FinishReason::ToolCalls),
            }],
            usage: Usage {
                prompt_tokens: 4,
                completion_tokens: 3,
                total_tokens: 7,
                cost_usd: Some(0.01),
                cache_read_tokens: 0,
                cache_creation_tokens: 0,
            },
        };
        let view = response_to_anthropic(&resp, "msg_42");
        let v = serde_json::to_value(&view).unwrap();
        assert_eq!(v["id"], "msg_42");
        assert_eq!(v["type"], "message");
        assert_eq!(v["role"], "assistant");
        assert_eq!(v["model"], "claude-opus-4-8");
        assert_eq!(v["content"][0]["type"], "text");
        assert_eq!(v["content"][0]["text"], "hi there");
        assert_eq!(v["content"][1]["type"], "tool_use");
        assert_eq!(v["content"][1]["id"], "tu_1");
        assert_eq!(v["content"][1]["input"]["city"], "NYC");
        assert_eq!(v["stop_reason"], "tool_use");
        assert!(v["stop_sequence"].is_null());
        assert_eq!(v["usage"]["input_tokens"], 4);
        assert_eq!(v["usage"]["output_tokens"], 3);
        assert_eq!(v["usage"]["cost_usd"], 0.01);
    }

    /// Run a sequence of canonical chunks through the encoder (plus `finish`) and return the ordered
    /// `(event, parsed-data)` frames.
    fn run_encoder(chunks: Vec<StreamChunk>) -> Vec<(&'static str, Value)> {
        let mut enc = EventEncoder::new("msg_1", "claude-opus-4-8");
        let mut out = Vec::new();
        let mut frames = Vec::new();
        for chunk in &chunks {
            frames.clear();
            enc.encode(chunk, &mut frames);
            for f in &frames {
                out.push((f.event, serde_json::from_str(&f.data).unwrap()));
            }
        }
        frames.clear();
        enc.finish(&mut frames);
        for f in &frames {
            out.push((f.event, serde_json::from_str(&f.data).unwrap()));
        }
        out
    }

    #[test]
    fn streaming_text_event_sequence() {
        let frames = run_encoder(vec![
            StreamChunk::Start {
                id: "upstream".into(),
                model: "upstream-model".into(),
            },
            StreamChunk::Content {
                index: 0,
                delta: "Hello".into(),
            },
            StreamChunk::Finish {
                index: 0,
                reason: FinishReason::Stop,
            },
            StreamChunk::Usage(Usage {
                prompt_tokens: 5,
                completion_tokens: 2,
                total_tokens: 7,
                cost_usd: None,
                cache_read_tokens: 0,
                cache_creation_tokens: 0,
            }),
        ]);
        let names: Vec<&str> = frames.iter().map(|(e, _)| *e).collect();
        assert_eq!(
            names,
            vec![
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            ]
        );
        // The id/model come from the encoder (the consumer-facing identity), not the upstream Start.
        assert_eq!(frames[0].1["message"]["id"], "msg_1");
        assert_eq!(frames[0].1["message"]["model"], "claude-opus-4-8");
        assert_eq!(frames[1].1["content_block"]["type"], "text");
        assert_eq!(frames[2].1["delta"]["type"], "text_delta");
        assert_eq!(frames[2].1["delta"]["text"], "Hello");
        assert_eq!(frames[4].1["delta"]["stop_reason"], "end_turn");
        assert_eq!(frames[4].1["usage"]["input_tokens"], 5);
        assert_eq!(frames[4].1["usage"]["output_tokens"], 2);
    }

    #[test]
    fn streaming_tool_call_brackets_blocks() {
        let frames = run_encoder(vec![
            StreamChunk::Start {
                id: "u".into(),
                model: "m".into(),
            },
            StreamChunk::Content {
                index: 0,
                delta: "let me check".into(),
            },
            StreamChunk::ToolCall {
                index: 0,
                call: ToolCallDelta {
                    index: 0,
                    id: Some("tu_1".into()),
                    name: Some("get_weather".into()),
                    arguments: Some("{\"city\":\"NYC\"}".into()),
                },
            },
            StreamChunk::Finish {
                index: 0,
                reason: FinishReason::ToolCalls,
            },
        ]);
        let names: Vec<&str> = frames.iter().map(|(e, _)| *e).collect();
        // text block (start, delta) -> stop -> tool block (start, input delta) -> stop -> delta -> stop
        assert_eq!(
            names,
            vec![
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            ]
        );
        // The tool block is index 1 (text was index 0) and carries id+name on start.
        assert_eq!(frames[4].1["index"], 1);
        assert_eq!(frames[4].1["content_block"]["type"], "tool_use");
        assert_eq!(frames[4].1["content_block"]["id"], "tu_1");
        assert_eq!(frames[4].1["content_block"]["name"], "get_weather");
        assert_eq!(frames[5].1["delta"]["type"], "input_json_delta");
        assert_eq!(frames[5].1["delta"]["partial_json"], "{\"city\":\"NYC\"}");
        assert_eq!(frames[7].1["delta"]["stop_reason"], "tool_use");
    }

    #[test]
    fn empty_stream_still_well_formed() {
        // No chunks at all: finish() alone must still emit a valid message envelope.
        let frames = run_encoder(vec![]);
        let names: Vec<&str> = frames.iter().map(|(e, _)| *e).collect();
        assert_eq!(
            names,
            vec!["message_start", "message_delta", "message_stop"]
        );
        assert_eq!(frames[1].1["delta"]["stop_reason"], "end_turn");
    }
}
