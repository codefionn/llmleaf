//! The OpenAI Responses API compat surface (`POST /v1/responses`) — served statelessly.
//!
//! In: an OpenAI Responses request JSON → canonical [`ChatRequest`]. The Responses dialect is its own
//! shape (decision filter: match the documented dialect even when awkward): `instructions` is a
//! top-level system prompt, `input` is either a bare string or an ordered list of typed *items*
//! (messages, function calls, function-call outputs, reasoning), tools are *flat* (`type`/`name` at the
//! top level, not nested under `function`), and reasoning rides in a `reasoning` object. All of that is
//! mapped here so the core stays dialect-free (principle 3). Unknown fields are preserved verbatim in
//! [`ChatRequest::extra`] (principle 7: transparent — we never silently drop).
//!
//! Out: a canonical [`ChatResponse`] → a single Responses `response` object ([`response_to_responses`],
//! a collected stream — principle 4), or the Responses streaming-event sequence ([`EventEncoder`]):
//! `response.created`/`response.in_progress`, then per output item an
//! `output_item.added` … `output_item.done` bracket (with `output_text`/`reasoning_text`/
//! `function_call_arguments` deltas inside), then a terminal `response.completed` (or
//! `response.incomplete`/`response.failed`) carrying the full response snapshot.
//!
//! **Deliberate deviations — llmleaf stores nothing (SOUL: the core is not a database):**
//! - `store` is accepted but the response *always* reports `"store": false`; the field is stripped, so
//!   no provider edge ever sees it.
//! - `previous_response_id` (non-null), an `item_reference` input item, and `background: true` are
//!   rejected with a clear 400 — there is no stored state to continue from, resolve, or poll.
//! - Responses-only knobs we do not model (`text`, `truncation`, `include`, `metadata`,
//!   `parallel_tool_calls`, …) ride through `extra` verbatim (P7): a Responses-speaking upstream honors
//!   them natively, while a chat-completions upstream may reject them. The consumer picks the fields
//!   that match the route it targets; llmleaf never silently drops them.
//!
//! This is the *serving* inverse of any Responses-speaking provider: this maps Responses → canonical to
//! serve a Responses-shaped consumer; a provider maps canonical → its own wire to *call* an upstream.

use std::collections::BTreeMap;

use llmleaf_model::{
    ChatRequest, ChatResponse, Choice, ContentPart, FinishReason, Message, ModelError, Role,
    StreamChunk, Thinking, ToolCall, ToolCallDelta, ToolChoice, ToolDef, Usage,
};
use serde_json::{json, Map, Value};

use crate::compat::openai;

fn mapping(msg: impl Into<String>) -> ModelError {
    ModelError::Mapping(msg.into())
}

// ---------------------------------------------------------------------------------------------
// Inbound: OpenAI Responses request JSON -> canonical ChatRequest
// ---------------------------------------------------------------------------------------------

/// Map an OpenAI Responses request into the canonical [`ChatRequest`].
///
/// `model` is required (as on the chat surface). `instructions` becomes a leading [`Role::System`]
/// message; `input` (a string, or an ordered list of typed items) expands into the conversation.
/// Everything unmodeled rides through in [`ChatRequest::extra`] (principle 7). See the module docs for
/// the stateless rejections (`previous_response_id`, `item_reference`, `background: true`) and the
/// always-`false` `store`.
pub fn parse_responses_request(value: Value) -> Result<ChatRequest, ModelError> {
    let Value::Object(mut obj) = value else {
        return Err(mapping("request body must be a JSON object"));
    };

    let model = match obj.remove("model") {
        Some(Value::String(s)) => s,
        _ => return Err(mapping("`model` is required and must be a string")),
    };

    // Stateless rejections (SOUL: the core is not a database). Remove-and-check so a rejected field
    // never lingers in `extra` either.
    if let Some(v) = obj.remove("previous_response_id") {
        if !v.is_null() {
            return Err(mapping(
                "`previous_response_id` is not supported: llmleaf is stateless and stores no \
                 responses to continue from",
            ));
        }
    }
    if obj.remove("background").and_then(|v| v.as_bool()) == Some(true) {
        return Err(mapping(
            "`background: true` is not supported: llmleaf does not store responses for later retrieval",
        ));
    }

    // Build the conversation: `instructions` hoists to a leading System message (like Anthropic's
    // top-level `system`), then the `input` items follow in order.
    let mut messages = Vec::new();
    if let Some(Value::String(instr)) = obj.remove("instructions") {
        if !instr.is_empty() {
            messages.push(Message::text(Role::System, instr));
        }
    }
    parse_input(obj.remove("input"), &mut messages)?;

    let max_tokens = obj
        .remove("max_output_tokens")
        .and_then(|v| v.as_u64())
        .map(|n| n as u32);
    let temperature = obj
        .remove("temperature")
        .and_then(|v| v.as_f64())
        .map(|n| n as f32);
    let top_p = obj
        .remove("top_p")
        .and_then(|v| v.as_f64())
        .map(|n| n as f32);
    let stream = obj
        .remove("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let tools = parse_tools(obj.remove("tools"))?;
    let tool_choice = parse_tool_choice(obj.remove("tool_choice"))?;

    // `reasoning.effort` becomes the canonical [`Thinking`] ladder *only* when `effort` is the object's
    // sole key and maps onto the ladder via the SAME vocabulary the chat surface uses
    // ([`openai::parse_reasoning_effort`] — reused, never duplicated). Otherwise the whole `reasoning`
    // object (a `summary`, `generate_summary`, an off-ladder effort, …) rides through `extra` verbatim,
    // exactly as `parse_chat_request` leaves an unrecognized `reasoning_effort` alone (principle 7).
    let thinking = match obj.remove("reasoning") {
        Some(Value::Object(r)) => {
            let only_effort = r.len() == 1 && r.contains_key("effort");
            let mapped = r
                .get("effort")
                .and_then(Value::as_str)
                .and_then(openai::parse_reasoning_effort);
            match (only_effort, mapped) {
                (true, Some(t)) => Some(t),
                _ => {
                    obj.insert("reasoning".into(), Value::Object(r));
                    None
                }
            }
        }
        Some(other) => {
            obj.insert("reasoning".into(), other);
            None
        }
        None => None,
    };

    // Remove-and-ignore: `store` (the response always reports `false`; documented) and `include` (we
    // always include what we can express). Everything else unmodeled rides through untouched.
    obj.remove("store");
    obj.remove("include");

    let extra = obj;

    Ok(ChatRequest {
        model,
        messages,
        max_tokens,
        temperature,
        top_p,
        // The Responses dialect has no `stop` field.
        stop: Vec::new(),
        stream,
        tools,
        tool_choice,
        thinking,
        extra,
    })
}

/// Expand the top-level `input`: a bare string is one [`Role::User`] text message; an array is a
/// sequence of typed items (see [`parse_input_array`]). Absent/null `input` contributes nothing.
fn parse_input(value: Option<Value>, out: &mut Vec<Message>) -> Result<(), ModelError> {
    match value {
        None | Some(Value::Null) => Ok(()),
        Some(Value::String(text)) => {
            out.push(Message::text(Role::User, text));
            Ok(())
        }
        Some(Value::Array(items)) => parse_input_array(items, out),
        Some(_) => Err(mapping("`input` must be a string or an array of items")),
    }
}

/// Expand an `input` array into canonical messages, in order.
///
/// The subtle piece is reasoning attachment: a `reasoning` item carries no message of its own, so its
/// thinking parts are held *pending* and prepended to the next assistant-side message (an `assistant`
/// `message` item, or an assistant message created/extended by a `function_call` item), preserving the
/// reasoning-before-answer order the upstream emitted and the replay contract needs. Trailing pending
/// reasoning with no following assistant turn becomes its own assistant message rather than being
/// dropped (principle 7).
fn parse_input_array(items: Vec<Value>, out: &mut Vec<Message>) -> Result<(), ModelError> {
    let mut pending: Vec<ContentPart> = Vec::new();
    for item in items {
        let Value::Object(obj) = item else {
            return Err(mapping("each `input` array item must be an object"));
        };
        match obj.get("type").and_then(Value::as_str) {
            // A message item: `type` absent or "message"; `role` required.
            None | Some("message") => {
                let role_str = obj
                    .get("role")
                    .and_then(Value::as_str)
                    .ok_or_else(|| mapping("`input` message item is missing `role`"))?;
                let role = parse_input_role(role_str)?;
                let content = parse_input_content(obj.get("content"))?;
                let mut msg = Message {
                    role,
                    content,
                    tool_calls: Vec::new(),
                    tool_call_id: None,
                    name: None,
                };
                if role == Role::Assistant {
                    prepend_pending(&mut msg, &mut pending);
                }
                out.push(msg);
            }
            // A tool call the assistant made: attach to the immediately preceding assistant turn when
            // there is one (text + calls share a turn, matching chat semantics), else start a new one.
            Some("function_call") => {
                let call = ToolCall {
                    id: str_field(&obj, "call_id"),
                    name: str_field(&obj, "name"),
                    arguments: str_field(&obj, "arguments"),
                };
                if matches!(out.last(), Some(m) if m.role == Role::Assistant) {
                    let msg = out.last_mut().expect("preceding assistant message present");
                    prepend_pending(msg, &mut pending);
                    msg.tool_calls.push(call);
                } else {
                    let mut msg = Message {
                        role: Role::Assistant,
                        content: Vec::new(),
                        tool_calls: vec![call],
                        tool_call_id: None,
                        name: None,
                    };
                    prepend_pending(&mut msg, &mut pending);
                    out.push(msg);
                }
            }
            // The result of a tool call → a [`Role::Tool`] message (as in the OpenAI/Anthropic dialects).
            Some("function_call_output") => {
                let call_id = str_field(&obj, "call_id");
                let text = function_output_text(obj.get("output"));
                out.push(Message {
                    role: Role::Tool,
                    content: vec![ContentPart::Text { text }],
                    tool_calls: Vec::new(),
                    tool_call_id: Some(call_id),
                    name: None,
                });
            }
            // Reasoning: held pending, attached to the next assistant-side message (below/above).
            Some("reasoning") => collect_reasoning(&obj, &mut pending),
            // A reference to a previously-stored item — unresolvable on a stateless gateway.
            Some("item_reference") => return Err(mapping(
                "`item_reference` input items cannot be resolved: llmleaf is stateless and stores \
                     no prior response items",
            )),
            Some(other) => return Err(mapping(format!("unsupported `input` item type {other:?}"))),
        }
    }
    // Reasoning with no following assistant turn is still preserved — as its own assistant message.
    if !pending.is_empty() {
        out.push(Message {
            role: Role::Assistant,
            content: std::mem::take(&mut pending),
            tool_calls: Vec::new(),
            tool_call_id: None,
            name: None,
        });
    }
    Ok(())
}

/// Prepend held reasoning parts to a message's content (reasoning leads the turn), draining `pending`.
fn prepend_pending(msg: &mut Message, pending: &mut Vec<ContentPart>) {
    if pending.is_empty() {
        return;
    }
    let mut parts = std::mem::take(pending);
    parts.append(&mut msg.content);
    msg.content = parts;
}

/// Read a string field, defaulting to `""` (matching the lenient chat-surface tool-call parsing).
fn str_field(obj: &Map<String, Value>, key: &str) -> String {
    obj.get(key)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

/// The Responses `input` message roles. `developer` collapses to [`Role::System`], exactly as the chat
/// surface treats it.
fn parse_input_role(role: &str) -> Result<Role, ModelError> {
    Ok(match role {
        "user" => Role::User,
        "system" | "developer" => Role::System,
        "assistant" => Role::Assistant,
        other => return Err(mapping(format!("unknown `input` message role `{other}`"))),
    })
}

/// Parse a message item's `content`: a string is one text part; an array is a list of typed parts.
fn parse_input_content(value: Option<&Value>) -> Result<Vec<ContentPart>, ModelError> {
    match value {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(Value::String(text)) => Ok(vec![ContentPart::Text { text: text.clone() }]),
        Some(Value::Array(parts)) => {
            let mut out = Vec::new();
            for part in parts {
                if let Some(p) = parse_input_content_part(part)? {
                    out.push(p);
                }
            }
            Ok(out)
        }
        Some(_) => Err(mapping("`content` must be a string, array, or null")),
    }
}

/// One typed content part. `input_text`/`output_text`/`text` are text; `input_image` with an inline
/// `image_url` is a canonical image; `refusal` carries no content and is skipped. A `file_id`-only image
/// (or an `input_file`/`input_audio` part) references bytes llmleaf does not hold, so it is a clear
/// mapping error naming the unsupported part type rather than a silent drop (principle 7).
fn parse_input_content_part(value: &Value) -> Result<Option<ContentPart>, ModelError> {
    let Value::Object(obj) = value else {
        return Err(mapping("content part must be an object"));
    };
    match obj.get("type").and_then(Value::as_str) {
        Some("input_text") | Some("output_text") | Some("text") => {
            let text = obj
                .get("text")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            Ok(Some(ContentPart::Text { text }))
        }
        Some("input_image") => match obj.get("image_url").and_then(Value::as_str) {
            Some(url) => {
                let detail = obj.get("detail").and_then(Value::as_str).map(str::to_owned);
                Ok(Some(ContentPart::ImageUrl {
                    url: url.to_string(),
                    detail,
                }))
            }
            None => Err(mapping(
                "unsupported `input` content part \"input_image\" without an inline `image_url` (a \
                 file_id-only image references stored bytes llmleaf does not hold)",
            )),
        },
        Some("refusal") => Ok(None),
        Some(other @ ("input_file" | "input_audio")) => {
            Err(mapping(format!("unsupported `input` content part type {other:?}")))
        }
        other => Err(mapping(format!("unsupported `input` content part type {other:?}"))),
    }
}

/// Accumulate a `reasoning` item's parts into the pending buffer, in order (summary text, then content
/// text, then any encrypted block). `summary[].text` and `content[].text` become open
/// [`ContentPart::Thinking`]; an item-level `signature` (the OpenRouter Responses dialect signs an item's
/// open reasoning — stock OpenAI never sends one) attaches to the item's last thinking part, so a signed
/// block replays signed; `encrypted_content` becomes a [`ContentPart::RedactedThinking`], verbatim.
fn collect_reasoning(obj: &Map<String, Value>, pending: &mut Vec<ContentPart>) {
    let before = pending.len();
    for key in ["summary", "content"] {
        if let Some(Value::Array(entries)) = obj.get(key) {
            for entry in entries {
                if let Some(text) = entry.get("text").and_then(Value::as_str) {
                    pending.push(ContentPart::Thinking {
                        thinking: text.to_string(),
                        signature: None,
                    });
                }
            }
        }
    }
    if let Some(sig) = obj.get("signature").and_then(Value::as_str) {
        if pending.len() > before {
            if let Some(ContentPart::Thinking { signature, .. }) = pending.last_mut() {
                *signature = Some(sig.to_string());
            }
        }
    }
    if let Some(data) = obj.get("encrypted_content").and_then(Value::as_str) {
        pending.push(ContentPart::RedactedThinking {
            data: data.to_string(),
        });
    }
}

/// Flatten a `function_call_output`'s `output`: a string, or an array of `output_text` parts
/// (concatenated). Anything else is empty text.
fn function_output_text(value: Option<&Value>) -> String {
    match value {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => {
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

/// Parse the FLAT Responses tools (`{ "type": "function", "name", "description"?, "parameters",
/// "strict"? }`). A non-`function` tool type is a hosted tool (`web_search`, `file_search`, …) the
/// gateway cannot execute — a clear mapping error naming it. `strict` is dropped at this edge: the
/// canonical model does not carry it, and the provider edge re-emits an explicit default — mirroring
/// [`openai::parse_tools`](super::openai), which likewise ignores it.
fn parse_tools(value: Option<Value>) -> Result<Vec<ToolDef>, ModelError> {
    let Some(Value::Array(items)) = value else {
        return Ok(Vec::new());
    };
    items
        .into_iter()
        .map(|t| {
            let Value::Object(o) = t else {
                return Err(mapping("each tool must be an object"));
            };
            if let Some(kind) = o.get("type").and_then(Value::as_str) {
                if kind != "function" {
                    return Err(mapping(format!(
                        "unsupported tool type {kind:?}: llmleaf cannot execute hosted tools"
                    )));
                }
            }
            let name = o
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| mapping("function tool missing `name`"))?
                .to_string();
            Ok(ToolDef {
                name,
                description: o
                    .get("description")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                parameters: o.get("parameters").cloned().unwrap_or(Value::Null),
            })
        })
        .collect()
}

/// Parse the Responses `tool_choice`: the strings `auto`/`none`/`required`, or the FLAT forced-function
/// object `{ "type": "function", "name" }`. A hosted-tool choice object (any other `type`) is a mapping
/// error — the gateway cannot execute hosted tools.
fn parse_tool_choice(value: Option<Value>) -> Result<Option<ToolChoice>, ModelError> {
    match value {
        Some(Value::String(s)) => Ok(match s.as_str() {
            "auto" => Some(ToolChoice::Auto),
            "none" => Some(ToolChoice::None),
            "required" => Some(ToolChoice::Required),
            _ => None,
        }),
        Some(Value::Object(o)) => match o.get("type").and_then(Value::as_str) {
            Some("function") => Ok(o
                .get("name")
                .and_then(Value::as_str)
                .map(|n| ToolChoice::Named(n.to_string()))),
            Some(other) => Err(mapping(format!(
                "unsupported tool_choice type {other:?}: hosted-tool choices are not executable by \
                 the gateway"
            ))),
            None => Ok(None),
        },
        _ => Ok(None),
    }
}

// ---------------------------------------------------------------------------------------------
// Outbound: canonical -> OpenAI Responses `response` object
// ---------------------------------------------------------------------------------------------

/// The canonical [`Thinking`] ladder → a Responses `reasoning.effort` string, for echoing the request's
/// effort back on the response. The inverse vocabulary of [`openai::parse_reasoning_effort`].
fn thinking_to_effort(t: Thinking) -> &'static str {
    match t {
        Thinking::Low => "low",
        Thinking::Med => "medium",
        Thinking::High => "high",
        Thinking::Highx => "xhigh",
        Thinking::Max => "max",
    }
}

/// What the Responses `response` object echoes back from the request. Built once from the parsed
/// [`ChatRequest`] and shared by the collected view and the streaming encoder, so both render an
/// identical envelope. The unmodeled knobs (`metadata`, `parallel_tool_calls`, `text`, `truncation`)
/// are echoed straight from `extra` — the same values that ride through to the upstream (P7).
pub struct RequestEcho {
    temperature: Option<f32>,
    top_p: Option<f32>,
    max_output_tokens: Option<u32>,
    /// Tools re-flattened back into the Responses shape.
    tools: Vec<Value>,
    /// `tool_choice` re-rendered in the Responses shape.
    tool_choice: Option<Value>,
    /// The reasoning effort, echoed as a string (from the canonical ladder, or the raw `reasoning.effort`
    /// left in `extra` when it was off-ladder).
    reasoning_effort: Option<String>,
    metadata: Option<Value>,
    parallel_tool_calls: Option<Value>,
    text: Option<Value>,
    truncation: Option<Value>,
}

impl RequestEcho {
    /// Capture what the response echoes from the request.
    pub fn from_request(req: &ChatRequest) -> Self {
        let reasoning_effort = match req.thinking {
            Some(t) => Some(thinking_to_effort(t).to_string()),
            // Off-ladder effort left in `extra` — echo it verbatim so the response reflects the ask.
            None => req
                .extra
                .get("reasoning")
                .and_then(|r| r.get("effort"))
                .and_then(Value::as_str)
                .map(str::to_owned),
        };
        RequestEcho {
            temperature: req.temperature,
            top_p: req.top_p,
            max_output_tokens: req.max_tokens,
            tools: tools_to_responses(&req.tools),
            tool_choice: req.tool_choice.as_ref().map(tool_choice_to_responses),
            reasoning_effort,
            metadata: req.extra.get("metadata").cloned(),
            parallel_tool_calls: req.extra.get("parallel_tool_calls").cloned(),
            text: req.extra.get("text").cloned(),
            truncation: req.extra.get("truncation").cloned(),
        }
    }
}

/// Re-flatten canonical tools into the Responses shape (`{ "type": "function", "name", … }`). `strict`
/// is not re-emitted: it was dropped inbound and the canonical model does not carry it.
fn tools_to_responses(tools: &[ToolDef]) -> Vec<Value> {
    tools
        .iter()
        .map(|t| {
            let mut o = Map::new();
            o.insert("type".into(), json!("function"));
            o.insert("name".into(), json!(t.name));
            if let Some(d) = &t.description {
                o.insert("description".into(), json!(d));
            }
            o.insert("parameters".into(), t.parameters.clone());
            Value::Object(o)
        })
        .collect()
}

/// Re-render a canonical [`ToolChoice`] in the Responses shape (a string, or a flat forced-function).
fn tool_choice_to_responses(tc: &ToolChoice) -> Value {
    match tc {
        ToolChoice::Auto => json!("auto"),
        ToolChoice::None => json!("none"),
        ToolChoice::Required => json!("required"),
        ToolChoice::Named(n) => json!({ "type": "function", "name": n }),
    }
}

/// Map the canonical [`FinishReason`] onto the Responses `status` (+ optional `incomplete_details`).
/// `Length` and `ContentFilter` are `incomplete` (the model stopped short); `Error` is `failed`;
/// everything else (including a normal `Stop`/`ToolCalls`) is `completed`.
fn status_for(finish: Option<FinishReason>) -> (&'static str, Option<&'static str>) {
    match finish {
        Some(FinishReason::Length) => ("incomplete", Some("max_output_tokens")),
        Some(FinishReason::ContentFilter) => ("incomplete", Some("content_filter")),
        Some(FinishReason::Error) => ("failed", None),
        _ => ("completed", None),
    }
}

/// The Responses `usage` object. `input_tokens_details.cached_tokens` relays the canonical cache-read
/// count. `output_tokens_details.reasoning_tokens` is always `0`: the canonical model carries no
/// reasoning-token count, and the core relays usage, never computes it (principle 5) — `0` is the honest
/// floor rather than a guess. `cost_usd` is the same llmleaf extension the chat surface carries (the
/// real-time price from the bundled dataset), present only when the model is priced — every consumer
/// surface reports cost identically.
fn usage_json(u: &Usage) -> Value {
    let mut usage = json!({
        "input_tokens": u.prompt_tokens,
        "input_tokens_details": { "cached_tokens": u.cache_read_tokens },
        "output_tokens": u.completion_tokens,
        "output_tokens_details": { "reasoning_tokens": 0 },
        "total_tokens": u.total_tokens,
    });
    if let Some(cost) = u.cost_usd {
        usage
            .as_object_mut()
            .expect("usage is an object")
            .insert("cost_usd".into(), json!(cost));
    }
    usage
}

/// Build the `output` array from a single [`Choice`]. The Responses dialect has no choices array — only
/// choice 0 is representable, and no further choice can arise from this surface (`n` is not part of the
/// dialect, so no request ever asks for more). Reasoning leads: a run of open [`ContentPart::Thinking`]
/// becomes one `reasoning` item (their text under `content` as `reasoning_text` entries — our text *is*
/// the reasoning, not a summary, so `summary` stays `[]`); a signature ends the block it signs, so a
/// signed part closes its item and the item carries `signature` (the OpenRouter Responses dialect's
/// signed open reasoning; the key is absent otherwise, keeping the stock-OpenAI shape untouched); a
/// [`ContentPart::RedactedThinking`] is its own `reasoning` item carrying `encrypted_content`. Then the
/// visible text becomes a `message` item, and each tool call a `function_call` item.
fn output_items(request_id: &str, choice: &Choice) -> Vec<Value> {
    let mut items = Vec::new();

    // Reasoning first. Group consecutive open thinking parts into one item; a signature or a redacted
    // block ends the group (a signature signs exactly the text before it; a redacted block is its own).
    let mut open_texts: Vec<&str> = Vec::new();
    for part in &choice.thinking {
        match part {
            ContentPart::Thinking {
                thinking,
                signature,
            } => {
                open_texts.push(thinking);
                if let Some(sig) = signature {
                    flush_reasoning(&mut items, request_id, &mut open_texts, Some(sig));
                }
            }
            ContentPart::RedactedThinking { data } => {
                flush_reasoning(&mut items, request_id, &mut open_texts, None);
                // The id carries the item's output position (`items.len()`), so every item id in one
                // response is unique and matches the `_<output_index>` convention the streaming
                // encoder's live events use — the terminal snapshot then names the same items.
                let idx = items.len();
                items.push(json!({
                    "type": "reasoning",
                    "id": format!("rs_{request_id}_{idx}"),
                    "summary": [],
                    "encrypted_content": data,
                }));
            }
            _ => {}
        }
    }
    flush_reasoning(&mut items, request_id, &mut open_texts, None);

    if !choice.text.is_empty() {
        let idx = items.len();
        items.push(json!({
            "type": "message",
            "id": format!("msg_{request_id}_{idx}"),
            "status": "completed",
            "role": "assistant",
            "content": [{ "type": "output_text", "text": choice.text, "annotations": [] }],
        }));
    }

    for call in &choice.tool_calls {
        let idx = items.len();
        items.push(json!({
            "type": "function_call",
            "id": format!("fc_{request_id}_{idx}"),
            "call_id": call.id,
            "name": call.name,
            "arguments": call.arguments,
            "status": "completed",
        }));
    }

    items
}

/// Emit the accumulated run of open reasoning text as one `reasoning` item, then clear the buffer. The
/// id carries the item's output position, like every item id here (see `output_items`). `signature`
/// (when the run ended on a signed part) is emitted as the item-level `signature` key — the OpenRouter
/// Responses dialect's signed-reasoning shape; the key is absent when there is nothing to sign with.
fn flush_reasoning(
    items: &mut Vec<Value>,
    request_id: &str,
    open_texts: &mut Vec<&str>,
    signature: Option<&str>,
) {
    if open_texts.is_empty() {
        return;
    }
    let content: Vec<Value> = open_texts
        .iter()
        .map(|t| json!({ "type": "reasoning_text", "text": t }))
        .collect();
    let idx = items.len();
    let mut item = json!({
        "type": "reasoning",
        "id": format!("rs_{request_id}_{idx}"),
        "summary": [],
        "content": content,
    });
    if let Some(sig) = signature {
        item["signature"] = json!(sig);
    }
    items.push(item);
    open_texts.clear();
}

/// Assemble the full Responses `response` object. Shared by the collected view and every streaming
/// snapshot (created / in_progress / completed), so there is one source of truth for the object shape
/// (principle 3). `usage` is `None` for the in-flight snapshots (`"usage": null`) and `Some` once known.
#[allow(clippy::too_many_arguments)]
fn build_response(
    request_id: &str,
    model: &str,
    output: Vec<Value>,
    usage: Option<&Usage>,
    status: &str,
    incomplete_reason: Option<&str>,
    echo: &RequestEcho,
    created: u64,
) -> Value {
    json!({
        "id": format!("resp_{request_id}"),
        "object": "response",
        "created_at": created,
        "status": status,
        "background": false,
        "error": Value::Null,
        "incomplete_details": incomplete_reason.map(|r| json!({ "reason": r })),
        "instructions": Value::Null,
        "max_output_tokens": echo.max_output_tokens,
        "model": model,
        "output": output,
        "parallel_tool_calls": echo.parallel_tool_calls.clone().unwrap_or(json!(true)),
        "previous_response_id": Value::Null,
        "reasoning": { "effort": echo.reasoning_effort, "summary": Value::Null },
        // llmleaf stores nothing — always false, whatever the request asked (documented deviation).
        "store": false,
        "temperature": echo.temperature,
        "text": echo.text.clone().unwrap_or(json!({ "format": { "type": "text" } })),
        "tool_choice": echo.tool_choice.clone().unwrap_or(json!("auto")),
        "tools": echo.tools,
        "top_p": echo.top_p,
        "truncation": echo.truncation.clone().unwrap_or(json!("disabled")),
        "usage": usage.map(usage_json),
        "metadata": echo.metadata.clone().unwrap_or(json!({})),
    })
}

/// Collect-mode mapping: a canonical [`ChatResponse`] to a Responses `response` object. The Responses
/// dialect keys the object on the consumer request id (`resp_<id>`) — the edge stamps that id onto
/// `resp.id` before calling — so `id`, the output-item ids, and the streaming encoder all agree.
pub fn response_to_responses(resp: &ChatResponse, echo: &RequestEcho, created: u64) -> Value {
    let choice = resp.choices.first();
    let output = choice
        .map(|c| output_items(&resp.id, c))
        .unwrap_or_default();
    let (status, reason) = status_for(choice.and_then(|c| c.finish_reason));
    build_response(
        &resp.id,
        &resp.model,
        output,
        Some(&resp.usage),
        status,
        reason,
        echo,
        created,
    )
}

// ---------------------------------------------------------------------------------------------
// Outbound streaming: canonical stream -> Responses SSE event sequence
// ---------------------------------------------------------------------------------------------

/// One Responses SSE frame: a named event plus its serialized `data:` payload. The HTTP edge wraps each
/// in an `axum` SSE event (`event: <event>\ndata: <data>`); this module stays transport-free (mirrors
/// [`anthropic::Frame`](super::anthropic)).
pub struct Frame {
    pub event: &'static str,
    pub data: String,
}

/// Serialize `data` into a new [`Frame`]. The Responses dialect emits several structurally-distinct
/// events per chunk, so — unlike the OpenAI chat [`ChunkEncoder`](super::openai) with its zero-alloc
/// borrowed frames — each event is built as a small JSON value and serialized here. The per-delta events
/// are flat objects; the heavier snapshot events fire only at the stream's start and end.
fn push(out: &mut Vec<Frame>, event: &'static str, data: Value) {
    out.push(Frame {
        event,
        data: serde_json::to_string(&data).expect("responses event serialises"),
    });
}

/// A partially-built tool call, accumulated across [`StreamChunk::ToolCall`] deltas for the final
/// snapshot (mirrors the canonical collector's fold).
#[derive(Default, Clone)]
struct AccTool {
    id: String,
    name: String,
    arguments: String,
}

/// The output item currently open on the wire. Each carries the `output_index` and item id assigned when
/// it was opened, plus whatever the item's `*.done` event needs (the accumulated text; a tool reads its
/// final fields from [`EventEncoder::tools`] by canonical index).
enum OpenItem {
    Reasoning {
        index: u32,
        id: String,
        text: String,
        /// The item-level signature, set when a [`StreamChunk::ThinkingSignature`] arrives while this
        /// item is open (the OpenRouter Responses dialect's signed open reasoning).
        sig: Option<String>,
    },
    Message {
        index: u32,
        id: String,
        text: String,
    },
    Tool {
        index: u32,
        id: String,
        canon: u32,
    },
}

/// Maps the canonical stream to the Responses streaming-event sequence. Stateful by necessity: the
/// dialect brackets each output item with `output_item.added`/`output_item.done`, numbers every event
/// with a monotonic `sequence_number`, and closes with a single `response.completed`/`incomplete`/
/// `failed` carrying the full snapshot — so the encoder holds the open-item state, the running sequence
/// number, and enough accumulated content to rebuild the terminal snapshot via the *same* view builder
/// the collected path uses ([`response_to_responses`]).
pub struct EventEncoder {
    request_id: String,
    model: String,
    created: u64,
    echo: RequestEcho,
    seq: u64,
    started: bool,
    output_index: u32,
    open: Option<OpenItem>,
    // Accumulated choice-0 content, folded exactly like the canonical collector, for the final snapshot.
    text: String,
    think_text: String,
    think_sig: Option<String>,
    redacted: Vec<String>,
    tools: BTreeMap<u32, AccTool>,
    usage: Usage,
    finish: Option<FinishReason>,
}

impl EventEncoder {
    /// Construct the encoder for one response. `request_id`/`model` are the consumer-facing identity
    /// (the upstream [`StreamChunk::Start`] id/model are ignored, as on every other surface).
    pub fn new(
        request_id: impl Into<String>,
        model: impl Into<String>,
        created: u64,
        echo: RequestEcho,
    ) -> Self {
        EventEncoder {
            request_id: request_id.into(),
            model: model.into(),
            created,
            echo,
            seq: 0,
            started: false,
            output_index: 0,
            open: None,
            text: String::new(),
            think_text: String::new(),
            think_sig: None,
            redacted: Vec::new(),
            tools: BTreeMap::new(),
            usage: Usage::default(),
            finish: None,
        }
    }

    /// The next monotonic `sequence_number` (Responses events number from 0).
    fn next_seq(&mut self) -> u64 {
        let s = self.seq;
        self.seq += 1;
        s
    }

    /// An in-flight snapshot (empty output, `usage: null`) for the `response.created`/`in_progress`
    /// frames, built from the shared view builder so it matches the terminal snapshot's shape.
    fn in_progress_snapshot(&self) -> Value {
        build_response(
            &self.request_id,
            &self.model,
            Vec::new(),
            None,
            "in_progress",
            None,
            &self.echo,
            self.created,
        )
    }

    /// Emit `response.created` then `response.in_progress` once. Defensive against a stream that opens
    /// with content: the envelope events must precede any item regardless of a leading `Start`.
    fn ensure_started(&mut self, out: &mut Vec<Frame>) {
        if self.started {
            return;
        }
        self.started = true;
        let snapshot = self.in_progress_snapshot();
        let seq = self.next_seq();
        push(
            out,
            "response.created",
            json!({ "type": "response.created", "response": snapshot, "sequence_number": seq }),
        );
        let snapshot = self.in_progress_snapshot();
        let seq = self.next_seq();
        push(
            out,
            "response.in_progress",
            json!({ "type": "response.in_progress", "response": snapshot, "sequence_number": seq }),
        );
    }

    /// Close whatever item is open, emitting its `*.done` events. A no-op when nothing is open.
    fn close_open(&mut self, out: &mut Vec<Frame>) {
        let Some(item) = self.open.take() else {
            return;
        };
        match item {
            OpenItem::Reasoning {
                index,
                id,
                text,
                sig,
            } => {
                let seq = self.next_seq();
                push(
                    out,
                    "response.reasoning_text.done",
                    json!({
                        "type": "response.reasoning_text.done",
                        "item_id": id, "output_index": index, "content_index": 0,
                        "text": text, "sequence_number": seq,
                    }),
                );
                // The item-level `signature` (signed open reasoning, OpenRouter dialect) is emitted only
                // when present, so the stock-OpenAI item shape stays untouched without one.
                let mut item = json!({
                    "type": "reasoning", "id": id, "summary": [],
                    "content": [{ "type": "reasoning_text", "text": text }],
                });
                if let Some(sig) = sig {
                    item["signature"] = json!(sig);
                }
                let seq = self.next_seq();
                push(
                    out,
                    "response.output_item.done",
                    json!({
                        "type": "response.output_item.done",
                        "output_index": index,
                        "item": item,
                        "sequence_number": seq,
                    }),
                );
            }
            OpenItem::Message { index, id, text } => {
                let seq = self.next_seq();
                push(
                    out,
                    "response.output_text.done",
                    json!({
                        "type": "response.output_text.done",
                        "item_id": id, "output_index": index, "content_index": 0,
                        "text": text, "sequence_number": seq,
                    }),
                );
                let seq = self.next_seq();
                push(
                    out,
                    "response.content_part.done",
                    json!({
                        "type": "response.content_part.done",
                        "item_id": id, "output_index": index, "content_index": 0,
                        "part": { "type": "output_text", "text": text, "annotations": [] },
                        "sequence_number": seq,
                    }),
                );
                let seq = self.next_seq();
                push(
                    out,
                    "response.output_item.done",
                    json!({
                        "type": "response.output_item.done",
                        "output_index": index,
                        "item": {
                            "type": "message", "id": id, "status": "completed", "role": "assistant",
                            "content": [{ "type": "output_text", "text": text, "annotations": [] }],
                        },
                        "sequence_number": seq,
                    }),
                );
            }
            OpenItem::Tool { index, id, canon } => {
                // The final call fields live in the snapshot accumulator, keyed by canonical index.
                let acc = self.tools.get(&canon).cloned().unwrap_or_default();
                let seq = self.next_seq();
                push(
                    out,
                    "response.function_call_arguments.done",
                    json!({
                        "type": "response.function_call_arguments.done",
                        "item_id": id, "output_index": index,
                        "arguments": acc.arguments, "sequence_number": seq,
                    }),
                );
                let seq = self.next_seq();
                push(
                    out,
                    "response.output_item.done",
                    json!({
                        "type": "response.output_item.done",
                        "output_index": index,
                        "item": {
                            "type": "function_call", "id": id, "call_id": acc.id, "name": acc.name,
                            "arguments": acc.arguments, "status": "completed",
                        },
                        "sequence_number": seq,
                    }),
                );
            }
        }
    }

    /// Ensure an open reasoning item, opening one (after closing any other) when needed. Returns its
    /// id + output index.
    fn open_reasoning(&mut self, out: &mut Vec<Frame>) -> (String, u32) {
        if let Some(OpenItem::Reasoning { id, index, .. }) = &self.open {
            return (id.clone(), *index);
        }
        self.close_open(out);
        let index = self.output_index;
        self.output_index += 1;
        let id = format!("rs_{}_{}", self.request_id, index);
        let seq = self.next_seq();
        push(
            out,
            "response.output_item.added",
            json!({
                "type": "response.output_item.added",
                "output_index": index,
                "item": { "type": "reasoning", "id": id, "summary": [], "content": [] },
                "sequence_number": seq,
            }),
        );
        self.open = Some(OpenItem::Reasoning {
            index,
            id: id.clone(),
            text: String::new(),
            sig: None,
        });
        (id, index)
    }

    /// Ensure an open message item, opening one (after closing any other) when needed. Returns its
    /// id + output index.
    fn open_message(&mut self, out: &mut Vec<Frame>) -> (String, u32) {
        if let Some(OpenItem::Message { id, index, .. }) = &self.open {
            return (id.clone(), *index);
        }
        self.close_open(out);
        let index = self.output_index;
        self.output_index += 1;
        let id = format!("msg_{}_{}", self.request_id, index);
        let seq = self.next_seq();
        push(
            out,
            "response.output_item.added",
            json!({
                "type": "response.output_item.added",
                "output_index": index,
                "item": {
                    "type": "message", "id": id, "status": "in_progress", "role": "assistant",
                    "content": [],
                },
                "sequence_number": seq,
            }),
        );
        let seq = self.next_seq();
        push(
            out,
            "response.content_part.added",
            json!({
                "type": "response.content_part.added",
                "item_id": id, "output_index": index, "content_index": 0,
                "part": { "type": "output_text", "text": "", "annotations": [] },
                "sequence_number": seq,
            }),
        );
        self.open = Some(OpenItem::Message {
            index,
            id: id.clone(),
            text: String::new(),
        });
        (id, index)
    }

    /// Ensure the open function-call item matches `call`'s canonical index, opening a fresh one (after
    /// closing any other) when the call index changes. Returns its id + output index.
    fn open_tool(&mut self, out: &mut Vec<Frame>, call: &ToolCallDelta) -> (String, u32) {
        if let Some(OpenItem::Tool { id, index, canon }) = &self.open {
            if *canon == call.index {
                return (id.clone(), *index);
            }
        }
        self.close_open(out);
        let index = self.output_index;
        self.output_index += 1;
        let id = format!("fc_{}_{}", self.request_id, index);
        let seq = self.next_seq();
        push(
            out,
            "response.output_item.added",
            json!({
                "type": "response.output_item.added",
                "output_index": index,
                "item": {
                    "type": "function_call", "id": id,
                    "call_id": call.id.as_deref().unwrap_or(""),
                    "name": call.name.as_deref().unwrap_or(""),
                    "arguments": "", "status": "in_progress",
                },
                "sequence_number": seq,
            }),
        );
        self.open = Some(OpenItem::Tool {
            index,
            id: id.clone(),
            canon: call.index,
        });
        (id, index)
    }

    /// Translate one canonical chunk into zero or more Responses events, appended to `out` (cleared by
    /// the caller per chunk). `Usage`/`Finish` are buffered for the terminal snapshot; only choice 0 is
    /// representable, so chunks for other indices are ignored (the dialect has no multi-choice concept,
    /// so they cannot arise from this surface — guarded defensively).
    pub fn encode(&mut self, chunk: &StreamChunk, out: &mut Vec<Frame>) {
        match chunk {
            StreamChunk::Start { .. } => self.ensure_started(out),
            StreamChunk::Thinking { index, delta } => {
                if *index != 0 {
                    return;
                }
                self.ensure_started(out);
                self.think_text.push_str(delta);
                let (id, oidx) = self.open_reasoning(out);
                if let Some(OpenItem::Reasoning { text, .. }) = &mut self.open {
                    text.push_str(delta);
                }
                let seq = self.next_seq();
                push(
                    out,
                    "response.reasoning_text.delta",
                    json!({
                        "type": "response.reasoning_text.delta",
                        "item_id": id, "output_index": oidx, "content_index": 0,
                        "delta": delta, "sequence_number": seq,
                    }),
                );
            }
            StreamChunk::ThinkingSignature { index, signature } => {
                if *index != 0 {
                    return;
                }
                // The signature signs the reasoning streamed so far. It emits no event of its own (the
                // dialect has no signature delta — OpenRouter's included); it lands on the open
                // reasoning item's `output_item.done` and on the terminal snapshot, folded exactly as
                // the canonical collector folds it (one signature per choice).
                self.think_sig = Some(signature.clone());
                if let Some(OpenItem::Reasoning { sig, .. }) = &mut self.open {
                    *sig = Some(signature.clone());
                }
            }
            StreamChunk::RedactedThinking { index, data } => {
                if *index != 0 {
                    return;
                }
                self.ensure_started(out);
                self.redacted.push(data.clone());
                // A redacted block arrives whole: a self-contained reasoning item (added + done).
                self.close_open(out);
                let index = self.output_index;
                self.output_index += 1;
                let id = format!("rs_{}_{}", self.request_id, index);
                let item = json!({
                    "type": "reasoning", "id": id, "summary": [], "encrypted_content": data,
                });
                let seq = self.next_seq();
                push(
                    out,
                    "response.output_item.added",
                    json!({
                        "type": "response.output_item.added",
                        "output_index": index, "item": item, "sequence_number": seq,
                    }),
                );
                let item = json!({
                    "type": "reasoning", "id": id, "summary": [], "encrypted_content": data,
                });
                let seq = self.next_seq();
                push(
                    out,
                    "response.output_item.done",
                    json!({
                        "type": "response.output_item.done",
                        "output_index": index, "item": item, "sequence_number": seq,
                    }),
                );
            }
            StreamChunk::Content { index, delta } => {
                if *index != 0 {
                    return;
                }
                self.ensure_started(out);
                self.text.push_str(delta);
                let (id, oidx) = self.open_message(out);
                if let Some(OpenItem::Message { text, .. }) = &mut self.open {
                    text.push_str(delta);
                }
                let seq = self.next_seq();
                push(
                    out,
                    "response.output_text.delta",
                    json!({
                        "type": "response.output_text.delta",
                        "item_id": id, "output_index": oidx, "content_index": 0,
                        "delta": delta, "sequence_number": seq,
                    }),
                );
            }
            StreamChunk::ToolCall { index, call } => {
                if *index != 0 {
                    return;
                }
                self.ensure_started(out);
                // Fold into the snapshot accumulator (persists across items), then emit the live delta.
                let acc = self.tools.entry(call.index).or_default();
                if let Some(id) = &call.id {
                    acc.id = id.clone();
                }
                if let Some(name) = &call.name {
                    acc.name = name.clone();
                }
                if let Some(args) = &call.arguments {
                    acc.arguments.push_str(args);
                }
                let (id, oidx) = self.open_tool(out, call);
                if let Some(args) = &call.arguments {
                    if !args.is_empty() {
                        let seq = self.next_seq();
                        push(
                            out,
                            "response.function_call_arguments.delta",
                            json!({
                                "type": "response.function_call_arguments.delta",
                                "item_id": id, "output_index": oidx,
                                "delta": args, "sequence_number": seq,
                            }),
                        );
                    }
                }
            }
            StreamChunk::Usage(u) => self.usage = *u,
            StreamChunk::Finish { index, reason } => {
                if *index != 0 {
                    return;
                }
                self.close_open(out);
                self.finish = Some(*reason);
            }
        }
    }

    /// Fold the accumulated choice-0 content into a canonical [`Choice`], exactly as the stream collector
    /// would — so the terminal snapshot is built by the *same* view builder as the collected path.
    fn accumulated_choice(&self) -> Choice {
        let mut thinking = Vec::new();
        if !self.think_text.is_empty() || self.think_sig.is_some() {
            thinking.push(ContentPart::Thinking {
                thinking: self.think_text.clone(),
                signature: self.think_sig.clone(),
            });
        }
        for data in &self.redacted {
            thinking.push(ContentPart::RedactedThinking { data: data.clone() });
        }
        Choice {
            index: 0,
            text: self.text.clone(),
            thinking,
            tool_calls: self
                .tools
                .values()
                .map(|t| ToolCall {
                    id: t.id.clone(),
                    name: t.name.clone(),
                    arguments: t.arguments.clone(),
                })
                .collect(),
            finish_reason: self.finish,
        }
    }

    /// Emit the terminal frame: close any open item, then `response.completed` (or `response.incomplete`
    /// / `response.failed`) carrying the FULL snapshot — byte-identical in shape to [`response_to_responses`]
    /// for the same chunks, since both run [`build_response`] over the same folded choice. Always starts
    /// the envelope first, so even an empty stream yields a well-formed response.
    pub fn finish(&mut self, out: &mut Vec<Frame>) {
        self.ensure_started(out);
        self.close_open(out);
        let resp = ChatResponse {
            id: self.request_id.clone(),
            model: self.model.clone(),
            choices: vec![self.accumulated_choice()],
            usage: self.usage,
        };
        let snapshot = response_to_responses(&resp, &self.echo, self.created);
        let event = match snapshot.get("status").and_then(Value::as_str) {
            Some("incomplete") => "response.incomplete",
            Some("failed") => "response.failed",
            _ => "response.completed",
        };
        let seq = self.next_seq();
        push(
            out,
            event,
            json!({ "type": event, "response": snapshot, "sequence_number": seq }),
        );
    }

    /// The `data:` payload for a mid-stream `error` event. Like the Anthropic surface, an upstream
    /// failure surfaces as an `event: error` frame rather than a terminal `response.completed`. Takes
    /// `&mut self` for the running `sequence_number`.
    pub fn error_frame(&mut self, message: &str) -> Frame {
        let seq = self.next_seq();
        Frame {
            event: "error",
            data: serde_json::to_string(&json!({
                "type": "error",
                "code": Value::Null,
                "message": message,
                "param": Value::Null,
                "sequence_number": seq,
            }))
            .expect("responses error event serialises"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llmleaf_model::collect_chunks;

    // --- Inbound ---------------------------------------------------------------------------------

    #[test]
    fn string_input_becomes_one_user_message() {
        let req = parse_responses_request(json!({ "model": "gpt-5", "input": "hi" })).unwrap();
        assert_eq!(req.model, "gpt-5");
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.messages[0].role, Role::User);
        assert_eq!(req.messages[0].text_content(), "hi");
    }

    #[test]
    fn instructions_hoist_to_leading_system_message() {
        let req = parse_responses_request(json!({
            "model": "m", "instructions": "be terse", "input": "hi"
        }))
        .unwrap();
        assert_eq!(req.messages[0].role, Role::System);
        assert_eq!(req.messages[0].text_content(), "be terse");
        assert_eq!(req.messages[1].role, Role::User);
    }

    #[test]
    fn array_input_roles_parts_and_image() {
        let req = parse_responses_request(json!({
            "model": "m",
            "input": [
                { "role": "developer", "content": "sys" },
                { "role": "user", "content": [
                    { "type": "input_text", "text": "look" },
                    { "type": "input_image", "image_url": "http://x/y.png", "detail": "low" },
                    { "type": "refusal", "refusal": "nope" }
                ] }
            ]
        }))
        .unwrap();
        // developer collapses to System.
        assert_eq!(req.messages[0].role, Role::System);
        assert_eq!(req.messages[0].text_content(), "sys");
        let user = &req.messages[1];
        assert_eq!(user.role, Role::User);
        // refusal skipped → two parts (text + image).
        assert_eq!(user.content.len(), 2);
        match &user.content[1] {
            ContentPart::ImageUrl { url, detail } => {
                assert_eq!(url, "http://x/y.png");
                assert_eq!(detail.as_deref(), Some("low"));
            }
            other => panic!("expected image, got {other:?}"),
        }
    }

    #[test]
    fn function_call_merges_into_preceding_assistant() {
        let req = parse_responses_request(json!({
            "model": "m",
            "input": [
                { "role": "assistant", "content": [{ "type": "output_text", "text": "let me check" }] },
                { "type": "function_call", "call_id": "c1", "name": "get_weather", "arguments": "{\"city\":\"NYC\"}" }
            ]
        }))
        .unwrap();
        // One assistant message carrying both the text and the tool call (a shared turn).
        assert_eq!(req.messages.len(), 1);
        let m = &req.messages[0];
        assert_eq!(m.role, Role::Assistant);
        assert_eq!(m.text_content(), "let me check");
        assert_eq!(m.tool_calls.len(), 1);
        assert_eq!(m.tool_calls[0].id, "c1");
        assert_eq!(m.tool_calls[0].name, "get_weather");
        assert!(m.tool_calls[0].arguments.contains("NYC"));
    }

    #[test]
    fn function_call_without_preceding_assistant_starts_new_turn() {
        let req = parse_responses_request(json!({
            "model": "m",
            "input": [
                { "role": "user", "content": "hi" },
                { "type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}" }
            ]
        }))
        .unwrap();
        assert_eq!(req.messages[1].role, Role::Assistant);
        assert_eq!(req.messages[1].tool_calls[0].id, "c1");
    }

    #[test]
    fn function_call_output_becomes_tool_message() {
        // Both string and array-of-output_text forms.
        let req = parse_responses_request(json!({
            "model": "m",
            "input": [
                { "type": "function_call_output", "call_id": "c1", "output": "72F" },
                { "type": "function_call_output", "call_id": "c2", "output": [
                    { "type": "output_text", "text": "ab" }, { "type": "output_text", "text": "cd" }
                ] }
            ]
        }))
        .unwrap();
        assert_eq!(req.messages[0].role, Role::Tool);
        assert_eq!(req.messages[0].tool_call_id.as_deref(), Some("c1"));
        assert_eq!(req.messages[0].text_content(), "72F");
        assert_eq!(req.messages[1].tool_call_id.as_deref(), Some("c2"));
        assert_eq!(req.messages[1].text_content(), "abcd");
    }

    #[test]
    fn reasoning_item_attaches_to_next_assistant_message() {
        let req = parse_responses_request(json!({
            "model": "m",
            "input": [
                { "type": "reasoning",
                  "summary": [{ "type": "summary_text", "text": "plan" }],
                  "content": [{ "type": "reasoning_text", "text": "steps" }],
                  "encrypted_content": "ENC" },
                { "role": "assistant", "content": [{ "type": "output_text", "text": "answer" }] }
            ]
        }))
        .unwrap();
        assert_eq!(req.messages.len(), 1);
        let m = &req.messages[0];
        assert_eq!(m.role, Role::Assistant);
        // Reasoning leads: two thinking blocks (summary + content) then the redacted block, then text.
        assert_eq!(m.content.len(), 4);
        assert!(
            matches!(&m.content[0], ContentPart::Thinking { thinking, .. } if thinking == "plan")
        );
        assert!(
            matches!(&m.content[1], ContentPart::Thinking { thinking, .. } if thinking == "steps")
        );
        assert!(matches!(&m.content[2], ContentPart::RedactedThinking { data } if data == "ENC"));
        assert!(matches!(&m.content[3], ContentPart::Text { text } if text == "answer"));
    }

    #[test]
    fn reasoning_item_signature_attaches_to_last_thinking_part() {
        // The OpenRouter Responses dialect signs an item's open reasoning at the item level; the
        // signature lands on the item's last thinking part so a signed block replays signed.
        let req = parse_responses_request(json!({
            "model": "m",
            "input": [
                { "type": "reasoning",
                  "content": [{ "type": "reasoning_text", "text": "signed thought" }],
                  "signature": "SIG" },
                { "role": "assistant", "content": [{ "type": "output_text", "text": "answer" }] }
            ]
        }))
        .unwrap();
        assert!(matches!(
            &req.messages[0].content[0],
            ContentPart::Thinking { thinking, signature }
                if thinking == "signed thought" && signature.as_deref() == Some("SIG")
        ));

        // A signature on an item with no thinking text has nothing to attach to and is not invented.
        let req = parse_responses_request(json!({
            "model": "m",
            "input": [
                { "type": "reasoning", "encrypted_content": "ENC", "signature": "SIG" },
                { "role": "assistant", "content": [{ "type": "output_text", "text": "a" }] }
            ]
        }))
        .unwrap();
        assert!(matches!(
            &req.messages[0].content[0],
            ContentPart::RedactedThinking { data } if data == "ENC"
        ));
    }

    #[test]
    fn trailing_reasoning_becomes_its_own_assistant_message() {
        let req = parse_responses_request(json!({
            "model": "m",
            "input": [
                { "role": "user", "content": "hi" },
                { "type": "reasoning", "content": [{ "type": "reasoning_text", "text": "orphan" }] }
            ]
        }))
        .unwrap();
        // The dangling reasoning is emitted, never dropped (P7).
        assert_eq!(req.messages.len(), 2);
        assert_eq!(req.messages[1].role, Role::Assistant);
        assert!(
            matches!(&req.messages[1].content[0], ContentPart::Thinking { thinking, .. } if thinking == "orphan")
        );
    }

    #[test]
    fn reasoning_effort_consumed_only_when_sole_key() {
        // effort-only, on the ladder → consumed into canonical thinking, not left in extra.
        let req = parse_responses_request(json!({
            "model": "m", "input": "hi", "reasoning": { "effort": "high" }
        }))
        .unwrap();
        assert_eq!(req.thinking, Some(Thinking::High));
        assert!(!req.extra.contains_key("reasoning"));

        // extra keys present → the whole object rides through verbatim, thinking stays None.
        let req = parse_responses_request(json!({
            "model": "m", "input": "hi", "reasoning": { "effort": "high", "summary": "auto" }
        }))
        .unwrap();
        assert_eq!(req.thinking, None);
        assert_eq!(req.extra["reasoning"]["summary"], "auto");

        // off-ladder effort → not canonical, rides through.
        let req = parse_responses_request(json!({
            "model": "m", "input": "hi", "reasoning": { "effort": "galaxy" }
        }))
        .unwrap();
        assert_eq!(req.thinking, None);
        assert_eq!(req.extra["reasoning"]["effort"], "galaxy");
    }

    #[test]
    fn store_is_stripped_and_unknown_fields_ride_through() {
        let req = parse_responses_request(json!({
            "model": "m", "input": "hi", "store": true,
            "metadata": { "k": "v" }, "truncation": "auto", "text": { "format": { "type": "text" } }
        }))
        .unwrap();
        assert!(!req.extra.contains_key("store"));
        assert_eq!(req.extra["metadata"]["k"], "v");
        assert_eq!(req.extra["truncation"], "auto");
        assert!(req.extra.contains_key("text"));
    }

    #[test]
    fn flat_tools_and_tool_choice() {
        let req = parse_responses_request(json!({
            "model": "m", "input": "hi",
            "tools": [{ "type": "function", "name": "get_weather", "parameters": { "type": "object" }, "strict": true }],
            "tool_choice": { "type": "function", "name": "get_weather" }
        }))
        .unwrap();
        assert_eq!(req.tools.len(), 1);
        assert_eq!(req.tools[0].name, "get_weather");
        assert_eq!(
            req.tool_choice,
            Some(ToolChoice::Named("get_weather".to_string()))
        );
    }

    #[test]
    fn rejections() {
        let is_err = |v: Value| parse_responses_request(v).is_err();
        // Stateless continuation / retrieval knobs.
        assert!(is_err(
            json!({ "model": "m", "input": "hi", "previous_response_id": "resp_x" })
        ));
        assert!(is_err(
            json!({ "model": "m", "input": "hi", "background": true })
        ));
        // A referenced stored item.
        assert!(is_err(json!({
            "model": "m", "input": [{ "type": "item_reference", "id": "msg_x" }]
        })));
        // A hosted tool the gateway cannot execute.
        assert!(is_err(json!({
            "model": "m", "input": "hi", "tools": [{ "type": "web_search" }]
        })));
        // An unsupported content part.
        assert!(is_err(json!({
            "model": "m",
            "input": [{ "role": "user", "content": [{ "type": "input_file", "file_id": "f_1" }] }]
        })));
        // A file_id-only image (no inline url).
        assert!(is_err(json!({
            "model": "m",
            "input": [{ "role": "user", "content": [{ "type": "input_image", "file_id": "f_1" }] }]
        })));
    }

    // --- Outbound collected view -----------------------------------------------------------------

    fn sample_response() -> ChatResponse {
        ChatResponse {
            id: "req-1".into(),
            model: "gpt-5".into(),
            choices: vec![Choice {
                index: 0,
                text: "the answer".into(),
                thinking: vec![
                    ContentPart::Thinking {
                        thinking: "reasoning".into(),
                        signature: Some("sig".into()),
                    },
                    ContentPart::RedactedThinking { data: "ENC".into() },
                ],
                tool_calls: vec![ToolCall {
                    id: "call_1".into(),
                    name: "get_weather".into(),
                    arguments: "{\"city\":\"NYC\"}".into(),
                }],
                finish_reason: Some(FinishReason::ToolCalls),
            }],
            usage: Usage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
                cost_usd: Some(0.01),
                cache_read_tokens: 3,
                cache_creation_tokens: 0,
            },
        }
    }

    fn empty_echo() -> RequestEcho {
        RequestEcho::from_request(
            &parse_responses_request(json!({ "model": "m", "input": "" })).unwrap(),
        )
    }

    #[test]
    fn collected_view_items_ids_status_and_usage() {
        let v = response_to_responses(&sample_response(), &empty_echo(), 1234);
        assert_eq!(v["id"], "resp_req-1");
        assert_eq!(v["object"], "response");
        assert_eq!(v["created_at"], 1234);
        assert_eq!(v["status"], "completed"); // ToolCalls → completed
        assert_eq!(v["store"], false);
        assert_eq!(v["model"], "gpt-5");

        let output = v["output"].as_array().unwrap();
        // reasoning (open text) → reasoning (encrypted) → message → function_call. Item ids carry the
        // output position, so they are unique within the response and match the streaming encoder's
        // `_<output_index>` convention.
        assert_eq!(output[0]["type"], "reasoning");
        assert_eq!(output[0]["id"], "rs_req-1_0");
        assert_eq!(output[0]["content"][0]["type"], "reasoning_text");
        assert_eq!(output[0]["content"][0]["text"], "reasoning");
        assert_eq!(output[0]["summary"].as_array().unwrap().len(), 0);
        assert_eq!(output[1]["type"], "reasoning");
        assert_eq!(output[1]["id"], "rs_req-1_1");
        assert_eq!(output[1]["encrypted_content"], "ENC");
        assert_eq!(output[2]["type"], "message");
        assert_eq!(output[2]["id"], "msg_req-1_2");
        assert_eq!(output[2]["status"], "completed");
        assert_eq!(output[2]["content"][0]["type"], "output_text");
        assert_eq!(output[2]["content"][0]["text"], "the answer");
        assert_eq!(output[3]["type"], "function_call");
        assert_eq!(output[3]["id"], "fc_req-1_3");
        assert_eq!(output[3]["call_id"], "call_1");
        assert_eq!(output[3]["name"], "get_weather");
        assert_eq!(output[3]["status"], "completed");

        // Usage incl. the cached-token relay and the honest reasoning-token floor.
        assert_eq!(v["usage"]["input_tokens"], 10);
        assert_eq!(v["usage"]["input_tokens_details"]["cached_tokens"], 3);
        assert_eq!(v["usage"]["output_tokens"], 5);
        assert_eq!(v["usage"]["output_tokens_details"]["reasoning_tokens"], 0);
        assert_eq!(v["usage"]["total_tokens"], 15);
        // The llmleaf cost extension rides on this surface exactly as it does on the chat surface.
        assert_eq!(v["usage"]["cost_usd"], 0.01);
    }

    #[test]
    fn collected_view_signature_signs_and_splits_reasoning_items() {
        // A signature ends the block it signs: two signed thinking parts become two reasoning items,
        // each carrying its own item-level `signature` (the OpenRouter Responses dialect); an unsigned
        // trailing part is its own unsigned item, with no `signature` key at all.
        let mut resp = sample_response();
        resp.choices[0].thinking = vec![
            ContentPart::Thinking {
                thinking: "first".into(),
                signature: Some("SIG_A".into()),
            },
            ContentPart::Thinking {
                thinking: "second".into(),
                signature: Some("SIG_B".into()),
            },
            ContentPart::Thinking {
                thinking: "unsigned".into(),
                signature: None,
            },
        ];
        let v = response_to_responses(&resp, &empty_echo(), 0);
        let output = v["output"].as_array().unwrap();
        assert_eq!(output[0]["content"][0]["text"], "first");
        assert_eq!(output[0]["signature"], "SIG_A");
        assert_eq!(output[1]["content"][0]["text"], "second");
        assert_eq!(output[1]["signature"], "SIG_B");
        assert_eq!(output[2]["content"][0]["text"], "unsigned");
        assert!(output[2].get("signature").is_none());
    }

    #[test]
    fn length_finish_is_incomplete() {
        let mut resp = sample_response();
        resp.choices[0].finish_reason = Some(FinishReason::Length);
        let v = response_to_responses(&resp, &empty_echo(), 0);
        assert_eq!(v["status"], "incomplete");
        assert_eq!(v["incomplete_details"]["reason"], "max_output_tokens");
    }

    #[test]
    fn echo_reflects_request_params() {
        let req = parse_responses_request(json!({
            "model": "m", "input": "hi",
            // Values chosen exactly representable in f32 so the echo round-trips byte-for-byte.
            "temperature": 0.5, "top_p": 0.25, "max_output_tokens": 256,
            "reasoning": { "effort": "high" },
            "tools": [{ "type": "function", "name": "f", "parameters": {} }],
            "tool_choice": "required",
            "metadata": { "k": "v" }
        }))
        .unwrap();
        let echo = RequestEcho::from_request(&req);
        let resp = ChatResponse {
            id: "r".into(),
            model: "m".into(),
            choices: vec![],
            usage: Usage::default(),
        };
        let v = response_to_responses(&resp, &echo, 0);
        assert_eq!(v["temperature"], 0.5);
        assert_eq!(v["top_p"], 0.25);
        assert_eq!(v["max_output_tokens"], 256);
        assert_eq!(v["reasoning"]["effort"], "high");
        assert_eq!(v["tools"][0]["type"], "function");
        assert_eq!(v["tools"][0]["name"], "f");
        assert_eq!(v["tool_choice"], "required");
        assert_eq!(v["metadata"]["k"], "v");
    }

    // --- Outbound streaming ----------------------------------------------------------------------

    fn run_encoder(chunks: &[StreamChunk]) -> Vec<(&'static str, Value)> {
        let echo = empty_echo();
        let mut enc = EventEncoder::new("req-1", "gpt-5", 1000, echo);
        let mut frames = Vec::new();
        let mut out = Vec::new();
        for chunk in chunks {
            frames.clear();
            enc.encode(chunk, &mut frames);
            for f in &frames {
                out.push((f.event, serde_json::from_str::<Value>(&f.data).unwrap()));
            }
        }
        frames.clear();
        enc.finish(&mut frames);
        for f in &frames {
            out.push((f.event, serde_json::from_str::<Value>(&f.data).unwrap()));
        }
        out
    }

    #[test]
    fn streaming_lifecycle_seq_and_snapshot_matches_collected() {
        let chunks = vec![
            // A faithful provider echoes the logical model on `Start`; the collected view reads
            // `resp.model` from it (as the chat surface does), while the encoder holds the logical
            // model — they agree, so the terminal snapshot matches the collected view.
            StreamChunk::Start {
                id: "upstream".into(),
                model: "gpt-5".into(),
            },
            StreamChunk::Thinking {
                index: 0,
                delta: "think".into(),
            },
            StreamChunk::Content {
                index: 0,
                delta: "hello ".into(),
            },
            StreamChunk::Content {
                index: 0,
                delta: "world".into(),
            },
            StreamChunk::ToolCall {
                index: 0,
                call: ToolCallDelta {
                    index: 0,
                    id: Some("call_1".into()),
                    name: Some("f".into()),
                    arguments: Some("{}".into()),
                },
            },
            StreamChunk::Usage(Usage {
                prompt_tokens: 4,
                completion_tokens: 3,
                total_tokens: 7,
                cost_usd: None,
                cache_read_tokens: 2,
                cache_creation_tokens: 0,
            }),
            StreamChunk::Finish {
                index: 0,
                reason: FinishReason::Stop,
            },
        ];
        let events = run_encoder(&chunks);
        let names: Vec<&str> = events.iter().map(|(n, _)| *n).collect();

        // The envelope opens the stream and `response.completed` terminates it.
        assert_eq!(names[0], "response.created");
        assert_eq!(names[1], "response.in_progress");
        assert_eq!(names.last().copied(), Some("response.completed"));
        // Item lifecycle appeared for each kind.
        assert!(names.contains(&"response.reasoning_text.delta"));
        assert!(names.contains(&"response.output_text.delta"));
        assert!(names.contains(&"response.function_call_arguments.delta"));
        assert!(
            names
                .iter()
                .filter(|n| **n == "response.output_item.added")
                .count()
                >= 3
        );
        assert!(
            names
                .iter()
                .filter(|n| **n == "response.output_item.done")
                .count()
                >= 3
        );

        // Every event carries a strictly-increasing sequence_number.
        let seqs: Vec<u64> = events
            .iter()
            .map(|(_, d)| d["sequence_number"].as_u64().unwrap())
            .collect();
        for w in seqs.windows(2) {
            assert!(
                w[1] > w[0],
                "sequence_number not strictly increasing: {seqs:?}"
            );
        }

        // The terminal snapshot equals the collected view for the same chunks (one shared builder).
        let snapshot = &events.last().unwrap().1["response"];
        let mut resp = collect_chunks(chunks);
        resp.id = "req-1".into(); // the edge stamps the consumer request id
        let collected = response_to_responses(&resp, &empty_echo(), 1000);
        assert_eq!(snapshot, &collected);

        // The delta carried the streamed text; the snapshot carries the whole answer + usage.
        let text_delta = events
            .iter()
            .find(|(n, _)| *n == "response.output_text.delta")
            .unwrap();
        assert_eq!(text_delta.1["delta"], "hello ");
        assert_eq!(snapshot["output"][1]["content"][0]["text"], "hello world");
        assert_eq!(snapshot["usage"]["output_tokens"], 3);
        assert_eq!(
            snapshot["usage"]["input_tokens_details"]["cached_tokens"],
            2
        );
    }

    #[test]
    fn streaming_signature_lands_on_done_item_and_snapshot() {
        // A ThinkingSignature chunk (signed open reasoning relayed from an OpenRouter-flavor upstream)
        // emits no event of its own; it rides on the reasoning item's `output_item.done` and folds into
        // the terminal snapshot exactly as the canonical collector folds it.
        let chunks = vec![
            StreamChunk::Start {
                id: "upstream".into(),
                model: "gpt-5".into(),
            },
            StreamChunk::Thinking {
                index: 0,
                delta: "signed think".into(),
            },
            StreamChunk::ThinkingSignature {
                index: 0,
                signature: "SIG".into(),
            },
            StreamChunk::Content {
                index: 0,
                delta: "answer".into(),
            },
            StreamChunk::Finish {
                index: 0,
                reason: FinishReason::Stop,
            },
        ];
        let events = run_encoder(&chunks);
        let done = events
            .iter()
            .find(|(n, d)| *n == "response.output_item.done" && d["item"]["type"] == "reasoning")
            .expect("reasoning item closed");
        assert_eq!(done.1["item"]["signature"], "SIG");
        assert_eq!(done.1["item"]["content"][0]["text"], "signed think");

        // Snapshot parity with the collected path (the shared fold carries the signature).
        let snapshot = &events.last().unwrap().1["response"];
        let mut resp = collect_chunks(chunks);
        resp.id = "req-1".into();
        let collected = response_to_responses(&resp, &empty_echo(), 1000);
        assert_eq!(snapshot, &collected);
        assert_eq!(snapshot["output"][0]["signature"], "SIG");
    }

    #[test]
    fn empty_stream_still_well_formed() {
        let events = run_encoder(&[]);
        let names: Vec<&str> = events.iter().map(|(n, _)| *n).collect();
        assert_eq!(
            names,
            vec![
                "response.created",
                "response.in_progress",
                "response.completed"
            ]
        );
        assert_eq!(events.last().unwrap().1["response"]["status"], "completed");
    }

    #[test]
    fn error_frame_shape() {
        let mut enc = EventEncoder::new("req-1", "m", 0, empty_echo());
        let frame = enc.error_frame("upstream exploded");
        assert_eq!(frame.event, "error");
        let v: Value = serde_json::from_str(&frame.data).unwrap();
        assert_eq!(v["type"], "error");
        assert_eq!(v["message"], "upstream exploded");
        assert!(v["code"].is_null());
        assert!(v["param"].is_null());
        assert!(v["sequence_number"].is_u64());
    }
}
