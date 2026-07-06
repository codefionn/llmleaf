//! The OpenAI Chat Completions compat surface.
//!
//! In: an OpenAI `chat.completion` request JSON → canonical [`ChatRequest`]. Unknown fields are
//! preserved verbatim in [`ChatRequest::extra`] (principle 7: transparent — we never silently drop).
//! Out: a canonical [`ResponseStream`]/[`ChatResponse`] → OpenAI `chat.completion.chunk` SSE frames
//! or a collected `chat.completion` object.

use llmleaf_model::{
    ChatRequest, ChatResponse, ContentPart, FinishReason, Message, ModelError, Role, StreamChunk,
    Thinking, ToolCall, ToolChoice, ToolDef, Usage,
};
use serde::Serialize;
use serde_json::Value;
use std::borrow::Cow;

fn mapping(msg: impl Into<String>) -> ModelError {
    ModelError::Mapping(msg.into())
}

// ---------------------------------------------------------------------------------------------
// Inbound: OpenAI request JSON -> canonical ChatRequest
// ---------------------------------------------------------------------------------------------

pub fn parse_chat_request(value: Value) -> Result<ChatRequest, ModelError> {
    let Value::Object(mut obj) = value else {
        return Err(mapping("request body must be a JSON object"));
    };

    let model = match obj.remove("model") {
        Some(Value::String(s)) => s,
        _ => return Err(mapping("`model` is required and must be a string")),
    };

    let messages = parse_messages(obj.remove("messages"))?;

    let max_tokens = obj
        .remove("max_completion_tokens")
        .or_else(|| obj.remove("max_tokens"))
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
    let stop = parse_stop(obj.remove("stop"));
    let stream = obj
        .remove("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let tools = parse_tools(obj.remove("tools"))?;
    let tool_choice = parse_tool_choice(obj.remove("tool_choice"));

    // A `reasoning_effort` the ladder can express becomes the canonical `thinking` knob, so *every*
    // downstream dialect can speak it — the OpenAI wire re-emits its own `reasoning_effort` at that
    // edge, while the budget-style providers (Anthropic, Gemini, Cohere) map it to a token budget they
    // could never see from a passthrough. A vocabulary outside the ladder rides through `extra`
    // untouched, exactly as before (principle 7).
    let thinking = match obj.remove("reasoning_effort") {
        Some(v) => match v.as_str().and_then(parse_reasoning_effort) {
            Some(t) => Some(t),
            None => {
                obj.insert("reasoning_effort".into(), v);
                None
            }
        },
        None => None,
    };

    // Whatever the consumer sent that we don't model rides through untouched.
    let extra = obj;

    Ok(ChatRequest {
        model,
        messages,
        max_tokens,
        temperature,
        top_p,
        stop,
        stream,
        tools,
        tool_choice,
        thinking,
        extra,
    })
}

/// An OpenAI-style `reasoning_effort` string → the canonical [`Thinking`] ladder. Accepts OpenAI's
/// vocabulary (`minimal`/`low`/`medium`/`high`), the ladder's own serde names (`med`/`highx`), and the
/// common extended rungs (`xhigh`/`max`). Anything else is not an error — the caller leaves it in
/// `extra` to ride through verbatim.
///
/// `pub(crate)` so the Responses dialect ([`super::responses`]) reuses the exact same vocabulary when
/// mapping its `reasoning.effort` onto the ladder — one source of truth, never a second copy that could
/// drift (principle 3).
pub(crate) fn parse_reasoning_effort(s: &str) -> Option<Thinking> {
    Some(match s {
        "minimal" | "low" => Thinking::Low,
        "medium" | "med" => Thinking::Med,
        "high" => Thinking::High,
        "xhigh" | "highx" => Thinking::Highx,
        "max" => Thinking::Max,
        _ => return None,
    })
}

fn parse_messages(value: Option<Value>) -> Result<Vec<Message>, ModelError> {
    let Some(Value::Array(items)) = value else {
        return Err(mapping("`messages` is required and must be an array"));
    };
    items.into_iter().map(parse_message).collect()
}

fn parse_message(value: Value) -> Result<Message, ModelError> {
    let Value::Object(mut obj) = value else {
        return Err(mapping("each message must be an object"));
    };
    // `parse_role` only matches fixed literals and returns a `Copy` `Role`, so borrow the role string
    // already owned by the map instead of allocating a throwaway copy of it.
    let role = match obj.remove("role") {
        Some(Value::String(r)) => parse_role(&r)?,
        _ => return Err(mapping("message is missing `role`")),
    };
    let content = parse_content(obj.remove("content"))?;
    let tool_calls = parse_tool_calls(obj.remove("tool_calls"))?;
    let tool_call_id = obj
        .remove("tool_call_id")
        .and_then(|v| v.as_str().map(str::to_owned));
    let name = obj
        .remove("name")
        .and_then(|v| v.as_str().map(str::to_owned));
    Ok(Message {
        role,
        content,
        tool_calls,
        tool_call_id,
        name,
    })
}

fn parse_role(role: &str) -> Result<Role, ModelError> {
    Ok(match role {
        "system" | "developer" => Role::System,
        "user" => Role::User,
        "assistant" => Role::Assistant,
        "tool" | "function" => Role::Tool,
        other => return Err(mapping(format!("unknown role `{other}`"))),
    })
}

fn parse_content(value: Option<Value>) -> Result<Vec<ContentPart>, ModelError> {
    match value {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(Value::String(text)) => Ok(vec![ContentPart::Text { text }]),
        Some(Value::Array(parts)) => parts.into_iter().map(parse_content_part).collect(),
        Some(_) => Err(mapping("`content` must be a string, array, or null")),
    }
}

fn parse_content_part(value: Value) -> Result<ContentPart, ModelError> {
    let Value::Object(obj) = value else {
        return Err(mapping("content part must be an object"));
    };
    match obj.get("type").and_then(Value::as_str) {
        Some("text") => {
            let text = obj
                .get("text")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            Ok(ContentPart::Text { text })
        }
        Some("image_url") => {
            let img = obj.get("image_url");
            let url = img
                .and_then(|v| v.get("url"))
                .and_then(Value::as_str)
                .ok_or_else(|| mapping("image_url part missing `image_url.url`"))?
                .to_string();
            let detail = img
                .and_then(|v| v.get("detail"))
                .and_then(Value::as_str)
                .map(str::to_owned);
            Ok(ContentPart::ImageUrl { url, detail })
        }
        other => Err(mapping(format!("unsupported content part type {other:?}"))),
    }
}

fn parse_tool_calls(value: Option<Value>) -> Result<Vec<ToolCall>, ModelError> {
    let Some(Value::Array(items)) = value else {
        return Ok(Vec::new());
    };
    items
        .into_iter()
        .map(|v| {
            let func = v.get("function");
            Ok(ToolCall {
                id: v
                    .get("id")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                name: func
                    .and_then(|f| f.get("name"))
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                arguments: func
                    .and_then(|f| f.get("arguments"))
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
            })
        })
        .collect()
}

fn parse_tools(value: Option<Value>) -> Result<Vec<ToolDef>, ModelError> {
    let Some(Value::Array(items)) = value else {
        return Ok(Vec::new());
    };
    items
        .into_iter()
        .filter_map(|v| v.get("function").cloned())
        .map(|f| {
            let name = f
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| mapping("tool function missing `name`"))?
                .to_string();
            Ok(ToolDef {
                name,
                description: f
                    .get("description")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                parameters: f.get("parameters").cloned().unwrap_or(Value::Null),
            })
        })
        .collect()
}

fn parse_tool_choice(value: Option<Value>) -> Option<ToolChoice> {
    match value {
        Some(Value::String(s)) => match s.as_str() {
            "auto" => Some(ToolChoice::Auto),
            "none" => Some(ToolChoice::None),
            "required" => Some(ToolChoice::Required),
            _ => None,
        },
        Some(Value::Object(o)) => o
            .get("function")
            .and_then(|f| f.get("name"))
            .and_then(Value::as_str)
            .map(|n| ToolChoice::Named(n.to_string())),
        _ => None,
    }
}

fn parse_stop(value: Option<Value>) -> Vec<String> {
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
// Outbound: canonical -> OpenAI wire format
// ---------------------------------------------------------------------------------------------

fn finish_str(reason: FinishReason) -> &'static str {
    match reason {
        FinishReason::Stop => "stop",
        FinishReason::Length => "length",
        FinishReason::ToolCalls => "tool_calls",
        FinishReason::ContentFilter => "content_filter",
        FinishReason::Error => "stop",
    }
}

// The borrowed wire shape of an OpenAI `chat.completion.chunk`. Serialising these `&str`-holding
// structs straight into a reused buffer is what keeps the streaming hot path allocation-free
// (principle 1): unlike a `serde_json::Value`, none of these owns a heap `String` or a `Map`. Fields
// are declared in the sorted key order `serde_json::Map` (a `BTreeMap`) emits, so the wire bytes are
// identical to the old `Value`-building encoder.
#[derive(Serialize)]
struct ChunkFrame<'a> {
    choices: &'a [ChoiceFrame<'a>],
    created: u64,
    id: &'a str,
    model: &'a str,
    object: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<UsageFrame>,
}

#[derive(Serialize)]
struct ChoiceFrame<'a> {
    delta: Delta<'a>,
    // Always emitted — `null` until the choice finishes — matching OpenAI's frames.
    finish_reason: Option<&'static str>,
    index: u32,
}

/// The per-chunk `delta` object. Untagged so each variant serialises as just its inner field map
/// (`{"role":..}`, `{"content":..}`, `{"tool_calls":..}`, or `{}`).
#[derive(Serialize)]
#[serde(untagged)]
enum Delta<'a> {
    Role {
        role: &'static str,
    },
    Content {
        content: &'a str,
    },
    ToolCalls {
        tool_calls: [ToolCallFrame<'a>; 1],
    },
    // Open reasoning text — the OpenRouter `delta.reasoning` extension; there is no stock-OpenAI
    // chunk field for it. A client that doesn't know the field ignores the frame's empty-content
    // delta harmlessly.
    Reasoning {
        reasoning: &'a str,
    },
    // A signed / redacted reasoning block, as one OpenRouter-style `reasoning_details` entry.
    ReasoningDetails {
        reasoning_details: [ReasoningDetailFrame<'a>; 1],
    },
    Empty {},
}

/// One OpenRouter-style `reasoning_details[]` entry (`reasoning.text` carries `signature`,
/// `reasoning.encrypted` carries `data`). Field order is sorted, like every frame here.
#[derive(Serialize)]
struct ReasoningDetailFrame<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    data: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    signature: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<&'a str>,
    #[serde(rename = "type")]
    kind: &'static str,
}

#[derive(Serialize)]
struct ToolCallFrame<'a> {
    function: FunctionFrame<'a>,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<&'a str>,
    index: u32,
    #[serde(rename = "type")]
    kind: &'static str,
}

#[derive(Serialize)]
struct FunctionFrame<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    arguments: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<&'a str>,
}

#[derive(Serialize)]
struct UsageFrame {
    completion_tokens: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    cost_usd: Option<f64>,
    prompt_tokens: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_tokens_details: Option<PromptTokensDetails>,
    total_tokens: u64,
    // Cache *creation* (cache writes) is Anthropic-only — OpenAI has no field for it — so it rides as
    // an llmleaf extension, omitted when zero (a non-caching response stays byte-identical).
    #[serde(skip_serializing_if = "is_zero")]
    cache_creation_tokens: u64,
}

/// OpenAI's `usage.prompt_tokens_details` — the standard home for the cached-read count, which is
/// what OpenAI and OpenRouter emit and the shape a stock OpenAI client expects.
#[derive(Serialize)]
struct PromptTokensDetails {
    cached_tokens: u64,
}

fn is_zero(n: &u64) -> bool {
    *n == 0
}

impl From<&Usage> for UsageFrame {
    fn from(u: &Usage) -> Self {
        UsageFrame {
            completion_tokens: u.completion_tokens,
            cost_usd: u.cost_usd,
            prompt_tokens: u.prompt_tokens,
            // Cache reads surface OpenAI-style under `prompt_tokens_details.cached_tokens`; omitted
            // entirely when there were none, so non-caching responses serialise exactly as before.
            prompt_tokens_details: (u.cache_read_tokens > 0).then_some(PromptTokensDetails {
                cached_tokens: u.cache_read_tokens,
            }),
            total_tokens: u.total_tokens,
            cache_creation_tokens: u.cache_creation_tokens,
        }
    }
}

// The borrowed wire shape of an OpenAI collected `chat.completion` object. Same technique as the
// streaming `ChunkFrame` above (principle 1): serialising these `&str`-holding structs straight into a
// buffer never builds a `serde_json::Value` tree, so the only per-request allocations are the leaf
// `Vec`s — no owned `String` clones. Fields are declared in the sorted key order `serde_json::Map` (a
// `BTreeMap`) emits, so the wire bytes are byte-identical to the old `Value`-building mapping. Reuses
// `UsageFrame` + `From<&Usage>` above, whose fields are already in sorted order.
#[derive(Serialize)]
pub struct CompletionView<'a> {
    choices: Vec<ChoiceView<'a>>,
    created: u64,
    id: &'a str,
    model: &'a str,
    object: &'static str,
    usage: UsageFrame,
}

#[derive(Serialize)]
struct ChoiceView<'a> {
    // Always emitted; `null` when `None` — matches the old `Value::Null` branch (no skip).
    finish_reason: Option<&'static str>,
    index: u32,
    message: MessageView<'a>,
}

#[derive(Serialize)]
struct MessageView<'a> {
    content: &'a str,
    // Open reasoning text and structured blocks — the OpenRouter `message.reasoning` /
    // `message.reasoning_details` extensions, mirroring the streaming deltas above. Both are
    // omitted for a choice with no thinking, so a non-reasoning response serialises exactly as
    // before. `reasoning` is a `Cow`: one open block (the overwhelmingly common shape) borrows,
    // multiple blocks concatenate.
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<Cow<'a, str>>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    reasoning_details: Vec<ReasoningDetailFrame<'a>>,
    role: &'static str,
    // Omitted entirely when the choice has no tool calls — matches the old
    // `if !c.tool_calls.is_empty()` insert-or-skip.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tool_calls: Vec<ToolCallView<'a>>,
}

/// The choice's open reasoning text: `None` when it has no open thinking block, a borrow of the one
/// block's text in the common case, an owned concatenation when there are several.
fn choice_reasoning(thinking: &[ContentPart]) -> Option<Cow<'_, str>> {
    let mut open = thinking.iter().filter_map(|p| match p {
        ContentPart::Thinking { thinking, .. } => Some(thinking.as_str()),
        _ => None,
    });
    let first = open.next()?;
    Some(match open.next() {
        None => Cow::Borrowed(first),
        Some(second) => {
            let mut all = String::from(first);
            all.push_str(second);
            open.for_each(|t| all.push_str(t));
            Cow::Owned(all)
        }
    })
}

/// The choice's thinking blocks as OpenRouter-style `reasoning_details[]` entries.
fn choice_reasoning_details(thinking: &[ContentPart]) -> Vec<ReasoningDetailFrame<'_>> {
    thinking
        .iter()
        .filter_map(|p| match p {
            ContentPart::Thinking {
                thinking,
                signature,
            } => Some(ReasoningDetailFrame {
                data: None,
                signature: signature.as_deref(),
                text: Some(thinking),
                kind: "reasoning.text",
            }),
            ContentPart::RedactedThinking { data } => Some(ReasoningDetailFrame {
                data: Some(data),
                signature: None,
                text: None,
                kind: "reasoning.encrypted",
            }),
            _ => None,
        })
        .collect()
}

#[derive(Serialize)]
struct ToolCallView<'a> {
    function: FunctionView<'a>,
    id: &'a str,
    #[serde(rename = "type")]
    kind: &'static str,
}

#[derive(Serialize)]
struct FunctionView<'a> {
    arguments: &'a str,
    name: &'a str,
}

/// Build the borrowed `chat.completion` view, borrowing every string leaf straight from `resp`. Shared
/// by the live chat edge ([`response_to_openai`]) and the batch surface, so there is one source of
/// truth for the collected shape (principle 3: one internal model).
pub(crate) fn completion_view(resp: &ChatResponse, created: u64) -> CompletionView<'_> {
    let choices = resp
        .choices
        .iter()
        .map(|c| ChoiceView {
            finish_reason: c.finish_reason.map(finish_str),
            index: c.index,
            message: MessageView {
                content: &c.text,
                reasoning: choice_reasoning(&c.thinking),
                reasoning_details: choice_reasoning_details(&c.thinking),
                role: "assistant",
                tool_calls: c
                    .tool_calls
                    .iter()
                    .map(|t| ToolCallView {
                        function: FunctionView {
                            arguments: &t.arguments,
                            name: &t.name,
                        },
                        id: &t.id,
                        kind: "function",
                    })
                    .collect(),
            },
        })
        .collect();
    CompletionView {
        choices,
        created,
        id: &resp.id,
        model: &resp.model,
        object: "chat.completion",
        usage: UsageFrame::from(&resp.usage),
    }
}

/// Maps the canonical stream to OpenAI `chat.completion.chunk` frames. Holds the response identity
/// so every frame is consistent. `created` is unix-seconds supplied by the edge.
pub struct ChunkEncoder {
    pub id: String,
    pub model: String,
    pub created: u64,
}

impl ChunkEncoder {
    pub fn new(id: impl Into<String>, model: impl Into<String>, created: u64) -> Self {
        ChunkEncoder {
            id: id.into(),
            model: model.into(),
            created,
        }
    }

    /// Translate one canonical chunk to its OpenAI `chat.completion.chunk` frame, serialised straight
    /// into `buf`. Returns `false` if the chunk produces no frame. `buf` is cleared first and its
    /// capacity is reused across a stream, so a steady stream allocates nothing per chunk — the hot
    /// path principle 1 calls sacred. Output is byte-identical to the old `Value`-building encoder.
    pub fn encode_into(&self, chunk: &StreamChunk, buf: &mut Vec<u8>) -> bool {
        buf.clear();
        match chunk {
            // The resolved id/model are already captured at construction; surface the opening role.
            StreamChunk::Start { .. } => self.write_frame(
                buf,
                &[ChoiceFrame {
                    delta: Delta::Role { role: "assistant" },
                    finish_reason: None,
                    index: 0,
                }],
                None,
            ),
            StreamChunk::Content { index, delta } => self.write_frame(
                buf,
                &[ChoiceFrame {
                    delta: Delta::Content { content: delta },
                    finish_reason: None,
                    index: *index,
                }],
                None,
            ),
            StreamChunk::ToolCall { index, call } => self.write_frame(
                buf,
                &[ChoiceFrame {
                    delta: Delta::ToolCalls {
                        tool_calls: [ToolCallFrame {
                            function: FunctionFrame {
                                arguments: call.arguments.as_deref(),
                                name: call.name.as_deref(),
                            },
                            id: call.id.as_deref(),
                            index: call.index,
                            kind: "function",
                        }],
                    },
                    finish_reason: None,
                    index: *index,
                }],
                None,
            ),
            // Reasoning surfaces as the OpenRouter extension fields — `delta.reasoning` for open
            // text, `delta.reasoning_details` for signed/encrypted blocks — since stock OpenAI
            // chunks have no representation for it and dropping it blinds every consumer downstream
            // of the gateway (a reasoning model's visible thinking would vanish here).
            StreamChunk::Thinking { index, delta } => self.write_frame(
                buf,
                &[ChoiceFrame {
                    delta: Delta::Reasoning { reasoning: delta },
                    finish_reason: None,
                    index: *index,
                }],
                None,
            ),
            StreamChunk::ThinkingSignature { index, signature } => self.write_frame(
                buf,
                &[ChoiceFrame {
                    delta: Delta::ReasoningDetails {
                        reasoning_details: [ReasoningDetailFrame {
                            data: None,
                            signature: Some(signature),
                            text: None,
                            kind: "reasoning.text",
                        }],
                    },
                    finish_reason: None,
                    index: *index,
                }],
                None,
            ),
            StreamChunk::RedactedThinking { index, data } => self.write_frame(
                buf,
                &[ChoiceFrame {
                    delta: Delta::ReasoningDetails {
                        reasoning_details: [ReasoningDetailFrame {
                            data: Some(data),
                            signature: None,
                            text: None,
                            kind: "reasoning.encrypted",
                        }],
                    },
                    finish_reason: None,
                    index: *index,
                }],
                None,
            ),
            StreamChunk::Usage(u) => self.write_frame(buf, &[], Some(UsageFrame::from(u))),
            StreamChunk::Finish { index, reason } => self.write_frame(
                buf,
                &[ChoiceFrame {
                    delta: Delta::Empty {},
                    finish_reason: Some(finish_str(*reason)),
                    index: *index,
                }],
                None,
            ),
        }
        true
    }

    /// Serialise one envelope (`{choices, created, id, model, object, usage?}`) into `buf`.
    fn write_frame(&self, buf: &mut Vec<u8>, choices: &[ChoiceFrame], usage: Option<UsageFrame>) {
        let frame = ChunkFrame {
            choices,
            created: self.created,
            id: &self.id,
            model: &self.model,
            object: "chat.completion.chunk",
            usage,
        };
        serde_json::to_writer(buf, &frame).expect("chunk frame serialises");
    }

    /// The terminal SSE data payload OpenAI clients expect.
    pub const DONE: &'static str = "[DONE]";
}

/// Collect-mode mapping: a canonical [`ChatResponse`] to an OpenAI `chat.completion` object.
///
/// Returns a borrowed [`CompletionView`] rather than a `serde_json::Value`: serialising it straight to
/// bytes (the caller wraps it in `axum::Json`) never builds a `Value` tree, so the per-request cost
/// drops to the leaf `Vec`s alone — the same proven technique the streaming encoder uses (principle 1).
/// Output is byte-identical to the old `Value`-building mapping (see `collected_view_bytes_match_legacy`).
pub fn response_to_openai(resp: &ChatResponse, created: u64) -> CompletionView<'_> {
    completion_view(resp, created)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Map};

    #[test]
    fn parse_minimal_request() {
        let v = json!({
            "model": "gpt-4o",
            "messages": [{ "role": "user", "content": "hi" }]
        });
        let req = parse_chat_request(v).unwrap();
        assert_eq!(req.model, "gpt-4o");
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.messages[0].text_content(), "hi");
        assert!(!req.stream);
    }

    #[test]
    fn unknown_fields_pass_through_to_extra() {
        let v = json!({
            "model": "gpt-4o",
            "messages": [{ "role": "user", "content": "hi" }],
            "seed": 42,
            "logit_bias": { "50256": -100 }
        });
        let req = parse_chat_request(v).unwrap();
        assert_eq!(req.extra.get("seed"), Some(&json!(42)));
        assert!(req.extra.contains_key("logit_bias"));
    }

    #[test]
    fn reasoning_effort_maps_to_canonical_thinking() {
        let req = |effort: &str| {
            parse_chat_request(json!({
                "model": "gpt-4o",
                "messages": [{ "role": "user", "content": "hi" }],
                "reasoning_effort": effort
            }))
            .unwrap()
        };
        assert_eq!(req("low").thinking, Some(Thinking::Low));
        assert_eq!(req("minimal").thinking, Some(Thinking::Low));
        assert_eq!(req("medium").thinking, Some(Thinking::Med));
        assert_eq!(req("high").thinking, Some(Thinking::High));
        assert_eq!(req("xhigh").thinking, Some(Thinking::Highx));
        assert_eq!(req("max").thinking, Some(Thinking::Max));
        // A mapped effort is consumed — it must not ALSO ride through `extra`, or the OpenAI wire
        // edge would emit the consumer's raw string alongside its own canonical re-emission.
        assert!(!req("low").extra.contains_key("reasoning_effort"));
        // Outside the ladder → not canonical, rides through verbatim (the old behaviour).
        let exotic = req("galaxy-brain");
        assert_eq!(exotic.thinking, None);
        assert_eq!(
            exotic.extra.get("reasoning_effort"),
            Some(&json!("galaxy-brain"))
        );
    }

    #[test]
    fn encode_reasoning_frames() {
        let enc = ChunkEncoder::new("resp-1", "gpt-4o", 1000);
        let mut buf = Vec::new();

        // Open reasoning text → OpenRouter-style `delta.reasoning`.
        assert!(enc.encode_into(
            &StreamChunk::Thinking {
                index: 0,
                delta: "weighing options".into()
            },
            &mut buf
        ));
        let frame: Value = serde_json::from_slice(&buf).unwrap();
        assert_eq!(
            frame["choices"][0]["delta"]["reasoning"],
            "weighing options"
        );

        // A signature rides as a `reasoning.text` details entry.
        assert!(enc.encode_into(
            &StreamChunk::ThinkingSignature {
                index: 0,
                signature: "sig-abc".into()
            },
            &mut buf
        ));
        let frame: Value = serde_json::from_slice(&buf).unwrap();
        let d = &frame["choices"][0]["delta"]["reasoning_details"][0];
        assert_eq!(d["type"], "reasoning.text");
        assert_eq!(d["signature"], "sig-abc");

        // A redacted block rides as a `reasoning.encrypted` details entry.
        assert!(enc.encode_into(
            &StreamChunk::RedactedThinking {
                index: 0,
                data: "opaque".into()
            },
            &mut buf
        ));
        let frame: Value = serde_json::from_slice(&buf).unwrap();
        let d = &frame["choices"][0]["delta"]["reasoning_details"][0];
        assert_eq!(d["type"], "reasoning.encrypted");
        assert_eq!(d["data"], "opaque");
    }

    #[test]
    fn collected_response_carries_reasoning() {
        let resp = ChatResponse {
            id: "r1".into(),
            model: "m".into(),
            choices: vec![llmleaf_model::Choice {
                index: 0,
                text: "answer".into(),
                thinking: vec![
                    ContentPart::Thinking {
                        thinking: "step by step".into(),
                        signature: Some("sig-xyz".into()),
                    },
                    ContentPart::RedactedThinking {
                        data: "opaque".into(),
                    },
                ],
                tool_calls: Vec::new(),
                finish_reason: Some(FinishReason::Stop),
            }],
            usage: Usage::default(),
        };
        let v = serde_json::to_value(response_to_openai(&resp, 1000)).unwrap();
        let msg = &v["choices"][0]["message"];
        assert_eq!(msg["content"], "answer");
        assert_eq!(msg["reasoning"], "step by step");
        assert_eq!(msg["reasoning_details"][0]["type"], "reasoning.text");
        assert_eq!(msg["reasoning_details"][0]["signature"], "sig-xyz");
        assert_eq!(msg["reasoning_details"][1]["type"], "reasoning.encrypted");

        // A choice with no thinking serialises without either field — byte-compatible with the
        // pre-reasoning shape.
        let bare = ChatResponse {
            id: "r2".into(),
            model: "m".into(),
            choices: vec![llmleaf_model::Choice {
                index: 0,
                text: "plain".into(),
                thinking: Vec::new(),
                tool_calls: Vec::new(),
                finish_reason: Some(FinishReason::Stop),
            }],
            usage: Usage::default(),
        };
        let v = serde_json::to_value(response_to_openai(&bare, 1000)).unwrap();
        let msg = v["choices"][0]["message"].as_object().unwrap();
        assert!(!msg.contains_key("reasoning"));
        assert!(!msg.contains_key("reasoning_details"));
    }

    #[test]
    fn parses_array_content_and_tools() {
        let v = json!({
            "model": "gpt-4o",
            "stream": true,
            "messages": [{
                "role": "user",
                "content": [
                    { "type": "text", "text": "look" },
                    { "type": "image_url", "image_url": { "url": "http://x/y.png", "detail": "low" } }
                ]
            }],
            "tools": [{ "type": "function", "function": { "name": "get_weather", "parameters": { "type": "object" } } }],
            "tool_choice": "required"
        });
        let req = parse_chat_request(v).unwrap();
        assert!(req.stream);
        assert_eq!(req.messages[0].content.len(), 2);
        assert_eq!(req.tools[0].name, "get_weather");
        assert_eq!(req.tool_choice, Some(ToolChoice::Required));
    }

    #[test]
    fn encode_content_and_finish_frames() {
        let enc = ChunkEncoder::new("resp-1", "gpt-4o", 1000);
        // One buffer, reused across chunks — `encode_into` clears it each time.
        let mut buf = Vec::new();

        assert!(enc.encode_into(
            &StreamChunk::Content {
                index: 0,
                delta: "Hello".into()
            },
            &mut buf
        ));
        let frame: Value = serde_json::from_slice(&buf).unwrap();
        assert_eq!(frame["object"], "chat.completion.chunk");
        assert_eq!(frame["choices"][0]["delta"]["content"], "Hello");

        assert!(enc.encode_into(
            &StreamChunk::Finish {
                index: 0,
                reason: FinishReason::Stop
            },
            &mut buf
        ));
        let fin: Value = serde_json::from_slice(&buf).unwrap();
        assert_eq!(fin["choices"][0]["finish_reason"], "stop");
    }

    #[test]
    fn encode_tool_call_frame_shape() {
        use llmleaf_model::ToolCallDelta;
        let enc = ChunkEncoder::new("resp-1", "gpt-4o", 1000);
        let mut buf = Vec::new();
        assert!(enc.encode_into(
            &StreamChunk::ToolCall {
                index: 0,
                call: ToolCallDelta {
                    index: 0,
                    id: Some("call_1".into()),
                    name: Some("get_weather".into()),
                    arguments: Some("{\"city\":\"NYC\"}".into()),
                },
            },
            &mut buf,
        ));
        let frame: Value = serde_json::from_slice(&buf).unwrap();
        let tc = &frame["choices"][0]["delta"]["tool_calls"][0];
        assert_eq!(tc["type"], "function");
        assert_eq!(tc["id"], "call_1");
        assert_eq!(tc["function"]["name"], "get_weather");
        assert_eq!(tc["function"]["arguments"], "{\"city\":\"NYC\"}");

        // The Usage chunk carries an empty choices array and the usage object.
        assert!(enc.encode_into(
            &StreamChunk::Usage(Usage {
                prompt_tokens: 1,
                completion_tokens: 2,
                total_tokens: 3,
                cost_usd: Some(0.5),
                cache_read_tokens: 0,
                cache_creation_tokens: 0,
            }),
            &mut buf,
        ));
        let u: Value = serde_json::from_slice(&buf).unwrap();
        assert_eq!(u["choices"].as_array().unwrap().len(), 0);
        assert_eq!(u["usage"]["total_tokens"], 3);
        assert_eq!(u["usage"]["cost_usd"], 0.5);
    }

    #[test]
    fn collected_response_shape() {
        use llmleaf_model::{Choice, Usage};
        let resp = ChatResponse {
            id: "r1".into(),
            model: "gpt-4o".into(),
            choices: vec![Choice {
                index: 0,
                text: "hi there".into(),
                thinking: vec![],
                tool_calls: vec![],
                finish_reason: Some(FinishReason::Stop),
            }],
            usage: Usage {
                prompt_tokens: 3,
                completion_tokens: 2,
                total_tokens: 5,
                cost_usd: Some(0.01),
                cache_read_tokens: 0,
                cache_creation_tokens: 0,
            },
        };
        // `to_value` here is a test-only convenience for the readable `v[..]` assertions — the hot
        // path serialises the view straight to bytes, never building a `Value`.
        let view = response_to_openai(&resp, 123);
        let v = serde_json::to_value(&view).unwrap();
        assert_eq!(v["object"], "chat.completion");
        assert_eq!(v["choices"][0]["message"]["content"], "hi there");
        assert_eq!(v["choices"][0]["finish_reason"], "stop");
        assert_eq!(v["usage"]["total_tokens"], 5);
        assert_eq!(v["usage"]["cost_usd"], 0.01);
    }

    /// The equivalence guarantee the streaming encoder added, ported to the collected mapping: the
    /// borrowed view must serialise to the *exact* bytes the old `Value`-building path produced. Covers
    /// every byte-identity branch: a choice with multiple tool calls + `finish_reason=Some`, a second
    /// choice with empty tool calls + `finish_reason=None` (emits `null`, omits `tool_calls`), and
    /// `cost_usd=Some`.
    #[test]
    fn collected_view_bytes_match_legacy() {
        use llmleaf_model::{Choice, ToolCall, Usage};
        let resp = ChatResponse {
            id: "r1".into(),
            model: "gpt-4o".into(),
            choices: vec![
                Choice {
                    index: 0,
                    text: "answer".into(),
                    thinking: vec![],
                    tool_calls: vec![
                        ToolCall {
                            id: "call_a".into(),
                            name: "get_weather".into(),
                            arguments: "{\"city\":\"NYC\"}".into(),
                        },
                        ToolCall {
                            id: "call_b".into(),
                            name: "get_time".into(),
                            arguments: "{}".into(),
                        },
                    ],
                    finish_reason: Some(FinishReason::ToolCalls),
                },
                Choice {
                    index: 1,
                    text: "more".into(),
                    thinking: vec![],
                    tool_calls: vec![],
                    finish_reason: None,
                },
            ],
            usage: Usage {
                prompt_tokens: 7,
                completion_tokens: 11,
                total_tokens: 18,
                cost_usd: Some(0.0042),
                cache_read_tokens: 0,
                cache_creation_tokens: 0,
            },
        };

        // The pre-refactor `Value`-building path, pasted here verbatim as the legacy oracle.
        let legacy = {
            let choices: Vec<Value> = resp
                .choices
                .iter()
                .map(|c| {
                    let mut message = Map::new();
                    message.insert("role".into(), Value::from("assistant"));
                    message.insert("content".into(), Value::from(c.text.clone()));
                    if !c.tool_calls.is_empty() {
                        let calls: Vec<Value> = c
                            .tool_calls
                            .iter()
                            .map(|t| {
                                let mut function = Map::new();
                                function.insert("name".into(), Value::from(t.name.clone()));
                                function
                                    .insert("arguments".into(), Value::from(t.arguments.clone()));
                                let mut call = Map::new();
                                call.insert("id".into(), Value::from(t.id.clone()));
                                call.insert("type".into(), Value::from("function"));
                                call.insert("function".into(), Value::Object(function));
                                Value::Object(call)
                            })
                            .collect();
                        message.insert("tool_calls".into(), Value::Array(calls));
                    }
                    let mut choice = Map::new();
                    choice.insert("index".into(), Value::from(c.index));
                    choice.insert("message".into(), Value::Object(message));
                    choice.insert(
                        "finish_reason".into(),
                        match c.finish_reason {
                            Some(r) => Value::from(finish_str(r)),
                            None => Value::Null,
                        },
                    );
                    Value::Object(choice)
                })
                .collect();
            let mut usage = json!({
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "total_tokens": resp.usage.total_tokens,
            });
            if let Some(cost) = resp.usage.cost_usd {
                usage
                    .as_object_mut()
                    .unwrap()
                    .insert("cost_usd".into(), json!(cost));
            }
            let mut root = Map::new();
            root.insert("id".into(), Value::from(resp.id.clone()));
            root.insert("object".into(), Value::from("chat.completion"));
            root.insert("created".into(), Value::from(123u64));
            root.insert("model".into(), Value::from(resp.model.clone()));
            root.insert("choices".into(), Value::Array(choices));
            root.insert("usage".into(), usage);
            Value::Object(root)
        };

        assert_eq!(
            serde_json::to_vec(&response_to_openai(&resp, 123)).unwrap(),
            serde_json::to_vec(&legacy).unwrap(),
        );
    }
}
