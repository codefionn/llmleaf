//! The OpenAI **Responses** API (`POST /v1/responses`) *wire* mapping — the canonical ⇄ OpenAI-Responses
//! translation, sibling of [`crate::openai_wire`] (which speaks the older `/chat/completions` dialect).
//!
//! OpenAI now treats the Responses API as its primary chat surface: a flat `input[]` of typed items
//! instead of `messages[]`, reasoning as first-class `reasoning` items (replayable across turns when the
//! caller opts out of server-side state), and a typed SSE event stream instead of `chat.completion.chunk`
//! frames. This is a second edge mapping onto the *same* canonical [`ChatRequest`]/[`StreamChunk`] types —
//! no dialect gets a shortcut through the core (principle 3); the Responses-specific knowledge lives here
//! at the edge, exactly like the chat-completions mapping (decision filter: quirks at the edge).
//!
//! llmleaf is a *stateless* proxy (principles 5, 9): it never lets the upstream store request payloads on
//! its behalf, and it makes reasoning replayable by carrying the encrypted reasoning back out and in — so
//! multi-turn reasoning round-trips through a stateless client without any node-local conversation state.
//!
//! Several upstreams speak this wire, differing only at the edges ([`ResponsesFlavor`]): stock OpenAI —
//! a flavor xAI (whose `POST /responses` is its documented-preferred chat surface) and Azure OpenAI (on
//! its resource-scoped v1 surface) speak verbatim, statelessness defaults included — OpenRouter's beta
//! `POST /responses` (an OpenAI-Responses-compatible surface over its multi-provider
//! router), and Groq's beta `POST /responses`. OpenRouter's dialect additionally carries *open*
//! reasoning — a reasoning item with plaintext `content` and a per-item `signature` (an Anthropic model
//! behind the router needs both replayed) — and reports the upstream cost on `usage.cost`. Groq's
//! dialect carries the same open reasoning but unsigned and never encrypted: it documents `include` (and
//! with it `reasoning.encrypted_content`) as unsupported, so the `include` statelessness default is
//! omitted there. Parsing all of those is unconditional (stock OpenAI simply never sends them);
//! *emitting* replayed reasoning is flavor-gated, because each upstream rejects request fields it does
//! not model.

use futures::StreamExt;
use llmleaf_model::{
    ChatRequest, ContentPart, FinishReason, Message, ModelError, ResponseStream, Role, StreamChunk,
    ToolCallDelta, ToolChoice, Usage,
};
use serde_json::{json, Map, Value};
use std::collections::HashMap;

/// Which Responses-speaking upstream the request body is built for. The wire is shared; only the
/// reasoning-replay capability differs (see the module docs). This is a *brand* property, fixed in
/// [`crate::compat::Brand`] — never sniffed per request.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ResponsesFlavor {
    /// Stock OpenAI: open reasoning has no request representation (OpenAI keeps reasoning server-side or
    /// encrypted, and rejects unknown item fields), so replay carries only `encrypted_content`. xAI and
    /// Azure OpenAI speak this flavor verbatim (both document `store` and
    /// `include: ["reasoning.encrypted_content"]`, so the statelessness defaults apply unchanged).
    #[default]
    OpenAi,
    /// OpenRouter's beta `POST /responses`: reasoning items carry plaintext `content` and a per-item
    /// `signature`, and both replay — that is what ports reasoning across the router's providers (an
    /// Anthropic model requires the signed thinking block back verbatim).
    OpenRouter,
    /// Groq's beta `POST /responses` (`https://api.groq.com/openai/v1/responses`): reasoning is open
    /// like OpenRouter's — plaintext `content` replays — but never signed (Groq models no `signature`)
    /// and never encrypted (`include`, and with it `reasoning.encrypted_content`, is
    /// documented-unsupported), so the `include` statelessness default is omitted and encrypted replay
    /// items have no representation.
    Groq,
}

impl ResponsesFlavor {
    /// Whether the dialect models *open* (plaintext) reasoning on request `reasoning` items — a prior
    /// turn's thinking replayed as `content` of `reasoning_text` parts. Stock OpenAI rejects it
    /// (reasoning stays server-side or encrypted); OpenRouter and Groq accept back exactly what they
    /// emit.
    fn replays_open_reasoning(self) -> bool {
        matches!(self, Self::OpenRouter | Self::Groq)
    }

    /// Whether the dialect models the per-item reasoning `signature`. OpenRouter alone — it is what
    /// ports signed reasoning across the router's providers; OpenAI and Groq reject the unknown field.
    fn models_signature(self) -> bool {
        self == Self::OpenRouter
    }

    /// Whether the dialect models encrypted reasoning — the `include: ["reasoning.encrypted_content"]`
    /// statelessness default and `encrypted_content` replay items. Groq documents `include` as
    /// unsupported and never emits encrypted reasoning, so both are omitted there.
    fn models_encrypted_reasoning(self) -> bool {
        matches!(self, Self::OpenAi | Self::OpenRouter)
    }
}

/// Canonical request → OpenAI **Responses** wire JSON. Dialect-specific `extra` fields ride back out
/// verbatim (principle 7); the two statelessness defaults (`store`, `include`) are the exception — a
/// consumer may override them (see below).
///
/// `stream` selects the upstream transport exactly as [`crate::openai_wire::request_to_openai`] does:
/// `false` collects the whole response in one shot, `true` asks upstream to stream its typed SSE. Unlike
/// the chat wire there is no `stream_options` — the Responses stream always carries usage on its terminal
/// `response.completed` event.
pub fn request_to_openai_responses(
    req: &ChatRequest,
    stream: bool,
    flavor: ResponsesFlavor,
) -> Value {
    let mut obj = Map::new();
    obj.insert("model".into(), json!(req.model));
    // The chat wire's `max_tokens`/`max_completion_tokens` is `max_output_tokens` here.
    if let Some(v) = req.max_tokens {
        obj.insert("max_output_tokens".into(), json!(v));
    }
    if let Some(v) = req.temperature {
        obj.insert("temperature".into(), json!(v));
    }
    if let Some(v) = req.top_p {
        obj.insert("top_p".into(), json!(v));
    }
    // Canonical `stop` has no Responses representation. This function is only reached once the caller has
    // checked `needs_chat_completions` (which lists `stop` among the chat-only features), so a request
    // that gets here already has an empty `stop` — there is nothing to drop and nothing to map.

    // `messages` → the flat `input[]` of typed items, in order. One canonical message can expand into
    // several Responses items (an assistant turn with replayed reasoning + text + tool calls).
    let mut input: Vec<Value> = Vec::new();
    for msg in &req.messages {
        match msg.role {
            Role::System => input.push(json!({ "role": "system", "content": input_content(msg) })),
            Role::User => input.push(json!({ "role": "user", "content": input_content(msg) })),
            Role::Assistant => push_assistant_items(msg, &mut input, flavor),
            // A tool result is a `function_call_output` item keyed by the call it answers.
            Role::Tool => input.push(json!({
                "type": "function_call_output",
                "call_id": msg.tool_call_id.as_deref().unwrap_or_default(),
                "output": msg.text_content(),
            })),
        }
    }
    obj.insert("input".into(), Value::Array(input));

    if !req.tools.is_empty() {
        let tools: Vec<Value> = req
            .tools
            .iter()
            .map(|t| {
                let mut fun = Map::new();
                // Flat in the Responses dialect: `type`/`name`/`parameters` at the top level, not nested
                // under a `function` object like the chat wire.
                fun.insert("type".into(), json!("function"));
                fun.insert("name".into(), json!(t.name));
                if let Some(desc) = &t.description {
                    fun.insert("description".into(), json!(desc));
                }
                fun.insert("parameters".into(), t.parameters.clone());
                // `strict` defaults to TRUE in the Responses dialect (the chat wire defaults it false). A
                // strict-mode schema that isn't strict-clean would start being rejected on the switch to
                // Responses, so send `strict: false` explicitly — schemas keep behaving exactly as they
                // did over chat completions (principle 7: the switch is transparent, not a behavior change).
                fun.insert("strict".into(), json!(false));
                Value::Object(fun)
            })
            .collect();
        obj.insert("tools".into(), json!(tools));
    }
    if let Some(tc) = &req.tool_choice {
        obj.insert("tool_choice".into(), tool_choice_to_responses(tc));
    }
    if let Some(t) = req.thinking {
        // Same lossy ladder as the chat wire's `reasoning_effort`, reused rather than duplicated.
        obj.insert(
            "reasoning".into(),
            json!({ "effort": crate::openai_wire::thinking_effort(t) }),
        );
    }
    obj.insert("stream".into(), json!(stream));

    // Pass through anything the consumer sent that we don't model (principle 7). Canonical fields were
    // inserted first, so `or_insert_with` lets a modeled key win over an `extra` collision — same rule
    // as `request_to_openai`.
    for (k, v) in &req.extra {
        obj.entry(k.clone()).or_insert_with(|| v.clone());
    }

    // Statelessness defaults — the ONE place the consumer must be able to override a value we set, so
    // they go in AFTER the `extra` loop with override-aware logic (consumer wins), the reverse of every
    // key above. `store` defaults TRUE upstream, which would have OpenAI silently retain the payload —
    // llmleaf is a stateless proxy and must not let that happen (principles 5, 9), so default it false.
    // With `store: false` there is no server-side reasoning to reference next turn, so ask for the
    // encrypted reasoning inline (`include`): that entry is exactly what makes reasoning replayable
    // across turns for a stateless client (it comes back as a `reasoning` item we echo on the next turn).
    // Flavor-gated: Groq documents `include` as unsupported (its reasoning comes back open, with no
    // encrypted form to ask for), so the default is omitted there — a consumer's explicit `include`
    // still rides through above (their field, principle 7).
    obj.entry("store").or_insert(json!(false));
    if flavor.models_encrypted_reasoning() && !obj.contains_key("include") {
        obj.insert("include".into(), json!(["reasoning.encrypted_content"]));
    }

    Value::Object(obj)
}

/// A system/user message's `content`: a bare string for the single-text common case (the cheapest shape,
/// mirroring [`crate::openai_wire`]'s `message_to_openai`), else an array of Responses *input* parts.
fn input_content(msg: &Message) -> Value {
    if msg.content.len() == 1 {
        if let ContentPart::Text { text } = &msg.content[0] {
            return json!(text);
        }
    }
    Value::Array(msg.content.iter().filter_map(input_part).collect())
}

/// One system/user content part → a Responses input part. Reasoning blocks never appear on a
/// system/user turn (they are assistant-only), so they map to nothing here.
fn input_part(p: &ContentPart) -> Option<Value> {
    match p {
        ContentPart::Text { text } => Some(json!({ "type": "input_text", "text": text })),
        ContentPart::ImageUrl { url, detail } => {
            let mut o = Map::new();
            o.insert("type".into(), json!("input_image"));
            // NOTE: `image_url` is a plain STRING in the Responses dialect, not the chat wire's nested
            // `{ "url": … }` object.
            o.insert("image_url".into(), json!(url));
            if let Some(d) = detail {
                o.insert("detail".into(), json!(d));
            }
            Some(Value::Object(o))
        }
        ContentPart::Thinking { .. } | ContentPart::RedactedThinking { .. } => None,
    }
}

/// Expand one assistant message into its ordered Responses items: replayed reasoning first, then the
/// visible message, then tool calls — the order the upstream itself emits and the order replay restores.
fn push_assistant_items(msg: &Message, out: &mut Vec<Value>, flavor: ResponsesFlavor) {
    for part in &msg.content {
        match part {
            // (a) Each redacted (encrypted) reasoning block replays as its own `reasoning` item — the
            // exact stateless-replay shape our own parse side reads back out (see
            // `openai_responses_to_chunks`), so a client echoing the prior turn round-trips reasoning
            // across turns. Groq models no encrypted reasoning (and never emits one to round-trip), so
            // the block has no request representation there and is dropped like open thinking is on
            // stock OpenAI.
            ContentPart::RedactedThinking { data } if flavor.models_encrypted_reasoning() => {
                out.push(json!({ "type": "reasoning", "summary": [], "encrypted_content": data }));
            }
            // (b) Open `Thinking` text/signature is flavor-gated. Stock OpenAI has NO request
            // representation for it (reasoning stays server-side or encrypted, and a foreign signature
            // means the reasoning came from another vendor and does not port) — drop it, mirroring
            // `content_parts` in `openai_wire.rs`. OpenRouter's beta dialect DOES model it — a
            // `reasoning` item with plaintext `content` and the per-item `signature` — and an Anthropic
            // model behind the router requires exactly that replayed, so emit it there. Groq's dialect
            // replays the same plaintext `content` but models no `signature` (its reasoning is never
            // signed) — an unknown field would be rejected, so it is withheld there.
            ContentPart::Thinking {
                thinking,
                signature,
            } if flavor.replays_open_reasoning() => {
                let mut item = Map::new();
                item.insert("type".into(), json!("reasoning"));
                item.insert("summary".into(), json!([]));
                item.insert(
                    "content".into(),
                    json!([{ "type": "reasoning_text", "text": thinking }]),
                );
                if flavor.models_signature() {
                    if let Some(sig) = signature {
                        item.insert("signature".into(), json!(sig));
                    }
                }
                out.push(Value::Object(item));
            }
            _ => {}
        }
    }
    // (c) The visible answer, if any, as a `message` item of `output_text` parts.
    let text = msg.text_content();
    if !text.is_empty() {
        out.push(json!({
            "type": "message",
            "role": "assistant",
            "content": [{ "type": "output_text", "text": text }],
        }));
    }
    // (d) Each tool call as a `function_call` item.
    for c in &msg.tool_calls {
        out.push(json!({
            "type": "function_call",
            "call_id": c.id,
            "name": c.name,
            "arguments": c.arguments,
        }));
    }
}

fn tool_choice_to_responses(tc: &ToolChoice) -> Value {
    match tc {
        ToolChoice::Auto => json!("auto"),
        ToolChoice::None => json!("none"),
        ToolChoice::Required => json!("required"),
        // Flat — no nested `function` object, unlike the chat wire's `tool_choice_to_openai`.
        ToolChoice::Named(name) => json!({ "type": "function", "name": name }),
    }
}

/// Chat-completions-only vocabulary the Responses endpoint rejects. Kept as a documented const so the
/// downgrade set is auditable in one place.
const CHAT_ONLY_EXTRA_KEYS: &[&str] = &[
    "response_format",
    "logit_bias",
    "frequency_penalty",
    "presence_penalty",
    "seed",
    "logprobs",
    "n",
    "modalities",
    "audio",
    "prediction",
    "functions",
    "function_call",
    "web_search_options",
    "stream_options",
    "stop",
    "max_completion_tokens",
    "verbosity",
    // The flat chat spelling of the reasoning knob (the Responses dialect nests it as
    // `reasoning.effort`), plus Groq's chat-only reasoning/search vocabulary.
    "reasoning_effort",
    "reasoning_format",
    "search_settings",
];

/// Whether this request uses chat-completions-only vocabulary the Responses endpoint rejects — a
/// non-empty canonical `stop`, or any `extra` key in [`CHAT_ONLY_EXTRA_KEYS`].
///
/// This is the per-request "use another endpoint" downgrade: a Responses-mode provider that sees such a
/// request serves it over `/chat/completions` instead, transparently, rather than dropping the field
/// (principle 7: no silent dropping of consumer fields — the request is served where its fields are legal).
pub fn needs_chat_completions(req: &ChatRequest) -> bool {
    !req.stop.is_empty()
        || req
            .extra
            .keys()
            .any(|k| CHAT_ONLY_EXTRA_KEYS.contains(&k.as_str()))
}

/// A Responses `usage` object → canonical [`Usage`]. `input_tokens`/`output_tokens` are the Responses
/// dialect's names for prompt/completion; cache hits live under `input_tokens_details.cached_tokens`;
/// `cost` is OpenRouter's upstream-reported USD price (absent on stock OpenAI). All relayed, never
/// computed (principle 5); there is no client-driven cache write, so creation is 0.
fn responses_usage(usage: &Value) -> Usage {
    Usage {
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
        cost_usd: usage.get("cost").and_then(Value::as_f64),
        cache_read_tokens: usage
            .get("input_tokens_details")
            .and_then(|d| d.get("cached_tokens"))
            .and_then(Value::as_u64)
            .unwrap_or(0),
        cache_creation_tokens: 0,
    }
}

/// The terminal finish reason from a Responses `response` object. `completed` is `ToolCalls` when the
/// output carried any `function_call` item (mirroring the chat wire's `tool_calls` finish), else `Stop`;
/// `incomplete` refines by `incomplete_details.reason`; `failed` is `Error`.
fn responses_finish_reason(response: &Value, saw_function_call: bool) -> FinishReason {
    match response.get("status").and_then(Value::as_str) {
        Some("completed") => {
            if saw_function_call {
                FinishReason::ToolCalls
            } else {
                FinishReason::Stop
            }
        }
        Some("incomplete") => match response
            .get("incomplete_details")
            .and_then(|d| d.get("reason"))
            .and_then(Value::as_str)
        {
            Some("max_output_tokens") => FinishReason::Length,
            Some("content_filter") => FinishReason::ContentFilter,
            _ => FinishReason::Stop,
        },
        Some("failed") => FinishReason::Error,
        _ => FinishReason::Stop,
    }
}

/// The collected (non-streaming) Responses object → canonical chunks — the mirror of
/// [`crate::openai_wire::openai_to_chunks`]. The Responses dialect has no `n`/choices, so the choice
/// index is always 0. Emission order: [`StreamChunk::Start`], the `output[]` items in order, the usage
/// report, then the terminal [`StreamChunk::Finish`].
pub fn openai_responses_to_chunks(value: Value, fallback_model: &str) -> Vec<StreamChunk> {
    let id = value
        .get("id")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let model = value
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or(fallback_model)
        .to_string();
    let mut chunks = vec![StreamChunk::Start { id, model }];

    // A running counter across `function_call` items — the Responses object reports them in order but
    // carries no per-call index, so it is synthesized here (the chat wire does the same by enumeration).
    let mut call_index: u32 = 0;
    let mut saw_function_call = false;

    if let Some(output) = value.get("output").and_then(Value::as_array) {
        for item in output {
            match item.get("type").and_then(Value::as_str) {
                Some("reasoning") => {
                    // Reasoning leads the answer it justifies. `summary[].text` (`summary_text`) and
                    // `content[].text` (`reasoning_text`) are open reasoning; a per-item `signature`
                    // (OpenRouter routing an Anthropic model) signs that open text and must survive for
                    // replay; `encrypted_content` is the redacted block we replay verbatim next turn.
                    for key in ["summary", "content"] {
                        if let Some(arr) = item.get(key).and_then(Value::as_array) {
                            for entry in arr {
                                if let Some(t) = entry.get("text").and_then(Value::as_str) {
                                    if !t.is_empty() {
                                        chunks.push(StreamChunk::Thinking {
                                            index: 0,
                                            delta: t.to_string(),
                                        });
                                    }
                                }
                            }
                        }
                    }
                    if let Some(sig) = item.get("signature").and_then(Value::as_str) {
                        if !sig.is_empty() {
                            chunks.push(StreamChunk::ThinkingSignature {
                                index: 0,
                                signature: sig.to_string(),
                            });
                        }
                    }
                    if let Some(data) = item.get("encrypted_content").and_then(Value::as_str) {
                        if !data.is_empty() {
                            chunks.push(StreamChunk::RedactedThinking {
                                index: 0,
                                data: data.to_string(),
                            });
                        }
                    }
                }
                Some("message") => {
                    if let Some(content) = item.get("content").and_then(Value::as_array) {
                        for part in content {
                            // Only `output_text` is visible answer; `refusal` parts are noted but not
                            // emitted as content (they carry no assistant answer).
                            if part.get("type").and_then(Value::as_str) == Some("output_text") {
                                if let Some(t) = part.get("text").and_then(Value::as_str) {
                                    if !t.is_empty() {
                                        chunks.push(StreamChunk::Content {
                                            index: 0,
                                            delta: t.to_string(),
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
                Some("function_call") => {
                    saw_function_call = true;
                    chunks.push(StreamChunk::ToolCall {
                        index: 0,
                        call: ToolCallDelta {
                            index: call_index,
                            id: item
                                .get("call_id")
                                .and_then(Value::as_str)
                                .map(str::to_owned),
                            name: item.get("name").and_then(Value::as_str).map(str::to_owned),
                            arguments: item
                                .get("arguments")
                                .and_then(Value::as_str)
                                .map(str::to_owned),
                        },
                    });
                    call_index += 1;
                }
                // Other item types (`web_search_call`, etc.) carry no canonical output — ignore them.
                _ => {}
            }
        }
    }

    if let Some(usage) = value.get("usage") {
        chunks.push(StreamChunk::Usage(responses_usage(usage)));
    }
    chunks.push(StreamChunk::Finish {
        index: 0,
        reason: responses_finish_reason(&value, saw_function_call),
    });

    chunks
}

// ---------------------------------------------------------------------------------------------
// Incremental streaming (SSE): one Responses event JSON → canonical chunks
// ---------------------------------------------------------------------------------------------

/// State threaded through a Responses SSE stream. Unlike the chat wire — whose tool-call deltas each
/// carry their own index — the Responses stream identifies a tool call by its output-item id, so the map
/// from that id to a canonical call index is rebuilt here as `output_item.added` events arrive.
pub struct ResponsesSseState {
    /// Fallback model id stamped on [`StreamChunk::Start`] when the `response.created` event omits one.
    fallback_model: String,
    /// Guards the single opening [`StreamChunk::Start`].
    seen_start: bool,
    /// Output-item id (of a `function_call`) → its canonical tool-call index.
    call_indices: HashMap<String, u32>,
    /// Next canonical tool-call index to assign.
    next_call_index: u32,
    /// Whether any `function_call` output item was seen (drives the `completed` → `ToolCalls` finish).
    saw_function_call: bool,
    /// Set once the terminal event (`completed`/`incomplete`/`failed`) has emitted its [`StreamChunk::Finish`].
    finished: bool,
}

impl ResponsesSseState {
    /// Fresh state carrying the fallback model for the opening [`StreamChunk::Start`].
    pub fn new(fallback_model: String) -> Self {
        Self {
            fallback_model,
            seen_start: false,
            call_indices: HashMap::new(),
            next_call_index: 0,
            saw_function_call: false,
            finished: false,
        }
    }

    /// Whether the terminal event has been processed — the stream loop stops once this is set (the
    /// Responses SSE has no `[DONE]` sentinel).
    pub fn is_finished(&self) -> bool {
        self.finished
    }
}

/// One Responses SSE event JSON (the object after `data: `) → canonical chunks. The streaming mirror of
/// [`openai_responses_to_chunks`], dispatched on the event's `"type"` discriminator. Every `data:`
/// payload is a self-describing JSON object, so the SSE `event:` lines can be ignored. A mid-stream
/// `error` event is NOT handled here (this returns only chunks) — [`openai_responses_sse_to_stream`]
/// intercepts it and surfaces an upstream failure, mirroring how [`crate::ollama`] handles its inline
/// error object.
pub fn openai_responses_event_to_canonical(
    value: &Value,
    st: &mut ResponsesSseState,
) -> Vec<StreamChunk> {
    let mut out = Vec::new();
    match value.get("type").and_then(Value::as_str).unwrap_or("") {
        // A second `response.created` (guard false) is ignored — the single `Start` is already out.
        "response.created" if !st.seen_start => {
            let response = value.get("response");
            let id = response
                .and_then(|r| r.get("id"))
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            let model = response
                .and_then(|r| r.get("model"))
                .and_then(Value::as_str)
                .unwrap_or(&st.fallback_model)
                .to_string();
            out.push(StreamChunk::Start { id, model });
            st.seen_start = true;
        }
        "response.output_item.added" => {
            let item = value.get("item");
            if item.and_then(|i| i.get("type")).and_then(Value::as_str) == Some("function_call") {
                st.saw_function_call = true;
                let idx = st.next_call_index;
                st.next_call_index += 1;
                // Register the output-item id so the later `function_call_arguments.delta` events resolve
                // to this same canonical call index.
                if let Some(item_id) = item.and_then(|i| i.get("id")).and_then(Value::as_str) {
                    st.call_indices.insert(item_id.to_string(), idx);
                }
                out.push(StreamChunk::ToolCall {
                    index: 0,
                    call: ToolCallDelta {
                        index: idx,
                        id: item
                            .and_then(|i| i.get("call_id"))
                            .and_then(Value::as_str)
                            .map(str::to_owned),
                        name: item
                            .and_then(|i| i.get("name"))
                            .and_then(Value::as_str)
                            .map(str::to_owned),
                        arguments: None,
                    },
                });
            }
        }
        "response.function_call_arguments.delta" => {
            if let Some(delta) = value.get("delta").and_then(Value::as_str) {
                let index = value
                    .get("item_id")
                    .and_then(Value::as_str)
                    .and_then(|id| st.call_indices.get(id).copied())
                    .unwrap_or(0);
                out.push(StreamChunk::ToolCall {
                    index: 0,
                    call: ToolCallDelta {
                        index,
                        id: None,
                        name: None,
                        arguments: Some(delta.to_string()),
                    },
                });
            }
        }
        "response.output_text.delta" => {
            if let Some(delta) = value.get("delta").and_then(Value::as_str) {
                out.push(StreamChunk::Content {
                    index: 0,
                    delta: delta.to_string(),
                });
            }
        }
        "response.reasoning_text.delta" | "response.reasoning_summary_text.delta" => {
            if let Some(delta) = value.get("delta").and_then(Value::as_str) {
                out.push(StreamChunk::Thinking {
                    index: 0,
                    delta: delta.to_string(),
                });
            }
        }
        "response.output_item.done" => {
            // A completed `reasoning` item carries what only arrives whole: the per-item `signature`
            // signing the streamed reasoning text (OpenRouter routing an Anthropic model — no delta
            // event exists for it), and the redacted block to replay next turn.
            let item = value.get("item");
            if item.and_then(|i| i.get("type")).and_then(Value::as_str) == Some("reasoning") {
                if let Some(sig) = item
                    .and_then(|i| i.get("signature"))
                    .and_then(Value::as_str)
                {
                    if !sig.is_empty() {
                        out.push(StreamChunk::ThinkingSignature {
                            index: 0,
                            signature: sig.to_string(),
                        });
                    }
                }
                if let Some(data) = item
                    .and_then(|i| i.get("encrypted_content"))
                    .and_then(Value::as_str)
                {
                    if !data.is_empty() {
                        out.push(StreamChunk::RedactedThinking {
                            index: 0,
                            data: data.to_string(),
                        });
                    }
                }
            }
        }
        "response.completed" | "response.incomplete" | "response.failed" => {
            if let Some(response) = value.get("response") {
                if let Some(usage) = response.get("usage") {
                    out.push(StreamChunk::Usage(responses_usage(usage)));
                }
                out.push(StreamChunk::Finish {
                    index: 0,
                    reason: responses_finish_reason(response, st.saw_function_call),
                });
            }
            st.finished = true;
        }
        // Everything else is a duplicate or structural event with no canonical content:
        // `response.in_progress`, `content_part.*`, the `*.done` text duplicates, `reasoning_summary_part.*`.
        _ => {}
    }
    out
}

/// Stream an OpenAI **Responses** SSE response into canonical chunks — the Responses mirror of
/// [`crate::openai_wire::openai_sse_to_stream`]. It owns the rolling SSE buffer, rejoins each `data:`
/// frame via [`crate::openai_wire::sse_payloads`] (a frame split across TCP reads is stitched back
/// together), and dispatches every event JSON through [`openai_responses_event_to_canonical`] — tokens
/// flow as they arrive (principle 4). There is no `[DONE]` sentinel: the loop ends once the terminal
/// `response.completed`/`incomplete`/`failed` event arrives (or the transport ends). A mid-stream `error`
/// event surfaces as [`ModelError::Upstream`] (`status: 500`), the same taxonomy [`crate::ollama`] uses
/// for its inline stream error; a transport error rides through verbatim; an undecodable frame becomes
/// [`ModelError::Mapping`].
pub fn openai_responses_sse_to_stream(
    body: crate::transport::BytesStream,
    model: String,
) -> ResponseStream {
    Box::pin(async_stream::stream! {
        let mut bytes = body;
        let mut buf: Vec<u8> = Vec::with_capacity(1024);
        let mut st = ResponsesSseState::new(model);
        while let Some(item) = bytes.next().await {
            let chunk = match item {
                Ok(b) => b,
                Err(e) => {
                    // The transport already mapped this to the canonical taxonomy.
                    yield Err(e);
                    return;
                }
            };
            for payload in crate::openai_wire::sse_payloads(&mut buf, &chunk) {
                // The Responses dialect ends on its terminal event, not a `[DONE]` sentinel — but some
                // OpenAI-wire intermediaries append one anyway. Skip it rather than fail the decode.
                if payload == "[DONE]" {
                    continue;
                }
                match serde_json::from_str::<Value>(&payload) {
                    Ok(v) => {
                        // A mid-stream `error` event is an upstream failure, not a parse failure —
                        // surface it the way `crate::ollama` surfaces its inline stream error.
                        if v.get("type").and_then(Value::as_str) == Some("error") {
                            let message = v
                                .get("message")
                                .and_then(Value::as_str)
                                .map(str::to_string)
                                .unwrap_or_else(|| v.to_string());
                            yield Err(ModelError::Upstream { status: 500, message });
                            return;
                        }
                        for c in openai_responses_event_to_canonical(&v, &mut st) {
                            yield Ok(c);
                        }
                    }
                    Err(e) => {
                        yield Err(ModelError::Mapping(e.to_string()));
                        return;
                    }
                }
            }
            if st.is_finished() {
                break;
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use llmleaf_model::{Message, Thinking, ToolCall, ToolDef};

    fn user_req(text: &str) -> ChatRequest {
        ChatRequest {
            model: "gpt-5".into(),
            messages: vec![Message::text(Role::User, text)],
            max_tokens: Some(64),
            temperature: None,
            top_p: None,
            stop: vec![],
            stream: false,
            tools: vec![],
            tool_choice: None,
            thinking: None,
            extra: Default::default(),
        }
    }

    #[test]
    fn request_maps_core_fields_and_stateless_defaults() {
        let req = user_req("hi");
        let wire = request_to_openai_responses(&req, false, ResponsesFlavor::OpenAi);
        assert_eq!(wire["model"], "gpt-5");
        // `max_tokens` becomes `max_output_tokens`; no chat-wire `max_tokens` leaks through.
        assert_eq!(wire["max_output_tokens"], 64);
        assert!(wire.get("max_tokens").is_none());
        // Single text collapses to a bare string content, mirroring the chat wire.
        assert_eq!(wire["input"][0]["role"], "user");
        assert_eq!(wire["input"][0]["content"], "hi");
        assert_eq!(wire["stream"], false);
        // No `stream_options` on the Responses wire.
        assert!(wire.get("stream_options").is_none());
        // Statelessness defaults: never let the upstream store the payload; make reasoning replayable.
        assert_eq!(wire["store"], false);
        assert_eq!(wire["include"], json!(["reasoning.encrypted_content"]));
    }

    #[test]
    fn request_maps_system_user_image_and_tools() {
        let mut req = user_req("describe");
        req.messages = vec![
            Message::text(Role::System, "be terse"),
            Message {
                role: Role::User,
                content: vec![
                    ContentPart::Text {
                        text: "what is this".into(),
                    },
                    ContentPart::ImageUrl {
                        url: "https://img/x.png".into(),
                        detail: Some("high".into()),
                    },
                ],
                tool_calls: vec![],
                tool_call_id: None,
                name: None,
            },
        ];
        req.tools = vec![ToolDef {
            name: "get_weather".into(),
            description: Some("look up weather".into()),
            parameters: json!({ "type": "object" }),
        }];
        req.tool_choice = Some(ToolChoice::Named("get_weather".into()));
        let wire = request_to_openai_responses(&req, false, ResponsesFlavor::OpenAi);

        // System is a bare-string content item.
        assert_eq!(
            wire["input"][0],
            json!({ "role": "system", "content": "be terse" })
        );
        // User multipart: input_text + input_image with a plain-STRING image_url and a detail.
        let user = &wire["input"][1];
        assert_eq!(user["role"], "user");
        assert_eq!(
            user["content"][0],
            json!({ "type": "input_text", "text": "what is this" })
        );
        assert_eq!(
            user["content"][1],
            json!({ "type": "input_image", "image_url": "https://img/x.png", "detail": "high" })
        );
        // Flat tool shape with an explicit `strict: false` (Responses defaults it true).
        assert_eq!(
            wire["tools"][0],
            json!({
                "type": "function",
                "name": "get_weather",
                "description": "look up weather",
                "parameters": { "type": "object" },
                "strict": false
            })
        );
        // Flat tool_choice: no nested `function` object.
        assert_eq!(
            wire["tool_choice"],
            json!({ "type": "function", "name": "get_weather" })
        );
    }

    #[test]
    fn request_replays_reasoning_and_tool_round_trip() {
        // A prior assistant turn: encrypted reasoning + visible text + a tool call, then the tool result.
        let mut req = user_req("continue");
        req.messages = vec![
            Message::text(Role::User, "weather?"),
            Message {
                role: Role::Assistant,
                content: vec![
                    ContentPart::RedactedThinking {
                        data: "ENC1".into(),
                    },
                    ContentPart::Text {
                        text: "let me check".into(),
                    },
                ],
                tool_calls: vec![ToolCall {
                    id: "call_1".into(),
                    name: "get_weather".into(),
                    arguments: "{\"city\":\"Paris\"}".into(),
                }],
                tool_call_id: None,
                name: None,
            },
            Message {
                role: Role::Tool,
                content: vec![ContentPart::Text {
                    text: "sunny".into(),
                }],
                tool_calls: vec![],
                tool_call_id: Some("call_1".into()),
                name: None,
            },
        ];
        let wire = request_to_openai_responses(&req, false, ResponsesFlavor::OpenAi);
        let input = wire["input"].as_array().unwrap();
        // user, reasoning, message, function_call, function_call_output — in that order.
        assert_eq!(input[0]["role"], "user");
        assert_eq!(
            input[1],
            json!({ "type": "reasoning", "summary": [], "encrypted_content": "ENC1" })
        );
        assert_eq!(
            input[2],
            json!({
                "type": "message",
                "role": "assistant",
                "content": [{ "type": "output_text", "text": "let me check" }]
            })
        );
        assert_eq!(
            input[3],
            json!({
                "type": "function_call",
                "call_id": "call_1",
                "name": "get_weather",
                "arguments": "{\"city\":\"Paris\"}"
            })
        );
        assert_eq!(
            input[4],
            json!({ "type": "function_call_output", "call_id": "call_1", "output": "sunny" })
        );
    }

    /// An assistant turn with open thinking (text + signature), a redacted block, and visible text —
    /// the replay shape whose handling is the one flavor difference on this wire.
    fn thinking_replay_req() -> ChatRequest {
        let mut req = user_req("go");
        req.messages = vec![Message {
            role: Role::Assistant,
            content: vec![
                ContentPart::Thinking {
                    thinking: "internal".into(),
                    signature: Some("sig".into()),
                },
                ContentPart::RedactedThinking { data: "ENC".into() },
                ContentPart::Text {
                    text: "answer".into(),
                },
            ],
            tool_calls: vec![],
            tool_call_id: None,
            name: None,
        }];
        req
    }

    #[test]
    fn openai_flavor_drops_open_thinking_but_keeps_redacted() {
        // Stock OpenAI: open thinking (text/signature) has no request representation and is dropped;
        // the redacted block replays.
        let req = thinking_replay_req();
        let input = request_to_openai_responses(&req, false, ResponsesFlavor::OpenAi)["input"]
            .as_array()
            .unwrap()
            .clone();
        assert_eq!(input.len(), 2); // reasoning + message; the open thinking produced nothing
        assert_eq!(input[0]["type"], "reasoning");
        assert_eq!(input[0]["encrypted_content"], "ENC");
        assert_eq!(input[1]["type"], "message");
    }

    #[test]
    fn openrouter_flavor_replays_open_thinking_with_signature() {
        // OpenRouter's beta dialect models open reasoning: the thinking text replays as a `reasoning`
        // item with plaintext `content` and the per-item `signature` (an Anthropic model behind the
        // router requires the signed block back verbatim), ahead of the redacted block and the message.
        let req = thinking_replay_req();
        let input = request_to_openai_responses(&req, false, ResponsesFlavor::OpenRouter)["input"]
            .as_array()
            .unwrap()
            .clone();
        assert_eq!(input.len(), 3);
        assert_eq!(
            input[0],
            json!({
                "type": "reasoning",
                "summary": [],
                "content": [{ "type": "reasoning_text", "text": "internal" }],
                "signature": "sig",
            })
        );
        assert_eq!(input[1]["encrypted_content"], "ENC");
        assert_eq!(input[2]["type"], "message");

        // Without a signature the item still replays — just unsigned.
        let mut req = user_req("go");
        req.messages = vec![Message {
            role: Role::Assistant,
            content: vec![ContentPart::Thinking {
                thinking: "unsigned".into(),
                signature: None,
            }],
            tool_calls: vec![],
            tool_call_id: None,
            name: None,
        }];
        let input = request_to_openai_responses(&req, false, ResponsesFlavor::OpenRouter)["input"]
            .as_array()
            .unwrap()
            .clone();
        assert_eq!(input[0]["content"][0]["text"], "unsigned");
        assert!(input[0].get("signature").is_none());
    }

    #[test]
    fn groq_flavor_replays_open_thinking_unsigned_and_drops_encrypted() {
        // Groq's beta dialect: open reasoning replays as plaintext `content` like OpenRouter's, but
        // UNSIGNED (Groq models no `signature`; the foreign signature is withheld, not sent to be
        // rejected) — and the redacted block has no Groq representation (encrypted reasoning does not
        // exist there), so it is dropped.
        let req = thinking_replay_req();
        let wire = request_to_openai_responses(&req, false, ResponsesFlavor::Groq);
        let input = wire["input"].as_array().unwrap();
        assert_eq!(input.len(), 2); // reasoning + message; the redacted block produced nothing
        assert_eq!(
            input[0],
            json!({
                "type": "reasoning",
                "summary": [],
                "content": [{ "type": "reasoning_text", "text": "internal" }],
            })
        );
        assert_eq!(input[1]["type"], "message");
        // The `store: false` statelessness default still applies (Groq accepts false/null); the
        // `include` default does NOT — Groq documents `include` as unsupported.
        assert_eq!(wire["store"], false);
        assert!(wire.get("include").is_none());
    }

    #[test]
    fn groq_flavor_lets_an_explicit_consumer_include_ride_through() {
        // Only OUR `include` statelessness default is withheld for Groq; a consumer's explicit
        // `include` is their field and rides through verbatim (principle 7).
        let mut req = user_req("hi");
        req.extra =
            serde_json::from_value(json!({ "include": ["message.output_text.logprobs"] })).unwrap();
        let wire = request_to_openai_responses(&req, false, ResponsesFlavor::Groq);
        assert_eq!(wire["include"], json!(["message.output_text.logprobs"]));
    }

    #[test]
    fn request_thinking_maps_to_reasoning_effort() {
        let effort = |t| {
            let mut req = user_req("hi");
            req.thinking = Some(t);
            request_to_openai_responses(&req, false, ResponsesFlavor::OpenAi)["reasoning"]["effort"]
                .clone()
        };
        assert_eq!(effort(Thinking::Low), "low");
        assert_eq!(effort(Thinking::Med), "medium");
        // The upper rungs collapse onto `high`, same lossy ladder as the chat wire.
        assert_eq!(effort(Thinking::Highx), "high");
        assert!(
            request_to_openai_responses(&user_req("hi"), false, ResponsesFlavor::OpenAi)
                .get("reasoning")
                .is_none()
        );
    }

    #[test]
    fn request_extra_passes_through_and_consumer_overrides_stateless_defaults() {
        let mut req = user_req("hi");
        req.extra = serde_json::from_value(json!({
            "prompt_cache_key": "abc",
            "store": true,
            "include": ["reasoning.encrypted_content", "message.output_text.logprobs"]
        }))
        .unwrap();
        let wire = request_to_openai_responses(&req, true, ResponsesFlavor::OpenAi);
        // Unmodeled field rides through verbatim (principle 7).
        assert_eq!(wire["prompt_cache_key"], "abc");
        // The consumer's explicit `store`/`include` win over our stateless defaults.
        assert_eq!(wire["store"], true);
        assert_eq!(
            wire["include"],
            json!([
                "reasoning.encrypted_content",
                "message.output_text.logprobs"
            ])
        );
    }

    #[test]
    fn needs_chat_completions_positives_and_negatives() {
        // A plain request stays on Responses.
        assert!(!needs_chat_completions(&user_req("hi")));
        // A non-empty `stop` forces the chat-completions downgrade.
        let mut req = user_req("hi");
        req.stop = vec!["\n".into()];
        assert!(needs_chat_completions(&req));
        // A chat-only `extra` key does too.
        for key in [
            "response_format",
            "logit_bias",
            "logprobs",
            "n",
            "verbosity",
            "reasoning_effort",
            "reasoning_format",
            "search_settings",
        ] {
            let mut req = user_req("hi");
            req.extra.insert(key.into(), json!(true));
            assert!(
                needs_chat_completions(&req),
                "{key} must force the downgrade"
            );
        }
        // An unrelated `extra` key does not.
        let mut req = user_req("hi");
        req.extra.insert("prompt_cache_key".into(), json!("x"));
        assert!(!needs_chat_completions(&req));
    }

    #[test]
    fn collected_response_maps_reasoning_text_toolcall_and_usage() {
        let resp = json!({
            "id": "resp_1",
            "model": "gpt-5",
            "status": "completed",
            "output": [
                { "type": "reasoning", "summary": [{ "type": "summary_text", "text": "think" }],
                  "encrypted_content": "ENC" },
                { "type": "message", "role": "assistant",
                  "content": [{ "type": "output_text", "text": "hello" }] },
                { "type": "function_call", "call_id": "call_1", "name": "get_weather",
                  "arguments": "{\"city\":\"Paris\"}" }
            ],
            "usage": {
                "input_tokens": 9, "output_tokens": 3, "total_tokens": 12,
                "input_tokens_details": { "cached_tokens": 4 }
            }
        });
        let chunks = openai_responses_to_chunks(resp, "fallback");
        assert!(
            matches!(&chunks[0], StreamChunk::Start { id, model } if id == "resp_1" && model == "gpt-5")
        );
        assert!(matches!(&chunks[1], StreamChunk::Thinking { delta, .. } if delta == "think"));
        assert!(matches!(&chunks[2], StreamChunk::RedactedThinking { data, .. } if data == "ENC"));
        assert!(matches!(&chunks[3], StreamChunk::Content { delta, .. } if delta == "hello"));
        match &chunks[4] {
            StreamChunk::ToolCall { index: 0, call } => {
                assert_eq!(call.index, 0);
                assert_eq!(call.id.as_deref(), Some("call_1"));
                assert_eq!(call.name.as_deref(), Some("get_weather"));
                assert_eq!(call.arguments.as_deref(), Some("{\"city\":\"Paris\"}"));
            }
            other => panic!("expected ToolCall, got {other:?}"),
        }
        // Usage, then a ToolCalls finish (a function_call was seen), in that terminal order.
        match &chunks[5] {
            StreamChunk::Usage(u) => {
                assert_eq!(u.prompt_tokens, 9);
                assert_eq!(u.completion_tokens, 3);
                assert_eq!(u.total_tokens, 12);
                assert_eq!(u.cache_read_tokens, 4);
            }
            other => panic!("expected Usage, got {other:?}"),
        }
        assert!(matches!(
            chunks[6],
            StreamChunk::Finish {
                reason: FinishReason::ToolCalls,
                ..
            }
        ));
    }

    #[test]
    fn collected_response_incomplete_maps_to_length() {
        let resp = json!({
            "id": "resp_2",
            "model": "gpt-5",
            "status": "incomplete",
            "incomplete_details": { "reason": "max_output_tokens" },
            "output": [
                { "type": "message", "role": "assistant",
                  "content": [{ "type": "output_text", "text": "partial" }] }
            ]
        });
        let chunks = openai_responses_to_chunks(resp, "gpt-5");
        assert!(matches!(&chunks[1], StreamChunk::Content { delta, .. } if delta == "partial"));
        // No usage block → no Usage chunk; the finish is Length.
        assert!(matches!(
            chunks.last().unwrap(),
            StreamChunk::Finish {
                reason: FinishReason::Length,
                ..
            }
        ));
    }

    #[test]
    fn collected_response_maps_openrouter_signature_and_cost() {
        // OpenRouter extras on the same wire: a per-item reasoning `signature` and `usage.cost`. Both
        // relay into the canonical stream (signature after the thinking text it signs, cost verbatim).
        let resp = json!({
            "id": "resp_or",
            "model": "anthropic/claude-4.5-sonnet",
            "status": "completed",
            "output": [
                { "type": "reasoning",
                  "content": [{ "type": "reasoning_text", "text": "mull" }],
                  "signature": "SIG", "format": "anthropic-claude-v1" },
                { "type": "message", "role": "assistant",
                  "content": [{ "type": "output_text", "text": "hi" }] }
            ],
            "usage": {
                "input_tokens": 5, "output_tokens": 2, "total_tokens": 7,
                "cost": 0.00042,
                "cost_details": { "upstream_inference_cost": null }
            }
        });
        let chunks = openai_responses_to_chunks(resp, "fallback");
        assert!(matches!(&chunks[1], StreamChunk::Thinking { delta, .. } if delta == "mull"));
        assert!(
            matches!(&chunks[2], StreamChunk::ThinkingSignature { signature, .. } if signature == "SIG")
        );
        assert!(matches!(&chunks[3], StreamChunk::Content { delta, .. } if delta == "hi"));
        assert!(
            matches!(&chunks[4], StreamChunk::Usage(u) if u.cost_usd == Some(0.00042) && u.total_tokens == 7)
        );
    }

    #[test]
    fn sse_event_sequence_maps_to_canonical_chunks() {
        let mut st = ResponsesSseState::new("gpt-5".into());
        let feed =
            |st: &mut ResponsesSseState, v: Value| openai_responses_event_to_canonical(&v, st);

        // response.created → Start.
        let c = feed(
            &mut st,
            json!({
                "type": "response.created",
                "response": { "id": "resp_9", "model": "gpt-5-2025" }
            }),
        );
        assert!(
            matches!(&c[0], StreamChunk::Start { id, model } if id == "resp_9" && model == "gpt-5-2025")
        );

        // output_item.added(function_call) → an opening ToolCall delta (id + name, no args yet).
        let c = feed(
            &mut st,
            json!({
                "type": "response.output_item.added",
                "item": { "type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "get_weather" }
            }),
        );
        match &c[0] {
            StreamChunk::ToolCall { call, .. } => {
                assert_eq!(call.index, 0);
                assert_eq!(call.id.as_deref(), Some("call_1"));
                assert_eq!(call.name.as_deref(), Some("get_weather"));
                assert!(call.arguments.is_none());
            }
            other => panic!("expected ToolCall, got {other:?}"),
        }

        // Two argument deltas, keyed by the same item id → the same call index.
        for frag in ["{\"city\":", "\"Paris\"}"] {
            let c = feed(
                &mut st,
                json!({
                    "type": "response.function_call_arguments.delta",
                    "item_id": "fc_1", "delta": frag
                }),
            );
            assert!(matches!(&c[0], StreamChunk::ToolCall { call, .. }
                if call.index == 0 && call.arguments.as_deref() == Some(frag)));
        }

        // Two output_text deltas → Content.
        for frag in ["Hel", "lo"] {
            let c = feed(
                &mut st,
                json!({ "type": "response.output_text.delta", "delta": frag }),
            );
            assert!(matches!(&c[0], StreamChunk::Content { delta, .. } if delta == frag));
        }

        // reasoning_text delta → Thinking.
        let c = feed(
            &mut st,
            json!({ "type": "response.reasoning_text.delta", "delta": "hmm" }),
        );
        assert!(matches!(&c[0], StreamChunk::Thinking { delta, .. } if delta == "hmm"));

        // output_item.done(reasoning) → the per-item signature (OpenRouter; no delta event exists for
        // it) then the encrypted block, in that order — signature first, since it signs the thinking
        // text already streamed.
        let c = feed(
            &mut st,
            json!({
                "type": "response.output_item.done",
                "item": { "type": "reasoning", "signature": "SIG", "encrypted_content": "ENC" }
            }),
        );
        assert!(
            matches!(&c[0], StreamChunk::ThinkingSignature { signature, .. } if signature == "SIG")
        );
        assert!(matches!(&c[1], StreamChunk::RedactedThinking { data, .. } if data == "ENC"));

        // response.completed → Usage then Finish(ToolCalls); the state marks the stream finished.
        assert!(!st.is_finished());
        let c = feed(
            &mut st,
            json!({
                "type": "response.completed",
                "response": {
                    "id": "resp_9", "status": "completed",
                    "usage": { "input_tokens": 5, "output_tokens": 7, "total_tokens": 12 }
                }
            }),
        );
        assert!(matches!(&c[0], StreamChunk::Usage(u) if u.total_tokens == 12));
        assert!(matches!(
            &c[1],
            StreamChunk::Finish {
                reason: FinishReason::ToolCalls,
                ..
            }
        ));
        assert!(st.is_finished());
    }
}
