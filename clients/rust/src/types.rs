//! Public SDK types — serde structs that mirror `proto/llmleaf/v1/llmleaf.proto` and
//! serialise to / parse from the exact OpenAI/OpenRouter-shaped JSON the gateway speaks
//! (see SPEC.md).
//!
//! Why not the prost-generated types? Prost emits protobuf-binary-shaped types: oneofs
//! become Rust enums with no JSON union behaviour, proto enums are `i32`, and there is
//! no OpenAI casing. These structs are the wire model; [`crate::pb`] is the codegen
//! proof / canonical-proto mirror.
//!
//! ## Enum ⇄ wire mapping
//!
//! SPEC.md asks for a single `enumToWire`/`enumFromWire` pair that lowercases the enum
//! value name. In serde the mechanical equivalent is `#[serde(rename_all =
//! "snake_case")]` on every closed-set enum: it turns `TOOL_CALLS` → `"tool_calls"`,
//! `IN_PROGRESS` → `"in_progress"`, etc. The proto `*_UNSPECIFIED` zero value ⇔ field
//! absent is modelled by making the field `Option<_>` and skipping it when `None`.
//!
//! ## Free-form JSON fields
//!
//! `ChatRequest.extra`, `FunctionDef.parameters`, `ResponseFormat.json_schema`,
//! `EmbeddingRequest.extra`, `SpeechRequest.extra`, `ModelEntry.default_parameters`
//! are carried as [`serde_json::Value`] (a raw JSON value), so they are spliced into the
//! body verbatim on encode and captured back as a sub-object on decode — never
//! double-encoded as a string. `ChatRequest.extra` is additionally flattened so its keys
//! merge at the top level of the request object.

use serde::de::{self, Deserializer};
use serde::ser::{SerializeSeq, Serializer};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Common
// ---------------------------------------------------------------------------

/// Token accounting echoed on every response. `cost_usd` is an llmleaf addition and is
/// absent when the model has no known price.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Usage {
    #[serde(default)]
    pub prompt_tokens: u32,
    #[serde(default)]
    pub completion_tokens: u32,
    #[serde(default)]
    pub total_tokens: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost_usd: Option<f64>,
    /// Prompt-cache hit accounting (OpenAI `usage.prompt_tokens_details`). Absent when the upstream
    /// reported no caching; [`Usage::cached_tokens`] flattens it to a plain count.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
    /// Input tokens written to the provider's prompt cache this request — a cache *write* (creation).
    /// An llmleaf extension (Anthropic reports it; OpenAI/OpenRouter do not); absent when there were none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_creation_tokens: Option<u32>,
}

impl Usage {
    /// Prompt tokens served from the provider's cache this request — a cache *read* (hit). `0` when
    /// the upstream reported no caching.
    pub fn cached_tokens(&self) -> u32 {
        self.prompt_tokens_details
            .as_ref()
            .and_then(|d| d.cached_tokens)
            .unwrap_or(0)
    }

    /// Input tokens written to the provider's cache this request — a cache *write* (creation). `0`
    /// when there were none (or the provider does not report writes).
    pub fn cache_writes(&self) -> u32 {
        self.cache_creation_tokens.unwrap_or(0)
    }
}

/// Breakdown of [`Usage::prompt_tokens`]. Today only the cache-read (hit) share is surfaced — the
/// count of prompt tokens served from the provider's cache rather than processed fresh.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PromptTokensDetails {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cached_tokens: Option<u32>,
}

// ---------------------------------------------------------------------------
// Enums (closed-set; snake_case wire tokens; absent == unspecified)
// ---------------------------------------------------------------------------

/// Message author role.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

/// Why the model stopped generating.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    Length,
    ToolCalls,
    ContentFilter,
}

/// Batch lifecycle status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BatchStatus {
    Validating,
    InProgress,
    Finalizing,
    Completed,
    Failed,
    Expired,
    Canceling,
    Canceled,
}

// ---------------------------------------------------------------------------
// Chat — content parts (multimodal)
// ---------------------------------------------------------------------------

/// `{"url":"...","detail":"auto"}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageUrl {
    pub url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// A single content part of a multimodal message.
///
/// Serialises to the tagged OpenAI shapes:
/// `{"type":"text","text":"..."}` and
/// `{"type":"image_url","image_url":{...}}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text { text: String },
    ImageUrl { image_url: ImageUrl },
}

impl ContentPart {
    /// Convenience constructor for a text part.
    pub fn text(text: impl Into<String>) -> Self {
        ContentPart::Text { text: text.into() }
    }

    /// Convenience constructor for an image-url part.
    pub fn image_url(url: impl Into<String>) -> Self {
        ContentPart::ImageUrl {
            image_url: ImageUrl {
                url: url.into(),
                detail: None,
            },
        }
    }
}

/// Message `content`: a plain string when there is only text, else an array of parts.
/// Untagged so it serialises as a bare string or a bare array, matching the wire.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Content {
    Text(String),
    Parts(Vec<ContentPart>),
}

impl From<String> for Content {
    fn from(s: String) -> Self {
        Content::Text(s)
    }
}

impl From<&str> for Content {
    fn from(s: &str) -> Self {
        Content::Text(s.to_string())
    }
}

impl From<Vec<ContentPart>> for Content {
    fn from(p: Vec<ContentPart>) -> Self {
        Content::Parts(p)
    }
}

// ---------------------------------------------------------------------------
// Chat — tool calls
// ---------------------------------------------------------------------------

/// A function the model called. `arguments` is a JSON-encoded string (OpenAI shape).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

/// A tool call emitted by the model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub function: FunctionCall,
}

/// Incremental tool-call fragment on a streaming delta.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FunctionCallDelta {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

/// Streaming tool-call delta; fields arrive piecemeal.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCallDelta {
    pub index: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none", rename = "type")]
    pub kind: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub function: Option<FunctionCallDelta>,
}

// ---------------------------------------------------------------------------
// Chat — reasoning ("thinking") blocks
// ---------------------------------------------------------------------------

/// One structured reasoning ("thinking") block (OpenRouter `reasoning_details[]`). It expresses both
/// *open* reasoning — visible text, optionally signed — and *hidden* reasoning — an encrypted/redacted
/// blob the provider returns in place of the text. `kind` (wire `type`) is the discriminator:
///
/// - `"reasoning.text"` → [`text`](Self::text) (+ optional [`signature`](Self::signature)) — **open**
/// - `"reasoning.summary"` → [`summary`](Self::summary) — **open** (a summarised view)
/// - `"reasoning.encrypted"` → [`data`](Self::data) — **hidden** (redacted / opaque)
///
/// `signature` and `data` are opaque and MUST be sent back verbatim in the next request's
/// `reasoning_details` to continue a signed/encrypted reasoning turn (the upstream rejects an altered
/// or dropped block — e.g. before a tool call). Use [`is_hidden`](Self::is_hidden) /
/// [`open_text`](Self::open_text) to branch without matching on the raw `kind` string.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReasoningDetail {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index: Option<u32>,
}

impl ReasoningDetail {
    /// Whether this block is hidden (redacted / encrypted) rather than open visible reasoning.
    pub fn is_hidden(&self) -> bool {
        self.kind == "reasoning.encrypted" || (self.data.is_some() && self.text.is_none())
    }

    /// The visible reasoning text of an open block — its `text`, falling back to its `summary`.
    /// `None` for a hidden block.
    pub fn open_text(&self) -> Option<&str> {
        self.text.as_deref().or(self.summary.as_deref())
    }
}

// ---------------------------------------------------------------------------
// Chat — messages
// ---------------------------------------------------------------------------

/// A chat message (request or response).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: Role,
    /// `content` is a plain string or an array of content parts. Optional because an
    /// assistant message that only calls tools may omit it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<Content>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    /// Set when `role == Tool`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Open reasoning text the assistant emitted (OpenRouter `reasoning`), if any. The flat,
    /// human-readable form; the structured [`reasoning_details`](Self::reasoning_details) is the
    /// replay-safe one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
    /// Structured reasoning blocks (open and hidden, with signatures — see [`ReasoningDetail`]). Echo
    /// these back verbatim on the next request to preserve signed reasoning across a turn.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub reasoning_details: Vec<ReasoningDetail>,
}

impl ChatMessage {
    /// A `system` message with text content.
    pub fn system(text: impl Into<String>) -> Self {
        Self::with_text(Role::System, text)
    }

    /// A `user` message with text content.
    pub fn user(text: impl Into<String>) -> Self {
        Self::with_text(Role::User, text)
    }

    /// An `assistant` message with text content.
    pub fn assistant(text: impl Into<String>) -> Self {
        Self::with_text(Role::Assistant, text)
    }

    fn with_text(role: Role, text: impl Into<String>) -> Self {
        ChatMessage {
            role,
            content: Some(Content::Text(text.into())),
            name: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
            reasoning: None,
            reasoning_details: Vec::new(),
        }
    }

    /// Accumulated plain-text content, if the message carries a text body.
    pub fn text(&self) -> Option<&str> {
        match &self.content {
            Some(Content::Text(s)) => Some(s.as_str()),
            _ => None,
        }
    }

    /// The visible (open) reasoning for this message: the flat `reasoning` text if present, else the
    /// concatenation of the open `reasoning_details` blocks. `None` when the turn carried no visible
    /// reasoning (it may still carry hidden blocks — see [`reasoning_details`](Self::reasoning_details)).
    pub fn reasoning_text(&self) -> Option<String> {
        if let Some(r) = &self.reasoning {
            return Some(r.clone());
        }
        let joined: String = self
            .reasoning_details
            .iter()
            .filter_map(ReasoningDetail::open_text)
            .collect();
        (!joined.is_empty()).then_some(joined)
    }
}

// ---------------------------------------------------------------------------
// Chat — tool / response-format definitions
// ---------------------------------------------------------------------------

/// A function the model MAY call. `parameters` is a raw JSON Schema value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FunctionDef {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Raw JSON object spliced verbatim.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
}

/// A tool definition (`{"type":"function","function":{...}}`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolDef {
    #[serde(rename = "type")]
    pub kind: String,
    pub function: FunctionDef,
}

impl ToolDef {
    /// A `function`-typed tool definition.
    pub fn function(function: FunctionDef) -> Self {
        ToolDef {
            kind: "function".to_string(),
            function,
        }
    }
}

/// Pin a specific function: `{"name":"..."}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FunctionName {
    pub name: String,
}

/// Named tool choice: `{"type":"function","function":{"name":"..."}}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NamedToolChoice {
    #[serde(rename = "type")]
    pub kind: String,
    pub function: FunctionName,
}

/// `tool_choice`: a bare mode string (`"auto"`/`"none"`/`"required"`) or a named object.
/// Untagged so it serialises as a bare string or the object, matching the wire.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolChoice {
    Mode(String),
    Named(NamedToolChoice),
}

impl ToolChoice {
    /// `"auto"`, `"none"` or `"required"`.
    pub fn mode(mode: impl Into<String>) -> Self {
        ToolChoice::Mode(mode.into())
    }

    /// Pin a named function.
    pub fn named(name: impl Into<String>) -> Self {
        ToolChoice::Named(NamedToolChoice {
            kind: "function".to_string(),
            function: FunctionName { name: name.into() },
        })
    }
}

/// `response_format`: `{"type":"text"|"json_object"|"json_schema","json_schema":{...}}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResponseFormat {
    #[serde(rename = "type")]
    pub kind: String,
    /// Raw JSON object, present when `type == "json_schema"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub json_schema: Option<serde_json::Value>,
}

// ---------------------------------------------------------------------------
// Chat — request
// ---------------------------------------------------------------------------

/// `stop`: a bare string for one element, else an array (the wire accepts either; we
/// emit a string for a single element and an array otherwise). Untagged.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Stop {
    One(String),
    Many(Vec<String>),
}

impl Stop {
    /// Build the wire-appropriate `stop` from a list: a bare string for exactly one
    /// element, an array otherwise. Returns `None` for an empty list.
    pub fn from_vec(mut v: Vec<String>) -> Option<Self> {
        match v.len() {
            0 => None,
            1 => Some(Stop::One(v.pop().unwrap())),
            _ => Some(Stop::Many(v)),
        }
    }
}

/// `POST /v1/chat/completions` request body.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Legacy token cap. Prefer `max_completion_tokens`; this is still sent when the
    /// caller set only it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Modern token cap (takes precedence on the wire).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop: Option<Stop>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ToolDef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
    /// `"low" | "medium" | "high"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    /// Dialect-specific passthrough. Its keys are merged at the top level of the request
    /// object (P7 transparent passthrough), so it is `#[serde(flatten)]`ed here.
    #[serde(default, flatten, skip_serializing_if = "Option::is_none")]
    pub extra: Option<serde_json::Map<String, serde_json::Value>>,
}

impl ChatRequest {
    /// Start a chat request for `model` with `messages`.
    pub fn new(model: impl Into<String>, messages: Vec<ChatMessage>) -> Self {
        ChatRequest {
            model: model.into(),
            messages,
            ..Default::default()
        }
    }
}

// ---------------------------------------------------------------------------
// Chat — non-streaming response
// ---------------------------------------------------------------------------

/// One completion choice.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Choice {
    #[serde(default)]
    pub index: u32,
    pub message: ChatMessage,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,
}

/// `POST /v1/chat/completions` non-streaming response (`object:"chat.completion"`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<Choice>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

impl ChatResponse {
    /// Plain text of the first choice, if any.
    pub fn first_text(&self) -> Option<&str> {
        self.choices.first().and_then(|c| c.message.text())
    }
}

// ---------------------------------------------------------------------------
// Chat — streaming chunk
// ---------------------------------------------------------------------------

/// Incremental delta on a streaming choice.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Delta {
    /// First chunk only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<Role>,
    /// Incremental text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCallDelta>,
    /// Incremental open reasoning text, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
    /// Incremental structured reasoning blocks (open / hidden — see [`ReasoningDetail`]).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub reasoning_details: Vec<ReasoningDetail>,
}

/// One streaming choice.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChunkChoice {
    #[serde(default)]
    pub index: u32,
    pub delta: Delta,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,
}

/// A streaming SSE frame (`object:"chat.completion.chunk"`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
    /// Terminal chunk only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

impl ChatCompletionChunk {
    /// The incremental text of the first choice's delta, if any.
    pub fn first_delta_text(&self) -> Option<&str> {
        self.choices
            .first()
            .and_then(|c| c.delta.content.as_deref())
    }

    /// The incremental open reasoning text of the first choice's delta, if any.
    pub fn first_delta_reasoning(&self) -> Option<&str> {
        self.choices
            .first()
            .and_then(|c| c.delta.reasoning.as_deref())
    }
}

// ---------------------------------------------------------------------------
// Responses (POST /v1/responses) — the OpenAI Responses dialect
// ---------------------------------------------------------------------------
//
// Same canonical core as chat, a different edge dialect. The wire quirks (SPEC.md):
// `input` is a bare string or an item array; message items are role-keyed objects with
// NO `"type"`, every other item carries its `"type"`; tools and the named `tool_choice`
// are FLAT (no nested `function` object); an `input_image.image_url` is a plain string;
// reasoning `summary[]`/`content[]` entries take their `"summary_text"`/`"reasoning_text"`
// token from the list they live in; output-text parts always carry an `annotations` array.

/// The caller's answer to a function call, replayed on the next turn
/// (`{"type":"function_call_output","call_id":"...","output":"..."}`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResponseFunctionCallOutputItem {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub call_id: String,
    pub output: String,
}

/// A function call the model made (`{"type":"function_call","call_id","name","arguments"}`).
/// `call_id` pairs it with its [`ResponseFunctionCallOutputItem`]; `arguments` is the raw
/// JSON string exactly as emitted.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResponseFunctionCallItem {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub call_id: String,
    pub name: String,
    pub arguments: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
}

/// One reasoning text entry. Its wire `"type"` token is decided by the list it sits in on a
/// [`ResponseReasoningItem`]: `summary[]` → `"summary_text"`, `content[]` → `"reasoning_text"`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResponseReasoningText {
    pub text: String,
}

impl ResponseReasoningText {
    /// Wrap a piece of reasoning text.
    pub fn new(text: impl Into<String>) -> Self {
        ResponseReasoningText { text: text.into() }
    }
}

/// A reasoning ("thinking") item. `encrypted_content` is opaque and MUST be echoed back
/// verbatim in the next request's input to continue an encrypted reasoning turn (SPEC.md).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ResponseReasoningItem {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Summarised reasoning; each entry serialises as `{"type":"summary_text","text"}`.
    #[serde(
        default,
        skip_serializing_if = "Vec::is_empty",
        serialize_with = "serialize_summary_texts"
    )]
    pub summary: Vec<ResponseReasoningText>,
    /// Full reasoning; each entry serialises as `{"type":"reasoning_text","text"}`.
    #[serde(
        default,
        skip_serializing_if = "Vec::is_empty",
        serialize_with = "serialize_content_texts"
    )]
    pub content: Vec<ResponseReasoningText>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encrypted_content: Option<String>,
}

/// A single content part of a Responses message. The wire `"type"` token matches the set
/// variant; note `input_image.image_url` is a plain **string** (not the chat dialect's
/// `{url}` object) and `output_text` always carries an `annotations` array.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseContentPart {
    InputText {
        text: String,
    },
    InputImage {
        image_url: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        detail: Option<String>,
    },
    OutputText {
        text: String,
        /// Always emitted (as `[]` when empty), matching the wire; contents are opaque.
        #[serde(default)]
        annotations: Vec<serde_json::Value>,
    },
}

impl ResponseContentPart {
    /// An `input_text` part.
    pub fn input_text(text: impl Into<String>) -> Self {
        ResponseContentPart::InputText { text: text.into() }
    }

    /// An `input_image` part; `image_url` is a plain string here.
    pub fn input_image(image_url: impl Into<String>) -> Self {
        ResponseContentPart::InputImage {
            image_url: image_url.into(),
            detail: None,
        }
    }

    /// An `output_text` part; `annotations` starts empty (serialises as `[]`).
    pub fn output_text(text: impl Into<String>) -> Self {
        ResponseContentPart::OutputText {
            text: text.into(),
            annotations: Vec::new(),
        }
    }

    /// The text of a text part (`input_text` or `output_text`), if this is one.
    pub fn text(&self) -> Option<&str> {
        match self {
            ResponseContentPart::InputText { text }
            | ResponseContentPart::OutputText { text, .. } => Some(text.as_str()),
            ResponseContentPart::InputImage { .. } => None,
        }
    }
}

/// A Responses message `content`: a bare string, or an array of parts. Untagged.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ResponseContent {
    Text(String),
    Parts(Vec<ResponseContentPart>),
}

impl From<String> for ResponseContent {
    fn from(s: String) -> Self {
        ResponseContent::Text(s)
    }
}

impl From<&str> for ResponseContent {
    fn from(s: &str) -> Self {
        ResponseContent::Text(s.to_string())
    }
}

impl From<Vec<ResponseContentPart>> for ResponseContent {
    fn from(p: Vec<ResponseContentPart>) -> Self {
        ResponseContent::Parts(p)
    }
}

/// A conversation message item. On the wire it is a role-keyed object with NO `"type"`
/// field: `{"role":"user","content":"..."}`. `id`/`status` are output-only.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResponseMessageItem {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// `"user" | "system" | "developer" | "assistant"` — a plain wire string (the dialect's
    /// `"developer"` role collides with no chat enum, so it stays free-form, per SPEC.md).
    pub role: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<ResponseContent>,
    /// Output only: `"in_progress" | "completed"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
}

impl ResponseMessageItem {
    fn with_role(role: &str, content: impl Into<ResponseContent>) -> Self {
        ResponseMessageItem {
            id: None,
            role: role.to_string(),
            content: Some(content.into()),
            status: None,
        }
    }

    /// A `user` message.
    pub fn user(content: impl Into<ResponseContent>) -> Self {
        Self::with_role("user", content)
    }

    /// A `system` message.
    pub fn system(content: impl Into<ResponseContent>) -> Self {
        Self::with_role("system", content)
    }

    /// A `developer` message (Responses-only role).
    pub fn developer(content: impl Into<ResponseContent>) -> Self {
        Self::with_role("developer", content)
    }

    /// An `assistant` message.
    pub fn assistant(content: impl Into<ResponseContent>) -> Self {
        Self::with_role("assistant", content)
    }
}

/// One item of the request `input` array or the response `output` array. The wire
/// discriminator is `"type"`; a message is the role-keyed object emitted with no `"type"`
/// (and decoded from an absent or `"message"` type). [`ResponseItem::Other`] preserves any
/// item type this SDK version does not model, verbatim.
#[derive(Debug, Clone, PartialEq)]
pub enum ResponseItem {
    Message(ResponseMessageItem),
    FunctionCall(ResponseFunctionCallItem),
    FunctionCallOutput(ResponseFunctionCallOutputItem),
    Reasoning(ResponseReasoningItem),
    /// An item type not modelled by this SDK version, kept verbatim (forward compatibility).
    Other(serde_json::Value),
}

impl ResponseItem {
    /// A `user` message item.
    pub fn user(content: impl Into<ResponseContent>) -> Self {
        ResponseItem::Message(ResponseMessageItem::user(content))
    }

    /// A `system` message item.
    pub fn system(content: impl Into<ResponseContent>) -> Self {
        ResponseItem::Message(ResponseMessageItem::system(content))
    }

    /// A `developer` message item.
    pub fn developer(content: impl Into<ResponseContent>) -> Self {
        ResponseItem::Message(ResponseMessageItem::developer(content))
    }

    /// An `assistant` message item.
    pub fn assistant(content: impl Into<ResponseContent>) -> Self {
        ResponseItem::Message(ResponseMessageItem::assistant(content))
    }

    /// A `function_call` item to replay a tool call the model made.
    pub fn function_call(
        call_id: impl Into<String>,
        name: impl Into<String>,
        arguments: impl Into<String>,
    ) -> Self {
        ResponseItem::FunctionCall(ResponseFunctionCallItem {
            id: None,
            call_id: call_id.into(),
            name: name.into(),
            arguments: arguments.into(),
            status: None,
        })
    }

    /// A `function_call_output` item — the caller's answer to a function call.
    pub fn function_call_output(call_id: impl Into<String>, output: impl Into<String>) -> Self {
        ResponseItem::FunctionCallOutput(ResponseFunctionCallOutputItem {
            id: None,
            call_id: call_id.into(),
            output: output.into(),
        })
    }
}

/// Internally-tagged, borrowing view used to serialise the typed (non-message) items:
/// the variant name becomes the `"type"` token and the struct's fields are inlined.
#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum TypedItemRef<'a> {
    FunctionCall(&'a ResponseFunctionCallItem),
    FunctionCallOutput(&'a ResponseFunctionCallOutputItem),
    Reasoning(&'a ResponseReasoningItem),
}

impl Serialize for ResponseItem {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            // A message is a bare role-keyed object with no `"type"` (SPEC.md).
            ResponseItem::Message(m) => m.serialize(serializer),
            ResponseItem::FunctionCall(f) => TypedItemRef::FunctionCall(f).serialize(serializer),
            ResponseItem::FunctionCallOutput(f) => {
                TypedItemRef::FunctionCallOutput(f).serialize(serializer)
            }
            ResponseItem::Reasoning(r) => TypedItemRef::Reasoning(r).serialize(serializer),
            ResponseItem::Other(v) => v.serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for ResponseItem {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = serde_json::Value::deserialize(deserializer)?;
        // The `"type"` token selects the variant; absent (or "message") means a message.
        let ty = value
            .get("type")
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned);
        let item = match ty.as_deref() {
            None | Some("message") => {
                ResponseItem::Message(serde_json::from_value(value).map_err(de::Error::custom)?)
            }
            Some("function_call") => {
                ResponseItem::FunctionCall(serde_json::from_value(value).map_err(de::Error::custom)?)
            }
            Some("function_call_output") => ResponseItem::FunctionCallOutput(
                serde_json::from_value(value).map_err(de::Error::custom)?,
            ),
            Some("reasoning") => {
                ResponseItem::Reasoning(serde_json::from_value(value).map_err(de::Error::custom)?)
            }
            // An unmodelled item type is kept verbatim rather than dropped or erroring.
            Some(_) => ResponseItem::Other(value),
        };
        Ok(item)
    }
}

/// Serialise a reasoning entry with its list-derived `"type"` token.
#[derive(Serialize)]
struct TaggedReasoningText<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    text: &'a str,
}

fn serialize_reasoning_texts<S: Serializer>(
    texts: &[ResponseReasoningText],
    kind: &'static str,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    let mut seq = serializer.serialize_seq(Some(texts.len()))?;
    for t in texts {
        seq.serialize_element(&TaggedReasoningText {
            kind,
            text: &t.text,
        })?;
    }
    seq.end()
}

fn serialize_summary_texts<S: Serializer>(
    texts: &[ResponseReasoningText],
    serializer: S,
) -> Result<S::Ok, S::Error> {
    serialize_reasoning_texts(texts, "summary_text", serializer)
}

fn serialize_content_texts<S: Serializer>(
    texts: &[ResponseReasoningText],
    serializer: S,
) -> Result<S::Ok, S::Error> {
    serialize_reasoning_texts(texts, "reasoning_text", serializer)
}

/// A tool the model MAY call — FLAT in this dialect (`type`/`name`/`parameters` at the top
/// level, no nested `function` object). `parameters` is a raw JSON Schema value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResponsesToolDef {
    #[serde(rename = "type")]
    pub kind: String,
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Raw JSON object spliced verbatim.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

impl ResponsesToolDef {
    /// A `function`-typed tool definition.
    pub fn function(name: impl Into<String>) -> Self {
        ResponsesToolDef {
            kind: "function".to_string(),
            name: name.into(),
            description: None,
            parameters: None,
            strict: None,
        }
    }
}

/// The FLAT named tool choice: `{"type":"function","name":"..."}` (no nested `function`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResponsesNamedToolChoice {
    #[serde(rename = "type")]
    pub kind: String,
    pub name: String,
}

/// `tool_choice`: a bare mode string (`"auto"`/`"none"`/`"required"`) or the flat named
/// object. Untagged so it serialises as a bare string or the object, matching the wire.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ResponsesToolChoice {
    Mode(String),
    Named(ResponsesNamedToolChoice),
}

impl ResponsesToolChoice {
    /// `"auto"`, `"none"` or `"required"`.
    pub fn mode(mode: impl Into<String>) -> Self {
        ResponsesToolChoice::Mode(mode.into())
    }

    /// Pin a named function (flat form).
    pub fn named(name: impl Into<String>) -> Self {
        ResponsesToolChoice::Named(ResponsesNamedToolChoice {
            kind: "function".to_string(),
            name: name.into(),
        })
    }
}

/// `reasoning`: `{"effort":"minimal"|"low"|"medium"|"high"|..., "summary":...}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ResponsesReasoning {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effort: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
}

/// `input`: a bare string (one user message) or an array of items. Untagged so it
/// serialises as a bare string or a bare array, matching the wire.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ResponsesInput {
    Text(String),
    Items(Vec<ResponseItem>),
}

impl From<String> for ResponsesInput {
    fn from(s: String) -> Self {
        ResponsesInput::Text(s)
    }
}

impl From<&str> for ResponsesInput {
    fn from(s: &str) -> Self {
        ResponsesInput::Text(s.to_string())
    }
}

impl From<Vec<ResponseItem>> for ResponsesInput {
    fn from(items: Vec<ResponseItem>) -> Self {
        ResponsesInput::Items(items)
    }
}

/// `POST /v1/responses` request body.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResponsesRequest {
    pub model: String,
    pub input: ResponsesInput,
    /// Becomes a leading system message.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ResponsesToolDef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ResponsesToolChoice>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ResponsesReasoning>,
    /// Accepted but always answered `false` — llmleaf stores nothing (SPEC.md).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
    /// Dialect-specific passthrough; its keys are merged at the top level of the request
    /// object (P7 transparent passthrough), so it is `#[serde(flatten)]`ed here.
    #[serde(default, flatten, skip_serializing_if = "Option::is_none")]
    pub extra: Option<serde_json::Map<String, serde_json::Value>>,
}

impl ResponsesRequest {
    /// Start a responses request for `model` with `input` (a bare string or item array).
    pub fn new(model: impl Into<String>, input: impl Into<ResponsesInput>) -> Self {
        ResponsesRequest {
            model: model.into(),
            input: input.into(),
            instructions: None,
            stream: None,
            temperature: None,
            top_p: None,
            max_output_tokens: None,
            tools: Vec::new(),
            tool_choice: None,
            reasoning: None,
            store: None,
            extra: None,
        }
    }
}

/// Prompt-cache hit accounting inside [`ResponsesUsage`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResponsesInputTokensDetails {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cached_tokens: Option<u32>,
}

/// Reasoning-token accounting inside [`ResponsesUsage`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResponsesOutputTokensDetails {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<u32>,
}

/// Token accounting in the Responses dialect's own names (`input_tokens`/`output_tokens`,
/// not the chat dialect's `prompt_tokens`/`completion_tokens`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResponsesUsage {
    #[serde(default)]
    pub input_tokens: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_tokens_details: Option<ResponsesInputTokensDetails>,
    #[serde(default)]
    pub output_tokens: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_tokens_details: Option<ResponsesOutputTokensDetails>,
    #[serde(default)]
    pub total_tokens: u32,
}

impl ResponsesUsage {
    /// Input tokens served from the provider's cache this request — a cache *read* (hit).
    /// `0` when the upstream reported no caching.
    pub fn cached_tokens(&self) -> u32 {
        self.input_tokens_details
            .as_ref()
            .and_then(|d| d.cached_tokens)
            .unwrap_or(0)
    }

    /// Output tokens spent on reasoning this request. `0` when none were reported.
    pub fn reasoning_tokens(&self) -> u32 {
        self.output_tokens_details
            .as_ref()
            .and_then(|d| d.reasoning_tokens)
            .unwrap_or(0)
    }
}

/// `status:"incomplete"` refinement: `"max_output_tokens" | "content_filter"`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResponsesIncompleteDetails {
    pub reason: String,
}

/// The inline error body carried on a failed [`ResponsesResponse`] (`status:"failed"`);
/// the proto's `ErrorBody`. Distinct from an HTTP error envelope (which surfaces as
/// [`crate::Error::Api`]) and from the streaming `"error"` event.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResponsesError {
    pub message: String,
    #[serde(rename = "type", default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
}

/// The response object (`object:"response"`), also the snapshot carried by the
/// `response.created` / `response.in_progress` / terminal stream events. Wire fields this
/// SDK does not model (tools, truncation, parallel_tool_calls, …) are ignored on decode.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResponsesResponse {
    pub id: String,
    /// `"response"`.
    pub object: String,
    #[serde(default)]
    pub created_at: i64,
    /// `"completed" | "in_progress" | "incomplete" | "failed"` — a plain wire string.
    pub status: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub incomplete_details: Option<ResponsesIncompleteDetails>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<ResponsesError>,
    #[serde(default)]
    pub model: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub output: Vec<ResponseItem>,
    /// `null` on in-flight snapshots; set on the terminal one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<ResponsesUsage>,
    /// llmleaf always answers `false` (SPEC.md).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ResponsesReasoning>,
}

impl ResponsesResponse {
    /// The assembled assistant text: every `output_text` part of every output message,
    /// concatenated in order.
    pub fn output_text(&self) -> String {
        let mut out = String::new();
        for item in &self.output {
            let ResponseItem::Message(msg) = item else {
                continue;
            };
            match &msg.content {
                Some(ResponseContent::Text(t)) => out.push_str(t),
                Some(ResponseContent::Parts(parts)) => {
                    for part in parts {
                        if let ResponseContentPart::OutputText { text, .. } = part {
                            out.push_str(text);
                        }
                    }
                }
                None => {}
            }
        }
        out
    }

    /// The function calls in `output`, in order.
    pub fn function_calls(&self) -> Vec<&ResponseFunctionCallItem> {
        self.output
            .iter()
            .filter_map(|item| match item {
                ResponseItem::FunctionCall(f) => Some(f),
                _ => None,
            })
            .collect()
    }
}

/// One streaming SSE event. Unlike chat streaming there is NO `data: [DONE]` sentinel: the
/// stream ends after the terminal `response.completed` / `response.incomplete` /
/// `response.failed` event. This is a flat superset of every event's fields — [`kind`] says
/// which are meaningful; see the accessor helpers.
///
/// [`kind`]: Self::kind
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResponsesStreamEvent {
    /// The event type, e.g. `"response.created"`, `"response.output_text.delta"`, `"error"`.
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub sequence_number: u64,
    /// Present on `response.created` / `response.in_progress` / the terminal events.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response: Option<ResponsesResponse>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_index: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub item_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content_index: Option<u32>,
    /// Present on `response.output_item.added` / `.done`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub item: Option<ResponseItem>,
    /// Present on `response.content_part.added` / `.done`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub part: Option<ResponseContentPart>,
    /// Present on `*.delta` events (text / reasoning / arguments).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub delta: Option<String>,
    /// Present on `response.output_text.done` / `response.reasoning_text.done`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// Present on `response.function_call_arguments.done`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
    /// Present on the `"error"` event.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

impl ResponsesStreamEvent {
    /// Whether this is a terminal event (`response.completed` / `.incomplete` / `.failed`),
    /// after which the stream ends.
    pub fn is_terminal(&self) -> bool {
        matches!(
            self.kind.as_str(),
            "response.completed" | "response.incomplete" | "response.failed"
        )
    }

    /// Whether this is the `"error"` event.
    pub fn is_error(&self) -> bool {
        self.kind == "error"
    }

    /// The incremental output text of a `response.output_text.delta` event; `None` otherwise.
    /// Accumulate these for the assembled text.
    pub fn output_text_delta(&self) -> Option<&str> {
        (self.kind == "response.output_text.delta")
            .then_some(self.delta.as_deref())
            .flatten()
    }

    /// The terminal `response` snapshot (with the full output and usage), if this is a
    /// terminal event carrying one.
    pub fn terminal_response(&self) -> Option<&ResponsesResponse> {
        self.is_terminal().then_some(self.response.as_ref()).flatten()
    }

    /// Whether the stream parser recognises this event's namespace (`response.*` or
    /// `"error"`). Unrecognised events are skipped by the transport (SPEC.md forward compat).
    pub(crate) fn is_recognised(&self) -> bool {
        self.kind == "error" || self.kind.starts_with("response.")
    }

    /// The message an `"error"` event carries, falling back to a failed snapshot's error body.
    pub(crate) fn error_message(&self) -> String {
        self.message
            .clone()
            .or_else(|| {
                self.response
                    .as_ref()
                    .and_then(|r| r.error.as_ref().map(|e| e.message.clone()))
            })
            .unwrap_or_default()
    }
}

// ---------------------------------------------------------------------------
// Embeddings
// ---------------------------------------------------------------------------

/// `input`: the wire accepts a bare string or an array of strings. Untagged.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    One(String),
    Many(Vec<String>),
}

impl From<String> for EmbeddingInput {
    fn from(s: String) -> Self {
        EmbeddingInput::One(s)
    }
}

impl From<&str> for EmbeddingInput {
    fn from(s: &str) -> Self {
        EmbeddingInput::One(s.to_string())
    }
}

impl From<Vec<String>> for EmbeddingInput {
    fn from(v: Vec<String>) -> Self {
        EmbeddingInput::Many(v)
    }
}

/// `POST /v1/embeddings` request body.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingRequest {
    pub model: String,
    pub input: EmbeddingInput,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<u32>,
    /// `"float"` | `"base64"`. When `"base64"`, the transport decodes each embedding back
    /// into a float vector before returning (see SPEC.md).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encoding_format: Option<String>,
    /// Raw JSON object passthrough.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extra: Option<serde_json::Value>,
}

impl EmbeddingRequest {
    /// Start an embedding request.
    pub fn new(model: impl Into<String>, input: impl Into<EmbeddingInput>) -> Self {
        EmbeddingRequest {
            model: model.into(),
            input: input.into(),
            dimensions: None,
            encoding_format: None,
            extra: None,
        }
    }
}

/// A single embedding vector. The transport always presents `embedding` as floats, even
/// when the wire carried base64.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Embedding {
    pub object: String,
    #[serde(default)]
    pub index: u32,
    pub embedding: Vec<f32>,
}

/// `POST /v1/embeddings` response (`object:"list"`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    pub object: String,
    pub data: Vec<Embedding>,
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

// ---------------------------------------------------------------------------
// Rerank
// ---------------------------------------------------------------------------

/// A rerank document. Usually a plain string; the wire also accepts a structured
/// multimodal object (`{ text?, image? }`), captured here as raw JSON. Untagged, so
/// it round-trips either shape (mirrors how [`EmbeddingInput`] stays JSON-flexible).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RerankDocument {
    Text(String),
    Object(serde_json::Value),
}

impl From<String> for RerankDocument {
    fn from(s: String) -> Self {
        RerankDocument::Text(s)
    }
}

impl From<&str> for RerankDocument {
    fn from(s: &str) -> Self {
        RerankDocument::Text(s.to_string())
    }
}

impl From<serde_json::Value> for RerankDocument {
    fn from(v: serde_json::Value) -> Self {
        RerankDocument::Object(v)
    }
}

/// `POST /v1/rerank` request body.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RerankRequest {
    pub model: String,
    pub query: String,
    pub documents: Vec<RerankDocument>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_n: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub return_documents: Option<bool>,
}

impl RerankRequest {
    /// Start a rerank request. `documents` accepts plain strings (`vec!["a", "b"]`) or
    /// structured multimodal objects (via [`serde_json::Value`]).
    pub fn new(
        model: impl Into<String>,
        query: impl Into<String>,
        documents: impl IntoIterator<Item = impl Into<RerankDocument>>,
    ) -> Self {
        RerankRequest {
            model: model.into(),
            query: query.into(),
            documents: documents.into_iter().map(Into::into).collect(),
            top_n: None,
            return_documents: None,
        }
    }
}

/// A single reranked document: its position in the request `documents` and its score.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RerankResult {
    #[serde(default)]
    pub index: u32,
    pub relevance_score: f32,
    /// Present only when the request set `return_documents: true`. Echoes the original
    /// document, which may be a plain string or a structured object.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub document: Option<RerankDocument>,
}

/// `POST /v1/rerank` response (`object:"list"`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RerankResponse {
    pub model: String,
    pub results: Vec<RerankResult>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

// ---------------------------------------------------------------------------
// Audio — speech / voices
// ---------------------------------------------------------------------------

/// `POST /v1/audio/speech` request body.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeechRequest {
    pub model: String,
    pub input: String,
    pub voice: String,
    /// `mp3|opus|aac|flac|wav|pcm`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_format: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speed: Option<f32>,
    /// Raw JSON object passthrough.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extra: Option<serde_json::Value>,
}

impl SpeechRequest {
    /// Start a speech request.
    pub fn new(
        model: impl Into<String>,
        input: impl Into<String>,
        voice: impl Into<String>,
    ) -> Self {
        SpeechRequest {
            model: model.into(),
            input: input.into(),
            voice: voice.into(),
            response_format: None,
            speed: None,
            extra: None,
        }
    }
}

/// A TTS voice.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Voice {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub languages: Vec<String>,
}

/// `GET /v1/audio/voices` response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VoicesResponse {
    pub model: String,
    pub voices: Vec<Voice>,
}

// ---------------------------------------------------------------------------
// Audio — transcriptions (STT)
// ---------------------------------------------------------------------------

/// The accompanying form fields for `POST /v1/audio/transcriptions`. The audio bytes are
/// the multipart `file` part and are passed separately to the SDK call.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TranscriptionRequest {
    pub model: String,
    /// ISO-639-1 hint.
    pub language: Option<String>,
    /// Decoding bias.
    pub prompt: Option<String>,
    /// `json|text|verbose_json|srt|vtt`.
    pub response_format: Option<String>,
    pub temperature: Option<f32>,
}

impl TranscriptionRequest {
    /// Start a transcription request for `model`.
    pub fn new(model: impl Into<String>) -> Self {
        TranscriptionRequest {
            model: model.into(),
            ..Default::default()
        }
    }
}

/// Structured transcription result (for `response_format` json / verbose_json).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TranscriptionResponse {
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

/// Result of a transcription: a structured object (json/verbose_json) or a plain-text
/// body (text/srt/vtt) — SPEC.md returns text directly for the latter.
#[derive(Debug, Clone, PartialEq)]
pub enum Transcription {
    Json(TranscriptionResponse),
    Text(String),
}

// ---------------------------------------------------------------------------
// Model catalog
// ---------------------------------------------------------------------------

/// Model architecture metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Architecture {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub input_modalities: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub output_modalities: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub modality: Option<String>,
    #[serde(default)]
    pub tokenizer: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instruct_type: Option<String>,
}

/// Per-token pricing (decimal strings, USD).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Pricing {
    pub prompt: String,
    pub completion: String,
}

/// Top-provider capabilities for a model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TopProvider {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_length: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,
    #[serde(default)]
    pub is_moderated: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_thinking_tokens: Option<u32>,
}

/// Admin-only fallback-chain entry (present only with a valid `x-admin-token`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelEndpoint {
    pub provider: String,
    pub model: String,
    #[serde(default)]
    pub down: bool,
    /// `"route" | "prefix"`.
    pub source: String,
}

/// One model in the catalog.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelEntry {
    pub id: String,
    #[serde(default)]
    pub canonical_slug: String,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub created: i64,
    #[serde(default)]
    pub description: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_length: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub architecture: Option<Architecture>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pricing: Option<Pricing>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_provider: Option<TopProvider>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub supported_parameters: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub unsupported_parameters: Vec<String>,
    /// Raw JSON object.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_parameters: Option<serde_json::Value>,
    /// Admin-only.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub endpoints: Vec<ModelEndpoint>,
}

/// `GET /v1/models` response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ListModelsResponse {
    #[serde(default)]
    pub data: Vec<ModelEntry>,
}

/// `type` query filter for `GET /v1/models`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelType {
    All,
    Llm,
    Tts,
    Stt,
    Embedding,
    Rerank,
}

impl ModelType {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            ModelType::All => "all",
            ModelType::Llm => "llm",
            ModelType::Tts => "tts",
            ModelType::Stt => "stt",
            ModelType::Embedding => "embedding",
            ModelType::Rerank => "rerank",
        }
    }
}

// ---------------------------------------------------------------------------
// Batches
// ---------------------------------------------------------------------------

/// One item in a batch create request: a `custom_id` plus a `ChatRequest` body.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BatchRequestItem {
    pub custom_id: String,
    pub body: ChatRequest,
}

/// `POST /v1/batches` request body.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BatchCreateRequest {
    pub requests: Vec<BatchRequestItem>,
}

/// Aggregate counts on a batch handle.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct BatchCounts {
    #[serde(default)]
    pub total: u64,
    #[serde(default)]
    pub processing: u64,
    #[serde(default)]
    pub succeeded: u64,
    #[serde(default)]
    pub errored: u64,
    #[serde(default)]
    pub canceled: u64,
    #[serde(default)]
    pub expired: u64,
}

/// A batch handle returned by create / retrieve / cancel.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BatchHandle {
    pub id: String,
    pub status: BatchStatus,
    #[serde(default)]
    pub counts: BatchCounts,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created_at: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ended_at: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub endpoint: Option<String>,
}

/// A successful per-item batch response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BatchResponse {
    pub status_code: u32,
    pub body: ChatResponse,
}

/// A failed per-item batch response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BatchError {
    pub code: String,
    pub message: String,
}

/// One line of the JSONL results stream.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BatchResultLine {
    pub custom_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response: Option<BatchResponse>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<BatchError>,
}
