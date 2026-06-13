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
        }
    }

    /// Accumulated plain-text content, if the message carries a text body.
    pub fn text(&self) -> Option<&str> {
        match &self.content {
            Some(Content::Text(s)) => Some(s.as_str()),
            _ => None,
        }
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
}

impl ModelType {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            ModelType::All => "all",
            ModelType::Llm => "llm",
            ModelType::Tts => "tts",
            ModelType::Stt => "stt",
            ModelType::Embedding => "embedding",
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
