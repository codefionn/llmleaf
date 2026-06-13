use serde::{Deserialize, Serialize};

/// One element of the canonical response stream.
///
/// This is the unit the core moves on the hot path. Output edge mappings turn a sequence of these
/// into a dialect's wire format (OpenAI `chat.completion.chunk`, etc.); provider extensions produce
/// them from upstream bytes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamChunk {
    /// The stream has opened. Carries the resolved response id and the model that actually served.
    Start { id: String, model: String },
    /// Incremental assistant text for choice `index`.
    Content { index: u32, delta: String },
    /// Incremental tool-call construction for choice `index`.
    ToolCall { index: u32, call: ToolCallDelta },
    /// A usage report. May arrive once at the end, or be updated as the provider reports it.
    /// The core never computes this — it only relays what the provider emits (principle 5).
    Usage(Usage),
    /// Choice `index` has finished.
    Finish { index: u32, reason: FinishReason },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCallDelta {
    pub index: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Argument fragment to append.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    Length,
    ToolCalls,
    ContentFilter,
    Error,
}

/// Token accounting as reported by the provider. The optional `cost_usd` is filled at the edge from
/// the bundled pricing dataset (the sidecar crate) when available — never computed on the hot path
/// beyond a multiply against a looked-up rate.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct Usage {
    #[serde(default)]
    pub prompt_tokens: u64,
    #[serde(default)]
    pub completion_tokens: u64,
    #[serde(default)]
    pub total_tokens: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost_usd: Option<f64>,
}

/// A fully collected, non-streaming response (principle 4: a collected stream).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatResponse {
    pub id: String,
    pub model: String,
    pub choices: Vec<Choice>,
    #[serde(default)]
    pub usage: Usage,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Choice {
    pub index: u32,
    pub text: String,
    #[serde(default)]
    pub tool_calls: Vec<crate::ToolCall>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,
}
