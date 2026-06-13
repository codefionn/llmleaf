use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// A canonical chat request. Every consumer dialect maps *into* this; every provider extension maps
/// *out of* it. Fields are the common denominator across modern chat APIs; anything dialect- or
/// provider-specific that we don't model rides verbatim in [`ChatRequest::extra`] (principle 7:
/// transparent — we never silently drop what we don't understand).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChatRequest {
    /// The logical model the consumer asked for. The router resolves this to provider targets;
    /// a provider extension may rewrite it to its own upstream model id.
    pub model: String,

    pub messages: Vec<Message>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stop: Vec<String>,

    /// Whether the *consumer* asked to stream. The internal representation is a stream regardless
    /// (principle 4); this records the consumer's wire preference for the output edge mapping.
    #[serde(default)]
    pub stream: bool,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ToolDef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,

    /// Optional reasoning ("thinking") effort — a coarse, canonical ladder each dialect maps into its
    /// own vocabulary at the edge (an effort string for some, a thinking token budget for others). Leave
    /// `None` for the upstream default — or to drive thinking yourself with the dialect's own field
    /// carried verbatim in [`ChatRequest::extra`] (principle 7).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<Thinking>,

    /// Dialect-/provider-specific fields preserved verbatim through the core.
    #[serde(default, skip_serializing_if = "Map::is_empty")]
    pub extra: Map<String, Value>,
}

/// Reasoning ("thinking") effort — an optional, deliberately coarse, canonical ladder.
///
/// Five rungs the core understands; the mapping into each dialect's own wire shape (an effort string,
/// a token budget, …) lives at the edge, never here (principle 2). The mapping is lossy by design — a
/// dialect with only three effort tiers collapses the upper rungs. For exact, dialect-native control,
/// leave [`ChatRequest::thinking`] `None` and pass the upstream's own field through
/// [`ChatRequest::extra`] verbatim (principle 7: transparent).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Thinking {
    Low,
    Med,
    High,
    /// A rung above `High` for dialects whose vocabulary can express it; collapses to the top tier
    /// where it cannot.
    Highx,
    Max,
}

/// The user-facing modality of a model, as surfaced by the model-catalog surface (`GET /v1/models`)
/// and the `?type=` filter on it.
///
/// This is *catalog* vocabulary — the coarse "what does this model do" a client picks from. It is
/// deliberately distinct from the engine's internal dispatch taxonomy (chat/embed/speech/transcribe):
/// this names the capability a consumer browses, not the code path a request takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Modality {
    /// A text-in, text-out language model (chat/completion).
    Llm,
    /// Text-to-speech (text in, audio out).
    Tts,
    /// Speech-to-text / transcription (audio in, text out).
    Stt,
    /// An embedding model (text in, vector out).
    Embedding,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    #[serde(default)]
    pub content: Vec<ContentPart>,
    /// Tool calls emitted by an assistant turn.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    /// For `role = tool`: which call this message answers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Optional author name (e.g. a named function/tool).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl Message {
    pub fn text(role: Role, text: impl Into<String>) -> Self {
        Message {
            role,
            content: vec![ContentPart::Text { text: text.into() }],
            tool_calls: Vec::new(),
            tool_call_id: None,
            name: None,
        }
    }

    /// Concatenate all text parts — convenience for providers/surfaces that want a flat string.
    pub fn text_content(&self) -> String {
        let mut out = String::new();
        for part in &self.content {
            if let ContentPart::Text { text } = part {
                out.push_str(text);
            }
        }
        out
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text {
        text: String,
    },
    ImageUrl {
        url: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        detail: Option<String>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    /// Arguments as a raw JSON string, exactly as the model emitted them.
    pub arguments: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolDef {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// JSON Schema for the tool's parameters.
    #[serde(default)]
    pub parameters: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoice {
    Auto,
    None,
    Required,
    /// Force a specific named tool.
    Named(String),
}
