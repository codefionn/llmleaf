//! The one internal canonical model for llmleaf.
//!
//! SOUL.md principle 3: "One internal model, many dialects." This crate IS that model. It knows
//! nothing of OpenAI, OpenRouter, Anthropic, or any provider — those are mappings at the edge.
//! No external dialect is "native" here; this is the lingua franca every dialect maps to and from.
//!
//! SOUL.md principle 4: "Streaming is the default." The canonical response [`ResponseStream`] is a
//! stream. A non-streaming response is a *collected* stream ([`collect`]), never the other way around.

mod audio;
mod batch;
mod catalog;
mod embed;
mod error;
mod request;
mod response;
mod stream;

pub use audio::{
    collect_audio, AudioBytes, AudioChunk, AudioStream, SpeechRequest, TranscriptionRequest,
    TranscriptionResponse, VoiceInfo,
};
pub use batch::{
    BatchCounts, BatchHandle, BatchItem, BatchOutcome, BatchResult, BatchResultStream, BatchSpec,
    BatchStatus,
};
pub use catalog::ModelInfo;
pub use embed::{Embedding, EmbeddingRequest, EmbeddingResponse};
pub use error::ModelError;
pub use request::{
    ChatRequest, ContentPart, Message, Modality, Role, Thinking, ToolCall, ToolChoice, ToolDef,
};
pub use response::{ChatResponse, Choice, FinishReason, StreamChunk, ToolCallDelta, Usage};
pub use stream::{collect, collect_chunks, ResponseStream};
