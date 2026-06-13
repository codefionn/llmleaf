//! Canonical audio model: text-to-speech (TTS) and speech-to-text (STT).
//!
//! TTS is genuinely incremental — audio bytes arrive over time — so the canonical speech response is
//! a **stream** of [`AudioChunk`]s, exactly as chat is a stream of `StreamChunk`s (principle 4). A
//! non-streaming consumer gets it collected ([`collect_audio`]), never the other way around.
//!
//! STT produces a single transcript object: there is no token-by-token surface in the common dialect,
//! so [`TranscriptionResponse`] is a batch object (the same reasoning as embeddings).

use std::pin::Pin;

use bytes::{Bytes, BytesMut};
use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::{ModelError, Usage};

// ---------------------------------------------------------------------------------------------
// Text-to-speech (TTS)
// ---------------------------------------------------------------------------------------------

/// A canonical speech-synthesis request (`/v1/audio/speech`). Maps in from the consumer dialect;
/// speech-capable providers map out of it.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SpeechRequest {
    /// The logical model (routed exactly as chat/embeddings).
    pub model: String,
    /// The text to speak.
    pub input: String,
    /// Provider voice id (e.g. `alloy`). Required by the OpenAI dialect.
    pub voice: String,
    /// Container/codec the consumer asked for (`mp3`, `opus`, `aac`, `flac`, `wav`, `pcm`). The output
    /// edge maps this to the right `Content-Type`; the provider honors it upstream.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_format: Option<String>,
    /// Playback speed multiplier.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speed: Option<f32>,
    /// Dialect-/provider-specific fields preserved verbatim (principle 7).
    #[serde(default, skip_serializing_if = "Map::is_empty")]
    pub extra: Map<String, Value>,
}

/// One element of the canonical audio stream — the unit the core moves on the TTS hot path.
///
/// `Start` carries the resolved MIME type, `Data` frames carry raw audio bytes, `Usage` is an
/// optional provider-reported accounting, and `Finish` terminates the stream.
#[derive(Debug, Clone, PartialEq)]
pub enum AudioChunk {
    /// The stream has opened; carries the audio MIME type (e.g. `audio/mpeg`).
    Start { content_type: String },
    /// A frame of audio bytes.
    Data(Bytes),
    /// Provider-reported usage, if any (the core relays, never computes — principle 5).
    Usage(Usage),
    /// The audio is complete.
    Finish,
}

/// The canonical streaming audio response (cf. [`crate::ResponseStream`] for chat). A trait object so
/// providers (compiled or WASM) and edge mappings produce/consume it without the core knowing their
/// concrete stream types.
pub type AudioStream = Pin<Box<dyn Stream<Item = Result<AudioChunk, ModelError>> + Send>>;

/// A fully collected audio response (principle 4: non-streaming is *defined as* a collected stream).
#[derive(Debug, Clone, PartialEq)]
pub struct AudioBytes {
    pub content_type: String,
    pub data: Bytes,
    pub usage: Usage,
}

/// Collect an [`AudioStream`] into the whole-object [`AudioBytes`]. The single sanctioned way to go
/// from the streaming representation to a non-streaming one (there is no separate non-streaming path).
pub async fn collect_audio(mut stream: AudioStream) -> Result<AudioBytes, ModelError> {
    let mut content_type = String::new();
    let mut data = BytesMut::new();
    let mut usage = Usage::default();
    while let Some(item) = stream.next().await {
        match item? {
            AudioChunk::Start { content_type: ct } => content_type = ct,
            AudioChunk::Data(bytes) => data.extend_from_slice(&bytes),
            AudioChunk::Usage(u) => usage = u,
            AudioChunk::Finish => {}
        }
    }
    Ok(AudioBytes {
        content_type,
        data: data.freeze(),
        usage,
    })
}

// ---------------------------------------------------------------------------------------------
// Voice catalog (`GET /v1/audio/voices`)
// ---------------------------------------------------------------------------------------------

/// One voice a speech model can synthesize with. The voices listing returns these for a model; `id` is
/// exactly the value a consumer puts in [`SpeechRequest::voice`]. A provider with a static catalog
/// (e.g. OpenAI, whose voice set is documented, never fetchable) declares them in code; a provider
/// whose upstream has a live voices API maps that reply into these.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VoiceInfo {
    /// The voice id — the value to pass as [`SpeechRequest::voice`] (e.g. `alloy`).
    pub id: String,
    /// A human-friendly display name, when the provider distinguishes one from the id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Language tags the voice covers (BCP-47 / ISO-639), when the provider reports them.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub languages: Vec<String>,
    /// Dialect-/provider-specific fields preserved verbatim (principle 7) — e.g. gender, preview url.
    #[serde(default, skip_serializing_if = "Map::is_empty")]
    pub extra: Map<String, Value>,
}

// ---------------------------------------------------------------------------------------------
// Speech-to-text (STT)
// ---------------------------------------------------------------------------------------------

/// A canonical transcription request (`/v1/audio/transcriptions`). The consumer surface decodes the
/// multipart upload into this; the audio rides in-band as bytes.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TranscriptionRequest {
    /// The logical model (routed exactly as chat/embeddings).
    pub model: String,
    /// Raw audio bytes to transcribe. Skipped from serialization so payload-carrying events never
    /// dump the audio blob (transparent about *transformations*, never about leaking megabytes).
    #[serde(skip)]
    pub audio: Bytes,
    /// Upload filename — providers use its extension to detect the container/codec.
    pub filename: String,
    /// ISO-639-1 language hint.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    /// Optional prompt to bias decoding.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    /// Requested transcript format (`json`, `text`, `verbose_json`, `srt`, `vtt`). The output edge maps
    /// the canonical transcript onto this.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_format: Option<String>,
    /// Sampling temperature.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Dialect-/provider-specific fields preserved verbatim (principle 7).
    #[serde(default, skip_serializing_if = "Map::is_empty")]
    pub extra: Map<String, Value>,
}

/// A canonical transcription response. A single batch object (see module note).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TranscriptionResponse {
    /// The transcript text.
    pub text: String,
    /// Detected/declared language, when the provider reports it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    /// Audio duration in seconds, when reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration: Option<f32>,
    #[serde(default)]
    pub usage: Usage,
}
