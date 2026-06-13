//! The OpenAI Audio Transcription compat surface (`POST /v1/audio/transcriptions`, speech-to-text).
//!
//! In: a `multipart/form-data` upload — the server decodes the parts and calls [`build_request`]. Out:
//! a canonical [`TranscriptionResponse`] rendered per the requested `response_format`. JSON shapes
//! (`json` | `verbose_json`) are built here from the canonical transcript; the text shapes (`text` |
//! `srt` | `vtt`) return the provider's transcript body verbatim — a provider that supports those
//! formats carries the real subtitle text through (see the OpenAI-compat provider's `transcribe`).

use bytes::Bytes;
use llmleaf_model::{ModelError, TranscriptionRequest, TranscriptionResponse};
use serde_json::{json, Map, Value};

fn mapping(msg: impl Into<String>) -> ModelError {
    ModelError::Mapping(msg.into())
}

/// The decoded fields of a transcription multipart upload, gathered by the server edge. String-typed
/// because every multipart value arrives as text; this is where they become the canonical request.
#[derive(Default)]
pub struct TranscriptionForm {
    pub model: Option<String>,
    pub audio: Option<Bytes>,
    pub filename: Option<String>,
    pub language: Option<String>,
    pub prompt: Option<String>,
    pub response_format: Option<String>,
    pub temperature: Option<String>,
}

/// Turn the decoded form into a canonical [`TranscriptionRequest`], validating the two required parts
/// (`model`, `file`) and parsing the numeric `temperature` (a mapping concern, kept out of the server).
pub fn build_request(form: TranscriptionForm) -> Result<TranscriptionRequest, ModelError> {
    let model = form
        .model
        .ok_or_else(|| mapping("`model` form field is required"))?;
    let audio = form
        .audio
        .ok_or_else(|| mapping("`file` form field is required"))?;
    let temperature = match form.temperature {
        Some(t) => Some(
            t.parse::<f32>()
                .map_err(|_| mapping("`temperature` must be a number"))?,
        ),
        None => None,
    };
    Ok(TranscriptionRequest {
        model,
        audio,
        filename: form.filename.unwrap_or_else(|| "audio".to_string()),
        language: form.language,
        prompt: form.prompt,
        response_format: form.response_format,
        temperature,
        extra: Map::new(),
    })
}

/// The transcript rendered for the consumer, per the requested `response_format`.
pub enum TranscriptionBody {
    /// A JSON object (`json` / `verbose_json`).
    Json(Value),
    /// Plain UTF-8 text (`text`, and the `srt`/`vtt` fallback).
    Text(String),
}

pub fn encode_transcription(
    resp: &TranscriptionResponse,
    response_format: Option<&str>,
) -> TranscriptionBody {
    match response_format.unwrap_or("json") {
        // Plain-text and subtitle formats: the transcript text is already the verbatim provider body.
        "text" | "srt" | "vtt" => TranscriptionBody::Text(resp.text.clone()),
        "verbose_json" => {
            let mut o = Map::new();
            o.insert("task".into(), json!("transcribe"));
            if let Some(lang) = &resp.language {
                o.insert("language".into(), json!(lang));
            }
            if let Some(dur) = resp.duration {
                o.insert("duration".into(), json!(dur));
            }
            o.insert("text".into(), json!(resp.text));
            TranscriptionBody::Json(Value::Object(o))
        }
        // "json" (the default) and any unknown value: the minimal { text } object.
        _ => TranscriptionBody::Json(json!({ "text": resp.text })),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resp() -> TranscriptionResponse {
        TranscriptionResponse {
            text: "hello world".into(),
            language: Some("en".into()),
            duration: Some(1.5),
            usage: Default::default(),
        }
    }

    #[test]
    fn build_requires_model_and_file() {
        let mut form = TranscriptionForm {
            model: Some("whisper-1".into()),
            ..Default::default()
        };
        assert!(build_request(TranscriptionForm {
            ..Default::default()
        })
        .is_err()); // no model
        form.audio = Some(Bytes::from_static(b"RIFF...."));
        let req = build_request(form).unwrap();
        assert_eq!(req.model, "whisper-1");
        assert_eq!(req.filename, "audio"); // default when omitted
    }

    #[test]
    fn bad_temperature_is_rejected() {
        let form = TranscriptionForm {
            model: Some("whisper-1".into()),
            audio: Some(Bytes::from_static(b"x")),
            temperature: Some("hot".into()),
            ..Default::default()
        };
        assert!(build_request(form).is_err());
    }

    #[test]
    fn json_default_is_minimal_text_object() {
        match encode_transcription(&resp(), None) {
            TranscriptionBody::Json(v) => assert_eq!(v["text"], "hello world"),
            _ => panic!("expected json"),
        }
    }

    #[test]
    fn verbose_json_carries_language_and_duration() {
        match encode_transcription(&resp(), Some("verbose_json")) {
            TranscriptionBody::Json(v) => {
                assert_eq!(v["language"], "en");
                assert_eq!(v["duration"], 1.5);
                assert_eq!(v["text"], "hello world");
            }
            _ => panic!("expected json"),
        }
    }

    #[test]
    fn text_format_is_plain_body() {
        match encode_transcription(&resp(), Some("text")) {
            TranscriptionBody::Text(t) => assert_eq!(t, "hello world"),
            _ => panic!("expected text"),
        }
    }
}
