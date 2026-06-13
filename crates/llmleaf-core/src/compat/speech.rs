//! The OpenAI Audio Speech compat surface (`POST /v1/audio/speech`, text-to-speech).
//!
//! In: an OpenAI speech request JSON → canonical [`SpeechRequest`]. Out: the provider's audio bytes
//! stream verbatim — there is no JSON envelope, the consumer receives the raw audio file (the server
//! sets `Content-Type` from the stream's opening chunk). Unknown fields ride through in
//! [`SpeechRequest::extra`] (principle 7).

use llmleaf_model::{ModelError, SpeechRequest};
use serde_json::Value;

fn mapping(msg: impl Into<String>) -> ModelError {
    ModelError::Mapping(msg.into())
}

pub fn parse_speech_request(value: Value) -> Result<SpeechRequest, ModelError> {
    let Value::Object(mut obj) = value else {
        return Err(mapping("request body must be a JSON object"));
    };

    let model = match obj.remove("model") {
        Some(Value::String(s)) => s,
        _ => return Err(mapping("`model` is required and must be a string")),
    };
    let input = match obj.remove("input") {
        Some(Value::String(s)) => s,
        _ => return Err(mapping("`input` is required and must be a string")),
    };
    let voice = match obj.remove("voice") {
        Some(Value::String(s)) => s,
        _ => return Err(mapping("`voice` is required and must be a string")),
    };
    let response_format = obj
        .remove("response_format")
        .and_then(|v| v.as_str().map(str::to_owned));
    let speed = obj
        .remove("speed")
        .and_then(|v| v.as_f64())
        .map(|n| n as f32);

    let extra = obj;
    Ok(SpeechRequest {
        model,
        input,
        voice,
        response_format,
        speed,
        extra,
    })
}

/// The MIME type for a `response_format`, used as a fallback when the provider did not set one on its
/// opening [`llmleaf_model::AudioChunk::Start`]. Mirrors the formats the OpenAI speech endpoint accepts.
pub fn content_type_for_format(format: Option<&str>) -> &'static str {
    match format.unwrap_or("mp3") {
        "mp3" => "audio/mpeg",
        // opus is delivered Ogg-framed, not as a raw opus stream.
        "opus" => "audio/ogg",
        "aac" => "audio/aac",
        "flac" => "audio/flac",
        "wav" => "audio/wav",
        "pcm" => "audio/pcm",
        _ => "application/octet-stream",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_required_fields_and_extra() {
        let req = parse_speech_request(json!({
            "model": "tts-1",
            "input": "hello",
            "voice": "alloy",
            "response_format": "opus",
            "speed": 1.25,
            "stability": 0.7
        }))
        .unwrap();
        assert_eq!(req.model, "tts-1");
        assert_eq!(req.voice, "alloy");
        assert_eq!(req.response_format.as_deref(), Some("opus"));
        assert!((req.speed.unwrap() - 1.25).abs() < 1e-6);
        assert_eq!(req.extra.get("stability"), Some(&json!(0.7)));
    }

    #[test]
    fn missing_voice_is_an_error() {
        assert!(parse_speech_request(json!({ "model": "tts-1", "input": "hi" })).is_err());
    }

    #[test]
    fn content_types_cover_documented_formats() {
        assert_eq!(content_type_for_format(None), "audio/mpeg");
        assert_eq!(content_type_for_format(Some("wav")), "audio/wav");
        assert_eq!(content_type_for_format(Some("flac")), "audio/flac");
    }
}
