//! Wire-decoding helpers that don't belong on the public types: base64 embedding
//! decoding and the `response_format` → `Content-Type` table.

use crate::error::{Error, Result};
use crate::types::{Embedding, EmbeddingResponse, Usage};
use base64::Engine as _;
use serde::Deserialize;

/// Raw embedding response as it arrives on the wire. `data[].embedding` is either a
/// float array (`encoding_format:"float"`) or a base64 string of little-endian f32 bytes
/// (`encoding_format:"base64"`); we normalise both to `Vec<f32>` (SPEC.md).
#[derive(Deserialize)]
pub(crate) struct RawEmbeddingResponse {
    pub object: String,
    pub data: Vec<RawEmbedding>,
    pub model: String,
    #[serde(default)]
    pub usage: Option<Usage>,
}

#[derive(Deserialize)]
pub(crate) struct RawEmbedding {
    pub object: String,
    #[serde(default)]
    pub index: u32,
    pub embedding: RawEmbeddingData,
}

/// Either the JSON float array or the base64 string form.
#[derive(Deserialize)]
#[serde(untagged)]
pub(crate) enum RawEmbeddingData {
    Floats(Vec<f32>),
    Base64(String),
}

impl RawEmbeddingData {
    fn into_floats(self) -> Result<Vec<f32>> {
        match self {
            RawEmbeddingData::Floats(v) => Ok(v),
            RawEmbeddingData::Base64(s) => decode_base64_f32(&s),
        }
    }
}

impl RawEmbeddingResponse {
    /// Normalise into the public [`EmbeddingResponse`], decoding any base64 payloads.
    pub(crate) fn into_public(self) -> Result<EmbeddingResponse> {
        let data = self
            .data
            .into_iter()
            .map(|e| {
                Ok(Embedding {
                    object: e.object,
                    index: e.index,
                    embedding: e.embedding.into_floats()?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(EmbeddingResponse {
            object: self.object,
            data,
            model: self.model,
            usage: self.usage,
        })
    }
}

/// Decode a base64 string of little-endian f32 bytes into a float vector.
fn decode_base64_f32(s: &str) -> Result<Vec<f32>> {
    let bytes = base64::engine::general_purpose::STANDARD.decode(s)?;
    if bytes.len() % 4 != 0 {
        return Err(Error::Stream(format!(
            "base64 embedding length {} is not a multiple of 4",
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// Map a TTS `response_format` to the `Content-Type` SPEC.md expects, as a fallback when
/// the server omits the header.
pub(crate) fn speech_content_type(response_format: Option<&str>) -> &'static str {
    match response_format.unwrap_or("mp3") {
        "wav" => "audio/wav",
        "opus" => "audio/ogg",
        "aac" => "audio/aac",
        "flac" => "audio/flac",
        "pcm" => "audio/pcm",
        // "mp3" and anything unknown default to mpeg.
        _ => "audio/mpeg",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_base64_le_f32() {
        // [1.0, 2.0] little-endian f32 → base64.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&1.0f32.to_le_bytes());
        bytes.extend_from_slice(&2.0f32.to_le_bytes());
        let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
        let out = decode_base64_f32(&b64).unwrap();
        assert_eq!(out, vec![1.0, 2.0]);
    }

    #[test]
    fn rejects_misaligned_base64() {
        let b64 = base64::engine::general_purpose::STANDARD.encode([0u8, 1, 2]);
        assert!(decode_base64_f32(&b64).is_err());
    }

    #[test]
    fn content_type_table() {
        assert_eq!(speech_content_type(None), "audio/mpeg");
        assert_eq!(speech_content_type(Some("wav")), "audio/wav");
        assert_eq!(speech_content_type(Some("opus")), "audio/ogg");
        assert_eq!(speech_content_type(Some("pcm")), "audio/pcm");
    }
}
