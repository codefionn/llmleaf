//! Opaque, self-describing batch/file ids — the mechanism that keeps batch stateless and multi-node
//! safe (SOUL.md principles 5, 9).
//!
//! A batch is an async job whose state lives at the provider's upstream, not in llmleaf. So the id we
//! hand a consumer must carry, by itself, everything any node needs to serve a later poll / cancel /
//! results call: *which provider instance owns the job* and *the provider's own id for it*. We encode
//! exactly that — `<provider-instance>\x1f<upstream-id>` — as URL-safe base64 behind a short prefix.
//!
//! Because the routing lives in the id, never in node-local memory, a batch created on node A is fully
//! serviceable from node B with no shared state and no inter-node chatter (principle 9). The id is
//! opaque to the consumer, which simply passes it back verbatim.
//!
//! Security: the ids are *not* a capability. All of a node's virtual consumer keys already share one
//! real upstream credential, and every encodable provider instance is operator-configured and trusted,
//! so an id reveals nothing a consumer could not already reach and forging the provider half only
//! re-targets an operator-trusted instance. The upstream id half is the only real secret (guessing
//! another consumer's job id), exactly as it is at the provider directly. An optional HMAC signature
//! (keyed by `[server].batch_id_secret`) can be layered on later to make the token tamper-evident; it
//! is deliberately out of v1 to keep decode a pure, allocation-light transform.

use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine as _;
use thiserror::Error;

/// ASCII Unit Separator. Separates the provider instance from the upstream id inside the encoded
/// payload. Provider instance names come from a TOML config file (human-typed identifiers) and
/// upstream ids are provider-issued tokens, so neither ever contains a control byte — the split is
/// unambiguous.
const SEP: u8 = 0x1f;

/// Prefix on an encoded *batch* id. Mirrors the `batch_…` shape OpenAI clients expect while staying
/// our own opaque token.
const BATCH_PREFIX: &str = "batch_";
/// Prefix on an encoded *input/output file* id (the OpenAI files surface).
const FILE_PREFIX: &str = "file_";

#[derive(Debug, Error, PartialEq, Eq)]
pub enum BatchIdError {
    #[error("malformed batch id")]
    Malformed,
}

/// Encode a batch id: `batch_<base64url(provider \x1f upstream)>`.
pub fn encode_batch(provider: &str, upstream_id: &str) -> String {
    encode_with(BATCH_PREFIX, provider, upstream_id)
}

/// Encode a file id: `file_<base64url(provider \x1f upstream)>`.
pub fn encode_file(provider: &str, upstream_id: &str) -> String {
    encode_with(FILE_PREFIX, provider, upstream_id)
}

/// Decode a batch id into `(provider_instance, upstream_id)`. Rejects a token that is not a batch
/// token (e.g. a file token), so the two id spaces cannot be crossed.
pub fn decode_batch(token: &str) -> Result<(String, String), BatchIdError> {
    decode_with(BATCH_PREFIX, token)
}

/// Decode a file id into `(provider_instance, upstream_id)`.
pub fn decode_file(token: &str) -> Result<(String, String), BatchIdError> {
    decode_with(FILE_PREFIX, token)
}

fn encode_with(prefix: &str, provider: &str, upstream_id: &str) -> String {
    let mut payload = Vec::with_capacity(provider.len() + 1 + upstream_id.len());
    payload.extend_from_slice(provider.as_bytes());
    payload.push(SEP);
    payload.extend_from_slice(upstream_id.as_bytes());
    format!("{prefix}{}", URL_SAFE_NO_PAD.encode(&payload))
}

fn decode_with(prefix: &str, token: &str) -> Result<(String, String), BatchIdError> {
    let b64 = token.strip_prefix(prefix).ok_or(BatchIdError::Malformed)?;
    let bytes = URL_SAFE_NO_PAD
        .decode(b64)
        .map_err(|_| BatchIdError::Malformed)?;
    let sep = bytes
        .iter()
        .position(|&b| b == SEP)
        .ok_or(BatchIdError::Malformed)?;
    let provider = std::str::from_utf8(&bytes[..sep]).map_err(|_| BatchIdError::Malformed)?;
    let upstream = std::str::from_utf8(&bytes[sep + 1..]).map_err(|_| BatchIdError::Malformed)?;
    if provider.is_empty() {
        return Err(BatchIdError::Malformed);
    }
    Ok((provider.to_string(), upstream.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_round_trip() {
        let token = encode_batch("openai-main", "batch_abc123");
        assert!(token.starts_with("batch_"));
        assert_eq!(
            decode_batch(&token).unwrap(),
            ("openai-main".to_string(), "batch_abc123".to_string())
        );
    }

    #[test]
    fn file_round_trip() {
        let token = encode_file("mistral-eu", "00000000-1111-2222-3333-444444444444");
        assert!(token.starts_with("file_"));
        assert_eq!(
            decode_file(&token).unwrap(),
            (
                "mistral-eu".to_string(),
                "00000000-1111-2222-3333-444444444444".to_string()
            )
        );
    }

    #[test]
    fn upstream_id_may_contain_separator_chars_safely() {
        // Provider ids that themselves embed underscores/hyphens survive the round trip.
        let token = encode_batch("p", "msgbatch_01ABC_def-GHI");
        assert_eq!(
            decode_batch(&token).unwrap(),
            ("p".to_string(), "msgbatch_01ABC_def-GHI".to_string())
        );
    }

    #[test]
    fn empty_upstream_id_is_allowed() {
        // An upstream id is opaque; only a missing *provider* is malformed.
        let token = encode_batch("p", "");
        assert_eq!(
            decode_batch(&token).unwrap(),
            ("p".to_string(), String::new())
        );
    }

    #[test]
    fn cross_kind_is_rejected() {
        let file = encode_file("p", "f1");
        assert_eq!(decode_batch(&file), Err(BatchIdError::Malformed));
        let batch = encode_batch("p", "b1");
        assert_eq!(decode_file(&batch), Err(BatchIdError::Malformed));
    }

    #[test]
    fn foreign_and_garbage_tokens_are_rejected() {
        assert_eq!(decode_batch("batch_abc123"), Err(BatchIdError::Malformed)); // raw upstream id, not ours
        assert_eq!(decode_batch("not-a-token"), Err(BatchIdError::Malformed));
        assert_eq!(decode_batch("batch_"), Err(BatchIdError::Malformed)); // empty payload
        assert_eq!(decode_batch(""), Err(BatchIdError::Malformed));
    }
}
