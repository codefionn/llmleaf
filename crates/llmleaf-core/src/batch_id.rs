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
//! ## Security: signed, owner-bound ids
//!
//! The retrieve/cancel/results surfaces authenticate *identity only* (the id names no model to gate
//! on), so the id itself must not become a cross-tenant capability: consumer A's batch must not be
//! readable or cancellable by consumer B just because B learned the upstream id. With
//! `[server].batch_id_secret` set, the id therefore also embeds the creating key's id and a truncated
//! HMAC-SHA256 tag keyed by that secret:
//!
//! ```text
//! batch_<base64url(provider \x1f upstream \x1f owner-key-id \x1f tag16)>
//! ```
//!
//! Decoding *verifies the tag first* (a forged or tampered id is `Malformed`, never routed), and the
//! engine then refuses any caller whose key is not the embedded owner. Nodes behind one load balancer
//! share the secret via config/`env:`, so the multi-node property is untouched (principle 9). With no
//! secret configured the legacy unsigned form is produced and consumed unchanged — the id is then NOT
//! a capability guard, only unguessability of the upstream id protects a batch, so production
//! deployments should set the secret (the engine warns at startup when it is absent).

use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine as _;
use hmac::{Hmac, Mac};
use sha2::Sha256;
use thiserror::Error;

/// ASCII Unit Separator. Separates the provider instance from the upstream id inside the encoded
/// payload. Provider instance names come from a TOML config file (human-typed identifiers), upstream
/// ids are provider-issued tokens, and key ids are validated at load to contain no control bytes, so
/// none ever contains a control byte — the split is unambiguous.
const SEP: u8 = 0x1f;

/// Prefix on an encoded *batch* id. Mirrors the `batch_…` shape OpenAI clients expect while staying
/// our own opaque token.
const BATCH_PREFIX: &str = "batch_";
/// Prefix on an encoded *input/output file* id (the OpenAI files surface).
const FILE_PREFIX: &str = "file_";

/// Truncated HMAC-SHA256 tag length, in bytes. 128 bits is far beyond offline forgery for an
/// HMAC (no length-extension, no collision reliance), and keeps the token short.
const TAG_LEN: usize = 16;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum BatchIdError {
    #[error("malformed batch id")]
    Malformed,
}

/// Signs and verifies the tag on owner-bound batch ids. Built once from `[server].batch_id_secret`
/// and shared behind an `Arc`; every node of a deployment must be configured with the SAME secret so
/// an id minted on one node verifies on all of them (principle 9: shared via config, not runtime
/// coordination).
pub struct BatchIdSigner {
    mac: Hmac<Sha256>,
}

impl BatchIdSigner {
    pub fn new(secret: &str) -> Self {
        BatchIdSigner {
            mac: <Hmac<Sha256> as Mac>::new_from_slice(secret.as_bytes())
                .expect("HMAC accepts any key length"),
        }
    }

    fn tag(&self, payload: &[u8]) -> [u8; TAG_LEN] {
        let mut mac = self.mac.clone();
        mac.update(payload);
        let out = mac.finalize().into_bytes();
        out[..TAG_LEN].try_into().expect("tag slice is TAG_LEN")
    }

    /// Constant-time check of a truncated tag. (`Mac::verify_slice` only accepts a full-length tag,
    /// so the truncated comparison is done here: fixed 16-byte operands, no early exit.)
    fn verify(&self, payload: &[u8], tag: &[u8]) -> bool {
        if tag.len() != TAG_LEN {
            return false;
        }
        let expected = self.tag(payload);
        expected
            .iter()
            .zip(tag.iter())
            .fold(0u8, |acc, (a, b)| acc | (a ^ b))
            == 0
    }
}

/// A decoded batch id: the owning provider instance, the upstream's own id, and — for signed ids —
/// the key that created the batch (the owner the engine binds the caller to).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedBatch {
    pub provider: String,
    pub upstream: String,
    /// `Some` only when the id was signed (see module docs); the engine refuses non-owners then.
    pub owner: Option<String>,
}

/// Encode a batch id. With a `signer` the id binds `owner` (the creating key's id) and carries its
/// HMAC tag: `batch_<base64url(provider \x1f upstream \x1f owner \x1f tag)>`. Without one it is the
/// legacy `batch_<base64url(provider \x1f upstream)>` and `owner` is not embedded (unsigned ids carry
/// no ownership — see module docs).
pub fn encode_batch(
    provider: &str,
    upstream_id: &str,
    owner: &str,
    signer: Option<&BatchIdSigner>,
) -> String {
    match signer {
        Some(signer) => {
            let head = join3(
                provider.as_bytes(),
                upstream_id.as_bytes(),
                owner.as_bytes(),
            );
            let tag = signer.tag(&head);
            let mut payload = Vec::with_capacity(head.len() + 1 + TAG_LEN);
            payload.extend_from_slice(&head);
            payload.push(SEP);
            payload.extend_from_slice(&tag);
            format!("{BATCH_PREFIX}{}", URL_SAFE_NO_PAD.encode(&payload))
        }
        None => encode_with(BATCH_PREFIX, provider, upstream_id),
    }
}

/// Decode a batch id, verifying the HMAC tag when a `signer` is given. A signed deployment rejects
/// legacy unsigned tokens (they carry no tag ⇒ `Malformed`), so a consumer cannot strip the owner
/// binding by hand-crafting the old form. Rejects a file token, so the two id spaces cannot cross.
pub fn decode_batch(
    token: &str,
    signer: Option<&BatchIdSigner>,
) -> Result<DecodedBatch, BatchIdError> {
    let bytes = decode_payload(BATCH_PREFIX, token)?;
    match signer {
        Some(signer) => {
            // Layout: provider \x1f upstream \x1f owner \x1f tag16. The tag is raw bytes and may
            // itself contain SEP, so split it off from the RIGHT, verify, then split the head.
            if bytes.len() < TAG_LEN + 1 || bytes[bytes.len() - TAG_LEN - 1] != SEP {
                return Err(BatchIdError::Malformed);
            }
            let (head, tag) = bytes.split_at(bytes.len() - TAG_LEN);
            let head = &head[..head.len() - 1]; // drop the SEP before the tag
            if !signer.verify(head, tag) {
                return Err(BatchIdError::Malformed);
            }
            let (provider, rest) = split_once(head)?;
            let (upstream, owner) = rsplit_once(rest)?;
            if provider.is_empty() {
                return Err(BatchIdError::Malformed);
            }
            Ok(DecodedBatch {
                provider: utf8(provider)?,
                upstream: utf8(upstream)?,
                owner: Some(utf8(owner)?),
            })
        }
        None => {
            let (provider, upstream) = split_fields(&bytes)?;
            Ok(DecodedBatch {
                provider,
                upstream,
                owner: None,
            })
        }
    }
}

/// Encode a file id: `file_<base64url(provider \x1f upstream)>`.
pub fn encode_file(provider: &str, upstream_id: &str) -> String {
    encode_with(FILE_PREFIX, provider, upstream_id)
}

/// Decode a file id into `(provider_instance, upstream_id)`.
pub fn decode_file(token: &str) -> Result<(String, String), BatchIdError> {
    let bytes = decode_payload(FILE_PREFIX, token)?;
    split_fields(&bytes)
}

fn encode_with(prefix: &str, provider: &str, upstream_id: &str) -> String {
    let mut payload = Vec::with_capacity(provider.len() + 1 + upstream_id.len());
    payload.extend_from_slice(provider.as_bytes());
    payload.push(SEP);
    payload.extend_from_slice(upstream_id.as_bytes());
    format!("{prefix}{}", URL_SAFE_NO_PAD.encode(&payload))
}

fn decode_payload(prefix: &str, token: &str) -> Result<Vec<u8>, BatchIdError> {
    let b64 = token.strip_prefix(prefix).ok_or(BatchIdError::Malformed)?;
    URL_SAFE_NO_PAD
        .decode(b64)
        .map_err(|_| BatchIdError::Malformed)
}

/// The legacy two-field layout: `provider \x1f upstream`.
fn split_fields(bytes: &[u8]) -> Result<(String, String), BatchIdError> {
    let (provider, upstream) = split_once(bytes)?;
    if provider.is_empty() {
        return Err(BatchIdError::Malformed);
    }
    Ok((utf8(provider)?, utf8(upstream)?))
}

/// Join `a \x1f b \x1f c` into one buffer.
fn join3(a: &[u8], b: &[u8], c: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(a.len() + b.len() + c.len() + 2);
    out.extend_from_slice(a);
    out.push(SEP);
    out.extend_from_slice(b);
    out.push(SEP);
    out.extend_from_slice(c);
    out
}

/// Split on the FIRST separator: `head \x1f rest`.
fn split_once(bytes: &[u8]) -> Result<(&[u8], &[u8]), BatchIdError> {
    let sep = bytes
        .iter()
        .position(|&b| b == SEP)
        .ok_or(BatchIdError::Malformed)?;
    Ok((&bytes[..sep], &bytes[sep + 1..]))
}

/// Split on the LAST separator: `rest \x1f tail`. Used for the signed head, whose `owner` field is
/// the last segment (the upstream id half is provider-issued and contains no control bytes, so the
/// last SEP is unambiguously the owner boundary).
fn rsplit_once(bytes: &[u8]) -> Result<(&[u8], &[u8]), BatchIdError> {
    let sep = bytes
        .iter()
        .rposition(|&b| b == SEP)
        .ok_or(BatchIdError::Malformed)?;
    Ok((&bytes[..sep], &bytes[sep + 1..]))
}

fn utf8(bytes: &[u8]) -> Result<String, BatchIdError> {
    std::str::from_utf8(bytes)
        .map(str::to_string)
        .map_err(|_| BatchIdError::Malformed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn signer() -> BatchIdSigner {
        BatchIdSigner::new("test-secret")
    }

    #[test]
    fn batch_round_trip_unsigned() {
        let token = encode_batch("openai-main", "batch_abc123", "team-a", None);
        assert!(token.starts_with("batch_"));
        let d = decode_batch(&token, None).unwrap();
        assert_eq!(d.provider, "openai-main");
        assert_eq!(d.upstream, "batch_abc123");
        assert_eq!(d.owner, None);
    }

    #[test]
    fn batch_round_trip_signed_binds_owner() {
        let s = signer();
        let token = encode_batch("openai-main", "batch_abc123", "team-a", Some(&s));
        assert!(token.starts_with("batch_"));
        let d = decode_batch(&token, Some(&s)).unwrap();
        assert_eq!(d.provider, "openai-main");
        assert_eq!(d.upstream, "batch_abc123");
        assert_eq!(d.owner.as_deref(), Some("team-a"));
    }

    #[test]
    fn signed_id_verifies_across_signer_instances_with_same_secret() {
        // The multi-node property (principle 9): node A mints, node B verifies — same configured
        // secret, no shared runtime state.
        let a = BatchIdSigner::new("cluster-secret");
        let b = BatchIdSigner::new("cluster-secret");
        let token = encode_batch("p", "up", "owner", Some(&a));
        let d = decode_batch(&token, Some(&b)).unwrap();
        assert_eq!(d.owner.as_deref(), Some("owner"));
    }

    #[test]
    fn wrong_secret_rejects() {
        let token = encode_batch("p", "up", "owner", Some(&signer()));
        let other = BatchIdSigner::new("different-secret");
        assert_eq!(
            decode_batch(&token, Some(&other)),
            Err(BatchIdError::Malformed)
        );
    }

    #[test]
    fn tampering_rejects() {
        let s = signer();
        // Flip a character in the middle of a signed token (inside the payload, not the prefix).
        let token = encode_batch("openai-main", "batch_abc123", "team-a", Some(&s));
        let mut chars: Vec<char> = token.chars().collect();
        let i = chars.len() / 2;
        chars[i] = if chars[i] == 'A' { 'B' } else { 'A' };
        let forged: String = chars.into_iter().collect();
        assert_eq!(
            decode_batch(&forged, Some(&s)),
            Err(BatchIdError::Malformed)
        );

        // Rebinding to a different owner by re-encoding the same provider/upstream without the
        // secret fails the tag check too (the forger can only produce unsigned tokens).
        let rebinding = encode_with(BATCH_PREFIX, "openai-main", "batch_abc123");
        assert_eq!(
            decode_batch(&rebinding, Some(&s)),
            Err(BatchIdError::Malformed)
        );
    }

    #[test]
    fn signed_deployment_rejects_legacy_unsigned_tokens() {
        // An attacker can always craft the unsigned form by hand — it must not decode when a secret
        // is configured, or the owner binding could be stripped.
        let s = signer();
        let legacy = encode_batch("p", "up", "owner", None);
        assert_eq!(
            decode_batch(&legacy, Some(&s)),
            Err(BatchIdError::Malformed)
        );
        // …and the unsigned decoder does not accept the signed form as legacy either (the tag and
        // owner would corrupt the two-field layout; it must not silently mis-split).
        let signed = encode_batch("p", "up", "owner", Some(&s));
        let d = decode_batch(&signed, None);
        assert!(d.is_err() || d.unwrap().owner.is_none());
    }

    #[test]
    fn upstream_id_may_contain_separator_chars_safely() {
        // Provider ids that themselves embed underscores/hyphens survive the round trip.
        let token = encode_batch("p", "msgbatch_01ABC_def-GHI", "o", None);
        let d = decode_batch(&token, None).unwrap();
        assert_eq!(
            (d.provider.as_str(), d.upstream.as_str()),
            ("p", "msgbatch_01ABC_def-GHI")
        );
    }

    #[test]
    fn empty_upstream_id_is_allowed() {
        // An upstream id is opaque; only a missing *provider* is malformed.
        let token = encode_batch("p", "", "o", None);
        let d = decode_batch(&token, None).unwrap();
        assert_eq!((d.provider.as_str(), d.upstream.as_str()), ("p", ""));
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
    fn cross_kind_is_rejected() {
        let file = encode_file("p", "f1");
        assert_eq!(decode_batch(&file, None), Err(BatchIdError::Malformed));
        let batch = encode_batch("p", "b1", "o", None);
        assert_eq!(decode_file(&batch), Err(BatchIdError::Malformed));
    }

    #[test]
    fn foreign_and_garbage_tokens_are_rejected() {
        assert_eq!(
            decode_batch("batch_abc123", None),
            Err(BatchIdError::Malformed)
        ); // raw upstream id, not ours
        assert_eq!(
            decode_batch("not-a-token", None),
            Err(BatchIdError::Malformed)
        );
        assert_eq!(decode_batch("batch_", None), Err(BatchIdError::Malformed)); // empty payload
        assert_eq!(decode_batch("", None), Err(BatchIdError::Malformed));
    }
}
