//! The OpenAI Embeddings compat surface (`POST /v1/embeddings`).
//!
//! In: an OpenAI embeddings request JSON → canonical [`EmbeddingRequest`]. Unknown fields ride through
//! in [`EmbeddingRequest::extra`] (principle 7). Out: a canonical [`EmbeddingResponse`] → the OpenAI
//! `{ object: "list", data: [...] }` shape, honoring the requested `encoding_format` (`float` |
//! `base64`) exactly as OpenAI documents (decision filter: fidelity beats convenience).

use llmleaf_model::{EmbeddingRequest, EmbeddingResponse, ModelError};
use serde::Serialize;
use serde_json::Value;

fn mapping(msg: impl Into<String>) -> ModelError {
    ModelError::Mapping(msg.into())
}

// ---------------------------------------------------------------------------------------------
// Inbound: OpenAI embeddings request JSON -> canonical EmbeddingRequest
// ---------------------------------------------------------------------------------------------

pub fn parse_embedding_request(value: Value) -> Result<EmbeddingRequest, ModelError> {
    let Value::Object(mut obj) = value else {
        return Err(mapping("request body must be a JSON object"));
    };

    let model = match obj.remove("model") {
        Some(Value::String(s)) => s,
        _ => return Err(mapping("`model` is required and must be a string")),
    };

    let input = parse_input(obj.remove("input"))?;
    let dimensions = obj
        .remove("dimensions")
        .and_then(|v| v.as_u64())
        .map(|n| n as u32);
    let encoding_format = obj
        .remove("encoding_format")
        .and_then(|v| v.as_str().map(str::to_owned));

    // Whatever else the consumer sent (`user`, provider knobs) rides through untouched.
    let extra = obj;

    Ok(EmbeddingRequest {
        model,
        input,
        dimensions,
        encoding_format,
        extra,
    })
}

/// OpenAI accepts a bare string or an array of strings. Token-id arrays (integers) are a valid OpenAI
/// shape but have no provider-portable canonical form here, so they are rejected explicitly rather
/// than silently coerced (principle 7: never silently mutate).
fn parse_input(value: Option<Value>) -> Result<Vec<String>, ModelError> {
    match value {
        Some(Value::String(s)) => Ok(vec![s]),
        Some(Value::Array(items)) => items
            .into_iter()
            .map(|v| match v {
                Value::String(s) => Ok(s),
                _ => Err(mapping(
                    "`input` array must contain strings (token-id input is unsupported)",
                )),
            })
            .collect(),
        _ => Err(mapping(
            "`input` is required and must be a string or array of strings",
        )),
    }
}

// ---------------------------------------------------------------------------------------------
// Outbound: canonical EmbeddingResponse -> OpenAI wire format
// ---------------------------------------------------------------------------------------------

/// Borrowed view of the OpenAI `{ object: "list", data, model, usage }` response. Same technique as
/// the streaming `ChunkFrame`/collected `CompletionView` (principle 1): serialising it straight to
/// bytes never builds a `serde_json::Value` tree. Fields are declared in the sorted key order
/// `serde_json::Map` (a `BTreeMap`) emits — `data, model, object, usage` — so the wire bytes are
/// byte-identical to the old `Value`-building mapping.
#[derive(Serialize)]
pub struct EmbeddingListView<'a> {
    data: Vec<EmbeddingItemView<'a>>,
    model: &'a str,
    object: &'static str,
    usage: EmbUsageView,
}

/// Per-item view. Sorted order: `embedding, index, object`. The `embedding` field is a borrowed float
/// slice or an owned base64 `String` depending on the requested `encoding_format`.
#[derive(Serialize)]
struct EmbeddingItemView<'a> {
    embedding: EmbeddingData<'a>,
    index: u32,
    object: &'static str,
}

/// Untagged so each variant serialises as just its inner value (an array of `f32`, or a JSON string) —
/// matching the old `Value::Array(..)` vs `Value::String(..)`.
#[derive(Serialize)]
#[serde(untagged)]
enum EmbeddingData<'a> {
    /// Serialises a borrowed slice of `f32` directly — zero allocation.
    Float(&'a [f32]),
    /// The irreducible per-vector base64 `String` (built exactly as before).
    Base64(String),
}

/// Usage view for embeddings. Distinct from `openai::UsageFrame`: the old embeddings `usage_json`
/// emitted only `prompt_tokens`/`total_tokens` (+ optional `cost_usd`) and never `completion_tokens`,
/// so reusing `UsageFrame` here would change the bytes. Sorted order: `cost_usd?, prompt_tokens,
/// total_tokens`.
#[derive(Serialize)]
struct EmbUsageView {
    #[serde(skip_serializing_if = "Option::is_none")]
    cost_usd: Option<f64>,
    prompt_tokens: u64,
    total_tokens: u64,
}

/// Map the canonical response to the OpenAI `{ object: "list", data, model, usage }` object. When the
/// consumer asked for `encoding_format: "base64"`, each vector is the raw little-endian `f32` bytes,
/// base64-encoded — OpenAI's exact representation. Returns a borrowed [`EmbeddingListView`] rather than
/// a `serde_json::Value` so the caller serialises straight to bytes (principle 1).
pub fn response_to_openai<'a>(
    resp: &'a EmbeddingResponse,
    encoding_format: Option<&str>,
) -> EmbeddingListView<'a> {
    let base64 = encoding_format == Some("base64");
    let data = resp
        .embeddings
        .iter()
        .map(|e| EmbeddingItemView {
            embedding: if base64 {
                EmbeddingData::Base64(encode_f32_base64(&e.vector))
            } else {
                EmbeddingData::Float(&e.vector)
            },
            index: e.index,
            object: "embedding",
        })
        .collect();

    EmbeddingListView {
        data,
        model: &resp.model,
        object: "list",
        usage: EmbUsageView {
            cost_usd: resp.usage.cost_usd,
            prompt_tokens: resp.usage.prompt_tokens,
            total_tokens: resp.usage.total_tokens,
        },
    }
}

/// Encode `f32`s as OpenAI's base64 embedding: little-endian IEEE-754 bytes, standard base64.
fn encode_f32_base64(vector: &[f32]) -> String {
    let mut bytes = Vec::with_capacity(vector.len() * 4);
    for f in vector {
        bytes.extend_from_slice(&f.to_le_bytes());
    }
    base64_encode(&bytes)
}

/// Minimal standard-alphabet base64 (RFC 4648) encoder. Hand-rolled to keep faith with SOUL's
/// zero-dependency minimalism — a dozen lines beat a crate for one call site.
fn base64_encode(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = *chunk.get(1).unwrap_or(&0) as u32;
        let b2 = *chunk.get(2).unwrap_or(&0) as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(ALPHABET[((n >> 18) & 63) as usize] as char);
        out.push(ALPHABET[((n >> 12) & 63) as usize] as char);
        out.push(if chunk.len() > 1 {
            ALPHABET[((n >> 6) & 63) as usize] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            ALPHABET[(n & 63) as usize] as char
        } else {
            '='
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use llmleaf_model::{Embedding, Usage};
    use serde_json::{json, Map};

    #[test]
    fn parses_string_and_array_input() {
        let one = parse_embedding_request(json!({ "model": "m", "input": "hi" })).unwrap();
        assert_eq!(one.input, vec!["hi".to_string()]);
        let many = parse_embedding_request(
            json!({ "model": "m", "input": ["a", "b"], "dimensions": 256 }),
        )
        .unwrap();
        assert_eq!(many.input.len(), 2);
        assert_eq!(many.dimensions, Some(256));
    }

    #[test]
    fn rejects_token_array_input() {
        assert!(parse_embedding_request(json!({ "model": "m", "input": [1, 2, 3] })).is_err());
    }

    #[test]
    fn unknown_fields_ride_in_extra() {
        let req =
            parse_embedding_request(json!({ "model": "m", "input": "hi", "user": "u-1" })).unwrap();
        assert_eq!(req.extra.get("user"), Some(&json!("u-1")));
    }

    #[test]
    fn float_response_shape() {
        let resp = EmbeddingResponse {
            model: "m".into(),
            embeddings: vec![Embedding {
                index: 0,
                vector: vec![0.5, -0.5],
            }],
            usage: Usage {
                prompt_tokens: 3,
                total_tokens: 3,
                cost_usd: Some(0.0001),
                ..Default::default()
            },
        };
        // `to_value` for the readable assertions only; the hot path serialises the view to bytes.
        let v = serde_json::to_value(response_to_openai(&resp, Some("float"))).unwrap();
        assert_eq!(v["object"], "list");
        assert_eq!(v["data"][0]["index"], 0);
        assert_eq!(v["data"][0]["embedding"][0], 0.5);
        assert_eq!(v["usage"]["prompt_tokens"], 3);
        assert_eq!(v["usage"]["cost_usd"], 0.0001);
    }

    #[test]
    fn base64_response_encodes_le_f32() {
        let resp = EmbeddingResponse {
            model: "m".into(),
            embeddings: vec![Embedding {
                index: 0,
                vector: vec![1.0f32],
            }],
            usage: Usage::default(),
        };
        let v = serde_json::to_value(response_to_openai(&resp, Some("base64"))).unwrap();
        // 1.0f32 little-endian = 00 00 80 3F -> base64 "AACAPw==".
        assert_eq!(v["data"][0]["embedding"], "AACAPw==");
    }

    /// The equivalence guarantee: the borrowed view serialises to the *exact* bytes the old
    /// `Value`-building path produced, for both the float and base64 encodings and with `cost_usd`
    /// present and absent.
    #[test]
    fn embedding_view_bytes_match_legacy() {
        // The pre-refactor `Value`-building path, pasted here verbatim as the legacy oracle.
        fn legacy(resp: &EmbeddingResponse, encoding_format: Option<&str>) -> Value {
            let base64 = encoding_format == Some("base64");
            let data: Vec<Value> = resp
                .embeddings
                .iter()
                .map(|e| {
                    let embedding = if base64 {
                        Value::String(encode_f32_base64(&e.vector))
                    } else {
                        Value::Array(e.vector.iter().map(|f| json!(f)).collect())
                    };
                    json!({ "object": "embedding", "index": e.index, "embedding": embedding })
                })
                .collect();
            let mut usage = json!({
                "prompt_tokens": resp.usage.prompt_tokens,
                "total_tokens": resp.usage.total_tokens,
            });
            if let Some(cost) = resp.usage.cost_usd {
                usage
                    .as_object_mut()
                    .unwrap()
                    .insert("cost_usd".into(), json!(cost));
            }
            let mut obj = Map::new();
            obj.insert("object".into(), json!("list"));
            obj.insert("data".into(), Value::Array(data));
            obj.insert("model".into(), json!(resp.model));
            obj.insert("usage".into(), usage);
            Value::Object(obj)
        }

        let with_cost = EmbeddingResponse {
            model: "m".into(),
            embeddings: vec![
                Embedding {
                    index: 0,
                    vector: vec![0.5, -0.5, 1.25],
                },
                Embedding {
                    index: 1,
                    vector: vec![2.0, 3.5],
                },
            ],
            usage: Usage {
                prompt_tokens: 4,
                total_tokens: 4,
                cost_usd: Some(0.0001),
                ..Default::default()
            },
        };
        let without_cost = EmbeddingResponse {
            usage: Usage {
                prompt_tokens: 4,
                total_tokens: 4,
                cost_usd: None,
                ..Default::default()
            },
            ..with_cost.clone()
        };

        for resp in [&with_cost, &without_cost] {
            for fmt in [Some("float"), Some("base64")] {
                assert_eq!(
                    serde_json::to_vec(&response_to_openai(resp, fmt)).unwrap(),
                    serde_json::to_vec(&legacy(resp, fmt)).unwrap(),
                    "mismatch for fmt {fmt:?}",
                );
            }
        }
    }
}
