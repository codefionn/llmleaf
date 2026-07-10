//! The rerank compat surface (`POST /v1/rerank`).
//!
//! In: a Cohere/Jina/Voyage/OpenRouter-style rerank request JSON → canonical [`RerankRequest`]. Unknown
//! fields ride through in [`RerankRequest::extra`] (principle 7). Out: a canonical [`RerankResponse`] →
//! the results-list wire shape those clients read (`{ object: "list", model, results: [{ index,
//! relevance_score, document? }], usage }`) — the one dialect every rerank consumer already speaks
//! (there is no OpenAI rerank surface to mirror; fidelity beats convenience).

use llmleaf_model::{ModelError, RerankDocument, RerankRequest, RerankResponse};
use serde::Serialize;
use serde_json::Value;

fn mapping(msg: impl Into<String>) -> ModelError {
    ModelError::Mapping(msg.into())
}

// ---------------------------------------------------------------------------------------------
// Inbound: rerank request JSON -> canonical RerankRequest
// ---------------------------------------------------------------------------------------------

pub fn parse_rerank_request(value: Value) -> Result<RerankRequest, ModelError> {
    let Value::Object(mut obj) = value else {
        return Err(mapping("request body must be a JSON object"));
    };

    let model = match obj.remove("model") {
        Some(Value::String(s)) => s,
        _ => return Err(mapping("`model` is required and must be a string")),
    };
    let query = match obj.remove("query") {
        Some(Value::String(s)) => s,
        _ => return Err(mapping("`query` is required and must be a string")),
    };
    let documents = parse_documents(obj.remove("documents"))?;
    let top_n = obj
        .remove("top_n")
        .and_then(|v| v.as_u64())
        .map(|n| n as u32);
    let return_documents = obj.remove("return_documents").and_then(|v| v.as_bool());

    // Whatever else the consumer sent (`max_tokens_per_doc`, `truncation`, provider knobs) rides
    // through untouched.
    let extra = obj;

    Ok(RerankRequest {
        model,
        query,
        documents,
        top_n,
        return_documents,
        extra,
    })
}

/// Rerank documents are a non-empty array whose items are each a plain string or a structured object
/// (the OpenRouter/multimodal `{ text?, image? }` shape). A structured object is preserved verbatim as
/// [`RerankDocument::Rich`] (principle 7: never silently drop or coerce what we don't model). A
/// non-string, non-object item is rejected explicitly rather than dropped.
fn parse_documents(value: Option<Value>) -> Result<Vec<RerankDocument>, ModelError> {
    match value {
        Some(Value::Array(items)) if !items.is_empty() => items
            .into_iter()
            .map(|v| match v {
                Value::String(s) => Ok(RerankDocument::Text(s)),
                Value::Object(obj) => Ok(RerankDocument::Rich(obj)),
                _ => Err(mapping("`documents` items must be strings or objects")),
            })
            .collect(),
        _ => Err(mapping(
            "`documents` is required and must be a non-empty array of strings or objects",
        )),
    }
}

// ---------------------------------------------------------------------------------------------
// Outbound: canonical RerankResponse -> rerank wire format
// ---------------------------------------------------------------------------------------------

/// Borrowed view of the `{ model, object, results, usage }` rerank response. Same technique as the
/// embeddings `EmbeddingListView` (principle 1): serialising it straight to bytes never builds a
/// `serde_json::Value` tree.
#[derive(Serialize)]
pub struct RerankListView<'a> {
    model: &'a str,
    object: &'static str,
    results: Vec<RerankResultView<'a>>,
    usage: RerankUsageView,
}

/// Per-result view: the input index, the relevance score, and the echoed document when the consumer
/// asked for it (a borrowed [`Value`], so a multimodal document round-trips with no clone).
#[derive(Serialize)]
struct RerankResultView<'a> {
    index: u32,
    relevance_score: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    document: Option<&'a Value>,
}

/// Usage view for rerank. There is no completion side, so only the billed count is emitted:
/// `total_tokens` (tokens for token-billed upstreams, Cohere's search-unit count otherwise), plus the
/// edge-priced `cost_usd` when the bundled dataset knows a rate.
#[derive(Serialize)]
struct RerankUsageView {
    #[serde(skip_serializing_if = "Option::is_none")]
    cost_usd: Option<f64>,
    total_tokens: u64,
}

/// Map the canonical response to the results-list wire object. Returns a borrowed [`RerankListView`]
/// rather than a `serde_json::Value` so the caller serialises straight to bytes (principle 1).
pub fn response_to_wire(resp: &RerankResponse) -> RerankListView<'_> {
    let results = resp
        .results
        .iter()
        .map(|r| RerankResultView {
            index: r.index,
            relevance_score: r.relevance_score,
            document: r.document.as_ref(),
        })
        .collect();

    RerankListView {
        model: &resp.model,
        object: "list",
        results,
        usage: RerankUsageView {
            cost_usd: resp.usage.cost_usd,
            total_tokens: resp.usage.total_tokens,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llmleaf_model::{RerankResult, Usage};
    use serde_json::json;

    #[test]
    fn parses_string_documents() {
        let req = parse_rerank_request(json!({
            "model": "rerank-v3.5",
            "query": "what is the capital of france?",
            "documents": ["paris is the capital", "berlin is in germany"],
            "top_n": 1,
        }))
        .unwrap();
        assert_eq!(req.model, "rerank-v3.5");
        assert_eq!(req.query, "what is the capital of france?");
        assert_eq!(req.documents.len(), 2);
        assert_eq!(req.documents[0].as_text(), Some("paris is the capital"));
        assert_eq!(req.top_n, Some(1));
    }

    #[test]
    fn preserves_structured_documents() {
        let req = parse_rerank_request(json!({
            "model": "m",
            "query": "q",
            "documents": [{ "text": "hi", "image": "data:..." }],
        }))
        .unwrap();
        assert!(matches!(req.documents[0], RerankDocument::Rich(_)));
        assert_eq!(req.documents[0].as_text(), Some("hi"));
    }

    #[test]
    fn unknown_fields_ride_in_extra() {
        let req = parse_rerank_request(json!({
            "model": "m",
            "query": "q",
            "documents": ["a"],
            "max_tokens_per_doc": 512,
        }))
        .unwrap();
        assert_eq!(req.extra.get("max_tokens_per_doc"), Some(&json!(512)));
    }

    #[test]
    fn rejects_missing_or_empty_documents() {
        assert!(parse_rerank_request(json!({ "model": "m", "query": "q" })).is_err());
        assert!(
            parse_rerank_request(json!({ "model": "m", "query": "q", "documents": [] })).is_err()
        );
        assert!(
            parse_rerank_request(json!({ "model": "m", "query": "q", "documents": [1, 2] }))
                .is_err()
        );
    }

    #[test]
    fn response_shape_and_document_echo() {
        let resp = RerankResponse {
            model: "m".into(),
            results: vec![
                RerankResult {
                    index: 1,
                    relevance_score: 0.98,
                    document: Some(json!("berlin")),
                },
                RerankResult {
                    index: 0,
                    relevance_score: 0.12,
                    document: None,
                },
            ],
            usage: Usage {
                total_tokens: 26,
                cost_usd: Some(0.00005),
                ..Default::default()
            },
        };
        let v = serde_json::to_value(response_to_wire(&resp)).unwrap();
        assert_eq!(v["object"], "list");
        assert_eq!(v["model"], "m");
        assert_eq!(v["results"][0]["index"], 1);
        // f32 → JSON f64 isn't bit-exact; compare with tolerance.
        assert!((v["results"][0]["relevance_score"].as_f64().unwrap() - 0.98).abs() < 1e-6);
        assert_eq!(v["results"][0]["document"], "berlin");
        // A result without an echoed document omits the field entirely.
        assert!(v["results"][1].get("document").is_none());
        assert_eq!(v["usage"]["total_tokens"], 26);
        assert_eq!(v["usage"]["cost_usd"], 0.00005);
    }
}
