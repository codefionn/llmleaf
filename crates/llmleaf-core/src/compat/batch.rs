//! The batch compat surface — llmleaf's asynchronous batch dialect (SOUL.md principle 3).
//!
//! Consumers already speak the OpenAI chat dialect here, so a batch is expressed as an *inline* array
//! of OpenAI-shaped chat requests, each tagged with a `custom_id`:
//!
//! ```json
//! { "requests": [ { "custom_id": "a", "body": { "model": "gpt-4o", "messages": [ … ] } }, … ] }
//! ```
//!
//! Inline (single-call) submission is deliberate: it is the one shape that maps statelessly to *every*
//! upstream batch API — Anthropic (native inline), OpenAI and Mistral (the provider builds a JSONL file
//! and submits it, entirely inside one call, holding nothing afterward). OpenAI's own two-call
//! file-upload dialect (`/v1/files` + `input_file_id`) is a separate surface (a future increment) whose
//! file step can only target file-capable upstreams; it is omitted here so the core holds no state
//! between calls (principle 5) and any node can serve a batch from its opaque id alone (principle 9).
//!
//! Each request `body` is parsed by the very same [`openai::parse_chat_request`] the live chat surface
//! uses, and each result body is rendered by the same [`openai::response_to_openai`] — there is no
//! batch-specific request/response mapping, only this envelope (principle 3: one internal model).

use std::borrow::Cow;

use llmleaf_model::ModelError;
use llmleaf_model::{BatchHandle, BatchItem, BatchOutcome, BatchResult, BatchSpec, BatchStatus};
use serde::Serialize;
use serde_json::{Map, Value};

use super::openai;
use super::openai::CompletionView;

fn mapping(msg: impl Into<String>) -> ModelError {
    ModelError::Mapping(msg.into())
}

/// Parse a batch-create body into a canonical [`BatchSpec`]. Accepts `requests` (preferred) or `input`
/// as the array key; each entry needs a string `custom_id` and a `body` (an OpenAI chat request).
/// OpenAI's per-line `method`/`url` fields, if present, are ignored — the route is decided by the
/// request `model`, not a URL.
pub fn parse_batch_create(value: Value) -> Result<BatchSpec, ModelError> {
    let Value::Object(mut obj) = value else {
        return Err(mapping("batch body must be a JSON object"));
    };
    let requests = match obj.remove("requests").or_else(|| obj.remove("input")) {
        Some(Value::Array(a)) => a,
        _ => return Err(mapping("`requests` is required and must be an array")),
    };
    if requests.is_empty() {
        return Err(mapping("`requests` must not be empty"));
    }
    let mut items = Vec::with_capacity(requests.len());
    for entry in requests {
        let Value::Object(mut line) = entry else {
            return Err(mapping("each request must be an object"));
        };
        let custom_id = match line.remove("custom_id") {
            Some(Value::String(s)) if !s.is_empty() => s,
            _ => return Err(mapping("each request needs a non-empty string `custom_id`")),
        };
        let body = line
            .remove("body")
            .ok_or_else(|| mapping("each request needs a `body`"))?;
        let request = openai::parse_chat_request(body)?;
        items.push(BatchItem { custom_id, request });
    }
    Ok(BatchSpec { items })
}

/// Render a canonical [`BatchHandle`] as a batch job object (OpenAI-shaped, with llmleaf's superset of
/// request counts). The `id` is already the opaque, provider-encoding token the engine wrapped.
pub fn handle_to_json(handle: &BatchHandle) -> Value {
    let mut counts = Map::new();
    counts.insert("total".into(), Value::from(handle.counts.total));
    counts.insert("completed".into(), Value::from(handle.counts.succeeded));
    counts.insert("failed".into(), Value::from(handle.counts.errored));
    counts.insert("processing".into(), Value::from(handle.counts.processing));
    counts.insert("canceled".into(), Value::from(handle.counts.canceled));
    counts.insert("expired".into(), Value::from(handle.counts.expired));

    let mut root = Map::new();
    root.insert("id".into(), Value::from(handle.id.clone()));
    root.insert("object".into(), Value::from("batch"));
    root.insert("status".into(), Value::from(status_str(handle.status)));
    root.insert("request_counts".into(), Value::Object(counts));
    if let Some(t) = handle.created_at {
        root.insert("created_at".into(), Value::from(t));
    }
    if let Some(t) = handle.expires_at {
        root.insert("expires_at".into(), Value::from(t));
    }
    if let Some(t) = handle.ended_at {
        root.insert("completed_at".into(), Value::from(t));
    }
    if let Some(e) = &handle.endpoint {
        root.insert("endpoint".into(), Value::from(e.clone()));
    }
    // Anything dialect-specific the provider carried rides through verbatim (principle 7), without
    // overwriting the fields we set above.
    for (k, v) in &handle.extra {
        root.entry(k.clone()).or_insert_with(|| v.clone());
    }
    Value::Object(root)
}

/// Borrowed view of one JSONL result line (OpenAI batch-output shape). Same technique as the streaming
/// `ChunkFrame`/collected `CompletionView` (principle 1): serialised straight to bytes, never a
/// `serde_json::Value` tree. Fields are declared in the sorted key order `serde_json::Map` (a
/// `BTreeMap`) emits — `custom_id, error, response` — so the wire bytes are byte-identical to the old
/// mapping. `error` and `response` are always *both* present (one `null`), matching the old code which
/// inserted both keys in every arm; hence `Option<T>` with no skip (None serialises as JSON `null`).
#[derive(Serialize)]
pub struct ResultLineView<'a> {
    custom_id: &'a str,
    error: Option<ErrorView<'a>>,
    response: Option<ResponseView<'a>>,
}

#[derive(Serialize)]
struct ResponseView<'a> {
    // Sorted order: body, status_code. The success body is exactly the chat `CompletionView`,
    // rendered with `created = 0` as the old code did.
    body: CompletionView<'a>,
    status_code: u16,
}

#[derive(Serialize)]
struct ErrorView<'a> {
    // Sorted order: code, message. The errored arm formats `status.to_string()` (`Cow::Owned`);
    // the canceled/expired arms use borrowed static strings (`Cow::Borrowed`).
    code: Cow<'a, str>,
    message: &'a str,
}

/// Render one [`BatchResult`] as a JSONL output line (OpenAI batch-output shape): a `response` for a
/// success, an `error` otherwise. Cancellation/expiry surface as errors with a stable `code`. Returns a
/// borrowed [`ResultLineView`] so the caller serialises straight to bytes (principle 1); the success
/// arm reuses the chat [`CompletionView`] the live surface uses (principle 3: one internal model).
pub fn result_to_line(result: &BatchResult) -> ResultLineView<'_> {
    let custom_id = &result.custom_id;
    match &result.outcome {
        BatchOutcome::Succeeded(resp) => ResultLineView {
            custom_id,
            error: None,
            response: Some(ResponseView {
                body: openai::response_to_openai(resp, 0),
                status_code: 200,
            }),
        },
        BatchOutcome::Errored { status, message } => ResultLineView {
            custom_id,
            error: Some(ErrorView {
                code: Cow::Owned(status.to_string()),
                message,
            }),
            response: None,
        },
        BatchOutcome::Canceled => ResultLineView {
            custom_id,
            error: Some(ErrorView {
                code: Cow::Borrowed("canceled"),
                message: "request canceled before completion",
            }),
            response: None,
        },
        BatchOutcome::Expired => ResultLineView {
            custom_id,
            error: Some(ErrorView {
                code: Cow::Borrowed("expired"),
                message: "request expired before completion",
            }),
            response: None,
        },
    }
}

/// The wire string for a status. Mirrors OpenAI's vocabulary (note the British double-l on cancel).
fn status_str(status: BatchStatus) -> &'static str {
    match status {
        BatchStatus::Validating => "validating",
        BatchStatus::InProgress => "in_progress",
        BatchStatus::Finalizing => "finalizing",
        BatchStatus::Completed => "completed",
        BatchStatus::Failed => "failed",
        BatchStatus::Expired => "expired",
        BatchStatus::Canceling => "cancelling",
        BatchStatus::Canceled => "cancelled",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llmleaf_model::{BatchCounts, ChatResponse, Choice, FinishReason, Usage};
    use serde_json::json;

    #[test]
    fn parses_inline_requests() {
        let spec = parse_batch_create(json!({
            "requests": [
                { "custom_id": "a", "body": { "model": "gpt-4o", "messages": [{ "role": "user", "content": "hi" }] } },
                { "custom_id": "b", "method": "POST", "url": "/v1/chat/completions",
                  "body": { "model": "claude", "messages": [{ "role": "user", "content": "yo" }] } }
            ]
        }))
        .unwrap();
        assert_eq!(spec.items.len(), 2);
        assert_eq!(spec.items[0].custom_id, "a");
        assert_eq!(spec.items[0].request.model, "gpt-4o");
        assert_eq!(spec.items[1].request.model, "claude"); // method/url ignored, body parsed
    }

    #[test]
    fn rejects_missing_fields() {
        assert!(parse_batch_create(json!({ "requests": [] })).is_err());
        assert!(parse_batch_create(json!({ "requests": [{ "body": {} }] })).is_err()); // no custom_id
        assert!(parse_batch_create(json!({ "requests": [{ "custom_id": "a" }] })).is_err()); // no body
        assert!(parse_batch_create(json!({})).is_err()); // no requests
    }

    #[test]
    fn renders_handle_and_results() {
        let handle = BatchHandle {
            id: "batch_opaque".into(),
            status: BatchStatus::InProgress,
            counts: BatchCounts {
                total: 2,
                processing: 1,
                succeeded: 1,
                ..Default::default()
            },
            created_at: Some(100),
            expires_at: None,
            ended_at: None,
            endpoint: Some("/v1/chat/completions".into()),
            extra: Default::default(),
        };
        let j = handle_to_json(&handle);
        assert_eq!(j["id"], "batch_opaque");
        assert_eq!(j["object"], "batch");
        assert_eq!(j["status"], "in_progress");
        assert_eq!(j["request_counts"]["completed"], 1);
        assert_eq!(j["request_counts"]["total"], 2);

        let ok = BatchResult {
            custom_id: "a".into(),
            outcome: BatchOutcome::Succeeded(ChatResponse {
                id: "x".into(),
                model: "gpt-4o".into(),
                choices: vec![Choice {
                    index: 0,
                    text: "hello".into(),
                    tool_calls: vec![],
                    finish_reason: Some(FinishReason::Stop),
                }],
                usage: Usage::default(),
            }),
        };
        // `to_value` for the readable assertions only; the hot path serialises the view to bytes.
        let line = serde_json::to_value(result_to_line(&ok)).unwrap();
        assert_eq!(line["custom_id"], "a");
        assert_eq!(line["response"]["status_code"], 200);
        assert_eq!(
            line["response"]["body"]["choices"][0]["message"]["content"],
            "hello"
        );
        assert_eq!(line["error"], Value::Null);

        let err = BatchResult {
            custom_id: "b".into(),
            outcome: BatchOutcome::Errored {
                status: 429,
                message: "rate limited".into(),
            },
        };
        let line = serde_json::to_value(result_to_line(&err)).unwrap();
        assert_eq!(line["error"]["code"], "429");
        assert_eq!(line["response"], Value::Null);
    }

    /// The equivalence guarantee: the borrowed result-line view serialises to the *exact* bytes the
    /// old `Value`-building path produced, for both the success arm (embedding the chat view) and an
    /// errored arm (`Cow::Owned` code).
    #[test]
    fn result_line_bytes_match_legacy() {
        // The pre-refactor `Value`-building path, pasted here verbatim as the legacy oracle.
        fn error_obj(code: &str, message: &str) -> Value {
            let mut e = Map::new();
            e.insert("code".into(), Value::from(code.to_string()));
            e.insert("message".into(), Value::from(message.to_string()));
            Value::Object(e)
        }
        fn legacy(result: &BatchResult) -> Value {
            let mut line = Map::new();
            line.insert("custom_id".into(), Value::from(result.custom_id.clone()));
            match &result.outcome {
                BatchOutcome::Succeeded(resp) => {
                    let mut response = Map::new();
                    response.insert("status_code".into(), Value::from(200u16));
                    response.insert(
                        "body".into(),
                        serde_json::to_value(openai::response_to_openai(resp, 0)).unwrap(),
                    );
                    line.insert("response".into(), Value::Object(response));
                    line.insert("error".into(), Value::Null);
                }
                BatchOutcome::Errored { status, message } => {
                    line.insert("response".into(), Value::Null);
                    line.insert("error".into(), error_obj(&status.to_string(), message));
                }
                BatchOutcome::Canceled => {
                    line.insert("response".into(), Value::Null);
                    line.insert(
                        "error".into(),
                        error_obj("canceled", "request canceled before completion"),
                    );
                }
                BatchOutcome::Expired => {
                    line.insert("response".into(), Value::Null);
                    line.insert(
                        "error".into(),
                        error_obj("expired", "request expired before completion"),
                    );
                }
            }
            Value::Object(line)
        }

        let ok = BatchResult {
            custom_id: "a".into(),
            outcome: BatchOutcome::Succeeded(ChatResponse {
                id: "x".into(),
                model: "gpt-4o".into(),
                choices: vec![Choice {
                    index: 0,
                    text: "hello".into(),
                    tool_calls: vec![],
                    finish_reason: Some(FinishReason::Stop),
                }],
                usage: Usage::default(),
            }),
        };
        let err = BatchResult {
            custom_id: "b".into(),
            outcome: BatchOutcome::Errored {
                status: 429,
                message: "rate limited".into(),
            },
        };

        for result in [&ok, &err] {
            assert_eq!(
                serde_json::to_vec(&result_to_line(result)).unwrap(),
                serde_json::to_vec(&legacy(result)).unwrap(),
            );
        }
    }
}
