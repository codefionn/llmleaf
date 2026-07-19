//! Shared HTTP plumbing for first-party providers. One place to map transport/status failures into
//! the canonical [`ModelError`], so every provider classifies upstream failures identically — now over
//! the injectable [`HttpTransport`] seam rather than `reqwest` directly (see [`crate::transport`]).

use futures::StreamExt;
use llmleaf_model::{ModelError, ResponseStream, StreamChunk};
use serde_json::Value;

use crate::transport::{HttpRequest, HttpResponse, HttpTransport};

/// Execute `req` on `transport`, then enforce the canonical failure taxonomy:
/// - transport error before/while reading → [`ModelError::Unavailable`] (eligible for fallback),
/// - non-2xx status → [`ModelError::Upstream`] (carries the upstream body for the event stream),
/// - non-JSON 2xx body → [`ModelError::Mapping`].
pub async fn post_json(
    transport: &dyn HttpTransport,
    req: HttpRequest,
) -> Result<Value, ModelError> {
    let resp = transport.execute(req).await?;
    let status = resp.status;
    let bytes = resp.collect_body().await?;
    if !(200..=299).contains(&status) {
        return Err(ModelError::Upstream {
            status,
            message: String::from_utf8_lossy(&bytes).into_owned(),
        });
    }
    serde_json::from_slice(&bytes).map_err(|e| ModelError::Mapping(e.to_string()))
}

/// Execute `req` and enforce status, returning the streaming [`HttpResponse`] so the caller can consume
/// its body (SSE chat, binary audio, JSONL batch results). Same failure taxonomy as [`post_json`]:
/// transport error → [`ModelError::Unavailable`] (fallback-eligible), non-2xx → [`ModelError::Upstream`]
/// (carries the body, which is drained here to build the message).
pub async fn send_checked(
    transport: &dyn HttpTransport,
    req: HttpRequest,
) -> Result<HttpResponse, ModelError> {
    let resp = transport.execute(req).await?;
    if !resp.is_success() {
        let status = resp.status;
        let bytes = resp.collect_body().await.unwrap_or_default();
        return Err(ModelError::Upstream {
            status,
            message: String::from_utf8_lossy(&bytes).into_owned(),
        });
    }
    Ok(resp)
}

/// Drain complete SSE `data:` payloads out of a rolling byte buffer.
///
/// HTTP body chunks have arbitrary boundaries, so a JSON event (or a UTF-8 code point) may span
/// several chunks. Only newline-terminated fields are decoded; a trailing partial field stays in
/// `buf` for the next call. Event names and comments are deliberately ignored because every provider
/// we use puts a self-describing `type` in the JSON data object.
pub fn sse_payloads(buf: &mut Vec<u8>, incoming: &[u8]) -> Vec<String> {
    buf.extend_from_slice(incoming);
    let mut out = Vec::new();
    while let Some(nl) = buf.iter().position(|&b| b == b'\n') {
        let end = if nl > 0 && buf[nl - 1] == b'\r' {
            nl - 1
        } else {
            nl
        };
        if let Ok(line) = std::str::from_utf8(&buf[..end]) {
            let line = line.trim();
            if !line.is_empty() && !line.starts_with(':') {
                if let Some(rest) = line.strip_prefix("data:") {
                    out.push(rest.strip_prefix(' ').unwrap_or(rest).to_string());
                }
            }
        }
        buf.drain(..=nl);
    }
    out
}

/// Decode a JSON-over-SSE response body into the canonical response stream.
///
/// Provider modules supply only their event-to-chunk state machine. This shared loop owns transport
/// failures, arbitrary byte boundaries, JSON classification, and the optional OpenAI-style `[DONE]`
/// sentinel. A provider event mapper may return several canonical chunks for one upstream event.
pub fn json_sse_to_stream<S, F>(
    body: crate::transport::BytesStream,
    mut state: S,
    mut map: F,
) -> ResponseStream
where
    S: Send + 'static,
    F: FnMut(&Value, &mut S) -> Result<Vec<StreamChunk>, ModelError> + Send + 'static,
{
    Box::pin(async_stream::stream! {
        let mut bytes = body;
        let mut buf = Vec::with_capacity(1024);
        while let Some(item) = bytes.next().await {
            let chunk = match item {
                Ok(chunk) => chunk,
                Err(error) => {
                    yield Err(error);
                    return;
                }
            };
            for payload in sse_payloads(&mut buf, &chunk) {
                if payload == "[DONE]" {
                    return;
                }
                let value = match serde_json::from_str::<Value>(&payload) {
                    Ok(value) => value,
                    Err(error) => {
                        yield Err(ModelError::Mapping(error.to_string()));
                        return;
                    }
                };
                match map(&value, &mut state) {
                    Ok(chunks) => {
                        for chunk in chunks {
                            yield Ok(chunk);
                        }
                    }
                    Err(error) => {
                        yield Err(error);
                        return;
                    }
                }
            }
        }
    })
}
