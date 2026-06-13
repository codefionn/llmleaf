//! Shared HTTP plumbing for first-party providers. One place to map transport/status failures into
//! the canonical [`ModelError`], so every provider classifies upstream failures identically — now over
//! the injectable [`HttpTransport`] seam rather than `reqwest` directly (see [`crate::transport`]).

use llmleaf_model::ModelError;
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
