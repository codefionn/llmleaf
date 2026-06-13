//! The SDK's typed error.

use serde::Deserialize;

/// Errors returned by the SDK.
///
/// [`Error::Api`] is the structured gateway error parsed from the canonical
/// `{"error":{"message":"..."}}` envelope (see SPEC.md). Everything else is a
/// transport or decoding failure.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// A non-2xx response from the gateway, carrying its HTTP status and the
    /// `error.message` field from the envelope.
    ///
    /// Status codes (per SPEC.md): 400 bad request · 401 missing/invalid key ·
    /// 403 blocked or model-not-allowed · 404 no route for model ·
    /// 429 key suspended (limiter) · 502 all upstreams failed.
    #[error("api error {status}: {message}")]
    Api {
        /// HTTP status code of the failed response.
        status: u16,
        /// Human-readable message from `error.message`.
        message: String,
    },

    /// The configured base URL could not be parsed.
    #[error("invalid base url: {0}")]
    InvalidBaseUrl(String),

    /// An HTTP / transport-level failure (connection, TLS, timeout, …).
    #[error("http transport error: {0}")]
    Http(#[from] reqwest::Error),

    /// A JSON body could not be (de)serialised.
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    /// A base64 embedding payload could not be decoded.
    #[error("base64 decode error: {0}")]
    Base64(#[from] base64::DecodeError),

    /// The server sent a malformed SSE / NDJSON stream.
    #[error("malformed stream: {0}")]
    Stream(String),
}

/// The on-the-wire error envelope: `{"error":{"message":"..."}}`.
#[derive(Debug, Deserialize)]
pub(crate) struct ErrorEnvelope {
    pub error: ErrorBody,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ErrorBody {
    #[serde(default)]
    pub message: String,
    #[allow(dead_code)]
    #[serde(default)]
    pub r#type: Option<String>,
    #[allow(dead_code)]
    #[serde(default)]
    pub code: Option<String>,
}

/// A convenience result alias.
pub type Result<T> = std::result::Result<T, Error>;
