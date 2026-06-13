//! The provider transport seam — the "client of the provider clients".
//!
//! Every first-party provider speaks its upstream over HTTP (and, for native realtime, a WebSocket).
//! Historically each built a `reqwest::RequestBuilder` directly and owned a `reqwest::Client`, so the
//! real request-mapping and response-parsing code could only run against a live network. This module
//! lifts that boundary onto traits: [`HttpTransport`] (request → response) and [`RealtimeTransport`]
//! (a pumped WebSocket session). Production wires the real [`ReqwestTransport`] +
//! [`crate::realtime_ws::TungsteniteRealtimeTransport`]; tests, benches, and the seeded simulation
//! inject fakes that return canned responses or seeded errors — so a provider's build+parse path can be
//! unit-tested, benchmarked (time + allocations), and driven through typical upstream failures.
//!
//! The canonical failure taxonomy stays centralized in [`crate::http`]: the transport reports transport
//! failures as [`ModelError::Unavailable`]; status/parse classification (`Upstream`/`Mapping`) happens
//! in `post_json`/`send_checked` exactly as before.

use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use futures::{Stream, StreamExt, TryStreamExt};
use llmleaf_model::ModelError;
use llmleaf_provider::RealtimePeer;
use serde_json::Value;

/// HTTP method — the providers only ever GET or POST.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    Get,
    Post,
}

/// A transport-neutral outbound request the provider edge builds; the [`HttpTransport`] executes it.
/// Query parameters are baked into `url` (as every provider already does), so there is no separate
/// query field.
#[derive(Debug, Clone)]
pub struct HttpRequest {
    pub method: Method,
    pub url: String,
    pub headers: Vec<(String, String)>,
    pub body: HttpBody,
}

impl HttpRequest {
    pub fn get(url: impl Into<String>) -> Self {
        Self {
            method: Method::Get,
            url: url.into(),
            headers: Vec::new(),
            body: HttpBody::Empty,
        }
    }

    pub fn post(url: impl Into<String>) -> Self {
        Self {
            method: Method::Post,
            url: url.into(),
            headers: Vec::new(),
            body: HttpBody::Empty,
        }
    }

    /// Add a header (chainable). Mirrors `reqwest::RequestBuilder::header`. Takes `AsRef<str>` so a
    /// `&String` credential, a `&str`, or a `String` all work at the call site without conversion.
    pub fn header(mut self, name: impl AsRef<str>, value: impl AsRef<str>) -> Self {
        self.headers
            .push((name.as_ref().to_string(), value.as_ref().to_string()));
        self
    }

    /// Add an `Authorization: Bearer <token>` header. Accepts `&str` or `&String`.
    pub fn bearer(self, token: impl AsRef<str>) -> Self {
        self.header("Authorization", format!("Bearer {}", token.as_ref()))
    }

    /// Set a JSON body (the common case).
    pub fn json(mut self, value: Value) -> Self {
        self.body = HttpBody::Json(value);
        self
    }

    /// Set a multipart body (file upload / transcription).
    pub fn multipart(mut self, form: MultipartForm) -> Self {
        self.body = HttpBody::Multipart(form);
        self
    }
}

/// The request body.
#[derive(Debug, Clone)]
pub enum HttpBody {
    Empty,
    Json(Value),
    Multipart(MultipartForm),
}

/// A transport-neutral multipart form (the reqwest impl converts it to `reqwest::multipart::Form`).
#[derive(Debug, Clone, Default)]
pub struct MultipartForm {
    pub parts: Vec<MultipartPart>,
}

impl MultipartForm {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn text(mut self, name: impl AsRef<str>, value: impl AsRef<str>) -> Self {
        self.parts.push(MultipartPart::Text {
            name: name.as_ref().to_string(),
            value: value.as_ref().to_string(),
        });
        self
    }

    pub fn bytes(
        mut self,
        name: impl Into<String>,
        filename: Option<String>,
        mime: Option<String>,
        data: Bytes,
    ) -> Self {
        self.parts.push(MultipartPart::Bytes {
            name: name.into(),
            filename,
            mime,
            data,
        });
        self
    }
}

#[derive(Debug, Clone)]
pub enum MultipartPart {
    Text {
        name: String,
        value: String,
    },
    Bytes {
        name: String,
        filename: Option<String>,
        mime: Option<String>,
        data: Bytes,
    },
}

/// A streaming response body. Items are already mapped to the canonical [`ModelError`] (the transport
/// converts a mid-stream transport failure to [`ModelError::Unavailable`]), so the SSE/NDJSON/JSONL/audio
/// consumers downstream see canonical errors only — never a `reqwest::Error`.
pub type BytesStream = Pin<Box<dyn Stream<Item = Result<Bytes, ModelError>> + Send>>;

/// An executed response: status + headers + a streaming body. `post_json` collects the body; streaming
/// callers (SSE chat, audio, batch-results) consume it incrementally.
pub struct HttpResponse {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: BytesStream,
}

impl HttpResponse {
    /// Whether the status is 2xx.
    pub fn is_success(&self) -> bool {
        (200..=299).contains(&self.status)
    }

    /// Case-insensitive response-header lookup (e.g. the audio `Content-Type` probe).
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }

    /// Drain the entire body into one buffer (used by `post_json` and by collect-only callers).
    pub async fn collect_body(self) -> Result<Bytes, ModelError> {
        let mut body = self.body;
        let mut buf: Vec<u8> = Vec::new();
        while let Some(chunk) = body.next().await {
            buf.extend_from_slice(&chunk?);
        }
        Ok(Bytes::from(buf))
    }
}

/// The outbound HTTP transport a provider executes its requests through.
#[async_trait]
pub trait HttpTransport: Send + Sync {
    async fn execute(&self, req: HttpRequest) -> Result<HttpResponse, ModelError>;
}

/// The native-realtime transport: dial the upstream WebSocket and pump frames between the consumer
/// [`RealtimePeer`] and the upstream **verbatim** (principle 7) until either side closes. Keeping the
/// pump inside the transport lets the real impl reuse a duplex socket split while a fake scripts the
/// upstream side without a server.
#[async_trait]
pub trait RealtimeTransport: Send + Sync {
    async fn run(
        &self,
        url: String,
        headers: Vec<(String, String)>,
        peer: RealtimePeer,
    ) -> Result<(), ModelError>;
}

/// The transports injected into every provider at construction. Cheap to clone (all `Arc`).
#[derive(Clone)]
pub struct Transports {
    pub http: Arc<dyn HttpTransport>,
    pub realtime: Arc<dyn RealtimeTransport>,
}

impl Transports {
    /// The production transports: `reqwest` for HTTP, `tokio-tungstenite` for the realtime WebSocket.
    pub fn real() -> Self {
        Self {
            http: Arc::new(ReqwestTransport::new()),
            realtime: Arc::new(crate::realtime_ws::TungsteniteRealtimeTransport),
        }
    }
}

/// The production HTTP transport: a shared `reqwest::Client`. The only place `reqwest` request/response
/// types appear now (mirrored by `tokio-tungstenite` in [`crate::realtime_ws`]).
pub struct ReqwestTransport {
    client: reqwest::Client,
}

impl ReqwestTransport {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
        }
    }

    /// Reuse a caller-built client (e.g. with custom timeouts/proxy).
    pub fn with_client(client: reqwest::Client) -> Self {
        Self { client }
    }
}

impl Default for ReqwestTransport {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl HttpTransport for ReqwestTransport {
    async fn execute(&self, req: HttpRequest) -> Result<HttpResponse, ModelError> {
        let mut builder = match req.method {
            Method::Get => self.client.get(&req.url),
            Method::Post => self.client.post(&req.url),
        };
        for (name, value) in &req.headers {
            builder = builder.header(name, value);
        }
        builder = match req.body {
            HttpBody::Empty => builder,
            HttpBody::Json(ref v) => builder.json(v),
            HttpBody::Multipart(form) => builder.multipart(to_reqwest_form(form)),
        };

        let resp = builder
            .send()
            .await
            .map_err(|e| ModelError::Unavailable(e.to_string()))?;
        let status = resp.status().as_u16();
        let headers = resp
            .headers()
            .iter()
            .filter_map(|(k, v)| {
                v.to_str()
                    .ok()
                    .map(|v| (k.as_str().to_string(), v.to_string()))
            })
            .collect();
        let body = resp
            .bytes_stream()
            .map_err(|e| ModelError::Unavailable(e.to_string()))
            .boxed();
        Ok(HttpResponse {
            status,
            headers,
            body,
        })
    }
}

/// Convert the neutral multipart form to a `reqwest::multipart::Form`. The provider passes known-good
/// MIME strings; if one is somehow rejected we fall back to a part without an explicit MIME rather than
/// panicking (the body still rides with its filename).
fn to_reqwest_form(form: MultipartForm) -> reqwest::multipart::Form {
    let mut out = reqwest::multipart::Form::new();
    for part in form.parts {
        match part {
            MultipartPart::Text { name, value } => out = out.text(name, value),
            MultipartPart::Bytes {
                name,
                filename,
                mime,
                data,
            } => {
                let make_part = || {
                    let mut p = reqwest::multipart::Part::bytes(data.to_vec());
                    if let Some(fname) = &filename {
                        p = p.file_name(fname.clone());
                    }
                    p
                };
                let mut p = make_part();
                if let Some(m) = &mime {
                    p = p.mime_str(m).unwrap_or_else(|_| make_part());
                }
                out = out.part(name, p);
            }
        }
    }
    out
}
