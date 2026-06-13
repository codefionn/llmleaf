//! Injectable fake transports for tests, benches, and the seeded simulation (the `test-support`
//! feature). They satisfy [`HttpTransport`]/[`RealtimeTransport`] without a network, so a real
//! provider's request-mapping + response-parsing path can be unit-tested, benchmarked, and driven
//! through the typical upstream failures (`Unavailable`/`Upstream{status}`/`Mapping`/`Timeout`).

use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt;
use llmleaf_model::ModelError;
use llmleaf_provider::{RealtimePeer, RealtimeWire};
use serde_json::Value;

use crate::transport::{
    BytesStream, HttpRequest, HttpResponse, HttpTransport, RealtimeTransport, Transports,
};

impl Transports {
    /// A [`Transports`] backed by no-op fakes (empty-JSON HTTP, no-frame realtime). For constructing a
    /// provider in a unit test that only exercises pure request/URL/auth building — no request is
    /// actually sent. Tests that assert on a real round trip build a [`FakeHttpTransport`] with the
    /// specific canned response or error they need.
    pub fn fake() -> Self {
        Transports {
            http: Arc::new(FakeHttpTransport::json(serde_json::json!({}))),
            realtime: Arc::new(FakeRealtimeTransport::scripted(Vec::new())),
        }
    }
}

/// A canned HTTP response a [`FakeHttpTransport`] hands back.
#[derive(Debug, Clone)]
pub struct FakeResponse {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: Bytes,
}

impl FakeResponse {
    /// A 200 JSON response (sets `content-type: application/json`).
    pub fn ok_json(value: &Value) -> Self {
        Self {
            status: 200,
            headers: vec![("content-type".into(), "application/json".into())],
            body: Bytes::from(serde_json::to_vec(value).expect("Value serializes")),
        }
    }

    /// A 200 response with an explicit content type (e.g. `text/event-stream`, `audio/mpeg`).
    pub fn ok_bytes(content_type: &str, body: impl Into<Bytes>) -> Self {
        Self {
            status: 200,
            headers: vec![("content-type".into(), content_type.to_string())],
            body: body.into(),
        }
    }

    /// An explicit status + body (e.g. `429`/`500` with an error payload).
    pub fn status(code: u16, body: impl Into<Bytes>) -> Self {
        Self {
            status: code,
            headers: Vec::new(),
            body: body.into(),
        }
    }
}

type Responder = Arc<dyn Fn(&HttpRequest) -> Result<FakeResponse, ModelError> + Send + Sync>;

/// A [`HttpTransport`] that answers from a responder closure. Convenience constructors cover the common
/// shapes (a fixed JSON body, an SSE stream body, a status-with-body, or a transport error); for
/// request-dependent or seeded behaviour, build one with [`FakeHttpTransport::new`].
#[derive(Clone)]
pub struct FakeHttpTransport {
    responder: Responder,
}

impl FakeHttpTransport {
    pub fn new(
        f: impl Fn(&HttpRequest) -> Result<FakeResponse, ModelError> + Send + Sync + 'static,
    ) -> Self {
        Self {
            responder: Arc::new(f),
        }
    }

    /// Always return this JSON object as a 200.
    pub fn json(value: Value) -> Self {
        Self::new(move |_| Ok(FakeResponse::ok_json(&value)))
    }

    /// Always return this SSE body as a 200 `text/event-stream` (for streaming chat).
    pub fn sse(body: impl Into<Bytes>) -> Self {
        let body = body.into();
        Self::new(move |_| Ok(FakeResponse::ok_bytes("text/event-stream", body.clone())))
    }

    /// Always return this status + body (e.g. a 4xx/5xx error the provider must surface as `Upstream`).
    pub fn status(code: u16, body: impl Into<Bytes>) -> Self {
        let body = body.into();
        Self::new(move |_| Ok(FakeResponse::status(code, body.clone())))
    }

    /// Always fail before a response with this transport error (surfaces as the given `ModelError`).
    pub fn error(err: ModelError) -> Self {
        Self::new(move |_| Err(err.clone()))
    }
}

#[async_trait]
impl HttpTransport for FakeHttpTransport {
    async fn execute(&self, req: HttpRequest) -> Result<HttpResponse, ModelError> {
        let resp = (self.responder)(&req)?;
        let body: BytesStream = futures::stream::once(async move { Ok(resp.body) }).boxed();
        Ok(HttpResponse {
            status: resp.status,
            headers: resp.headers,
            body,
        })
    }
}

/// A [`RealtimeTransport`] that scripts the upstream side of a realtime session without a server: it
/// pushes `upstream_frames` to the consumer, drains whatever the consumer sends, then returns
/// `outcome`. Lets the native realtime path be exercised (frame forwarding, lifecycle, error surfacing)
/// in tests and simulation.
#[derive(Clone)]
pub struct FakeRealtimeTransport {
    pub upstream_frames: Vec<RealtimeWire>,
    pub outcome: Result<(), ModelError>,
}

impl FakeRealtimeTransport {
    /// A clean session that emits `frames` then closes.
    pub fn scripted(frames: Vec<RealtimeWire>) -> Self {
        Self {
            upstream_frames: frames,
            outcome: Ok(()),
        }
    }

    /// A session that fails to connect / dies mid-stream with `err`.
    pub fn failing(err: ModelError) -> Self {
        Self {
            upstream_frames: Vec::new(),
            outcome: Err(err),
        }
    }
}

#[async_trait]
impl RealtimeTransport for FakeRealtimeTransport {
    async fn run(
        &self,
        _url: String,
        _headers: Vec<(String, String)>,
        peer: RealtimePeer,
    ) -> Result<(), ModelError> {
        let RealtimePeer {
            mut inbound,
            outbound,
        } = peer;
        for frame in &self.upstream_frames {
            if outbound.send(frame.clone()).await.is_err() {
                break;
            }
        }
        // Drain anything the consumer sent so the channel doesn't wedge, until it closes.
        while inbound.try_recv().is_ok() {}
        self.outcome.clone()
    }
}
