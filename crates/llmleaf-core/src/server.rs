//! The HTTP surfaces (SOUL.md "Architecture soul", inverted control plane).
//!
//! Two surfaces on one listener, strictly separated in role:
//!   - **consumer** (`POST /v1/chat/completions`, …) — the OpenAI compat surface; bearer-key auth,
//!     map in → engine → map out.
//!   - **read-only admin** (`GET /admin/routes|/admin/health|/admin/keys`) — observability only. There
//!     is *no* mutation surface: runtime state is pulled from configured control endpoints, never
//!     pushed in. The usage/lifecycle stream leaves the node as an outbound push (the control crate's
//!     reporter taps the in-process bus), not as an inbound SSE subscription.
//!
//! The core never depends on the control plane: with no admin token configured the read-only admin
//! surface simply closes, and the proxy keeps proxying from config + last-good cache alone (principle 6).

use std::collections::{BTreeMap, HashSet};
use std::convert::Infallible;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::body::Body;
use axum::extract::ws::{Message as WsMessage, WebSocket, WebSocketUpgrade};
use axum::extract::{DefaultBodyLimit, Multipart, Path, Query, State};
use axum::http::header::{AUTHORIZATION, CONTENT_TYPE, RETRY_AFTER};
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use bytes::Bytes;
use futures::{SinkExt, Stream, StreamExt};
use llmleaf_model::{AudioChunk, Modality, ModelInfo, ResponseStream};
use llmleaf_pricing::{ModelCard, Pricing};
use llmleaf_provider::{ProviderRegistry, RealtimePeer, RealtimeWire};
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::mpsc;

use crate::admin::{AdminAccess, Observability};
use crate::compat::openai::{self, ChunkEncoder};
use crate::compat::realtime::{session as rt_session, wire as rt_wire};
use crate::compat::transcription::TranscriptionBody;
use crate::compat::{
    anthropic, batch, embeddings, openapi, rerank, responses, speech, transcription,
};
use crate::config::Config;
use crate::engine::{Engine, EngineError};
use crate::events::{Event, EventBus};
use crate::keys::{AuthError, KeyStore};

/// Shared, clone-able application state. Everything here is `Arc`/handle — cheap to clone per request.
#[derive(Clone)]
pub struct AppState {
    pub engine: Arc<Engine>,
    pub keys: Arc<KeyStore>,
    /// Optional OAuth2 resource-server verifier (`[oauth]`). Validates JWT bearers as a second,
    /// additive scheme beside `keys`; `None` ⇒ only static keys authenticate. The control plane wires
    /// its JWKS pull and introspector into this handle after the state is built.
    #[cfg(feature = "oauth")]
    pub oauth: Option<Arc<crate::oauth::OAuthVerifier>>,
    pub events: EventBus,
    /// Resolved admin token. `None` ⇒ the read-only admin surface is disabled.
    pub admin_token: Option<Arc<String>>,
    /// Maximum inbound request body size in bytes (`[server].max_body_bytes`), applied as the router's
    /// body limit so base64-inlined multimodal images don't 413 against axum's 2 MiB default.
    max_body_bytes: usize,
    request_seq: Arc<AtomicU64>,
}

/// Build the shared state from config + a provider registry handed in by the binary. No sync
/// interceptor — equivalent to `build_state_with(config, registry, None)`. Used by tests, benches, and
/// config-only embedding.
pub fn build_state(
    config: &Config,
    registry: Arc<ProviderRegistry>,
) -> Result<AppState, Box<dyn std::error::Error + Send + Sync>> {
    build_state_with(config, registry, None)
}

/// Build the shared state, optionally wiring in a sync interceptor (`[control.intercept]`). The binary
/// constructs the reqwest-backed interceptor in `llmleaf-control` and passes it here; the core only
/// ever sees the HTTP-free [`Interceptor`] trait object (principle 2).
pub fn build_state_with(
    config: &Config,
    registry: Arc<ProviderRegistry>,
    interceptor: Option<Arc<dyn crate::engine::Interceptor>>,
) -> Result<AppState, Box<dyn std::error::Error + Send + Sync>> {
    let events = EventBus::new(config.server.event_buffer);
    let pricing = Arc::new(Pricing::bundled()?);
    let keys = Arc::new(KeyStore::from_config(&config.keys));
    let engine = Arc::new(Engine::new(
        config,
        registry,
        events.clone(),
        pricing,
        interceptor,
    ));
    let admin_token = config
        .server
        .admin_token
        .as_ref()
        .and_then(|s| s.resolve())
        .map(Arc::new);

    // OAuth2 resource-server verifier (`[oauth]`), built crypto-only here; the control plane installs
    // its JWKS roster and introspector afterwards (mirrors how it feeds the key store).
    #[cfg(feature = "oauth")]
    let oauth = match &config.oauth {
        Some(c) => Some(Arc::new(
            crate::oauth::OAuthVerifier::from_config(c)
                .map_err(|e| format!("invalid [oauth] config: {e}"))?,
        )),
        None => None,
    };
    // A config asking for OAuth in a build that cannot validate it must fail loudly, never silently
    // admit on the wrong scheme or ignore the directive.
    #[cfg(not(feature = "oauth"))]
    if config.oauth.is_some() {
        return Err("[oauth] is configured but this build lacks the `oauth` feature".into());
    }

    Ok(AppState {
        engine,
        keys,
        #[cfg(feature = "oauth")]
        oauth,
        events,
        admin_token,
        max_body_bytes: config.server.max_body_bytes,
        request_seq: Arc::new(AtomicU64::new(0)),
    })
}

/// Assemble the router. Surfaces are mounted unconditionally; admin handlers self-guard on the token.
/// Every mounted admin endpoint is read-only — there is no mutation surface (the control plane is
/// outbound; the core pulls state, it is never pushed to).
pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/healthz", get(healthz))
        .route("/v1/openapi.json", get(openapi_spec))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/messages", post(anthropic_messages))
        .route("/v1/responses", post(responses_create))
        .route("/v1/responses/{id}", get(get_response))
        .route("/v1/embeddings", post(embeddings_handler))
        .route("/v1/rerank", post(rerank_handler))
        .route("/v1/audio/speech", post(audio_speech))
        .route("/v1/audio/voices", get(audio_voices))
        .route("/v1/models", get(list_models))
        .route("/v1/audio/transcriptions", post(audio_transcriptions))
        .route("/v1/realtime", get(realtime_ws))
        .route("/v1/batches", post(create_batch))
        .route("/v1/batches/{id}", get(retrieve_batch))
        .route("/v1/batches/{id}/cancel", post(cancel_batch))
        .route("/v1/batches/{id}/results", get(batch_results))
        .route("/admin/routes", get(admin_routes))
        .route("/admin/health", get(admin_health))
        .route("/admin/keys", get(admin_list_keys))
        // Accept base64-inlined multimodal images (a data: URI easily exceeds axum's 2 MiB default,
        // which would 413 the whole request). Sized by `[server].max_body_bytes`.
        .layer(DefaultBodyLimit::max(state.max_body_bytes))
        .with_state(state)
}

/// Build everything and serve until the process is told to stop. A convenience wrapper for embedding
/// or config-only operation; the binary uses [`build_state`] + [`serve_with_state`] so it can grab the
/// state handles, spawn the outbound control tasks, and drive graceful shutdown.
pub async fn serve(
    config: Config,
    registry: Arc<ProviderRegistry>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let listen = config.server.listen.clone();
    let state = build_state(&config, registry)?;
    serve_with_state(state, &listen, std::future::pending::<()>()).await
}

/// Serve a pre-built state, shutting down gracefully when `shutdown` resolves. The binary builds the
/// state first (so the control crate can tap [`AppState::keys`]/[`AppState::events`]), then passes a
/// shutdown future wired to SIGTERM/ctrl-c that also cancels the background control tasks.
pub async fn serve_with_state(
    state: AppState,
    listen: &str,
    shutdown: impl std::future::Future<Output = ()> + Send + 'static,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let app = build_router(state);
    let listener = tokio::net::TcpListener::bind(listen).await?;
    tracing::info!(%listen, "llmleaf listening");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown)
        .await?;
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Consumer surface
// ---------------------------------------------------------------------------------------------

async fn healthz() -> &'static str {
    "ok"
}

/// `GET /v1/openapi.json` — the static OpenAPI 3.1 description of this consumer surface. Public (no
/// bearer, like `/healthz`): it is the fixed API contract, carrying no runtime, tenant, or provider
/// data, so transparency (P7) and the OpenAPI convention both want it served credential-free. The
/// document is built once and cached; this handler writes the borrowed bytes with no per-request work.
async fn openapi_spec() -> Response {
    ([(CONTENT_TYPE, "application/json")], openapi::document()).into_response()
}

async fn chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    let Some(token) = bearer(&headers) else {
        return error(StatusCode::UNAUTHORIZED, "missing bearer token");
    };

    // Map in (principle 3: dialect → canonical at the edge).
    let req = match openai::parse_chat_request(body) {
        Ok(r) => r,
        Err(e) => return error(StatusCode::BAD_REQUEST, e.to_string()),
    };
    let logical_model = req.model.clone();
    let stream_requested = req.stream;
    let now = now_secs();

    // Authenticate: a verdict lookup, never arithmetic (principle 5).
    let key = match authorize_token(&state, &token, &logical_model, now).await {
        Ok(id) => id,
        Err(e) => return auth_error(e),
    };

    let request_id = next_request_id(&state.request_seq);
    let created = now;

    // Route → stream.
    let stream = match state.engine.run(req, key, request_id.clone(), now).await {
        Ok(s) => s,
        Err(e) => return engine_error(e),
    };

    // Map out (principle 4: a non-streaming response is a collected stream).
    if stream_requested {
        let encoder = ChunkEncoder::new(format!("chatcmpl-{request_id}"), logical_model, created);
        sse_from_response(stream, encoder).into_response()
    } else {
        match llmleaf_model::collect(stream).await {
            Ok(resp) => Json(openai::response_to_openai(&resp, created)).into_response(),
            Err(e) => error(StatusCode::BAD_GATEWAY, e.to_string()),
        }
    }
}

/// Turn a canonical response stream into an OpenAI SSE response.
fn sse_from_response(
    stream: ResponseStream,
    encoder: ChunkEncoder,
) -> Sse<impl Stream<Item = Result<SseEvent, Infallible>>> {
    let body = async_stream::stream! {
        let mut stream = stream;
        // One buffer for the whole connection: `encode_into` clears and refills it per chunk, so the
        // streaming hot path allocates nothing per frame (`axum` copies the borrowed `&str` into its
        // own event buffer). 256 bytes covers a typical content frame without growing.
        let mut buf: Vec<u8> = Vec::with_capacity(256);
        while let Some(item) = stream.next().await {
            match item {
                Ok(chunk) => {
                    if encoder.encode_into(&chunk, &mut buf) {
                        // serde_json always emits valid UTF-8.
                        let frame = std::str::from_utf8(&buf).expect("frame is utf-8");
                        yield Ok(SseEvent::default().data(frame));
                    }
                }
                Err(e) => {
                    let frame = json!({ "error": { "message": e.to_string() } });
                    yield Ok(SseEvent::default().data(frame.to_string()));
                    break;
                }
            }
        }
        yield Ok(SseEvent::default().data(ChunkEncoder::DONE));
    };
    Sse::new(body).keep_alive(KeepAlive::default())
}

// ---------------------------------------------------------------------------------------------
// Anthropic Messages surface (`POST /v1/messages`) — a second chat dialect on the same core
// ---------------------------------------------------------------------------------------------

/// The Anthropic Messages compat surface. Same hot path as [`chat_completions`] — authenticate → map in
/// → route → stream → map out — but in the Anthropic dialect: the credential is the `x-api-key` header
/// (an `Authorization: Bearer` token is also accepted), the body maps via [`anthropic::parse_messages_request`],
/// and the output is either an Anthropic `message` object or the Anthropic streaming-event sequence.
async fn anthropic_messages(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    let Some(token) = anthropic_key(&headers) else {
        return anthropic_error(
            StatusCode::UNAUTHORIZED,
            "authentication_error",
            "missing x-api-key",
        );
    };

    // Map in (principle 3: dialect → canonical at the edge).
    let req = match anthropic::parse_messages_request(body) {
        Ok(r) => r,
        Err(e) => {
            return anthropic_error(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                e.to_string(),
            )
        }
    };
    let logical_model = req.model.clone();
    let stream_requested = req.stream;
    let now = now_secs();

    // Authenticate: a verdict lookup, never arithmetic (principle 5). The same key store and verdict
    // overlay serve every surface — the dialect changes, the identity model does not.
    let key = match authorize_token(&state, &token, &logical_model, now).await {
        Ok(id) => id,
        Err(e) => return anthropic_auth_error(e),
    };

    let request_id = next_request_id(&state.request_seq);
    let id = format!("msg_{request_id}");

    // Route → stream.
    let stream = match state.engine.run(req, key, request_id, now).await {
        Ok(s) => s,
        Err(e) => return anthropic_engine_error(e),
    };

    // Map out (principle 4: a non-streaming response is a collected stream).
    if stream_requested {
        let encoder = anthropic::EventEncoder::new(id, logical_model);
        anthropic_sse_from_response(stream, encoder).into_response()
    } else {
        match llmleaf_model::collect(stream).await {
            Ok(resp) => Json(anthropic::response_to_anthropic(&resp, &id)).into_response(),
            Err(e) => anthropic_error(StatusCode::BAD_GATEWAY, "api_error", e.to_string()),
        }
    }
}

/// Turn a canonical response stream into the Anthropic streaming-event SSE response. Unlike the OpenAI
/// pump, one canonical chunk can yield several named events, the trailing `message_delta`/`message_stop`
/// are flushed by [`anthropic::EventEncoder::finish`] after the stream ends, and a mid-stream failure is
/// surfaced as an `event: error` frame (Anthropic has no `[DONE]` sentinel).
fn anthropic_sse_from_response(
    stream: ResponseStream,
    encoder: anthropic::EventEncoder,
) -> Sse<impl Stream<Item = Result<SseEvent, Infallible>>> {
    let body = async_stream::stream! {
        let mut stream = stream;
        let mut encoder = encoder;
        // One frame buffer for the whole connection: `encode`/`finish` clear and refill it, reusing the
        // Vec's capacity across the stream.
        let mut frames: Vec<anthropic::Frame> = Vec::new();
        let mut errored = false;
        while let Some(item) = stream.next().await {
            match item {
                Ok(chunk) => {
                    frames.clear();
                    encoder.encode(&chunk, &mut frames);
                    for frame in &frames {
                        yield Ok(SseEvent::default().event(frame.event).data(frame.data.as_str()));
                    }
                }
                Err(e) => {
                    yield Ok(SseEvent::default()
                        .event("error")
                        .data(anthropic::error_data(&e.to_string())));
                    errored = true;
                    break;
                }
            }
        }
        if !errored {
            frames.clear();
            encoder.finish(&mut frames);
            for frame in &frames {
                yield Ok(SseEvent::default().event(frame.event).data(frame.data.as_str()));
            }
        }
    };
    Sse::new(body).keep_alive(KeepAlive::default())
}

// ---------------------------------------------------------------------------------------------
// OpenAI Responses surface (`POST /v1/responses`) — a third chat dialect on the same core
// ---------------------------------------------------------------------------------------------

/// The OpenAI Responses compat surface. Same hot path as [`chat_completions`] — authenticate → map in →
/// route → stream → map out — but in the Responses dialect: the body maps via
/// [`responses::parse_responses_request`], and the output is either a single Responses `response` object
/// or the Responses streaming-event sequence. Served statelessly: `store` is always answered `false`,
/// and the stateless-continuation knobs are rejected at the map-in edge (see [`responses`]).
async fn responses_create(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    let Some(token) = bearer(&headers) else {
        return error(StatusCode::UNAUTHORIZED, "missing bearer token");
    };

    // Map in (principle 3: dialect → canonical at the edge).
    let req = match responses::parse_responses_request(body) {
        Ok(r) => r,
        Err(e) => return error(StatusCode::BAD_REQUEST, e.to_string()),
    };
    let logical_model = req.model.clone();
    let stream_requested = req.stream;
    // Capture what the response echoes from the request before the request is consumed by the engine.
    let echo = responses::RequestEcho::from_request(&req);
    let now = now_secs();

    // Authenticate: a verdict lookup, never arithmetic (principle 5).
    let key = match authorize_token(&state, &token, &logical_model, now).await {
        Ok(id) => id,
        Err(e) => return auth_error(e),
    };

    let request_id = next_request_id(&state.request_seq);
    let created = now;

    // Route → stream.
    let stream = match state.engine.run(req, key, request_id.clone(), now).await {
        Ok(s) => s,
        Err(e) => return engine_error(e),
    };

    // Map out (principle 4: a non-streaming response is a collected stream).
    if stream_requested {
        let encoder = responses::EventEncoder::new(request_id, logical_model, created, echo);
        responses_sse_from_response(stream, encoder).into_response()
    } else {
        match llmleaf_model::collect(stream).await {
            Ok(mut resp) => {
                // The Responses object is keyed by the consumer request id (`resp_<id>`), not the
                // upstream's — stamp it so `id`, the output-item ids, and the streaming encoder agree.
                resp.id = request_id;
                Json(responses::response_to_responses(&resp, &echo, created)).into_response()
            }
            Err(e) => error(StatusCode::BAD_GATEWAY, e.to_string()),
        }
    }
}

/// Turn a canonical response stream into the Responses streaming-event SSE response. Like the Anthropic
/// pump, one canonical chunk can yield several named events, the terminal `response.completed` (or
/// `response.incomplete`/`response.failed`) is flushed by [`responses::EventEncoder::finish`] after the
/// stream ends, and a mid-stream failure surfaces as an `event: error` frame. There is no `[DONE]`
/// sentinel — the Responses stream ends after the terminal event.
fn responses_sse_from_response(
    stream: ResponseStream,
    encoder: responses::EventEncoder,
) -> Sse<impl Stream<Item = Result<SseEvent, Infallible>>> {
    let body = async_stream::stream! {
        let mut stream = stream;
        let mut encoder = encoder;
        // One frame buffer for the whole connection: `encode`/`finish` clear and refill it, reusing the
        // Vec's capacity across the stream.
        let mut frames: Vec<responses::Frame> = Vec::new();
        let mut errored = false;
        while let Some(item) = stream.next().await {
            match item {
                Ok(chunk) => {
                    frames.clear();
                    encoder.encode(&chunk, &mut frames);
                    for frame in &frames {
                        yield Ok(SseEvent::default().event(frame.event).data(frame.data.as_str()));
                    }
                }
                Err(e) => {
                    let frame = encoder.error_frame(&e.to_string());
                    yield Ok(SseEvent::default().event(frame.event).data(frame.data.as_str()));
                    errored = true;
                    break;
                }
            }
        }
        if !errored {
            frames.clear();
            encoder.finish(&mut frames);
            for frame in &frames {
                yield Ok(SseEvent::default().event(frame.event).data(frame.data.as_str()));
            }
        }
    };
    Sse::new(body).keep_alive(KeepAlive::default())
}

/// `GET /v1/responses/{id}` — always 404. llmleaf is stateless and stores no responses (`store` is
/// always `false`), so retrieval is unsupported by design. This is P7 transparency: a client that
/// ignored `"store": false` is told exactly why, in the same OpenAI error envelope as the rest of the
/// surface.
async fn get_response(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Response {
    // Authenticate first, like every other consumer route — SOUL.md enumerates the unauthenticated
    // surface (`/healthz`, `/v1/openapi.json`) and this endpoint is not on it. Identity only:
    // retrieval names no model, so there is no allow-list to check.
    let Some(token) = bearer(&headers) else {
        return error(StatusCode::UNAUTHORIZED, "missing bearer token");
    };
    if let Err(e) = authorize_token_identity(&state, &token, now_secs()).await {
        return auth_error(e);
    }
    error(
        StatusCode::NOT_FOUND,
        format!(
            "response '{id}' not found: llmleaf is stateless and does not store responses \
             (`store` is always false), so retrieval is unsupported by design"
        ),
    )
}

// ---------------------------------------------------------------------------------------------
// Realtime surface (`GET /v1/realtime`, WebSocket) — the OpenAI Realtime dialect
// ---------------------------------------------------------------------------------------------

#[derive(Deserialize)]
struct RealtimeQuery {
    #[serde(default)]
    model: Option<String>,
}

/// The OpenAI Realtime WebSocket surface. Authenticates the handshake (mirroring an OpenAI Realtime
/// SDK client: a `Authorization: Bearer` header OR the `openai-insecure-api-key.<token>`
/// subprotocol), then upgrades and either proxies to a native realtime provider or runs the bridge.
async fn realtime_ws(
    State(state): State<AppState>,
    ws: WebSocketUpgrade,
    headers: HeaderMap,
    Query(q): Query<RealtimeQuery>,
) -> Response {
    let Some(token) = realtime_token(&headers) else {
        return error(StatusCode::UNAUTHORIZED, "missing realtime credential");
    };
    // The model is a handshake query param (as in OpenAI's `?model=`); we authorize and route on it.
    let Some(model) = q.model.filter(|m| !m.is_empty()) else {
        return error(StatusCode::BAD_REQUEST, "missing ?model= query parameter");
    };
    let now = now_secs();
    let key = match authorize_token(&state, &token, &model, now).await {
        Ok(id) => id,
        Err(e) => return auth_error(e),
    };
    let request_id = next_request_id(&state.request_seq);

    // Echo the GA `realtime` subprotocol when the client offered subprotocols (a no-op otherwise).
    ws.protocols(["realtime"])
        .on_upgrade(move |socket| handle_realtime(state, socket, model, key, request_id, now))
        .into_response()
}

/// Extract the consumer token from a realtime handshake: the generic `Authorization: Bearer` header,
/// else the browser-pattern subprotocol token (whose dialect-specific shape the realtime edge owns —
/// principle 2 keeps the provider-branded name out of the core's generic auth). A stray legacy
/// `OpenAI-Beta` header is tolerated and ignored.
fn realtime_token(headers: &HeaderMap) -> Option<String> {
    if let Some(tok) = bearer(headers) {
        return Some(tok);
    }
    let protocols = headers.get("sec-websocket-protocol")?.to_str().ok()?;
    rt_wire::token_from_subprotocols(protocols)
}

/// Pick the native realtime path when an eligible target serves it; otherwise bridge over chat.
async fn handle_realtime(
    state: AppState,
    socket: WebSocket,
    model: String,
    key: String,
    request_id: String,
    now: u64,
) {
    if let Some(target) = state.engine.realtime_target(&model, &request_id, now) {
        run_native_realtime(state, socket, target, model, key, request_id).await;
    } else {
        rt_session::run_bridge(
            socket,
            state.engine.clone(),
            state.events.clone(),
            key,
            model,
            request_id,
        )
        .await;
    }
}

/// Proxy a native realtime session in two phases, so a native transport that fails *before producing
/// any output* falls back to the chat bridge instead of erroring the consumer (principle 8: fail toward
/// availability), while a native session that has already committed is proxied verbatim (principle 7).
///
/// **Phase 1 — probe (socket unsplit).** The native session is "committed" only once the provider emits
/// its first frame. Until then the consumer has seen no upstream output, so any provider error is safe
/// to recover from: we run the chat bridge on the *same* socket, replaying the consumer frames buffered
/// during the probe. A genuinely broken key/route is not masked — it resurfaces as an ordinary bridge
/// turn error (principle 7).
///
/// **Phase 2 — committed native (post-first-frame).** Split, pump verbatim both ways, tap usage. A
/// MID-SESSION provider error keeps the error-frame-and-close behavior: the upstream holds session state
/// the core cannot replay, so a silent restart onto the bridge would fabricate a fresh context
/// (principle 7). Fallback is a strictly pre-output affordance.
async fn run_native_realtime(
    state: AppState,
    mut socket: WebSocket,
    target: crate::engine::RealtimeTarget,
    model: String,
    key: String,
    request_id: String,
) {
    let events = state.events.clone();
    let engine = state.engine.clone();
    let provider_name = target.provider_name.clone();
    events.emit(Event::RequestStarted {
        id: request_id.clone(),
        key: key.clone(),
        model: model.clone(),
        request: None,
    });
    events.emit(Event::RequestRouted {
        id: request_id.clone(),
        provider: provider_name.clone(),
        upstream_model: target.params.model.clone(),
    });

    // Two channels form the dialect-opaque peer; the provider owns the upstream socket.
    let (in_tx, in_rx) = mpsc::channel::<RealtimeWire>(64);
    let (out_tx, mut out_rx) = mpsc::channel::<RealtimeWire>(64);
    let peer = RealtimePeer {
        inbound: in_rx,
        outbound: out_tx,
    };

    let provider = target.provider;
    let cx = target.cx;
    let params = target.params;
    let provider_task = tokio::spawn(async move { provider.realtime(params, peer, &cx).await });

    // ---- Phase 1: probe. The socket stays UNSPLIT until the provider proves the session is live.
    // Every non-committed outcome is handled inline and returns here; the loop only *breaks* with the
    // first provider frame, so `socket`/`out_rx`/`in_tx`/`provider_task` all survive intact into phase 2.
    let mut replay: Vec<RealtimeWire> = Vec::new();
    let committed: RealtimeWire = loop {
        tokio::select! {
            maybe_wire = out_rx.recv() => {
                match maybe_wire {
                    // First provider frame ⇒ the native session is committed. Forward it in phase 2.
                    Some(wire) => break wire,
                    // The provider task ended before emitting a single frame. Nothing consumer-visible
                    // has been sent, so we are free to fall back or complete an empty session.
                    None => {
                        match provider_task.await {
                            // Pre-output failure ⇒ FALL BACK to the chat bridge on the same, still-unsplit
                            // socket, replaying the consumer frames buffered during the probe. The native
                            // reason stays visible as a warn; a genuinely broken key/route resurfaces as an
                            // ordinary bridge turn error (principle 7), so nothing is masked. No second
                            // `RequestStarted` — the bridge reports routing per-turn via `Engine::run`.
                            Ok(Err(e)) => {
                                tracing::warn!(
                                    provider = %provider_name, error = %e,
                                    "native realtime failed before first frame; falling back to chat bridge"
                                );
                                rt_session::run_bridge_replaying(
                                    socket,
                                    engine,
                                    key,
                                    model,
                                    &request_id,
                                    std::mem::take(&mut replay),
                                )
                                .await;
                                events.emit(Event::RequestCompleted {
                                    id: request_id,
                                    finish: None,
                                });
                            }
                            // A clean close with no output at all: treat as an empty native session.
                            Ok(Ok(())) => {
                                let _ = socket.send(WsMessage::Close(None)).await;
                                events.emit(Event::RequestCompleted {
                                    id: request_id,
                                    finish: None,
                                });
                            }
                            Err(_join) => {
                                events.emit(Event::RequestFailed {
                                    id: request_id,
                                    error: "realtime session task aborted".to_string(),
                                });
                            }
                        }
                        return;
                    }
                }
            }
            maybe_msg = socket.recv() => {
                match maybe_msg {
                    Some(Ok(msg)) => match msg {
                        // A fast client may send frames before the upstream is up. Forward them to the
                        // (connecting) provider AND buffer a copy so a fallback can replay them.
                        WsMessage::Text(t) => {
                            let wire = RealtimeWire::Text(t.as_str().to_owned());
                            replay.push(wire.clone());
                            let _ = in_tx.send(wire).await;
                        }
                        WsMessage::Binary(b) => {
                            let wire = RealtimeWire::Binary(b);
                            replay.push(wire.clone());
                            let _ = in_tx.send(wire).await;
                        }
                        // The consumer closed before the upstream produced anything: nothing to proxy.
                        WsMessage::Close(_) => {
                            provider_task.abort();
                            events.emit(Event::RequestCompleted {
                                id: request_id,
                                finish: None,
                            });
                            return;
                        }
                        WsMessage::Ping(_) | WsMessage::Pong(_) => {}
                    },
                    // Socket error or end before the upstream produced anything: abort and complete.
                    Some(Err(_)) | None => {
                        provider_task.abort();
                        events.emit(Event::RequestCompleted {
                            id: request_id,
                            finish: None,
                        });
                        return;
                    }
                }
            }
        }
    };

    // ---- Phase 2: committed native. Exactly the prior behavior — split, pump verbatim both ways, tap
    // usage — starting from the frame that committed the session.
    let (mut ws_tx, mut ws_rx) = socket.split();

    // Consumer → provider: forward every frame verbatim, then signal close.
    let in_pump = tokio::spawn(async move {
        while let Some(Ok(msg)) = ws_rx.next().await {
            let wire = match msg {
                WsMessage::Text(t) => RealtimeWire::Text(t.as_str().to_owned()),
                WsMessage::Binary(b) => RealtimeWire::Binary(b),
                WsMessage::Close(_) => break,
                WsMessage::Ping(_) | WsMessage::Pong(_) => continue,
            };
            if in_tx.send(wire).await.is_err() {
                return;
            }
        }
        let _ = in_tx.send(RealtimeWire::Close).await;
    });

    // Provider → consumer: forward verbatim, tapping usage from the terminal frame for the event bus.
    // The committed first frame is processed through the same match (so usage is tapped even on it),
    // then the loop drains the rest.
    let mut pending = Some(committed);
    loop {
        let wire = match pending.take() {
            Some(w) => w,
            None => match out_rx.recv().await {
                Some(w) => w,
                None => break,
            },
        };
        match wire {
            RealtimeWire::Text(t) => {
                if let Some(usage) = rt_wire::usage_from_server_frame(&t) {
                    events.emit(Event::Usage {
                        id: request_id.clone(),
                        key: key.clone(),
                        model: model.clone(),
                        usage: engine.price(&model, usage),
                    });
                }
                if ws_tx.send(WsMessage::Text(t.into())).await.is_err() {
                    break;
                }
            }
            RealtimeWire::Binary(b) => {
                if ws_tx.send(WsMessage::Binary(b)).await.is_err() {
                    break;
                }
            }
            RealtimeWire::Close => {
                let _ = ws_tx.send(WsMessage::Close(None)).await;
                break;
            }
        }
    }

    // The session ended (provider returned ⇒ out_tx dropped ⇒ loop drained). Report its outcome.
    let outcome = provider_task.await;
    in_pump.abort();
    match outcome {
        Ok(Ok(())) => {
            events.emit(Event::RequestCompleted {
                id: request_id,
                finish: None,
            });
        }
        Ok(Err(e)) => {
            // A MID-SESSION failure (the session had already committed): surface it to the consumer as a
            // Realtime error frame and close — never a silent restart onto the bridge, which would
            // fabricate a fresh context the client never asked for (principle 7).
            let frame = rt_wire::error_frame("upstream_error", &e.to_string());
            let _ = ws_tx.send(WsMessage::Text(frame.to_string().into())).await;
            events.emit(Event::RequestFailed {
                id: request_id,
                error: e.to_string(),
            });
        }
        Err(_join) => {
            events.emit(Event::RequestFailed {
                id: request_id,
                error: "realtime session task aborted".to_string(),
            });
        }
    }
    let _ = ws_tx.send(WsMessage::Close(None)).await;
}

// ---------------------------------------------------------------------------------------------
// Embeddings surface (`POST /v1/embeddings`)
// ---------------------------------------------------------------------------------------------

async fn embeddings_handler(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    let Some(token) = bearer(&headers) else {
        return error(StatusCode::UNAUTHORIZED, "missing bearer token");
    };
    let req = match embeddings::parse_embedding_request(body) {
        Ok(r) => r,
        Err(e) => return error(StatusCode::BAD_REQUEST, e.to_string()),
    };
    let model = req.model.clone();
    let encoding_format = req.encoding_format.clone();
    let now = now_secs();
    let key = match authorize_token(&state, &token, &model, now).await {
        Ok(id) => id,
        Err(e) => return auth_error(e),
    };
    let request_id = next_request_id(&state.request_seq);

    match state.engine.embed(req, key, request_id, now).await {
        Ok(resp) => Json(embeddings::response_to_openai(
            &resp,
            encoding_format.as_deref(),
        ))
        .into_response(),
        Err(e) => engine_error(e),
    }
}

// ---------------------------------------------------------------------------------------------
// Rerank surface (`POST /v1/rerank`) — query + documents in, relevance-ordered scores out
// ---------------------------------------------------------------------------------------------

async fn rerank_handler(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    let Some(token) = bearer(&headers) else {
        return error(StatusCode::UNAUTHORIZED, "missing bearer token");
    };
    let req = match rerank::parse_rerank_request(body) {
        Ok(r) => r,
        Err(e) => return error(StatusCode::BAD_REQUEST, e.to_string()),
    };
    let model = req.model.clone();
    let now = now_secs();
    let key = match authorize_token(&state, &token, &model, now).await {
        Ok(id) => id,
        Err(e) => return auth_error(e),
    };
    let request_id = next_request_id(&state.request_seq);

    match state.engine.rerank(req, key, request_id, now).await {
        Ok(resp) => Json(rerank::response_to_wire(&resp)).into_response(),
        Err(e) => engine_error(e),
    }
}

// ---------------------------------------------------------------------------------------------
// Speech surface (`POST /v1/audio/speech`, text-to-speech) — raw audio body, not SSE
// ---------------------------------------------------------------------------------------------

async fn audio_speech(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    let Some(token) = bearer(&headers) else {
        return error(StatusCode::UNAUTHORIZED, "missing bearer token");
    };
    let req = match speech::parse_speech_request(body) {
        Ok(r) => r,
        Err(e) => return error(StatusCode::BAD_REQUEST, e.to_string()),
    };
    let model = req.model.clone();
    let response_format = req.response_format.clone();
    let now = now_secs();
    let key = match authorize_token(&state, &token, &model, now).await {
        Ok(id) => id,
        Err(e) => return auth_error(e),
    };
    let request_id = next_request_id(&state.request_seq);

    let mut stream = match state.engine.speech(req, key, request_id, now).await {
        Ok(s) => s,
        Err(e) => return engine_error(e),
    };

    // Pull the stream to its opening `Start` to learn the audio MIME type before headers are sent.
    // Any `Data` that somehow precedes `Start` is buffered and re-emitted, never dropped (principle 7).
    let mut content_type: Option<String> = None;
    let mut pending: Vec<Bytes> = Vec::new();
    while content_type.is_none() {
        match stream.next().await {
            Some(Ok(AudioChunk::Start { content_type: ct })) => content_type = Some(ct),
            Some(Ok(AudioChunk::Data(b))) => pending.push(b),
            Some(Ok(_)) => {}
            Some(Err(e)) => return error(StatusCode::BAD_GATEWAY, e.to_string()),
            None => break,
        }
    }
    let content_type = content_type
        .unwrap_or_else(|| speech::content_type_for_format(response_format.as_deref()).to_string());

    // The remaining audio bytes stream straight to the client; the instrumented stream behind it emits
    // usage/completion events as they pass.
    let body = Body::from_stream(async_stream::stream! {
        for b in pending {
            yield Ok::<Bytes, std::convert::Infallible>(b);
        }
        while let Some(item) = stream.next().await {
            match item {
                Ok(AudioChunk::Data(b)) => yield Ok(b),
                Ok(_) => {}
                // Headers are already on the wire; we can only stop the body. The failure still
                // surfaces on the event stream (the instrumented stream emitted RequestFailed).
                Err(_) => break,
            }
        }
    });

    Response::builder()
        .header(CONTENT_TYPE, content_type)
        .body(body)
        .expect("valid audio response")
}

// ---------------------------------------------------------------------------------------------
// Voice catalog (`GET /v1/audio/voices?model=…`) — which voices a speech model can use
// ---------------------------------------------------------------------------------------------

#[derive(Deserialize)]
struct VoicesQuery {
    /// The logical model whose voices to list; routed exactly like a speech request.
    model: String,
}

/// List the voices available for a speech model. Bearer-authed like the other `/v1` surfaces; the model
/// routes to its provider chain and the first voice-aware target answers (principle 1), so for a
/// static-catalog provider (e.g. OpenAI) this never leaves the node. The envelope is
/// `{ "model": …, "voices": [ { "id", "name?", "languages?", … } ] }`.
async fn audio_voices(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(query): Query<VoicesQuery>,
) -> Response {
    let Some(token) = bearer(&headers) else {
        return error(StatusCode::UNAUTHORIZED, "missing bearer token");
    };
    let now = now_secs();
    let key = match authorize_token(&state, &token, &query.model, now).await {
        Ok(id) => id,
        Err(e) => return auth_error(e),
    };
    let request_id = next_request_id(&state.request_seq);

    match state
        .engine
        .voices(query.model.clone(), key, request_id, now)
        .await
    {
        Ok(voices) => Json(json!({ "model": query.model, "voices": voices })).into_response(),
        Err(e) => engine_error(e),
    }
}

// ---------------------------------------------------------------------------------------------
// Model catalog surface (`GET /v1/models`) — OpenRouter-shaped discovery
// ---------------------------------------------------------------------------------------------
//
// Lists the models this node ACTUALLY SERVES: the explicitly-routed models, plus — for each prefix
// namespace — the real upstream catalog the provider reports (passed through with the prefixed id).
// The bundled pricing dataset is NOT an availability source; it only ENRICHES served models (and fills
// gaps for providers whose list-models API is sparse). The wire shape mirrors OpenRouter's
// `GET /api/v1/models` (`{ "data": [ … ] }`); `?type=` / `?search=` are llmleaf extensions applied
// server-side.
//
// Two views on one endpoint. The PUBLIC view (any valid consumer bearer) carries no provider identity
// — only ids, capabilities, and pricing. The ADMIN view (a valid `x-admin-token` *on top of* the
// bearer) adds a per-model `endpoints` array: the provider chain with names and node-local health.
// Provider topology lives only behind the admin token, consistent with the `/admin/*` surface (SOUL
// P2/P7); `down` is this node's view, never a cluster verdict (P9).

/// Query string for `GET /v1/models`. Both filters are llmleaf extensions (OpenRouter's `/models`
/// takes no query params) and are applied server-side after enrichment.
#[derive(Debug, Default, Deserialize)]
struct ModelsQuery {
    /// Modality filter: absent / `all` ⇒ no filter; else `llm` | `tts` | `stt` | `embedding` |
    /// `rerank`.
    #[serde(default, rename = "type")]
    kind: Option<String>,
    /// Case-insensitive substring over the model id.
    #[serde(default)]
    search: Option<String>,
}

/// Provenance of a served-model entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Source {
    /// An explicitly-configured `[[routes]]` model (a logical id).
    Route,
    /// A model passed through from a prefix provider's upstream catalog, OR the non-enumerable
    /// `<prefix>/*` namespace marker.
    Prefix,
}

/// One served model, before rendering. `meta` is the merged metadata (provider-reported, gaps filled
/// from the bundled dataset); `None` for a namespace marker. `callable` is false only for the
/// `<prefix>/*` marker (a namespace is not a model id). For prefix entries, `provider`/`upstream_model`
/// name the upstream so the admin `endpoints` line needs no re-resolution.
struct ModelEntry {
    source: Source,
    meta: Option<ModelInfo>,
    callable: bool,
    provider: Option<String>,
    upstream_model: Option<String>,
}

/// List the models this node serves (`GET /v1/models`), OpenRouter-shaped. See the section header for
/// the served-only union and the public-vs-admin split.
async fn list_models(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(query): Query<ModelsQuery>,
) -> Response {
    // Every `/v1` surface requires a consumer bearer. Listing carries no single model, so authenticate
    // identity-only (like the batch surfaces); model scoping is applied to the result below.
    let Some(token) = bearer(&headers) else {
        return error(StatusCode::UNAUTHORIZED, "missing bearer token");
    };
    let now = now_secs();
    let key = match authorize_token_identity(&state, &token, now).await {
        Ok(id) => id,
        Err(e) => return auth_error(e),
    };

    // Parse the modality filter up front — an unknown value is a 400, never silently treated as `all`.
    let modality_filter = match parse_modality_filter(query.kind.as_deref()) {
        Ok(f) => f,
        Err(msg) => return error(StatusCode::BAD_REQUEST, msg),
    };
    let search = query.search.as_deref().map(str::to_ascii_lowercase);
    let admin = admin_view_authorized(&state, &headers);
    // Scope to the key's allowed models (OpenAI/OpenRouter return only accessible models; allow-list
    // entries may be `*` wildcard patterns). `None` ⇒ unrestricted.
    let allowed = state.keys.model_scope(&key);

    let engine = &state.engine;

    // Build the SERVED-model union, deduped by id; a BTreeMap gives deterministic, id-sorted output.
    let mut entries: BTreeMap<String, ModelEntry> = BTreeMap::new();

    // 1. Explicitly-routed models (logical ids). Enrich from the bundled dataset by the logical id,
    //    then by the first target's upstream model id — never inventing metadata, only looking it up.
    for model in engine.router().models() {
        let card = engine.pricing().card(model).or_else(|| {
            engine
                .resolve_targets(model)
                .and_then(|t| t.first().and_then(|t| t.model.clone()))
                .and_then(|upstream| engine.pricing().card(&upstream))
        });
        let meta = enrich(ModelInfo::new(model), card);
        entries.insert(
            model.to_string(),
            ModelEntry {
                source: Source::Route,
                meta: Some(meta),
                callable: true,
                provider: None,
                upstream_model: None,
            },
        );
    }

    // 2. Prefix namespaces: pass through each provider's REAL upstream catalog (prefixed ids), enhanced
    //    from the bundled dataset for whatever the provider's list-models API does not report. A
    //    provider that cannot enumerate (Unsupported) — or a fetch that fails — degrades to a single
    //    non-callable `<prefix>/*` marker rather than failing the whole listing or guessing members.
    let prefixes: Vec<(String, String)> = engine
        .router()
        .prefixes()
        .map(|(p, n)| (p.to_string(), n.to_string()))
        .collect();
    for (prefix, provider) in prefixes {
        let request_id = next_request_id(&state.request_seq);
        match engine.provider_models(&provider, &request_id).await {
            Ok(models) => {
                for info in models {
                    let bare = info.id.clone();
                    let id = format!("{prefix}/{bare}");
                    let mut meta = enrich(info, engine.pricing().card(&bare));
                    // Tag the display name with the namespace so a consumer can tell which prefix
                    // serves the model without parsing the id.
                    let display = meta.name.take().unwrap_or_else(|| bare.clone());
                    meta.name = Some(format!("[{prefix}] {display}"));
                    entries.entry(id).or_insert(ModelEntry {
                        source: Source::Prefix,
                        meta: Some(meta),
                        callable: true,
                        provider: Some(provider.clone()),
                        upstream_model: Some(bare),
                    });
                }
            }
            Err(_) => {
                entries.entry(format!("{prefix}/*")).or_insert(ModelEntry {
                    source: Source::Prefix,
                    meta: None,
                    callable: false,
                    provider: Some(provider.clone()),
                    upstream_model: Some("*".to_string()),
                });
            }
        }
    }

    // Per-key scoping, then prune namespace markers from the public view (admin keeps them).
    if let Some(allowed) = &allowed {
        entries.retain(|id, _| allowed.permits(id));
    }
    if !admin {
        entries.retain(|_, e| e.callable);
    }

    // Which prefix providers classify their catalog by type at all. A provider whose list-models API
    // reports a modality for at least one of its models "supports model types"; one that reports none
    // does not (e.g. a bare OpenAI-compatible endpoint, or a speech-only upstream that never tags its
    // models). For a provider that does not support model types, applying a `?type=` filter would hide
    // its whole catalog, so the filter below is IGNORED for it — its models pass through unfiltered.
    // Built only when a filter is active; route entries (no provider) keep the strict rule below.
    let typed_providers: HashSet<String> = match modality_filter {
        None => HashSet::new(),
        Some(_) => entries
            .values()
            .filter(|e| e.meta.as_ref().and_then(|m| m.modality).is_some())
            .filter_map(|e| e.provider.clone())
            .collect(),
    };

    // Filter (modality, then search) and render. Filters run after enrichment so they see merged meta.
    let data: Vec<Value> = entries
        .into_iter()
        .filter(|(_, e)| match modality_filter {
            None => true,
            // A provider that does not classify its catalog by type ignores the filter (pass through).
            Some(_) if matches!(&e.provider, Some(p) if !typed_providers.contains(p)) => true,
            // Otherwise a specific filter keeps only entries whose modality is known AND matches; an
            // unknown modality is excluded — you can only filter by what you know.
            Some(m) => e.meta.as_ref().and_then(|m| m.modality) == Some(m),
        })
        .filter(|(id, _)| match &search {
            None => true,
            Some(needle) => id.to_ascii_lowercase().contains(needle.as_str()),
        })
        .map(|(id, e)| render_model(&id, &e, admin, engine, now))
        .collect();

    Json(json!({ "data": data })).into_response()
}

/// Fill the `None` fields of a provider-reported [`ModelInfo`] from the bundled dataset card. The
/// provider's own value always wins (it knows its deployment); the dataset only fills gaps — the
/// "provider-specific enhancement" for upstreams whose list-models API does not report limits/pricing.
fn enrich(mut info: ModelInfo, card: Option<ModelCard>) -> ModelInfo {
    if let Some(c) = card {
        info.modality = info.modality.or(c.modality);
        info.max_context = info.max_context.or(c.max_context);
        info.max_output = info.max_output.or(c.max_output);
        info.max_thinking = info.max_thinking.or(c.max_thinking);
        info.input_per_mtok = info.input_per_mtok.or(c.input_per_mtok);
        info.output_per_mtok = info.output_per_mtok.or(c.output_per_mtok);
        // Param metadata: the provider's own report wins (a non-empty value), the dataset only fills a
        // gap — same "provider knows its deployment" rule as the scalars above.
        if info.unsupported_parameters.is_empty() {
            if let Some(u) = c.unsupported_parameters {
                info.unsupported_parameters = u;
            }
        }
        if info.default_parameters.is_empty() {
            if let Some(d) = c.default_parameters {
                info.default_parameters = d;
            }
        }
    }
    info
}

/// Parse the `?type=` value into a modality filter. Absent / empty / `all` ⇒ no filter (`Ok(None)`);
/// a known modality ⇒ `Ok(Some(m))`; anything else ⇒ `Err(message)` for a 400.
fn parse_modality_filter(raw: Option<&str>) -> Result<Option<Modality>, String> {
    match raw.map(str::trim) {
        None | Some("") | Some("all") => Ok(None),
        Some("llm") => Ok(Some(Modality::Llm)),
        Some("tts") => Ok(Some(Modality::Tts)),
        Some("stt") => Ok(Some(Modality::Stt)),
        Some("embedding") => Ok(Some(Modality::Embedding)),
        Some("rerank") => Ok(Some(Modality::Rerank)),
        Some(other) => Err(format!(
            "unknown type '{other}' (expected all|llm|tts|stt|embedding|rerank)"
        )),
    }
}

/// Whether to render the admin-enhanced view of `GET /v1/models` (the per-model `endpoints` chain +
/// node-local health). Unlike [`guard_admin`], this NEVER returns a response — it silently selects the
/// view, so a consumer with a valid bearer but no/invalid admin token still gets the public catalog
/// (this endpoint is public-by-default). Keep it DISTINCT from `guard_admin`: merging them would
/// 404/401 the public path. True only when an admin token is configured AND `x-admin-token` matches.
fn admin_view_authorized(state: &AppState, headers: &HeaderMap) -> bool {
    state.admin_view(presented_admin_token(headers))
}

/// Render one served-model entry to its OpenRouter-shaped JSON object. `admin` adds the gated
/// `endpoints` chain.
fn render_model(id: &str, entry: &ModelEntry, admin: bool, engine: &Engine, now: u64) -> Value {
    let meta = entry.meta.as_ref();
    let modality = meta.and_then(|m| m.modality);
    let max_context = meta.and_then(|m| m.max_context);
    let max_output = meta.and_then(|m| m.max_output);
    let max_thinking = meta.and_then(|m| m.max_thinking);
    let name = meta
        .and_then(|m| m.name.clone())
        .unwrap_or_else(|| id.to_string());

    let mut top_provider = serde_json::Map::new();
    top_provider.insert("context_length".into(), opt_u32(max_context));
    top_provider.insert("max_completion_tokens".into(), opt_u32(max_output));
    top_provider.insert("is_moderated".into(), json!(false));
    // llmleaf extension: a model's published thinking budget. Additive — OpenRouter clients ignore
    // unknown fields. Present only when known, so absence is truthful.
    if let Some(t) = max_thinking {
        top_provider.insert("max_thinking_tokens".into(), json!(t));
    }

    let mut obj = serde_json::Map::new();
    obj.insert("id".into(), json!(id));
    obj.insert("canonical_slug".into(), json!(id));
    obj.insert("name".into(), json!(name));
    obj.insert("created".into(), json!(0));
    obj.insert("description".into(), json!(""));
    obj.insert("context_length".into(), opt_u32(max_context));
    obj.insert("architecture".into(), architecture_value(meta, modality));
    obj.insert("pricing".into(), pricing_json(meta));
    obj.insert("top_provider".into(), Value::Object(top_provider));
    obj.insert("per_request_limits".into(), Value::Null);
    // `supported_parameters`: a provider's OWN positive list (e.g. an OpenRouter passthrough that
    // reports one) is authoritative and passes through verbatim; otherwise we compute it from the
    // modality baseline minus the model's `unsupported_parameters`. The negative list and recommended
    // defaults are surfaced alongside (additive — OpenRouter clients ignore unknown fields) so a
    // consumer can see *why* a param is absent and what value the model pins.
    let unsupported = meta
        .map(|m| m.unsupported_parameters.as_slice())
        .unwrap_or(&[]);
    let supported = match meta.and_then(|m| m.supported_parameters.as_ref()) {
        Some(list) => json!(list),
        None => supported_parameters(modality, max_thinking.is_some(), unsupported),
    };
    obj.insert("supported_parameters".into(), supported);
    if !unsupported.is_empty() {
        obj.insert("unsupported_parameters".into(), json!(unsupported));
    }
    if let Some(defaults) = meta
        .map(|m| &m.default_parameters)
        .filter(|d| !d.is_empty())
    {
        obj.insert("default_parameters".into(), Value::Object(defaults.clone()));
    }
    obj.insert("supported_voices".into(), Value::Null);
    if admin {
        obj.insert("endpoints".into(), endpoints_json(id, entry, engine, now));
    }
    Value::Object(obj)
}

fn opt_u32(v: Option<u32>) -> Value {
    v.map(|n| json!(n)).unwrap_or(Value::Null)
}

/// The `architecture` block for a served model. When the upstream published its OWN `architecture`
/// carrying real modality arrays — OpenRouter does, e.g. `input_modalities: ["text","image"]` for a
/// vision model — that block is AUTHORITATIVE and passes through verbatim, so fine-grained input/output
/// capabilities (which our coarse [`Modality`] enum cannot represent) survive to consumers. Only when no
/// such block exists — a bare routed model, or a stub `architecture` with no modality arrays (some
/// brands publish just `{modality, tokenizer}`) — do we fall back to [`architecture_json`], the coarse
/// block derived from our internal modality. Preserved verbatim from `openai_wire`'s `extra` (SOUL:
/// principle 7, no data loss).
fn architecture_value(meta: Option<&ModelInfo>, modality: Option<Modality>) -> Value {
    if let Some(arch) = meta
        .and_then(|m| m.extra.get("architecture"))
        .and_then(Value::as_object)
    {
        let has_modalities = ["input_modalities", "output_modalities"].iter().any(|k| {
            arch.get(*k)
                .and_then(Value::as_array)
                .is_some_and(|a| !a.is_empty())
        });
        if has_modalities {
            return Value::Object(arch.clone());
        }
    }
    architecture_json(modality)
}

/// The OpenRouter `architecture` block derived from our internal modality. An unknown modality (a
/// routed-but-uncatalogued model) yields empty arrays + null modality rather than a guess.
fn architecture_json(modality: Option<Modality>) -> Value {
    let (input, output, m): (&[&str], &[&str], Option<&str>) = match modality {
        Some(Modality::Llm) => (&["text"], &["text"], Some("text->text")),
        Some(Modality::Tts) => (&["text"], &["audio"], Some("text->audio")),
        Some(Modality::Stt) => (&["audio"], &["text"], Some("audio->text")),
        Some(Modality::Embedding) => (&["text"], &["embeddings"], Some("text->embeddings")),
        Some(Modality::Rerank) => (&["text"], &["scores"], Some("text->scores")),
        None => (&[], &[], None),
    };
    json!({
        "input_modalities": input,
        "output_modalities": output,
        "modality": m,
        "tokenizer": "Other",
        "instruct_type": Value::Null,
    })
}

/// The OpenRouter `pricing` block: per-TOKEN decimal strings. `null` when the model is not token-priced
/// (rate-less audio, or an upstream that reports no price) — never a misleading `"0"` object. A
/// present-but-zero rate (embeddings) renders `"0"`.
fn pricing_json(meta: Option<&ModelInfo>) -> Value {
    let Some(meta) = meta else {
        return Value::Null;
    };
    if meta.input_per_mtok.is_none() && meta.output_per_mtok.is_none() {
        return Value::Null;
    }
    json!({
        "prompt": per_token_str(meta.input_per_mtok.unwrap_or(0.0)),
        "completion": per_token_str(meta.output_per_mtok.unwrap_or(0.0)),
    })
}

/// Format a per-million-token rate as an OpenRouter-style per-TOKEN price string: fixed-decimal (never
/// scientific notation, which Rust's default `f64` Display emits for these tiny values), trailing zeros
/// trimmed, and a literal `"0"` for a genuine zero rate.
fn per_token_str(per_mtok: f64) -> String {
    let per_token = per_mtok / 1_000_000.0;
    if per_token == 0.0 {
        return "0".to_string();
    }
    // 12 decimal places covers the smallest realistic rate (~2e-8 per token) with margin.
    let s = format!("{per_token:.12}");
    if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        s
    }
}

/// The OpenRouter `supported_parameters` array, by modality, with the model's `unsupported` set
/// subtracted. Reasoning params are appended only when the model has a known thinking budget. An unknown
/// modality yields an empty array (no guessing). `supported` and `unsupported` stay exact complements
/// within the modality baseline, so a consumer reading either reaches the same conclusion.
fn supported_parameters(
    modality: Option<Modality>,
    has_thinking: bool,
    unsupported: &[String],
) -> Value {
    let mut params: Vec<&str> = match modality {
        Some(Modality::Llm) => vec![
            "max_tokens",
            "temperature",
            "top_p",
            "stop",
            "frequency_penalty",
            "presence_penalty",
            "tools",
            "tool_choice",
        ],
        Some(Modality::Tts) => vec!["voice", "response_format", "speed"],
        Some(Modality::Stt) => vec!["language", "prompt", "response_format", "temperature"],
        Some(Modality::Embedding) => vec!["encoding_format", "dimensions"],
        Some(Modality::Rerank) => vec!["query", "documents", "top_n", "return_documents"],
        None => vec![],
    };
    if has_thinking {
        params.extend(["reasoning", "include_reasoning", "reasoning_effort"]);
    }
    params.retain(|p| !unsupported.iter().any(|u| u == p));
    json!(params)
}

/// The admin-only `endpoints` array: where this model is served, with node-local health. A prefix
/// passthrough model (or the `<prefix>/*` marker) names its single upstream provider + bare model id; a
/// routed model lists its ordered fallback chain. Provider names appear ONLY here (admin-gated).
fn endpoints_json(id: &str, entry: &ModelEntry, engine: &Engine, now: u64) -> Value {
    if entry.source == Source::Prefix {
        let provider = entry.provider.as_deref().unwrap_or("?");
        let upstream = entry.upstream_model.as_deref().unwrap_or("*");
        return json!([{
            "provider": provider,
            "model": upstream,
            "down": engine.health().is_down(provider, now),
            "source": "prefix",
        }]);
    }
    match engine.resolve_targets(id) {
        Some(targets) => {
            let chain: Vec<Value> = targets
                .iter()
                .map(|t| {
                    json!({
                        "provider": t.provider,
                        "model": t.model.clone().unwrap_or_else(|| id.to_string()),
                        "down": engine.health().is_down(&t.provider, now),
                        "source": "route",
                    })
                })
                .collect();
            json!(chain)
        }
        None => json!([]),
    }
}

// ---------------------------------------------------------------------------------------------
// Transcription surface (`POST /v1/audio/transcriptions`, speech-to-text) — multipart upload
// ---------------------------------------------------------------------------------------------

async fn audio_transcriptions(
    State(state): State<AppState>,
    headers: HeaderMap,
    mut multipart: Multipart,
) -> Response {
    let Some(token) = bearer(&headers) else {
        return error(StatusCode::UNAUTHORIZED, "missing bearer token");
    };

    let mut form = transcription::TranscriptionForm::default();
    loop {
        match multipart.next_field().await {
            Ok(Some(field)) => {
                let name = field.name().unwrap_or_default().to_string();
                match name.as_str() {
                    "file" => {
                        form.filename = field.file_name().map(str::to_string);
                        match field.bytes().await {
                            Ok(b) => form.audio = Some(b),
                            Err(e) => {
                                return error(
                                    StatusCode::BAD_REQUEST,
                                    format!("reading `file` part: {e}"),
                                )
                            }
                        }
                    }
                    "model" => form.model = field.text().await.ok(),
                    "language" => form.language = field.text().await.ok(),
                    "prompt" => form.prompt = field.text().await.ok(),
                    "response_format" => form.response_format = field.text().await.ok(),
                    "temperature" => form.temperature = field.text().await.ok(),
                    // Drain and ignore parts we don't model (e.g. `timestamp_granularities[]`).
                    _ => {
                        let _ = field.bytes().await;
                    }
                }
            }
            Ok(None) => break,
            Err(e) => return error(StatusCode::BAD_REQUEST, format!("invalid multipart: {e}")),
        }
    }

    let response_format = form.response_format.clone();
    let req = match transcription::build_request(form) {
        Ok(r) => r,
        Err(e) => return error(StatusCode::BAD_REQUEST, e.to_string()),
    };
    let model = req.model.clone();
    let now = now_secs();
    let key = match authorize_token(&state, &token, &model, now).await {
        Ok(id) => id,
        Err(e) => return auth_error(e),
    };
    let request_id = next_request_id(&state.request_seq);

    match state.engine.transcribe(req, key, request_id, now).await {
        Ok(resp) => match transcription::encode_transcription(&resp, response_format.as_deref()) {
            TranscriptionBody::Json(v) => Json(v).into_response(),
            TranscriptionBody::Text(t) => {
                ([(CONTENT_TYPE, "text/plain; charset=utf-8")], t).into_response()
            }
        },
        Err(e) => engine_error(e),
    }
}

// ---------------------------------------------------------------------------------------------
// Batch surface (asynchronous jobs) — inline OpenAI-shaped requests in, opaque ids back
// ---------------------------------------------------------------------------------------------

/// `POST /v1/batches` — submit an inline batch. Authenticates identity once and checks *every* model
/// the batch references against the key's allow-list (a batch must not smuggle a disallowed model).
async fn create_batch(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    let Some(token) = bearer(&headers) else {
        return error(StatusCode::UNAUTHORIZED, "missing bearer token");
    };
    let spec = match batch::parse_batch_create(body) {
        Ok(s) => s,
        Err(e) => return error(StatusCode::BAD_REQUEST, e.to_string()),
    };
    let now = now_secs();

    // Gate on identity and on each distinct model. `parse_batch_create` guarantees ≥1 item, so the
    // loop authorizes at least once and `key` is always set.
    let mut models: Vec<&str> = spec
        .items
        .iter()
        .map(|i| i.request.model.as_str())
        .collect();
    models.sort_unstable();
    models.dedup();
    let mut key = None;
    for model in models {
        match authorize_token(&state, &token, model, now).await {
            Ok(id) => key = Some(id),
            Err(e) => return auth_error(e),
        }
    }
    let key = key.expect("non-empty batch authorizes at least one model");

    let request_id = next_request_id(&state.request_seq);
    match state.engine.batch_create(spec, key, request_id).await {
        Ok(handle) => Json(batch::handle_to_json(&handle)).into_response(),
        Err(e) => engine_error(e),
    }
}

/// `GET /v1/batches/{id}` — poll a batch's status. Routed by the opaque id; identity-only auth (no
/// model to gate on).
async fn retrieve_batch(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Response {
    let Some(token) = bearer(&headers) else {
        return error(StatusCode::UNAUTHORIZED, "missing bearer token");
    };
    let now = now_secs();
    if let Err(e) = authorize_token_identity(&state, &token, now).await {
        return auth_error(e);
    }
    let request_id = next_request_id(&state.request_seq);
    match state.engine.batch_retrieve(&id, request_id).await {
        Ok(handle) => Json(batch::handle_to_json(&handle)).into_response(),
        Err(e) => engine_error(e),
    }
}

/// `POST /v1/batches/{id}/cancel` — request cancellation. Routed by the opaque id; identity-only auth.
async fn cancel_batch(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Response {
    let Some(token) = bearer(&headers) else {
        return error(StatusCode::UNAUTHORIZED, "missing bearer token");
    };
    let now = now_secs();
    if let Err(e) = authorize_token_identity(&state, &token, now).await {
        return auth_error(e);
    }
    let request_id = next_request_id(&state.request_seq);
    match state.engine.batch_cancel(&id, request_id).await {
        Ok(handle) => Json(batch::handle_to_json(&handle)).into_response(),
        Err(e) => engine_error(e),
    }
}

/// `GET /v1/batches/{id}/results` — stream a completed batch's results as JSONL (one result per line).
/// Routed by the opaque id; identity-only auth. The body streams straight from the provider's result
/// file (principle 4 — never buffered whole).
async fn batch_results(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Response {
    let Some(token) = bearer(&headers) else {
        return error(StatusCode::UNAUTHORIZED, "missing bearer token");
    };
    let now = now_secs();
    let key = match authorize_token_identity(&state, &token, now).await {
        Ok(id) => id,
        Err(e) => return auth_error(e),
    };
    let request_id = next_request_id(&state.request_seq);
    let stream = match state.engine.batch_results(&id, key, request_id).await {
        Ok(s) => s,
        Err(e) => return engine_error(e),
    };

    let body = Body::from_stream(async_stream::stream! {
        let mut stream = stream;
        while let Some(item) = stream.next().await {
            match item {
                Ok(result) => {
                    // `result_to_line` returns a borrowed view now; serialise it straight to a String
                    // (it has no `Display`). Output is byte-identical to the old `Value::to_string`.
                    let mut line = serde_json::to_string(&batch::result_to_line(&result))
                        .expect("result line serialises");
                    line.push('\n');
                    yield Ok::<Bytes, std::convert::Infallible>(Bytes::from(line));
                }
                Err(e) => {
                    // Headers are already on the wire; surface the failure as a trailing JSON line and
                    // stop. The instrumented stream behind us emits no RequestFailed for batch results.
                    let mut line = json!({ "error": { "message": e.to_string() } }).to_string();
                    line.push('\n');
                    yield Ok(Bytes::from(line));
                    break;
                }
            }
        }
    });

    Response::builder()
        .header(CONTENT_TYPE, "application/x-ndjson")
        .body(body)
        .expect("valid ndjson response")
}

// ---------------------------------------------------------------------------------------------
// Read-only admin surface (observability only — no mutation path exists)
// ---------------------------------------------------------------------------------------------

async fn admin_routes(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if let Some(resp) = guard_admin(&state, &headers) {
        return resp;
    }
    Json(state.routes()).into_response()
}

async fn admin_health(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if let Some(resp) = guard_admin(&state, &headers) {
        return resp;
    }
    Json(state.health(now_secs())).into_response()
}

/// Read-only: a secret-free listing of the current key roster + verdict overlay (whatever the identity
/// and limits pulls last installed, or the config base). For dashboards/diagnostics only.
async fn admin_list_keys(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if let Some(resp) = guard_admin(&state, &headers) {
        return resp;
    }
    Json(state.keys()).into_response()
}

// ---------------------------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------------------------

fn bearer(headers: &HeaderMap) -> Option<String> {
    headers
        .get(AUTHORIZATION)?
        .to_str()
        .ok()?
        .strip_prefix("Bearer ")
        .map(str::to_string)
}

/// Extract the consumer credential for the Anthropic surface: the `x-api-key` header (as an Anthropic
/// SDK sends it), falling back to an `Authorization: Bearer` token. Either way it is the same consumer
/// virtual key every other surface authenticates — only the header it rides in differs.
fn anthropic_key(headers: &HeaderMap) -> Option<String> {
    if let Some(value) = headers.get("x-api-key").and_then(|v| v.to_str().ok()) {
        if !value.is_empty() {
            return Some(value.to_string());
        }
    }
    bearer(headers)
}

/// Authenticate a consumer bearer token, dispatching between the OAuth2 JWT verifier (`[oauth]`) and the
/// static key store. A JWT is three dot-separated segments; anything else — or any build without an
/// OAuth verifier — goes to the key store. Both schemes return the same `KeyId`/[`AuthError`], so every
/// handler downstream is scheme-agnostic. Always `async` so the call sites are identical whether or not
/// the `oauth` feature is compiled in.
async fn authorize_token(
    state: &AppState,
    token: &str,
    model: &str,
    now: u64,
) -> Result<crate::keys::KeyId, AuthError> {
    #[cfg(feature = "oauth")]
    if let Some(oauth) = &state.oauth {
        if token.split('.').count() == 3 {
            return oauth.authorize(token, model, now).await;
        }
    }
    state.keys.authorize(token, model, now)
}

/// Identity-only counterpart of [`authorize_token`], for batch surfaces that carry no model (poll /
/// cancel / results are routed by an opaque id, not a model). Same JWT-vs-static-key dispatch.
async fn authorize_token_identity(
    state: &AppState,
    token: &str,
    now: u64,
) -> Result<crate::keys::KeyId, AuthError> {
    #[cfg(feature = "oauth")]
    if let Some(oauth) = &state.oauth {
        if token.split('.').count() == 3 {
            return oauth.authorize_identity(token, now).await;
        }
    }
    state.keys.authorize_identity(token, now)
}

/// Extract the presented admin token (`x-admin-token`) from request headers, transport → value. The
/// transport-agnostic [`Observability`] surface takes this `Option<&str>`, never the `HeaderMap`.
fn presented_admin_token(headers: &HeaderMap) -> Option<&str> {
    headers.get("x-admin-token").and_then(|v| v.to_str().ok())
}

/// Guard a read-only admin request. Returns `Some(response)` to short-circuit when the token is
/// missing, the surface is disabled (no token configured), or the presented token does not match;
/// `None` when the caller is authorized to proceed. The decision itself lives in
/// [`Observability::admin_access`]; this only maps the verdict to a transport response.
fn guard_admin(state: &AppState, headers: &HeaderMap) -> Option<Response> {
    match state.admin_access(presented_admin_token(headers)) {
        AdminAccess::Disabled => Some(error(StatusCode::NOT_FOUND, "admin surface disabled")),
        AdminAccess::Unauthorized => Some(error(StatusCode::UNAUTHORIZED, "invalid admin token")),
        AdminAccess::Authorized => None,
    }
}

fn error(status: StatusCode, message: impl Into<String>) -> Response {
    (
        status,
        Json(json!({ "error": { "message": message.into() } })),
    )
        .into_response()
}

/// Attach a `Retry-After: <secs>` header to an already-built error response (RFC 9110 §10.2.3). Used for
/// the `429` the engine returns when every target on a chain is rate-limited and the bounded wait
/// elapsed — telling the consumer the soonest estimated moment capacity returns.
fn with_retry_after(mut resp: Response, secs: u64) -> Response {
    if let Ok(value) = HeaderValue::from_str(&secs.to_string()) {
        resp.headers_mut().insert(RETRY_AFTER, value);
    }
    resp
}

fn auth_error(err: AuthError) -> Response {
    match err {
        AuthError::Unknown => error(StatusCode::UNAUTHORIZED, "unknown api key"),
        AuthError::Blocked => error(StatusCode::FORBIDDEN, "key is blocked"),
        AuthError::Suspended { until } => error(
            StatusCode::TOO_MANY_REQUESTS,
            format!("key suspended until {until}"),
        ),
        AuthError::ModelNotAllowed => {
            error(StatusCode::FORBIDDEN, "model not allowed for this key")
        }
    }
}

fn engine_error(err: EngineError) -> Response {
    match err {
        EngineError::NoRoute(m) => {
            error(StatusCode::NOT_FOUND, format!("no route for model '{m}'"))
        }
        EngineError::NoTargets(m) => {
            error(StatusCode::NOT_FOUND, format!("model '{m}' has no targets"))
        }
        EngineError::AllTargetsFailed(e) => error(StatusCode::BAD_GATEWAY, e.to_string()),
        EngineError::RateLimited { retry_after_secs } => with_retry_after(
            error(
                StatusCode::TOO_MANY_REQUESTS,
                format!("rate limited; retry after {retry_after_secs}s"),
            ),
            retry_after_secs,
        ),
        EngineError::Blocked(reason) => error(StatusCode::FORBIDDEN, reason),
        EngineError::EmptyBatch => error(StatusCode::BAD_REQUEST, "batch has no requests"),
        EngineError::MixedProviderBatch => error(
            StatusCode::BAD_REQUEST,
            "batch mixes models that route to different providers",
        ),
        EngineError::BatchNotFound(id) => {
            error(StatusCode::NOT_FOUND, format!("batch '{id}' not found"))
        }
    }
}

/// The Anthropic-shaped error envelope (`{ "type": "error", "error": { "type", "message" } }`) — the
/// `/v1/messages` surface speaks the Anthropic dialect on errors too, so an Anthropic SDK parses them
/// (dialect fidelity beats convenience).
fn anthropic_error(status: StatusCode, err_type: &str, message: impl Into<String>) -> Response {
    (
        status,
        Json(json!({
            "type": "error",
            "error": { "type": err_type, "message": message.into() },
        })),
    )
        .into_response()
}

/// Auth verdicts → Anthropic error envelope. Same statuses [`auth_error`] uses, in the Anthropic shape.
fn anthropic_auth_error(err: AuthError) -> Response {
    match err {
        AuthError::Unknown => anthropic_error(
            StatusCode::UNAUTHORIZED,
            "authentication_error",
            "unknown api key",
        ),
        AuthError::Blocked => {
            anthropic_error(StatusCode::FORBIDDEN, "permission_error", "key is blocked")
        }
        AuthError::Suspended { until } => anthropic_error(
            StatusCode::TOO_MANY_REQUESTS,
            "rate_limit_error",
            format!("key suspended until {until}"),
        ),
        AuthError::ModelNotAllowed => anthropic_error(
            StatusCode::FORBIDDEN,
            "permission_error",
            "model not allowed for this key",
        ),
    }
}

/// Engine outcomes → Anthropic error envelope. Same statuses [`engine_error`] uses, in the Anthropic
/// shape. The batch variants cannot arise on the chat path but are mapped for exhaustiveness.
fn anthropic_engine_error(err: EngineError) -> Response {
    match err {
        EngineError::NoRoute(m) => anthropic_error(
            StatusCode::NOT_FOUND,
            "not_found_error",
            format!("no route for model '{m}'"),
        ),
        EngineError::NoTargets(m) => anthropic_error(
            StatusCode::NOT_FOUND,
            "not_found_error",
            format!("model '{m}' has no targets"),
        ),
        EngineError::AllTargetsFailed(e) => {
            anthropic_error(StatusCode::BAD_GATEWAY, "api_error", e.to_string())
        }
        EngineError::RateLimited { retry_after_secs } => with_retry_after(
            anthropic_error(
                StatusCode::TOO_MANY_REQUESTS,
                "rate_limit_error",
                format!("rate limited; retry after {retry_after_secs}s"),
            ),
            retry_after_secs,
        ),
        EngineError::Blocked(reason) => {
            anthropic_error(StatusCode::FORBIDDEN, "permission_error", reason)
        }
        EngineError::EmptyBatch => anthropic_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "batch has no requests",
        ),
        EngineError::MixedProviderBatch => anthropic_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "batch mixes models that route to different providers",
        ),
        EngineError::BatchNotFound(id) => anthropic_error(
            StatusCode::NOT_FOUND,
            "not_found_error",
            format!("batch '{id}' not found"),
        ),
    }
}

fn next_request_id(seq: &AtomicU64) -> String {
    let n = seq.fetch_add(1, Ordering::Relaxed);
    format!("{}-{}", now_ms(), n)
}

pub(crate) fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod architecture_tests {
    use super::{architecture_value, Modality, ModelInfo};
    use serde_json::json;

    #[test]
    fn upstream_architecture_with_modalities_passes_through() {
        // An OpenRouter vision model reports its real modality arrays; the internal `ModelInfo`
        // preserves that block verbatim in `extra`. It must survive to the rendered `architecture`
        // (our coarse `Modality::Llm` alone would have flattened it to `["text"]`, hiding `image`).
        let mut info = ModelInfo::new("moonshotai/kimi-k2-vision");
        info.extra.insert(
            "architecture".into(),
            json!({
                "input_modalities": ["text", "image"],
                "output_modalities": ["text"],
                "modality": "text+image->text",
                "tokenizer": "Other",
            }),
        );
        let arch = architecture_value(Some(&info), Some(Modality::Llm));
        assert_eq!(arch["input_modalities"], json!(["text", "image"]));
        assert_eq!(arch["output_modalities"], json!(["text"]));
    }

    #[test]
    fn missing_or_stub_architecture_falls_back_to_coarse() {
        // No architecture in `extra` → derive the coarse block from the modality.
        let bare = ModelInfo::new("acme/plain-llm");
        let arch = architecture_value(Some(&bare), Some(Modality::Llm));
        assert_eq!(arch["input_modalities"], json!(["text"]));

        // A stub architecture with no modality arrays (some brands publish just `{modality, tokenizer}`)
        // is NOT richer than the derivation, so we still fall back rather than emit empty arrays.
        let mut stub = ModelInfo::new("acme/stub");
        stub.extra.insert(
            "architecture".into(),
            json!({ "modality": "text", "tokenizer": "GPT" }),
        );
        let arch = architecture_value(Some(&stub), Some(Modality::Llm));
        assert_eq!(arch["input_modalities"], json!(["text"]));
    }
}
