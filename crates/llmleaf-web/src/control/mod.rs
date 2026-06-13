//! The machine-facing control endpoints — the heart of the inverted control plane (SOUL.md principle
//! 5). The core is always the client; this app only *serves what the core pulls* and *receives what the
//! core pushes*. There is NO endpoint here that reaches into the core.
//!
//!   - `GET  /llmleaf/keys`     → identity roster the core's `IdentityRefresher` pulls.
//!   - `GET  /llmleaf/verdicts` → verdict overlay the core's `VerdictRefresher` pulls.
//!   - `POST /llmleaf/usage`    ← usage/lifecycle batch the core's `UsageReporter` pushes.
//!
//! All three are guarded by the optional shared bearer token (`[control].token`). Identity/verdicts are
//! read-only; usage is append-only. None of them can mutate the core.

use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};

use crate::db;
use crate::state::AppState;
use crate::wire::{IdentityResponse, UsageBatch, VerdictResponse};

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/llmleaf/keys", get(identity))
        .route("/llmleaf/verdicts", get(verdicts))
        .route("/llmleaf/usage", post(usage))
}

/// Verify the `Authorization: Bearer <token>` against `[control].token`. With no token configured the
/// surface is open (logged once at startup); otherwise a mismatch is `401`. Constant-time compare so the
/// token isn't an timing oracle.
// The Err carries a ready axum `Response` (the 401) — intentional; it's returned straight to the client.
#[allow(clippy::result_large_err)]
fn check_auth(state: &AppState, headers: &HeaderMap) -> Result<(), Response> {
    let Some(expected) = state
        .config
        .control
        .token
        .as_ref()
        .and_then(|s| s.resolve())
    else {
        return Ok(()); // open surface (DEV)
    };
    let presented = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "));
    match presented {
        Some(tok) if tokens_match(tok, &expected) => Ok(()),
        _ => Err((
            StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({ "error": "invalid control token" })),
        )
            .into_response()),
    }
}

/// Constant-time token comparison that does NOT leak the token length. Both sides are SHA-256'd to a
/// fixed 32 bytes first, so the byte-wise compare always runs over equal lengths regardless of the
/// presented token's length (a plain length check would be a timing oracle on the secret's length).
fn tokens_match(presented: &str, expected: &str) -> bool {
    use sha2::{Digest, Sha256};
    let a = Sha256::digest(presented.as_bytes());
    let b = Sha256::digest(expected.as_bytes());
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

async fn identity(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if let Err(resp) = check_auth(&state, &headers) {
        return resp;
    }
    match db::keys::roster(&state.db).await {
        Ok(keys) => Json(IdentityResponse { keys }).into_response(),
        Err(e) => db_error(e),
    }
}

async fn verdicts(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if let Err(resp) = check_auth(&state, &headers) {
        return resp;
    }
    match db::keys::verdicts(&state.db).await {
        Ok(verdicts) => Json(VerdictResponse { verdicts }).into_response(),
        Err(e) => db_error(e),
    }
}

async fn usage(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(batch): Json<UsageBatch>,
) -> Response {
    if let Err(resp) = check_auth(&state, &headers) {
        return resp;
    }
    match db::usage::ingest(&state.db, &batch.events).await {
        Ok(n) => Json(serde_json::json!({ "ingested": n })).into_response(),
        Err(e) => db_error(e),
    }
}

fn db_error(e: sqlx::Error) -> Response {
    tracing::error!(error = %e, "control endpoint db error");
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(serde_json::json!({ "error": "internal error" })),
    )
        .into_response()
}
