//! Operator authentication: the master-password + OIDC login flows, the session cookie, the route gate
//! that protects the human UI, and the helpers server functions use to know who is logged in.
//!
//! Sessions are server-side (see [`crate::db::session`]); the cookie carries only an opaque token.

pub mod oidc;
pub mod password;

use axum::extract::{Query, Request, State};
use axum::http::StatusCode;
use axum::middleware::Next;
use axum::response::{IntoResponse, Redirect, Response};
use axum::routing::{get, post};
use axum::{Form, Router};
use axum_extra::extract::cookie::{Cookie, CookieJar, SameSite};
use leptos::prelude::{expect_context, ServerFnError};
use serde::Deserialize;

use crate::db;
use crate::dto::Session;
use crate::state::AppState;

/// The session cookie name.
pub const COOKIE: &str = "llmleaf_session";

/// Build the session cookie. `HttpOnly` + `SameSite=Lax` + root path. (Set `Secure` at your TLS
/// terminator / reverse proxy in production — left off here so config-less `http://localhost` dev works.)
fn session_cookie(token: String) -> Cookie<'static> {
    Cookie::build((COOKIE, token))
        .http_only(true)
        .same_site(SameSite::Lax)
        .path("/")
        .build()
}

/// Only allow same-origin relative redirects (`/...`, never `//host`) to avoid an open redirect.
fn safe_redirect(target: Option<&str>) -> String {
    match target {
        Some(t) if t.starts_with('/') && !t.starts_with("//") => t.to_string(),
        _ => "/".to_string(),
    }
}

// ---------------------------------------------------------------------------------------------
// Server-function helpers (run inside the server's reactive context)
// ---------------------------------------------------------------------------------------------

/// Resolve the logged-in operator from the request cookie, or `None`. Usable inside server functions.
pub async fn current_session() -> Option<Session> {
    let jar = leptos_axum::extract::<CookieJar>().await.ok()?;
    let token = jar.get(COOKIE)?.value().to_string();
    let state = expect_context::<AppState>();
    db::session::lookup_session(&state.db, &token)
        .await
        .ok()
        .flatten()
}

/// Like [`current_session`] but errors when unauthenticated — the guard every mutating/reading server
/// function calls first.
pub async fn require_session() -> Result<Session, ServerFnError> {
    current_session()
        .await
        .ok_or_else(|| ServerFnError::new("authentication required"))
}

// ---------------------------------------------------------------------------------------------
// The route gate (tower middleware over the whole app)
// ---------------------------------------------------------------------------------------------

/// Paths reachable without an operator session: the login page, the auth handlers, the machine control
/// endpoints (they carry their own bearer auth), static assets, and health.
fn is_public(path: &str) -> bool {
    path == "/login"
        || path == "/healthz"
        || path == "/favicon.ico"
        || path.starts_with("/auth/")
        || path.starts_with("/llmleaf/")
        || path.starts_with("/pkg/")
        || path.starts_with("/assets/")
}

/// Gate every non-public request on a valid session. Browser navigations get a redirect to `/login`
/// (remembering where they were going); `/api/*` server-function calls get a clean `401`.
pub async fn gate(
    State(state): State<AppState>,
    jar: CookieJar,
    req: Request,
    next: Next,
) -> Response {
    let path = req.uri().path().to_string();
    if is_public(&path) {
        return next.run(req).await;
    }
    let authed = match jar.get(COOKIE) {
        Some(c) => db::session::lookup_session(&state.db, c.value())
            .await
            .ok()
            .flatten()
            .is_some(),
        None => false,
    };
    if authed {
        return next.run(req).await;
    }
    if path.starts_with("/api/") {
        return (StatusCode::UNAUTHORIZED, "authentication required").into_response();
    }
    let encoded = urlencoding_min(&path);
    Redirect::to(&format!("/login?redirect_to={encoded}")).into_response()
}

/// Minimal percent-encoding for a path used as a query-string value (encode the few characters that
/// would break the query). Avoids pulling a urlencoding dep for one call site.
fn urlencoding_min(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' | b'/' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

// ---------------------------------------------------------------------------------------------
// Axum auth handlers (own their own cookies; not server functions)
// ---------------------------------------------------------------------------------------------

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/auth/login", post(login))
        .route("/auth/logout", get(logout))
        .route("/auth/oidc/login", get(oidc_login))
        .route("/auth/oidc/callback", get(oidc_callback))
}

#[derive(Deserialize)]
struct LoginForm {
    password: String,
    #[serde(default)]
    redirect_to: Option<String>,
}

async fn login(
    State(state): State<AppState>,
    jar: CookieJar,
    Form(form): Form<LoginForm>,
) -> Response {
    let ok = state
        .master_hash
        .as_ref()
        .as_ref()
        .map(|hash| password::verify(&form.password, hash))
        .unwrap_or(false);
    if !ok {
        return Redirect::to("/login?error=bad_password").into_response();
    }
    let ttl = state.config.auth.session_ttl_secs;
    match db::session::create_session(&state.db, "operator", "password", ttl).await {
        Ok(token) => (
            jar.add(session_cookie(token)),
            Redirect::to(&safe_redirect(form.redirect_to.as_deref())),
        )
            .into_response(),
        Err(e) => {
            tracing::error!(error = %e, "session creation failed");
            Redirect::to("/login?error=server").into_response()
        }
    }
}

async fn logout(State(state): State<AppState>, jar: CookieJar) -> Response {
    if let Some(c) = jar.get(COOKIE) {
        let _ = db::session::delete_session(&state.db, c.value()).await;
    }
    let cleared = jar.remove(Cookie::build((COOKIE, "")).path("/").build());
    (cleared, Redirect::to("/login")).into_response()
}

#[derive(Deserialize)]
struct RedirectQuery {
    #[serde(default)]
    redirect_to: Option<String>,
}

async fn oidc_login(State(state): State<AppState>, Query(q): Query<RedirectQuery>) -> Response {
    let Some(oidc) = state.config.oidc.as_ref() else {
        return (StatusCode::NOT_FOUND, "OIDC login is not configured").into_response();
    };
    match oidc::begin_login(&state, oidc, q.redirect_to.as_deref()).await {
        Ok(url) => Redirect::to(&url).into_response(),
        Err(e) => {
            tracing::warn!(error = %e, "OIDC begin_login failed");
            Redirect::to("/login?error=oidc").into_response()
        }
    }
}

#[derive(Deserialize)]
struct CallbackQuery {
    #[serde(default)]
    code: Option<String>,
    #[serde(default)]
    state: Option<String>,
    #[serde(default)]
    error: Option<String>,
}

async fn oidc_callback(
    State(state): State<AppState>,
    jar: CookieJar,
    Query(q): Query<CallbackQuery>,
) -> Response {
    if let Some(err) = q.error {
        tracing::warn!(error = %err, "OIDC provider returned an error");
        return Redirect::to("/login?error=oidc").into_response();
    }
    let (Some(code), Some(csrf)) = (q.code, q.state) else {
        return Redirect::to("/login?error=oidc").into_response();
    };
    let Some(oidc) = state.config.oidc.as_ref() else {
        return (StatusCode::NOT_FOUND, "OIDC login is not configured").into_response();
    };
    match oidc::complete_login(&state, oidc, &code, &csrf).await {
        Ok((subject, redirect_to)) => {
            let ttl = state.config.auth.session_ttl_secs;
            match db::session::create_session(&state.db, &subject, "oidc", ttl).await {
                Ok(token) => (
                    jar.add(session_cookie(token)),
                    Redirect::to(&safe_redirect(redirect_to.as_deref())),
                )
                    .into_response(),
                Err(e) => {
                    tracing::error!(error = %e, "session creation failed");
                    Redirect::to("/login?error=server").into_response()
                }
            }
        }
        Err(e) => {
            tracing::warn!(error = %e, "OIDC complete_login failed");
            Redirect::to("/login?error=oidc").into_response()
        }
    }
}
