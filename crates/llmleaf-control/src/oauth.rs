//! OAuth2 outbound: the HTTP half of the resource-server surface, kept out of the core (principle 2).
//!
//! Two jobs, both feeding the core's [`OAuthVerifier`] through its HTTP-free handles:
//!   - **PULL** the IdP's JWKS (explicit `jwks_uri`, else OIDC discovery from `issuer`) on an interval
//!     and install it via [`OAuthVerifier::install_jwks`] — the analogue of the identity pull. Primed
//!     synchronously before the listener opens so a cold node fails closed on an empty key set.
//!   - **INTROSPECT** (RFC 7662) per request, implementing the core's [`TokenIntrospector`] trait with
//!     reqwest — the analogue of the engine's HTTP interceptor.

use std::sync::Arc;
use std::time::Duration;

use llmleaf_core::{IntrospectionConfig, OAuthConfig, OAuthVerifier, TokenIntrospector};
use serde::Deserialize;
use serde_json::Value;
use tokio::task::JoinHandle;
use tokio::time::{interval, MissedTickBehavior};
use tokio_util::sync::CancellationToken;

use crate::get_json;

/// Build the HTTP client, wire the introspector (when configured), then prime + spawn the JWKS pull.
/// Priming is synchronous (the caller awaits this before opening the listener) so a cold node fails
/// closed rather than serving on an empty key set. An inline-only deployment (`[oauth].jwks` set, no
/// `jwks_uri`) is config-operable (principle 6) and spawns no background fetch.
pub async fn start_oauth(
    cfg: &OAuthConfig,
    verifier: Arc<OAuthVerifier>,
    shutdown: CancellationToken,
) -> Vec<JoinHandle<()>> {
    let http = reqwest::Client::new();
    let mut handles = Vec::new();

    if let Some(intro) = &cfg.introspection {
        verifier.install_introspector(Arc::new(HttpIntrospector::new(http.clone(), intro)));
        tracing::info!(url = %intro.url, "wired OAuth token introspection");
    }

    // Pull only when there is a remote source: an explicit jwks_uri, or no inline keys to fall back on.
    if cfg.jwks_uri.is_some() || cfg.jwks.is_none() {
        let refresher = JwksRefresher::new(http, cfg, verifier, shutdown);
        refresher.prime().await;
        handles.push(refresher.spawn());
    }
    handles
}

// ---------------------------------------------------------------------------------------------
// JWKS pull (`[oauth].jwks_uri`, else OIDC discovery from `issuer`)
// ---------------------------------------------------------------------------------------------

/// GETs the IdP's JWKS every `jwks_refresh_secs` and installs it as the core verifier's signing-key set.
/// Fail posture mirrors the identity pull: a failed poll keeps the last-good keys; a cold node that
/// cannot prime validates nothing (fail closed), unless an inline `[oauth].jwks` seeded the verifier.
struct JwksRefresher {
    http: reqwest::Client,
    issuer: String,
    /// Explicit endpoint; `None` ⇒ discovered from `issuer` on each fetch until it resolves.
    jwks_uri: Option<String>,
    verifier: Arc<OAuthVerifier>,
    interval: Duration,
    timeout: Duration,
    shutdown: CancellationToken,
}

/// The slice of the OIDC discovery document we need.
#[derive(Deserialize)]
struct OidcDiscovery {
    jwks_uri: String,
}

impl JwksRefresher {
    fn new(
        http: reqwest::Client,
        cfg: &OAuthConfig,
        verifier: Arc<OAuthVerifier>,
        shutdown: CancellationToken,
    ) -> Self {
        JwksRefresher {
            http,
            issuer: cfg.issuer.clone(),
            jwks_uri: cfg.jwks_uri.clone(),
            verifier,
            interval: Duration::from_secs(cfg.jwks_refresh_secs.max(1)),
            timeout: Duration::from_millis(cfg.timeout_ms.max(1)),
            shutdown,
        }
    }

    /// The synchronous first pull, run before the listener opens. On failure it logs and leaves the
    /// verifier on whatever inline `[oauth].jwks` seeded it (or empty ⇒ fail closed).
    async fn prime(&self) {
        match self.fetch().await {
            Ok(n) => tracing::info!(count = n, "primed JWKS from issuer"),
            Err(e) => tracing::warn!(error = %e,
                "JWKS prime failed; relying on inline [oauth].jwks if present (else failing closed)"),
        }
    }

    fn spawn(self) -> JoinHandle<()> {
        tokio::spawn(async move {
            let mut tick = interval(self.interval);
            tick.set_missed_tick_behavior(MissedTickBehavior::Delay);
            tick.tick().await; // discard the immediate tick; prime() already did the first pull
            loop {
                tokio::select! {
                    _ = self.shutdown.cancelled() => break,
                    _ = tick.tick() => self.refresh_once().await,
                }
            }
            tracing::info!("JWKS refresher stopped");
        })
    }

    async fn refresh_once(&self) {
        match self.fetch().await {
            Ok(n) => tracing::debug!(count = n, "installed JWKS"),
            Err(e) => tracing::warn!(error = %e, "JWKS pull failed; keeping last-good keys"),
        }
    }

    async fn fetch(&self) -> Result<usize, String> {
        let uri = match &self.jwks_uri {
            Some(u) => u.clone(),
            None => self.discover().await?,
        };
        let body: Value = get_json(&self.http, &uri, None, self.timeout)
            .await
            .map_err(|e| e.to_string())?;
        self.verifier.install_jwks(&body)
    }

    /// OIDC discovery: `<issuer>/.well-known/openid-configuration` → `jwks_uri`.
    async fn discover(&self) -> Result<String, String> {
        let url = format!(
            "{}/.well-known/openid-configuration",
            self.issuer.trim_end_matches('/')
        );
        let doc: OidcDiscovery = get_json(&self.http, &url, None, self.timeout)
            .await
            .map_err(|e| e.to_string())?;
        Ok(doc.jwks_uri)
    }
}

// ---------------------------------------------------------------------------------------------
// Token introspection (`[oauth].introspection`, RFC 7662)
// ---------------------------------------------------------------------------------------------

/// RFC 7662 introspector: POSTs `token=<jwt>` form-encoded and trusts the IdP's `active` field. The core
/// caches each answer for `cache_ttl_secs`, so this is not called per request once a token is warm.
struct HttpIntrospector {
    http: reqwest::Client,
    url: String,
    credential: Option<String>,
    timeout: Duration,
}

#[derive(Deserialize)]
struct IntrospectionResponse {
    #[serde(default)]
    active: bool,
}

impl HttpIntrospector {
    fn new(http: reqwest::Client, cfg: &IntrospectionConfig) -> Self {
        HttpIntrospector {
            http,
            url: cfg.url.clone(),
            credential: cfg.credential.as_ref().and_then(|s| s.resolve()),
            timeout: Duration::from_millis(cfg.timeout_ms.max(1)),
        }
    }
}

#[async_trait::async_trait]
impl TokenIntrospector for HttpIntrospector {
    async fn introspect(&self, token: &str) -> Result<bool, String> {
        // RFC 7662 wants the token in an `application/x-www-form-urlencoded` body. We set it directly
        // rather than pull in reqwest's `urlencoded` feature: only JWT-shaped bearers reach here, and a
        // JWT is base64url + `.` (`[A-Za-z0-9_.-]`), every byte of which is safe in a form value.
        let mut req = self
            .http
            .post(&self.url)
            .timeout(self.timeout)
            .header(
                reqwest::header::CONTENT_TYPE,
                "application/x-www-form-urlencoded",
            )
            .body(format!("token={token}"));
        if let Some(c) = &self.credential {
            req = req.bearer_auth(c);
        }
        let resp = req
            .send()
            .await
            .map_err(|e| e.to_string())?
            .error_for_status()
            .map_err(|e| e.to_string())?;
        let body: IntrospectionResponse = resp.json().await.map_err(|e| e.to_string())?;
        Ok(body.active)
    }
}
