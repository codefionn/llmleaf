//! llmleaf control-plane client — the outbound side of the inverted control plane (SOUL.md
//! principles 5 & 6).
//!
//! The core holds no inbound-mutated state and exposes no mutation surface. This crate is the one
//! place that does outbound control HTTP, keeping `llmleaf-core` HTTP-client-free (principle 2). It
//! drives the core only through narrow, HTTP-free handles: the [`KeyStore`] snapshot-swap methods, the
//! [`Engine`] topology-install method, and the [`EventBus`] tap. Three jobs, each independently
//! enabled by config:
//!   - **PULL** identity ([`control.identity`]), verdicts ([`control.limits`]), and dynamic topology
//!     ([`control.topology`] — providers + routes, diffed against the previous pull and reconciled)
//!     on an interval and install them as node-local snapshots. Fail-open (keep last-good) for limits
//!     and topology; fail-closed on cold start for identity (an empty identity cache must reject,
//!     never admit).
//!   - **PUSH** usage/lifecycle events ([`control.usage`]) to a sink, batched and async, tapping the
//!     in-process bus. Lossy by design; the hot path never waits.
//!   - **SYNC** intercept ([`control.intercept`]) — the in-flight hook, wired into the engine.
//!
//! [`control.identity`]: llmleaf_core::IdentitySource
//! [`control.limits`]: llmleaf_core::LimitsSource
//! [`control.topology`]: llmleaf_core::TopologySource
//! [`control.usage`]: llmleaf_core::UsageSink
//! [`control.intercept`]: llmleaf_core::InterceptHook

use std::sync::Arc;
use std::time::Duration;

use llmleaf_core::{
    ControlConfig, Engine, EventBus, InterceptPhase, Interceptor, KeyStore, ProviderFactory,
    ResolvedAuth,
};
use tokio_util::sync::CancellationToken;

mod intercept;
#[cfg(feature = "oauth")]
mod oauth;
mod refresh;
mod usage;

pub use intercept::HttpInterceptor;
#[cfg(feature = "oauth")]
pub use oauth::start_oauth;
pub use refresh::{IdentityRefresher, TopologyRefresher, VerdictRefresher};
pub use usage::UsageReporter;

/// Where a pulled `[control.topology]` layer is installed: the engine that swaps it in, and the
/// binary's provider factory that builds instances for pulled kinds (so a pulled provider behaves
/// exactly like a file-configured one). Pass `None` to [`start`] from an embedder without a factory —
/// a configured `[control.topology]` is then warned about and skipped, never half-applied.
pub struct TopologyTarget {
    pub engine: Arc<Engine>,
    pub factory: Arc<dyn ProviderFactory>,
}

/// Build the sync interceptor (`[control.intercept]`) the binary passes to
/// `llmleaf_core::build_state_with`. Returns `None` when the section is absent or has no `phases`
/// (disabled by data — it never fires, paying nothing). Response-phase screening is not yet wired into
/// the engine, so a config requesting it is honored only for the request phase (warned about here).
pub fn build_interceptor(cfg: &ControlConfig) -> Option<Arc<dyn Interceptor>> {
    let hook = cfg.intercept.as_ref()?;
    if hook.phases.is_empty() {
        tracing::info!("[control.intercept] configured but no `phases` set; interceptor disabled");
        return None;
    }
    if hook.phases.contains(&InterceptPhase::Response) {
        tracing::warn!(
            "[control.intercept] response-phase screening is not yet wired into the engine; \
             only request-phase will fire"
        );
    }
    let auth = cfg.resolve_auth(hook.auth.as_deref(), hook.credential.as_ref());
    Some(Arc::new(HttpInterceptor::new(
        new_http_client(),
        hook,
        auth,
    )))
}

/// Handles to the spawned background tasks. After cancelling the shared [`CancellationToken`], await
/// [`ControlHandles::join`] to drain them on shutdown.
pub struct ControlHandles {
    handles: Vec<compio::runtime::JoinHandle<()>>,
}

impl ControlHandles {
    /// Await every spawned task. Call after the shutdown token has been cancelled.
    pub async fn join(self) {
        for h in self.handles {
            let _ = h.await;
        }
    }
}

/// Build the shared HTTP client and spawn the configured outbound control tasks.
///
/// Identity priming runs *before this returns* (a synchronous first pull) so the listener never opens
/// on an empty identity cache — see [`IdentityRefresher::prime`]. Tasks that aren't configured aren't
/// spawned; with an empty `[control]` section this is a no-op returning empty handles.
pub async fn start(
    cfg: &ControlConfig,
    keys: Arc<KeyStore>,
    events: &EventBus,
    topology: Option<TopologyTarget>,
    shutdown: CancellationToken,
) -> ControlHandles {
    let http = new_http_client();
    let mut handles = Vec::new();

    if let Some(topo_cfg) = &cfg.topology {
        match topology {
            Some(target) => {
                let auth = cfg.resolve_auth(topo_cfg.auth.as_deref(), topo_cfg.credential.as_ref());
                let refresher = TopologyRefresher::new(
                    http.clone(),
                    topo_cfg,
                    auth,
                    target.engine,
                    target.factory,
                    shutdown.clone(),
                );
                handles.push(refresher.spawn());
            }
            None => tracing::warn!(
                "[control.topology] is configured but no provider factory was wired; \
                 the pulled topology is ignored"
            ),
        }
    }
    if let Some(identity) = &cfg.identity {
        let auth = cfg.resolve_auth(identity.auth.as_deref(), identity.credential.as_ref());
        let refresher =
            IdentityRefresher::new(http.clone(), identity, auth, keys.clone(), shutdown.clone());
        // Prime synchronously: fail-closed identity must not open the listener on a cold cache.
        refresher.prime().await;
        handles.push(refresher.spawn());
    }
    if let Some(limits) = &cfg.limits {
        let auth = cfg.resolve_auth(limits.auth.as_deref(), limits.credential.as_ref());
        let refresher =
            VerdictRefresher::new(http.clone(), limits, auth, keys.clone(), shutdown.clone());
        handles.push(refresher.spawn());
    }
    if let Some(usage) = &cfg.usage {
        let auth = cfg.resolve_auth(usage.auth.as_deref(), usage.credential.as_ref());
        let reporter = UsageReporter::new(
            http.clone(),
            usage,
            auth,
            events.subscribe(),
            shutdown.clone(),
        );
        handles.push(reporter.spawn());
    }

    ControlHandles { handles }
}

/// Shared HTTP GET → JSON helper. Optional resolved auth, per-request timeout, non-2xx ⇒ `Err`.
pub(crate) async fn get_json<T: serde::de::DeserializeOwned>(
    http: &cyper::Client,
    url: &str,
    auth: Option<&ResolvedAuth>,
    timeout: Duration,
) -> Result<T, String> {
    let req =
        apply_auth(http.get(url).map_err(|e| e.to_string())?, auth).map_err(|e| e.to_string())?;
    let response = compio::time::timeout(timeout, req.send())
        .await
        .map_err(|_| "control request timed out".to_string())?
        .map_err(|e| e.to_string())?;
    if !response.status().is_success() {
        return Err(format!(
            "control request failed with HTTP {}",
            response.status()
        ));
    }
    response.json::<T>().await.map_err(|e| e.to_string())
}

/// Attach a resolved control-plane auth to an outbound request: a `bearer` as `Authorization: Bearer
/// <token>`, a `header` scheme as its `<name>: <value>`. `None` ⇒ the request goes out unauthenticated.
/// The single place the core's [`ResolvedAuth`] meets the HTTP client, keeping `llmleaf-core` HTTP-free.
pub(crate) fn apply_auth(
    req: cyper::RequestBuilder,
    auth: Option<&ResolvedAuth>,
) -> cyper::Result<cyper::RequestBuilder> {
    match auth {
        Some(ResolvedAuth::Bearer(token)) => req.bearer_auth(token),
        Some(ResolvedAuth::Header { name, value }) => req.header(name, value),
        None => Ok(req),
    }
}

/// Render a URL for logs with any `user:pass@` userinfo stripped. Control/usage/intercept URLs are
/// operator config and may embed credentials (e.g. `https://user:secret@control/keys`); those must
/// never reach log collectors. The rest of the URL (host, path, query) is kept for diagnosability.
pub(crate) fn redact_url(url: &str) -> String {
    let Some(scheme_end) = url.find("://") else {
        return url.to_string();
    };
    let auth_start = scheme_end + 3;
    let auth_end = url[auth_start..]
        .find('/')
        .map(|i| auth_start + i)
        .unwrap_or(url.len());
    let authority = &url[auth_start..auth_end];
    match authority.rfind('@') {
        Some(at) => {
            let mut out = String::with_capacity(url.len());
            out.push_str(&url[..auth_start]);
            out.push_str("<redacted>@");
            out.push_str(&authority[at + 1..]);
            out.push_str(&url[auth_end..]);
            out
        }
        None => url.to_string(),
    }
}

pub(crate) fn new_http_client() -> cyper::Client {
    let _ = compio::rustls::crypto::ring::default_provider().install_default();
    cyper::Client::builder()
        // Credential protection: never follow a redirect off the request's origin — the control
        // bearer/credential header must not be replayed to whatever host a `Location` names (cyper
        // strips only `Authorization`-family headers cross-host, and not on same-host scheme
        // downgrades). Mirrors llmleaf-providers' transport policy.
        .redirect(same_origin_redirects(5))
        .build()
        .expect("cyper client configuration is valid")
}

/// A [`cyper::redirect::Policy`] that follows at most `max` hops and only within one origin
/// (same scheme + host + port). Cross-origin hops and https→http downgrades fail the request.
pub(crate) fn same_origin_redirects(max: usize) -> cyper::redirect::Policy {
    cyper::redirect::Policy::custom(move |attempt| {
        if attempt.previous().len() > max {
            return attempt.error("too many redirects");
        }
        let hop_is_same_origin = attempt.previous().last().is_some_and(|prev| {
            prev.scheme() == attempt.url().scheme()
                && prev.host_str() == attempt.url().host_str()
                && prev.port_or_known_default() == attempt.url().port_or_known_default()
        });
        if hop_is_same_origin {
            attempt.follow()
        } else {
            attempt.error("cross-origin redirect refused (credential protection)")
        }
    })
}

/// Await a `Send` request future with a Compio timer while retaining the `Send` contract required by
/// the interceptor and OAuth traits. Compio timers are local to their runtime, so the timer runs on a
/// short-lived local runtime and signals the caller through a runtime-neutral oneshot channel.
pub(crate) async fn with_timeout<T>(
    timeout: Duration,
    request: impl std::future::Future<Output = cyper::Result<T>> + Send,
) -> Result<T, String>
where
    T: Send,
{
    let (timer_tx, mut timer_rx) = tokio::sync::oneshot::channel();
    std::thread::Builder::new()
        .name("llmleaf-control-timeout".to_string())
        .spawn(move || {
            if let Ok(runtime) = compio::runtime::Runtime::new() {
                runtime.block_on(compio::time::sleep(timeout));
            }
            let _ = timer_tx.send(());
        })
        .map_err(|e| format!("failed to start control timeout: {e}"))?;
    tokio::select! {
        result = request => result.map_err(|e| e.to_string()),
        _ = &mut timer_rx => Err("control request timed out".to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redact_url_strips_userinfo_only() {
        assert_eq!(
            redact_url("https://user:secret@control.example/llmleaf/keys"),
            "https://<redacted>@control.example/llmleaf/keys"
        );
        assert_eq!(
            redact_url("http://control.example:3000/llmleaf/usage?x=1"),
            "http://control.example:3000/llmleaf/usage?x=1"
        );
        assert_eq!(redact_url("not-a-url"), "not-a-url");
        assert_eq!(
            redact_url("https://user@host/x"),
            "https://<redacted>@host/x"
        );
    }
}
