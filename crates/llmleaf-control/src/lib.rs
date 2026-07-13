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
        reqwest::Client::new(),
        hook,
        auth,
    )))
}

/// Handles to the spawned background tasks. After cancelling the shared [`CancellationToken`], await
/// [`ControlHandles::join`] to drain them on shutdown.
pub struct ControlHandles {
    handles: Vec<tokio::task::JoinHandle<()>>,
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
    let http = reqwest::Client::new();
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
    http: &reqwest::Client,
    url: &str,
    auth: Option<&ResolvedAuth>,
    timeout: Duration,
) -> Result<T, reqwest::Error> {
    let req = apply_auth(http.get(url).timeout(timeout), auth);
    req.send().await?.error_for_status()?.json::<T>().await
}

/// Attach a resolved control-plane auth to an outbound request: a `bearer` as `Authorization: Bearer
/// <token>`, a `header` scheme as its `<name>: <value>`. `None` ⇒ the request goes out unauthenticated.
/// The single place the core's [`ResolvedAuth`] meets reqwest, keeping `llmleaf-core` HTTP-free.
pub(crate) fn apply_auth(
    req: reqwest::RequestBuilder,
    auth: Option<&ResolvedAuth>,
) -> reqwest::RequestBuilder {
    match auth {
        Some(ResolvedAuth::Bearer(token)) => req.bearer_auth(token),
        Some(ResolvedAuth::Header { name, value }) => req.header(name, value),
        None => req,
    }
}
