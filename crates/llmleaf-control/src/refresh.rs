//! Pull refreshers: poll a control endpoint on an interval and install the result as a new node-local
//! snapshot in the core's [`KeyStore`]. A failed poll keeps the last-good snapshot in place — the
//! proxy keeps serving (principle 8). The hot path is never on the network; these run in the
//! background and only ever call the HTTP-free snapshot-swap methods.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use llmleaf_core::{
    Engine, IdentityInput, IdentitySource, KeyStore, LimitsSource, ProviderConfig, ProviderFactory,
    ResolvedAuth, RouteConfig, TopologySource, Verdict,
};
use serde::Deserialize;
use tokio::task::JoinHandle;
use tokio::time::{interval, MissedTickBehavior};
use tokio_util::sync::CancellationToken;

use crate::get_json;

// ---------------------------------------------------------------------------------------------
// Verdicts (`[control.limits]`)
// ---------------------------------------------------------------------------------------------

/// GETs `{"verdicts": {key_id: Verdict}}` every `refresh_secs` and installs it as the verdict overlay.
/// Identities are untouched. Fail-open: a failed poll keeps the last-good overlay; a cold node simply
/// has no overlay yet (every authenticated key runs unrestricted).
pub struct VerdictRefresher {
    http: reqwest::Client,
    url: String,
    auth: Option<ResolvedAuth>,
    interval: Duration,
    timeout: Duration,
    keys: Arc<KeyStore>,
    shutdown: CancellationToken,
}

#[derive(Deserialize)]
struct VerdictResponse {
    #[serde(default)]
    verdicts: HashMap<String, Verdict>,
}

impl VerdictRefresher {
    pub fn new(
        http: reqwest::Client,
        cfg: &LimitsSource,
        auth: Option<ResolvedAuth>,
        keys: Arc<KeyStore>,
        shutdown: CancellationToken,
    ) -> Self {
        VerdictRefresher {
            http,
            url: cfg.url.clone(),
            auth,
            interval: Duration::from_secs(cfg.refresh_secs.max(1)),
            timeout: Duration::from_millis(cfg.timeout_ms.max(1)),
            keys,
            shutdown,
        }
    }

    /// Spawn the poll loop. The first tick fires immediately, so verdicts land shortly after startup.
    pub fn spawn(self) -> JoinHandle<()> {
        tokio::spawn(async move {
            let mut tick = interval(self.interval);
            tick.set_missed_tick_behavior(MissedTickBehavior::Delay);
            loop {
                tokio::select! {
                    _ = self.shutdown.cancelled() => break,
                    _ = tick.tick() => self.refresh_once().await,
                }
            }
            tracing::info!(url = %self.url, "verdict refresher stopped");
        })
    }

    async fn refresh_once(&self) {
        match get_json::<VerdictResponse>(&self.http, &self.url, self.auth.as_ref(), self.timeout)
            .await
        {
            Ok(resp) => {
                let n = resp.verdicts.len();
                self.keys.install_verdicts(resp.verdicts);
                tracing::debug!(count = n, url = %self.url, "installed verdict overlay");
            }
            Err(e) => tracing::warn!(error = %e, url = %self.url,
                "verdict pull failed; keeping last-good overlay (fail open)"),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Topology (`[control.topology]`)
// ---------------------------------------------------------------------------------------------

/// GETs `{"providers": […], "routes": […]}` (the file `[[providers]]`/`[[routes]]` shapes) every
/// `refresh_secs` and installs it as the dynamic topology layer via [`Engine::install_topology`],
/// which DIFFS it against the previously installed layer: new resources are added, vanished ones
/// removed (their node-local state cleaned up), changed ones rebuilt — untouched providers keep their
/// live instances, rate buckets, and cooldowns. Fail-open: a failed poll keeps the last-good layer,
/// and a cold node runs from the config base alone; a *successful* poll with empty lists deliberately
/// removes every dynamic resource (the controller declared none). The file base is never touched.
pub struct TopologyRefresher {
    http: reqwest::Client,
    url: String,
    auth: Option<ResolvedAuth>,
    interval: Duration,
    timeout: Duration,
    engine: Arc<Engine>,
    /// The binary's `kind` → instance mapping — pulled providers are built exactly like file ones.
    factory: Arc<dyn ProviderFactory>,
    shutdown: CancellationToken,
}

#[derive(Deserialize)]
struct TopologyResponse {
    #[serde(default)]
    providers: Vec<ProviderConfig>,
    #[serde(default)]
    routes: Vec<RouteConfig>,
}

impl TopologyRefresher {
    pub fn new(
        http: reqwest::Client,
        cfg: &TopologySource,
        auth: Option<ResolvedAuth>,
        engine: Arc<Engine>,
        factory: Arc<dyn ProviderFactory>,
        shutdown: CancellationToken,
    ) -> Self {
        TopologyRefresher {
            http,
            url: cfg.url.clone(),
            auth,
            interval: Duration::from_secs(cfg.refresh_secs.max(1)),
            timeout: Duration::from_millis(cfg.timeout_ms.max(1)),
            engine,
            factory,
            shutdown,
        }
    }

    /// Spawn the poll loop. The first tick fires immediately, so a pulled topology lands shortly after
    /// startup (it is availability-additive, so unlike identity there is nothing to prime fail-closed).
    pub fn spawn(self) -> JoinHandle<()> {
        tokio::spawn(async move {
            let mut tick = interval(self.interval);
            tick.set_missed_tick_behavior(MissedTickBehavior::Delay);
            loop {
                tokio::select! {
                    _ = self.shutdown.cancelled() => break,
                    _ = tick.tick() => self.refresh_once().await,
                }
            }
            tracing::info!(url = %self.url, "topology refresher stopped");
        })
    }

    async fn refresh_once(&self) {
        match get_json::<TopologyResponse>(&self.http, &self.url, self.auth.as_ref(), self.timeout)
            .await
        {
            Ok(resp) => {
                let diff = self.engine.install_topology(
                    resp.providers,
                    resp.routes,
                    self.factory.as_ref(),
                );
                for reason in &diff.skipped {
                    tracing::warn!(url = %self.url, "pulled topology entry skipped: {reason}");
                }
                if diff.is_noop() {
                    tracing::debug!(url = %self.url, "pulled topology unchanged");
                } else {
                    tracing::info!(
                        providers_added = diff.providers_added,
                        providers_updated = diff.providers_updated,
                        providers_removed = diff.providers_removed,
                        routes_added = diff.routes_added,
                        routes_updated = diff.routes_updated,
                        routes_removed = diff.routes_removed,
                        url = %self.url,
                        "installed pulled topology"
                    );
                }
            }
            Err(e) => tracing::warn!(error = %e, url = %self.url,
                "topology pull failed; keeping last-good dynamic layer (fail open)"),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Identity (`[control.identity]`)
// ---------------------------------------------------------------------------------------------

/// GETs `{"keys": [{id, pw_hash, name?, allowed_models?}]}` every `refresh_secs` and installs it as the
/// identity layer (verdicts carried over for surviving keys). The config `[[keys]]` are always the
/// seed; this pull refreshes on top.
///
/// Cold-start posture (fail closed): a failed *first* pull installs nothing, so only the config-seeded
/// identities exist — unknown callers are rejected, never admitted. A warm node keeps its last-good
/// identity cache through an outage. There is no safe "fail open" for authentication, so a failed pull
/// never widens the roster.
pub struct IdentityRefresher {
    http: reqwest::Client,
    url: String,
    auth: Option<ResolvedAuth>,
    interval: Duration,
    timeout: Duration,
    keys: Arc<KeyStore>,
    shutdown: CancellationToken,
}

#[derive(Deserialize)]
struct IdentityResponse {
    #[serde(default)]
    keys: Vec<KeyDto>,
}

#[derive(Deserialize)]
struct KeyDto {
    id: String,
    /// A crypt(3) MCF hash, never plaintext (mirrors [`llmleaf_core::Config`] `[[keys]].pw_hash`).
    pw_hash: String,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    allowed_models: Option<Vec<String>>,
}

impl IdentityRefresher {
    pub fn new(
        http: reqwest::Client,
        cfg: &IdentitySource,
        auth: Option<ResolvedAuth>,
        keys: Arc<KeyStore>,
        shutdown: CancellationToken,
    ) -> Self {
        IdentityRefresher {
            http,
            url: cfg.url.clone(),
            auth,
            interval: Duration::from_secs(cfg.refresh_secs.max(1)),
            timeout: Duration::from_millis(cfg.timeout_ms.max(1)),
            keys,
            shutdown,
        }
    }

    /// The synchronous first pull, run before the listener opens (see [`crate::start`]). On failure it
    /// logs and leaves the config-seeded identities — a cold node fails closed by serving only the
    /// keys it already knows (or nothing, if the config declared none), never by admitting unknowns.
    pub async fn prime(&self) {
        match self.fetch().await {
            Ok(roster) => {
                let n = roster.len();
                self.keys.install_identities(roster);
                tracing::info!(count = n, url = %self.url, "primed identity roster from control plane");
            }
            Err(e) => tracing::warn!(error = %e, url = %self.url,
                "identity prime failed; starting with config-seeded keys only (fail closed)"),
        }
    }

    /// Spawn the poll loop. The immediate first tick is consumed (priming already pulled once), so the
    /// next pull is a full interval away.
    pub fn spawn(self) -> JoinHandle<()> {
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
            tracing::info!(url = %self.url, "identity refresher stopped");
        })
    }

    async fn refresh_once(&self) {
        match self.fetch().await {
            Ok(roster) => {
                let n = roster.len();
                self.keys.install_identities(roster);
                tracing::debug!(count = n, url = %self.url, "installed identity roster");
            }
            Err(e) => tracing::warn!(error = %e, url = %self.url,
                "identity pull failed; keeping last-good roster"),
        }
    }

    async fn fetch(&self) -> Result<Vec<IdentityInput>, reqwest::Error> {
        let resp: IdentityResponse =
            get_json(&self.http, &self.url, self.auth.as_ref(), self.timeout).await?;
        Ok(resp
            .keys
            .into_iter()
            .map(|k| IdentityInput {
                id: k.id,
                pw_hash: k.pw_hash,
                name: k.name,
                allowed_models: k.allowed_models,
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verdict_response_parses_contract_shape() {
        let body = r#"{
            "verdicts": {
                "demo-team":  { "blocked": false, "suspended_until": 1765000000, "allowed_models": ["gpt-4o"] },
                "noisy-team": { "blocked": true }
            }
        }"#;
        let resp: VerdictResponse = serde_json::from_str(body).unwrap();
        assert_eq!(resp.verdicts.len(), 2);
        let demo = &resp.verdicts["demo-team"];
        assert_eq!(demo.suspended_until, Some(1765000000));
        assert_eq!(demo.allowed_models.as_ref().unwrap().len(), 1);
        assert!(resp.verdicts["noisy-team"].blocked);
    }

    #[test]
    fn empty_verdict_response_is_ok() {
        let resp: VerdictResponse = serde_json::from_str("{}").unwrap();
        assert!(resp.verdicts.is_empty());
    }

    #[test]
    fn topology_response_parses_contract_shape() {
        // Exactly the file `[[providers]]`/`[[routes]]` shapes, as JSON.
        let body = r#"{
            "providers": [
                { "name": "openai-dyn", "kind": "openai", "credential": "env:OPENAI_API_KEY",
                  "settings": { "organization": "org-x" },
                  "limits": { "requests_per_min": 100 } },
                { "name": "echo-dyn", "kind": "echo", "prefix": "dyn" }
            ],
            "routes": [
                { "model": "gpt-4o-dyn", "targets": [
                    { "provider": "openai-dyn", "model": "gpt-4o" },
                    { "provider": "echo-dyn" }
                ] }
            ]
        }"#;
        let resp: TopologyResponse = serde_json::from_str(body).unwrap();
        assert_eq!(resp.providers.len(), 2);
        assert_eq!(resp.providers[0].kind, "openai");
        assert_eq!(
            resp.providers[0].limits.as_ref().unwrap().requests_per_min,
            Some(100)
        );
        assert_eq!(resp.providers[1].prefix.as_deref(), Some("dyn"));
        assert_eq!(resp.routes.len(), 1);
        assert_eq!(resp.routes[0].targets.len(), 2);
        assert_eq!(resp.routes[0].targets[1].model, None);

        // Both lists are optional; an empty body clears the dynamic layer.
        let empty: TopologyResponse = serde_json::from_str("{}").unwrap();
        assert!(empty.providers.is_empty());
        assert!(empty.routes.is_empty());

        // The config shapes are strict (`deny_unknown_fields`): a typo'd field is a parse error, never
        // a silently ignored knob.
        assert!(serde_json::from_str::<TopologyResponse>(
            r#"{ "providers": [ { "name": "p", "kind": "echo", "bogus": 1 } ] }"#
        )
        .is_err());
    }

    #[test]
    fn identity_response_parses_contract_shape() {
        let body = r#"{
            "keys": [
                { "id": "demo-team", "pw_hash": "$2y$12$abc", "name": "demo", "allowed_models": ["gpt-4o", "demo"] },
                { "id": "minimal", "pw_hash": "$6$xyz" }
            ]
        }"#;
        let resp: IdentityResponse = serde_json::from_str(body).unwrap();
        assert_eq!(resp.keys.len(), 2);
        assert_eq!(resp.keys[0].id, "demo-team");
        assert_eq!(resp.keys[0].allowed_models.as_deref().unwrap().len(), 2);
        assert_eq!(resp.keys[1].name, None);
        assert_eq!(resp.keys[1].allowed_models, None);
    }
}
