//! The hot path (SOUL.md principle 1: "The hot path is sacred").
//!
//! A request's whole life, and nothing else per-request in the core:
//!   authenticate → map in → **route → stream → emit events** → map out.
//! Authentication and the dialect mappings happen at the server edge; this module owns the middle:
//! pick a target from the route's ordered fallback chain (skipping providers this node currently
//! considers down), call it, and wrap the resulting canonical stream so usage/lifecycle events flow
//! out the bus as bytes pass. No allocation here that isn't a clone of already-owned config.
//!
//! Fallback (principle 8) happens at *connection* time: once a provider hands back a stream we are
//! committed to it (its bytes may already be on the wire). A connect failure penalizes that provider
//! (node-local health) and falls through to the next target.

use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::sync::{Arc, RwLock};
use std::time::Duration;

use async_stream::stream;
use futures::StreamExt;
use llmleaf_model::{
    AudioChunk, AudioStream, BatchHandle, BatchOutcome, BatchResultStream, BatchSpec, ChatRequest,
    EmbeddingRequest, EmbeddingResponse, FinishReason, ModelError, ModelInfo, RerankRequest,
    RerankResponse, ResponseStream, SpeechRequest, StreamChunk, TranscriptionRequest,
    TranscriptionResponse, VoiceInfo,
};
use llmleaf_pricing::Pricing;
use llmleaf_provider::{Provider, ProviderCx, ProviderFactory, ProviderRegistry, RealtimeParams};
use serde_json::Value;
use thiserror::Error;
use tokio::time::Instant;

use crate::batch_id;
use crate::config::{Config, InterceptPhase, ProviderConfig, RouteConfig, Target};
use crate::events::{Event, EventBus};
use crate::ratelimit::{RateGuard, RateLimiter};
use crate::route::{HealthTable, Router};

#[derive(Debug, Error)]
pub enum EngineError {
    #[error("no route configured for model '{0}'")]
    NoRoute(String),
    #[error("route for model '{0}' has no targets")]
    NoTargets(String),
    /// Every target in the chain was down or failed to connect.
    #[error("all targets failed: {0}")]
    AllTargetsFailed(ModelError),
    /// Every target on the chain was over its node-local rate limit, and the bounded wait for capacity
    /// (`server.rate_limit_max_wait_ms`) elapsed without a slot freeing up. `retry_after_secs` is the
    /// soonest estimated time capacity returns — surfaced to the consumer as a `Retry-After` header.
    #[error("rate limited; retry after {retry_after_secs}s")]
    RateLimited { retry_after_secs: u64 },
    /// A sync interceptor blocked this request in-flight (principle 1: the one sanctioned hot-path
    /// insertion). Carries the reason for the consumer.
    #[error("blocked by interceptor: {0}")]
    Blocked(String),
    /// A batch was submitted with no requests.
    #[error("batch has no requests")]
    EmptyBatch,
    /// A batch's requests do not all route to the same provider instance. A batch is one upstream job,
    /// so it cannot be split across providers (a future extension may fan out and merge; see SOUL).
    #[error("batch mixes models that route to different providers")]
    MixedProviderBatch,
    /// A batch id did not decode, or named a provider instance this node no longer has configured.
    /// Treated as not-found rather than a server error: the id is opaque and may be stale or foreign.
    #[error("batch '{0}' not found")]
    BatchNotFound(String),
}

/// The one sanctioned hot-path insertion (SOUL.md principle 1): an external service the core calls
/// in-flight to pass / block / rewrite a canonical payload. The trait lives in the core (HTTP-free,
/// principle 2); its implementation (reqwest) lives in `llmleaf-control` and is wired in by the binary.
#[async_trait::async_trait]
pub trait Interceptor: Send + Sync {
    /// Cheap scope check — `true` if this hook fires for the given phase/key/model. Called *before* any
    /// serialization, so an out-of-scope request pays nothing on the hot path.
    fn applies(&self, phase: InterceptPhase, key: &str, model: &str) -> bool;

    /// Screen a canonical payload in-flight. Called only when [`Interceptor::applies`] returned true.
    /// The implementation owns its own timeout and fail-open/closed policy.
    async fn screen(
        &self,
        phase: InterceptPhase,
        key: &str,
        model: &str,
        payload: &Value,
    ) -> InterceptOutcome;
}

/// An interceptor's verdict on a payload.
pub enum InterceptOutcome {
    /// Proceed unchanged.
    Pass,
    /// Refuse the request; `reason` is surfaced to the consumer.
    Block { reason: String },
    /// Replace the canonical payload with this one before proceeding (the only sanctioned non-transparent
    /// mutation — principle 7 — because it is explicit config plus a documented hook).
    Rewrite(Value),
}

/// What [`Engine::dispatch`] hands back on success: the served modality value plus the identity the
/// caller needs to finish the lifecycle (instrument a stream, or emit the batch tail). The chosen
/// provider has already had `RequestRouted` emitted and its health penalty cleared.
struct Dispatched<T> {
    value: T,
    request_id: String,
    key: String,
    logical_model: String,
    /// The config name of the provider instance that served, and the upstream model id it served as —
    /// what the tokens/min rate-limit debit keys on (per-model buckets are keyed by *upstream* model).
    provider: String,
    upstream_model: String,
    /// The limiter this request was admitted against — the one the tokens/min debit must land in. Held
    /// per-request so a topology swap mid-stream keeps debiting the buckets that admitted it.
    rate: Arc<RateLimiter>,
    /// The rate-limit admission guard (concurrency permits). Kept alive for the life of the request:
    /// moved into the instrumented stream for streaming modalities, held until return for batch ones.
    guard: RateGuard,
}

/// A native-realtime-capable target the realtime edge can drive directly (the analogue, for sessions,
/// of what [`Engine::dispatch`] picks for one-shot modalities). Resolved by [`Engine::realtime_target`].
pub struct RealtimeTarget {
    /// The chosen provider instance (already confirmed to serve native realtime).
    pub provider: Arc<dyn Provider>,
    /// The per-call context (resolved credential, endpoint, settings) for that provider.
    pub cx: ProviderCx,
    /// The upstream realtime params (the resolved upstream model id).
    pub params: RealtimeParams,
    /// The config name of the chosen provider (for `RequestRouted` events).
    pub provider_name: String,
}

/// One immutable view of everything routing needs: the provider registry, the route table, the
/// per-instance configs, and the node-local rate limiter. Built from the file base at startup;
/// [`Engine::install_topology`] reconciles a pulled dynamic layer onto that base and swaps in a new
/// snapshot. A handler grabs one `Arc` per request, so a mid-request swap never mixes two topologies.
pub struct Topology {
    registry: ProviderRegistry,
    router: Router,
    /// Merged per-instance configs (base + dynamic), what [`Topology::build_cx`] reads.
    providers: HashMap<String, ProviderConfig>,
    /// Node-local rate limiter (per-provider + per-model flow control). Shared into instrumented streams
    /// behind an `Arc` so the tokens/min debit can fire as usage is observed (principles 1, 8, 9).
    rate: Arc<RateLimiter>,
    /// The dynamic (pulled) layer this snapshot carries — what the *next* pull is diffed against.
    dynamic_providers: HashMap<String, ProviderConfig>,
    dynamic_routes: HashMap<String, RouteConfig>,
    /// Instances built for the dynamic providers, carried across installs while their `kind` holds.
    dynamic_instances: HashMap<String, Arc<dyn Provider>>,
}

impl Topology {
    /// Resolves logical models to fallback chains (and prefix namespaces) in this snapshot.
    pub fn router(&self) -> &Router {
        &self.router
    }

    fn build_cx(&self, name: &str, request_id: &str) -> ProviderCx {
        let cfg = self.providers.get(name);
        ProviderCx {
            request_id: request_id.to_string(),
            credential: cfg
                .and_then(|c| c.credential.as_ref())
                .and_then(|s| s.resolve()),
            endpoint: cfg.and_then(|c| c.endpoint.clone()),
            settings: cfg.map(|c| c.settings.clone()).unwrap_or_default(),
        }
    }
}

/// What one [`Engine::install_topology`] reconciliation did, for the control-plane refresher to log.
/// Counts are of *dynamic* resources only; the file base never changes.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct TopologyDiff {
    pub providers_added: usize,
    pub providers_updated: usize,
    pub providers_removed: usize,
    pub routes_added: usize,
    pub routes_updated: usize,
    pub routes_removed: usize,
    /// Pulled entries that could not be installed (a base-name collision, a duplicate, an unknown
    /// kind), each with its reason — surfaced so the operator sees exactly what was dropped.
    pub skipped: Vec<String>,
}

impl TopologyDiff {
    /// Whether this install changed nothing (skips are not changes — a skipped entry was never live).
    pub fn is_noop(&self) -> bool {
        self.providers_added == 0
            && self.providers_updated == 0
            && self.providers_removed == 0
            && self.routes_added == 0
            && self.routes_updated == 0
            && self.routes_removed == 0
    }
}

/// The data-plane engine. Cheap to share behind an `Arc`; holds the swappable topology snapshot plus
/// the node-local health table and the (clone-able) event bus.
pub struct Engine {
    /// The current topology, swapped wholesale by [`Engine::install_topology`] — the same snapshot
    /// pattern as [`crate::keys::KeyStore`]: the hot path takes one short read lock, bumps the `Arc`,
    /// and never holds the lock across an `.await` (principle 1: no allocation, no I/O).
    topology: RwLock<Arc<Topology>>,
    /// The immutable file base (principle 6). A pulled layer merges *onto* this; it can extend the
    /// base but never shadow one of its names.
    base_registry: Arc<ProviderRegistry>,
    base_providers: Vec<ProviderConfig>,
    base_routes: Vec<RouteConfig>,
    events: EventBus,
    pricing: Arc<Pricing>,
    health: HealthTable,
    /// Upper bound on the hot-path wait for rate-limit capacity when every target is saturated.
    rate_max_wait: Duration,
    include_payloads: bool,
    cooldown_secs: u64,
    /// The optional in-flight sync interceptor (`[control.intercept]`), wired by the binary. `None` ⇒
    /// no hot-path insertion at all.
    interceptor: Option<Arc<dyn Interceptor>>,
}

impl Engine {
    pub fn new(
        config: &Config,
        registry: Arc<ProviderRegistry>,
        events: EventBus,
        pricing: Arc<Pricing>,
        interceptor: Option<Arc<dyn Interceptor>>,
    ) -> Self {
        let providers = config
            .providers
            .iter()
            .map(|p| (p.name.clone(), p.clone()))
            .collect();
        let base = Topology {
            registry: (*registry).clone(),
            router: Router::new(&config.routes, &config.providers),
            providers,
            rate: Arc::new(RateLimiter::new(&config.providers)),
            dynamic_providers: HashMap::new(),
            dynamic_routes: HashMap::new(),
            dynamic_instances: HashMap::new(),
        };
        Engine {
            topology: RwLock::new(Arc::new(base)),
            base_registry: registry,
            base_providers: config.providers.clone(),
            base_routes: config.routes.clone(),
            events,
            pricing,
            health: HealthTable::new(),
            rate_max_wait: Duration::from_millis(config.server.rate_limit_max_wait_ms),
            include_payloads: config.server.include_payloads,
            cooldown_secs: config.server.fallback_cooldown_secs,
            interceptor,
        }
    }

    /// The current topology snapshot: one short read lock and an `Arc` bump. Callers hold one snapshot
    /// for a whole request, so a concurrent [`Engine::install_topology`] never mixes two topologies.
    pub fn topology(&self) -> Arc<Topology> {
        self.topology.read().unwrap().clone()
    }

    /// Install a pulled dynamic topology layer (`[control.topology]`): diff `providers`/`routes`
    /// against the previously installed layer and swap in a reconciled snapshot — new resources are
    /// added, vanished ones removed (instance dropped, rate and cooldown state cleaned up), changed
    /// ones rebuilt — while untouched providers keep their live instances, rate buckets, and cooldowns.
    ///
    /// The file base always wins: a pulled provider whose name (or route whose model) collides with a
    /// config-file entry is skipped, so the pulled layer extends the base but never overrides it
    /// (principle 6). `factory` builds instances for new kinds — the core itself still never names a
    /// provider (principle 2). Per-entry failures (unknown kind, duplicate) skip that entry and are
    /// reported in the returned [`TopologyDiff`], never poisoning the rest of the pull.
    ///
    /// Called by the control plane's single topology refresher; installs are serialized by having that
    /// one caller. The hot path only ever *reads* snapshots, so requests are unaffected either way.
    pub fn install_topology(
        &self,
        providers: Vec<ProviderConfig>,
        routes: Vec<RouteConfig>,
        factory: &dyn ProviderFactory,
    ) -> TopologyDiff {
        let prev = self.topology();
        let mut diff = TopologyDiff::default();

        // Filter the pulled providers (base shadowing, duplicates, unknown kinds), reusing the previous
        // instance while `kind` is unchanged: an instance is per-kind dialect logic — endpoint,
        // credential, and settings flow per-call via `ProviderCx` — so a config-only change needs no
        // rebuild, and a re-kinded or new provider goes through the factory.
        let base_names: HashSet<&str> = self
            .base_providers
            .iter()
            .map(|p| p.name.as_str())
            .collect();
        let mut dyn_providers: HashMap<String, ProviderConfig> = HashMap::new();
        let mut dyn_instances: HashMap<String, Arc<dyn Provider>> = HashMap::new();
        for p in providers {
            if base_names.contains(p.name.as_str()) {
                diff.skipped.push(format!(
                    "provider '{}' shadows a config-file provider; skipped",
                    p.name
                ));
                continue;
            }
            if dyn_providers.contains_key(&p.name) {
                diff.skipped.push(format!(
                    "duplicate provider '{}' in pulled topology; first entry wins",
                    p.name
                ));
                continue;
            }
            let reused = match prev.dynamic_providers.get(&p.name) {
                Some(old) if old.kind == p.kind => prev.dynamic_instances.get(&p.name).cloned(),
                _ => None,
            };
            let Some(instance) = reused.or_else(|| factory.build(&p.kind)) else {
                diff.skipped.push(format!(
                    "provider '{}' has unknown kind '{}'; skipped",
                    p.name, p.kind
                ));
                continue;
            };
            dyn_instances.insert(p.name.clone(), instance);
            dyn_providers.insert(p.name.clone(), p);
        }

        let base_models: HashSet<&str> =
            self.base_routes.iter().map(|r| r.model.as_str()).collect();
        let mut dyn_routes: HashMap<String, RouteConfig> = HashMap::new();
        for r in routes {
            if base_models.contains(r.model.as_str()) {
                diff.skipped.push(format!(
                    "route '{}' shadows a config-file route; skipped",
                    r.model
                ));
                continue;
            }
            if dyn_routes.contains_key(&r.model) {
                diff.skipped.push(format!(
                    "duplicate route '{}' in pulled topology; first entry wins",
                    r.model
                ));
                continue;
            }
            dyn_routes.insert(r.model.clone(), r);
        }

        // The diff proper, against the previously installed dynamic layer.
        for (name, cfg) in &dyn_providers {
            match prev.dynamic_providers.get(name) {
                None => diff.providers_added += 1,
                Some(old) if old != cfg => diff.providers_updated += 1,
                Some(_) => {}
            }
        }
        diff.providers_removed = prev
            .dynamic_providers
            .keys()
            .filter(|n| !dyn_providers.contains_key(*n))
            .count();
        for (model, cfg) in &dyn_routes {
            match prev.dynamic_routes.get(model) {
                None => diff.routes_added += 1,
                Some(old) if old != cfg => diff.routes_updated += 1,
                Some(_) => {}
            }
        }
        diff.routes_removed = prev
            .dynamic_routes
            .keys()
            .filter(|m| !dyn_routes.contains_key(*m))
            .count();

        // An identical pull is the common steady state: keep the current snapshot untouched — no swap,
        // no state churn, nothing rebuilt.
        if diff.is_noop() {
            return diff;
        }

        // Merge onto the base and rebuild the derived tables.
        let merged_providers: Vec<ProviderConfig> = self
            .base_providers
            .iter()
            .cloned()
            .chain(dyn_providers.values().cloned())
            .collect();
        let merged_routes: Vec<RouteConfig> = self
            .base_routes
            .iter()
            .cloned()
            .chain(dyn_routes.values().cloned())
            .collect();
        let providers_map: HashMap<String, ProviderConfig> = merged_providers
            .iter()
            .map(|p| (p.name.clone(), p.clone()))
            .collect();

        let mut registry = (*self.base_registry).clone();
        for (name, instance) in &dyn_instances {
            registry.register(name.clone(), instance.clone());
        }

        // Rate state carries over for every provider whose limit config is unchanged (the whole base
        // always qualifies), so a swap never resets buckets or drops held permits for untouched
        // providers — and in-flight streams still debit those same shared buckets.
        let rate = Arc::new(prev.rate.reconciled(&merged_providers, |name| {
            match (prev.providers.get(name), providers_map.get(name)) {
                (Some(old), Some(new)) => {
                    old.limits == new.limits && old.model_limits == new.model_limits
                }
                _ => false,
            }
        }));

        // Cooldown cleanup: a removed provider must not leave a stale health entry behind, and a
        // *changed* one is a new upstream contact whose old penalty no longer means anything.
        for (name, old) in &prev.providers {
            if providers_map.get(name).is_none_or(|new| new != old) {
                self.health.clear(name);
            }
        }

        let next = Topology {
            registry,
            router: Router::new(&merged_routes, &merged_providers),
            providers: providers_map,
            rate,
            dynamic_providers: dyn_providers,
            dynamic_routes: dyn_routes,
            dynamic_instances: dyn_instances,
        };
        *self.topology.write().unwrap() = Arc::new(next);
        diff
    }

    /// Run the request-phase sync interceptor, if one is configured and in scope for this `(key,
    /// model)`. Returns the request unchanged on `Pass` (or when no interceptor applies — the common
    /// case, paying nothing), the rewritten request on `Rewrite`, or [`EngineError::Blocked`] on
    /// `Block`. This is the single sanctioned hot-path insertion (principle 1); the operator pays its
    /// latency knowingly by opting the route/key in.
    async fn screen_request<T: serde::Serialize + serde::de::DeserializeOwned>(
        &self,
        req: T,
        key: &str,
        model: &str,
    ) -> Result<T, EngineError> {
        let Some(ic) = &self.interceptor else {
            return Ok(req);
        };
        if !ic.applies(InterceptPhase::Request, key, model) {
            return Ok(req);
        }
        // Serialize the canonical request for screening. A serialization hiccup is our bug, not a policy
        // decision, so we log and pass rather than block legitimate traffic.
        let value = match serde_json::to_value(&req) {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!(error = %e, "could not serialize request for interceptor; passing through");
                return Ok(req);
            }
        };
        match ic.screen(InterceptPhase::Request, key, model, &value).await {
            InterceptOutcome::Pass => Ok(req),
            InterceptOutcome::Block { reason } => Err(EngineError::Blocked(reason)),
            InterceptOutcome::Rewrite(v) => serde_json::from_value(v)
                .map_err(|e| EngineError::Blocked(format!("interceptor rewrite was invalid: {e}"))),
        }
    }

    pub fn health(&self) -> &HealthTable {
        &self.health
    }

    /// The current topology's node-local rate limiter, for read-only observability surfaces.
    pub fn rate(&self) -> Arc<RateLimiter> {
        self.topology().rate.clone()
    }

    /// Read-only access to the bundled model catalog (modality + limits + rates) for the model-listing
    /// surface. Pure lookup/iteration over the bundled dataset — never a fetch (principle 1).
    pub fn pricing(&self) -> &Pricing {
        &self.pricing
    }

    /// The ordered fallback chain for a logical `model`, or `None` if unrouted. Delegates to the router
    /// so a handler can read a model's targets *without* ever touching the provider registry (principle
    /// 2: the core knows no provider — this exposes only operator-declared config, not a catalog).
    /// Returned owned (targets are a handful of small strings, and only listing surfaces call this) so
    /// the chain outlives the topology snapshot it was resolved from.
    pub fn resolve_targets(&self, model: &str) -> Option<Vec<Target>> {
        self.topology()
            .router
            .resolve(model)
            .map(|t| t.into_owned())
    }

    /// Enumerate the upstream catalog of a configured provider instance by asking it
    /// ([`Provider::models`]). The handler passes a provider *name* from the router's prefix table; the
    /// engine resolves it against the registry and builds the call context — so a handler can list a
    /// prefix namespace's real models without ever naming or touching a provider itself (principle 2).
    /// A provider that does not enumerate returns [`ModelError::Unsupported`], which the listing renders
    /// as a non-enumerable namespace rather than a failure.
    pub async fn provider_models(
        &self,
        provider_name: &str,
        request_id: &str,
    ) -> Result<Vec<ModelInfo>, ModelError> {
        let topo = self.topology();
        let Some(provider) = topo.registry.get(provider_name) else {
            return Err(ModelError::Unsupported(format!(
                "no provider instance '{provider_name}'"
            )));
        };
        let cx = topo.build_cx(provider_name, request_id);
        provider.models(&cx).await
    }

    /// Pick the best target for `model` whose provider serves a *native* realtime session: prefer one
    /// that isn't cooling down, but — mirroring [`Self::dispatch`] — if every realtime-capable target
    /// is down, still return one rather than force the session onto the chat bridge over a stale,
    /// node-local cooldown (principle 8: fail toward availability). `None` ⇒ no realtime-capable target
    /// on the chain at all — the realtime edge should bridge over chat streaming instead (the
    /// `Unsupported` fall-through, applied at session granularity). The core never names the provider;
    /// it asks each candidate via [`Provider::supports_realtime`].
    pub fn realtime_target(
        &self,
        model: &str,
        request_id: &str,
        now: u64,
    ) -> Option<RealtimeTarget> {
        let topo = self.topology();
        let targets = topo.router.resolve(model)?;
        // First pass prefers realtime-capable targets that aren't cooling down; the second falls open
        // to a cooled-down one when none is fresh. A provider that lacks native realtime is skipped in
        // both passes — that is a capability gap, not a health signal, so it correctly falls through to
        // the chat bridge rather than being served natively.
        for honor_cooldown in [true, false] {
            for target in targets.iter() {
                if honor_cooldown && self.health.is_down(&target.provider, now) {
                    continue;
                }
                let Some(provider) = topo.registry.get(&target.provider) else {
                    continue;
                };
                if !provider.supports_realtime() {
                    continue;
                }
                let upstream = target.model.clone().unwrap_or_else(|| model.to_string());
                let cx = topo.build_cx(&target.provider, request_id);
                return Some(RealtimeTarget {
                    provider,
                    cx,
                    params: RealtimeParams { model: upstream },
                    provider_name: target.provider.clone(),
                });
            }
        }
        None
    }

    /// Price provider-reported usage from the bundled dataset (a lookup + multiply, never a fetch). The
    /// realtime edge uses this to price usage tapped from a native session's terminal frame; the bridge
    /// path is priced by [`Self::instrument`] like any other stream.
    pub fn price(&self, model: &str, usage: llmleaf_model::Usage) -> llmleaf_model::Usage {
        self.pricing.price(model, usage)
    }

    /// Run a chat request through the pipeline. `key` is the authorized key id (for events),
    /// `request_id` correlates the lifecycle, `now` is unix-seconds supplied by the caller (the core
    /// holds no clock for decisions — principle 9).
    pub async fn run(
        &self,
        req: ChatRequest,
        key: String,
        request_id: String,
        now: u64,
    ) -> Result<ResponseStream, EngineError> {
        let logical_model = req.model.clone();
        let req = self.screen_request(req, &key, &logical_model).await?;
        let payload = self.payload(&req);
        let dispatched = self
            .dispatch(
                logical_model,
                key,
                request_id,
                now,
                payload,
                move |provider, cx, upstream| {
                    let mut preq = req.clone();
                    preq.model = upstream;
                    async move { provider.chat(preq, &cx).await }
                },
            )
            .await?;
        Ok(self.instrument(dispatched))
    }

    /// Embed inputs through the pipeline. Same route→fallback→health→events skeleton as chat; a target
    /// that does not implement embeddings ([`ModelError::Unsupported`]) is skipped without a health
    /// penalty, so the chain naturally lands on the next embedding-capable provider.
    pub async fn embed(
        &self,
        req: EmbeddingRequest,
        key: String,
        request_id: String,
        now: u64,
    ) -> Result<EmbeddingResponse, EngineError> {
        let logical_model = req.model.clone();
        let req = self.screen_request(req, &key, &logical_model).await?;
        let payload = self.payload(&req);
        let dispatched = self
            .dispatch(
                logical_model,
                key,
                request_id,
                now,
                payload,
                move |provider, cx, upstream| {
                    let mut preq = req.clone();
                    preq.model = upstream;
                    async move { provider.embed(preq, &cx).await }
                },
            )
            .await?;
        let Dispatched {
            value: mut resp,
            request_id,
            key,
            logical_model,
            provider,
            upstream_model,
            rate,
            guard,
        } = dispatched;
        resp.usage = self.pricing.price(&logical_model, resp.usage);
        // Debit the observed tokens against this provider/model's tokens/min bucket (the cost was unknown
        // at admission); then release the concurrency permit by dropping the guard.
        rate.debit_tokens(
            &provider,
            &upstream_model,
            resp.usage.total_tokens,
            Instant::now(),
        );
        self.emit_batch_tail(&request_id, &key, &logical_model, resp.usage);
        drop(guard);
        Ok(resp)
    }

    /// Rerank a query's documents through the pipeline. Same route→fallback→health→events skeleton as
    /// embeddings — a batch modality: a target that does not implement rerank ([`ModelError::Unsupported`])
    /// is skipped without a health penalty, so the chain naturally lands on the next rerank-capable
    /// provider.
    pub async fn rerank(
        &self,
        req: RerankRequest,
        key: String,
        request_id: String,
        now: u64,
    ) -> Result<RerankResponse, EngineError> {
        let logical_model = req.model.clone();
        let req = self.screen_request(req, &key, &logical_model).await?;
        let payload = self.payload(&req);
        let dispatched = self
            .dispatch(
                logical_model,
                key,
                request_id,
                now,
                payload,
                move |provider, cx, upstream| {
                    let mut preq = req.clone();
                    preq.model = upstream;
                    async move { provider.rerank(preq, &cx).await }
                },
            )
            .await?;
        let Dispatched {
            value: mut resp,
            request_id,
            key,
            logical_model,
            provider,
            upstream_model,
            rate,
            guard,
        } = dispatched;
        resp.usage = self.pricing.price(&logical_model, resp.usage);
        rate.debit_tokens(
            &provider,
            &upstream_model,
            resp.usage.total_tokens,
            Instant::now(),
        );
        self.emit_batch_tail(&request_id, &key, &logical_model, resp.usage);
        drop(guard);
        Ok(resp)
    }

    /// Synthesize speech through the pipeline. Audio is a stream, so this mirrors chat: the chosen
    /// provider's [`AudioStream`] is instrumented so usage/lifecycle events flow as bytes pass.
    pub async fn speech(
        &self,
        req: SpeechRequest,
        key: String,
        request_id: String,
        now: u64,
    ) -> Result<AudioStream, EngineError> {
        let logical_model = req.model.clone();
        let req = self.screen_request(req, &key, &logical_model).await?;
        let payload = self.payload(&req);
        let dispatched = self
            .dispatch(
                logical_model,
                key,
                request_id,
                now,
                payload,
                move |provider, cx, upstream| {
                    let mut preq = req.clone();
                    preq.model = upstream;
                    async move { provider.speech(preq, &cx).await }
                },
            )
            .await?;
        Ok(self.instrument_audio(dispatched))
    }

    /// List the voices a speech model can synthesize with. A metadata lookup, not a billed completion:
    /// it rides the same route→fallback→health skeleton (principle 1) — a target that does not list
    /// voices ([`ModelError::Unsupported`]) is skipped without a penalty, so the chain lands on the next
    /// voice-aware provider — but there is no usage to price and no tail event to emit.
    pub async fn voices(
        &self,
        model: String,
        key: String,
        request_id: String,
        now: u64,
    ) -> Result<Vec<VoiceInfo>, EngineError> {
        let dispatched = self
            .dispatch(
                model,
                key,
                request_id,
                now,
                None,
                move |provider, cx, upstream| async move { provider.voices(&upstream, &cx).await },
            )
            .await?;
        // Close the lifecycle bracket `dispatch` opened with `RequestStarted`. There is no `Usage`
        // event: a voices listing consumes no tokens, and the core never fabricates accounting
        // (principle 5) — so this is the one tail that emits `RequestCompleted` without a preceding
        // `Usage`. A failure path already emitted `RequestFailed` inside `dispatch`.
        self.events.emit(Event::RequestCompleted {
            id: dispatched.request_id.clone(),
            finish: None,
        });
        Ok(dispatched.value)
    }

    /// Transcribe audio through the pipeline. A batch result like embeddings.
    pub async fn transcribe(
        &self,
        req: TranscriptionRequest,
        key: String,
        request_id: String,
        now: u64,
    ) -> Result<TranscriptionResponse, EngineError> {
        let logical_model = req.model.clone();
        let req = self.screen_request(req, &key, &logical_model).await?;
        let payload = self.payload(&req);
        let dispatched = self
            .dispatch(
                logical_model,
                key,
                request_id,
                now,
                payload,
                move |provider, cx, upstream| {
                    let mut preq = req.clone();
                    preq.model = upstream;
                    async move { provider.transcribe(preq, &cx).await }
                },
            )
            .await?;
        let Dispatched {
            value: mut resp,
            request_id,
            key,
            logical_model,
            provider,
            upstream_model,
            rate,
            guard,
        } = dispatched;
        resp.usage = self.pricing.price(&logical_model, resp.usage);
        rate.debit_tokens(
            &provider,
            &upstream_model,
            resp.usage.total_tokens,
            Instant::now(),
        );
        self.emit_batch_tail(&request_id, &key, &logical_model, resp.usage);
        drop(guard);
        Ok(resp)
    }

    // ---------------------------------------------------------------------------------------------
    // Batch (asynchronous jobs). Unlike the modalities above, batch does NOT use the fallback chain:
    // a batch is one long-lived upstream job, so it targets exactly one provider and never silently
    // re-submits to a fallback (that could double-bill a 100k-request job). State lives at the
    // provider's upstream; the only thing that crosses nodes is the opaque, provider-encoding id
    // (see `batch_id`), so retrieve/cancel/results are served from the id alone (principle 9).
    // ---------------------------------------------------------------------------------------------

    /// Submit a batch. Resolves the single provider instance that will own it (rejecting a batch whose
    /// requests route to different providers), remaps each item's logical model to its route's upstream
    /// id, submits to that provider, and wraps the returned upstream id into the opaque token the
    /// consumer sees. `RequestStarted`/`RequestRouted`/`RequestCompleted` bracket the *submission* (no
    /// `Usage`: nothing is billed at submit time — tokens are accounted when results are fetched).
    pub async fn batch_create(
        &self,
        spec: BatchSpec,
        key: String,
        request_id: String,
    ) -> Result<BatchHandle, EngineError> {
        if spec.items.is_empty() {
            return Err(EngineError::EmptyBatch);
        }
        let topo = self.topology();
        let (provider_name, routing_model, spec) = Self::resolve_batch(&topo, spec)?;
        let provider = topo
            .registry
            .get(&provider_name)
            .ok_or_else(|| EngineError::BatchNotFound(provider_name.clone()))?;
        let cx = topo.build_cx(&provider_name, &request_id);

        self.events.emit(Event::RequestStarted {
            id: request_id.clone(),
            key,
            model: routing_model,
            request: self.payload(&spec),
        });

        match provider.batch_create(spec, &cx).await {
            Ok(mut handle) => {
                self.events.emit(Event::RequestRouted {
                    id: request_id.clone(),
                    provider: provider_name.clone(),
                    upstream_model: handle.endpoint.clone().unwrap_or_default(),
                });
                handle.id = batch_id::encode_batch(&provider_name, &handle.id);
                self.events.emit(Event::RequestCompleted {
                    id: request_id,
                    finish: None,
                });
                Ok(handle)
            }
            Err(e) => {
                self.events.emit(Event::RequestFailed {
                    id: request_id,
                    error: e.to_string(),
                });
                Err(EngineError::AllTargetsFailed(e))
            }
        }
    }

    /// Poll a batch's status. Decodes the owning provider from the opaque id and forwards directly —
    /// no router, no fallback (only that instance holds the job). Deliberately event-free: status is
    /// polled frequently and the core keeps the poll a thin proxy (principle 1). The returned handle's
    /// id is re-wrapped into the opaque token so the consumer keeps seeing a stable id.
    pub async fn batch_retrieve(
        &self,
        batch_id: &str,
        request_id: String,
    ) -> Result<BatchHandle, EngineError> {
        let topo = self.topology();
        let (provider, provider_name, upstream_id) = Self::batch_target(&topo, batch_id)?;
        let cx = topo.build_cx(&provider_name, &request_id);
        let mut handle = provider
            .batch_retrieve(&upstream_id, &cx)
            .await
            .map_err(EngineError::AllTargetsFailed)?;
        handle.id = batch_id::encode_batch(&provider_name, &handle.id);
        Ok(handle)
    }

    /// Request cancellation of a batch. Like [`Self::batch_retrieve`], routed by the opaque id.
    pub async fn batch_cancel(
        &self,
        batch_id: &str,
        request_id: String,
    ) -> Result<BatchHandle, EngineError> {
        let topo = self.topology();
        let (provider, provider_name, upstream_id) = Self::batch_target(&topo, batch_id)?;
        let cx = topo.build_cx(&provider_name, &request_id);
        let mut handle = provider
            .batch_cancel(&upstream_id, &cx)
            .await
            .map_err(EngineError::AllTargetsFailed)?;
        handle.id = batch_id::encode_batch(&provider_name, &handle.id);
        Ok(handle)
    }

    /// Stream a completed batch's results. Routed by the opaque id; the provider's result stream is
    /// instrumented so each succeeded line's provider-reported usage is priced and pushed as a `Usage`
    /// event (principle 5: relay, never compute). Because results may be fetched more than once, those
    /// events repeat — downstream dedupes by the event id, which is `"<batch-id>:<custom-id>"`.
    pub async fn batch_results(
        &self,
        batch_id: &str,
        key: String,
        request_id: String,
    ) -> Result<BatchResultStream, EngineError> {
        let topo = self.topology();
        let (provider, provider_name, upstream_id) = Self::batch_target(&topo, batch_id)?;
        let cx = topo.build_cx(&provider_name, &request_id);
        let stream = provider
            .batch_results(&upstream_id, &cx)
            .await
            .map_err(EngineError::AllTargetsFailed)?;
        Ok(self.instrument_batch(stream, batch_id.to_string(), key))
    }

    /// Resolve the single provider instance a batch targets and rewrite each item's logical model to
    /// its route's upstream id. Every item must route to the same provider (a batch is one upstream
    /// job). Returns `(provider_name, routing_model, rewritten_spec)`.
    fn resolve_batch(
        topo: &Topology,
        mut spec: BatchSpec,
    ) -> Result<(String, String, BatchSpec), EngineError> {
        let routing_model = spec.items[0].request.model.clone();
        let mut chosen: Option<String> = None;
        for item in spec.items.iter_mut() {
            let targets = topo
                .router
                .resolve(&item.request.model)
                .ok_or_else(|| EngineError::NoRoute(item.request.model.clone()))?;
            let primary = targets
                .first()
                .ok_or_else(|| EngineError::NoTargets(item.request.model.clone()))?;
            match &chosen {
                Some(p) if *p != primary.provider => return Err(EngineError::MixedProviderBatch),
                Some(_) => {}
                None => chosen = Some(primary.provider.clone()),
            }
            if let Some(upstream) = &primary.model {
                item.request.model = upstream.clone();
            }
        }
        // `chosen` is always set: `batch_create` rejected the empty batch before calling us.
        Ok((
            chosen.expect("non-empty batch has a provider"),
            routing_model,
            spec,
        ))
    }

    /// Decode the opaque batch id into `(provider, provider_instance_name, upstream_id)`. A decode
    /// failure or an instance no longer in this node's registry is reported as not-found (the id is
    /// opaque and may be stale or foreign — never a 5xx). The provider name is the one we encoded into
    /// the id at create time, i.e. the config instance name `build_cx` expects.
    fn batch_target(
        topo: &Topology,
        batch_id: &str,
    ) -> Result<(Arc<dyn Provider>, String, String), EngineError> {
        let (provider_name, upstream_id) = batch_id::decode_batch(batch_id)
            .map_err(|_| EngineError::BatchNotFound(batch_id.to_string()))?;
        let provider = topo
            .registry
            .get(&provider_name)
            .ok_or_else(|| EngineError::BatchNotFound(batch_id.to_string()))?;
        Ok((provider, provider_name, upstream_id))
    }

    /// Wrap a provider batch-result stream so each succeeded line's usage is priced and emitted as a
    /// `Usage` event. Mirrors [`Self::instrument`] for the streaming chat path.
    fn instrument_batch(
        &self,
        inner: BatchResultStream,
        batch_id: String,
        key: String,
    ) -> BatchResultStream {
        let events = self.events.clone();
        let pricing = self.pricing.clone();
        Box::pin(stream! {
            let mut inner = inner;
            while let Some(item) = inner.next().await {
                match item {
                    Ok(mut result) => {
                        if let BatchOutcome::Succeeded(resp) = &mut result.outcome {
                            let priced = pricing.price(&resp.model, resp.usage);
                            resp.usage = priced;
                            events.emit(Event::Usage {
                                id: format!("{batch_id}:{}", result.custom_id),
                                key: key.clone(),
                                model: resp.model.clone(),
                                usage: priced,
                            });
                        }
                        yield Ok(result);
                    }
                    // A line-level error is data, not a request failure — pass it through.
                    Err(e) => yield Err(e),
                }
            }
        })
    }

    /// The whole per-request life, written once for every modality (principle 1). Resolve the route,
    /// emit `RequestStarted`, then walk the ordered fallback chain: skip cooled-down targets, skip
    /// unregistered ones, hand the first willing provider to `op`. A connect/transport failure
    /// penalizes that provider (node-local switchover, principle 8) and falls through; an
    /// [`ModelError::Unsupported`] falls through *without* a penalty (the provider is healthy, it just
    /// lacks this modality). On success the penalty is cleared and `RequestRouted` is emitted; the
    /// caller finishes the modality-specific event tail.
    async fn dispatch<T, Op, Fut>(
        &self,
        logical_model: String,
        key: String,
        request_id: String,
        now: u64,
        payload: Option<Value>,
        op: Op,
    ) -> Result<Dispatched<T>, EngineError>
    where
        Op: Fn(Arc<dyn Provider>, ProviderCx, String) -> Fut,
        Fut: Future<Output = Result<T, ModelError>> + Send,
    {
        // One topology snapshot for the whole dispatch: the chain walked, the registry consulted, and
        // the limiter admitted against all agree even if a pulled swap lands mid-request.
        let topo = self.topology();
        let targets = topo
            .router
            .resolve(&logical_model)
            .ok_or_else(|| EngineError::NoRoute(logical_model.clone()))?;
        if targets.is_empty() {
            return Err(EngineError::NoTargets(logical_model));
        }

        self.events.emit(Event::RequestStarted {
            id: request_id.clone(),
            key: key.clone(),
            model: logical_model.clone(),
            request: payload,
        });

        // Walk the chain, now also honoring node-local rate limits. An over-limit target is skipped
        // exactly like a cooled-down one (fall toward availability, principle 8); only when *every*
        // target is over its limit — and nothing actually failed — do we wait, bounded by
        // `rate_max_wait`, for the soonest target to free up, then retry the whole walk. A real
        // connect/transport failure still short-circuits to `AllTargetsFailed` (waiting cannot help it).
        let mut wait_budget = self.rate_max_wait;
        loop {
            // Fresh per walk: a monotonic clock for the rate buckets (advances across each wait), and the
            // soonest moment any rate-limited target could admit.
            let now_instant = Instant::now();
            let mut last_err: Option<ModelError> = None;
            let mut soonest_wait: Option<Duration> = None;

            // Two passes, unchanged in spirit (principle 1: one health read per attempted target). The
            // first honors cooldown; the second runs only when the first attempted nothing — every target
            // cooling down or rate-limited — so a stale cooldown never blacks out the only provider we
            // have. Rate-limited targets count as "attempted nothing", so a fully saturated chain reaches
            // the wait path below rather than the fail-open retry quietly consuming the budget.
            for honor_cooldown in [true, false] {
                for target in targets.iter() {
                    if honor_cooldown && self.health.is_down(&target.provider, now) {
                        continue;
                    }
                    let Some(provider) = topo.registry.get(&target.provider) else {
                        last_err = Some(ModelError::Unavailable(format!(
                            "provider '{}' is not registered",
                            target.provider
                        )));
                        continue;
                    };

                    let upstream_model = target
                        .model
                        .clone()
                        .unwrap_or_else(|| logical_model.clone());

                    // Node-local admission: take a request token + concurrency permit and check the
                    // tokens/min floor (principle 9 — a fast local decision, never a round-trip). Over
                    // limit ⇒ skip this target and remember the soonest moment it could admit, just like a
                    // cooldown skip. The guard holds the concurrency permits until the request's stream ends.
                    let guard =
                        match topo
                            .rate
                            .try_admit(&target.provider, &upstream_model, now_instant)
                        {
                            Ok(g) => g,
                            Err(wait) => {
                                soonest_wait = Some(soonest_wait.map_or(wait, |w| w.min(wait)));
                                continue;
                            }
                        };

                    let cx = topo.build_cx(&target.provider, &request_id);

                    match op(provider, cx, upstream_model.clone()).await {
                        Ok(value) => {
                            self.health.clear(&target.provider);
                            self.events.emit(Event::RequestRouted {
                                id: request_id.clone(),
                                provider: target.provider.clone(),
                                upstream_model: upstream_model.clone(),
                            });
                            return Ok(Dispatched {
                                value,
                                request_id,
                                key,
                                logical_model,
                                provider: target.provider.clone(),
                                upstream_model,
                                rate: topo.rate.clone(),
                                guard,
                            });
                        }
                        // The provider lacks this modality: not a failure, not a health signal. Release the
                        // admission guard and try the next link without penalizing.
                        Err(e @ ModelError::Unsupported(_)) => {
                            drop(guard);
                            last_err = Some(e);
                        }
                        Err(e) => {
                            // Node-local switchover: release the permit, penalize, try the next link.
                            drop(guard);
                            self.health
                                .penalize(&target.provider, now, self.cooldown_secs);
                            self.events.emit(Event::ProviderHealth {
                                provider: target.provider.clone(),
                                status: "degraded".to_string(),
                            });
                            last_err = Some(e);
                        }
                    }
                }
                // The cooldown-honoring pass attempted at least one target — its outcome stands. Only when
                // it attempted nothing do we fall through to the fail-open pass.
                if last_err.is_some() {
                    break;
                }
            }

            // Pure rate-limit saturation: nothing failed, but every target was over its limit. Wait
            // (bounded) for the soonest target to free up, then retry; a single `sleep`, never a spin.
            // The wait is the operator's opted-in latency (principle 1), capped by `rate_max_wait` —
            // exhaust it (or `rate_limit_max_wait_ms = 0`) and the consumer gets a `429`. `soonest_wait`
            // is only ever `Some` when a configured limit threw a target back (an unconfigured limiter
            // returns `Ok` from every `try_admit`), so this branch — and `rate_max_wait`'s non-zero
            // default — is dead weight unless `[providers.limits]`/`model_limits` are set.
            if last_err.is_none() {
                if let Some(wait) = soonest_wait {
                    let nap = wait.min(wait_budget);
                    if nap.is_zero() {
                        let retry_after_secs = wait.as_secs().max(1);
                        self.events.emit(Event::RequestFailed {
                            id: request_id.clone(),
                            error: format!("rate limited; retry after {retry_after_secs}s"),
                        });
                        return Err(EngineError::RateLimited { retry_after_secs });
                    }
                    tokio::time::sleep(nap).await;
                    wait_budget -= nap;
                    continue;
                }
            }

            let err = last_err
                .unwrap_or_else(|| ModelError::Unavailable("no eligible targets".to_string()));
            self.events.emit(Event::RequestFailed {
                id: request_id,
                error: err.to_string(),
            });
            return Err(EngineError::AllTargetsFailed(err));
        }
    }

    /// Serialize a request for the lifecycle event, but only when the operator opted into payloads.
    fn payload<T: serde::Serialize>(&self, req: &T) -> Option<Value> {
        if self.include_payloads {
            serde_json::to_value(req).ok()
        } else {
            None
        }
    }

    /// Emit the closing `Usage` + `RequestCompleted` for a *batch* (non-streaming) modality. Streaming
    /// modalities (chat, speech) emit these from their instrumented stream instead, as bytes pass.
    fn emit_batch_tail(
        &self,
        request_id: &str,
        key: &str,
        model: &str,
        usage: llmleaf_model::Usage,
    ) {
        self.events.emit(Event::Usage {
            id: request_id.to_string(),
            key: key.to_string(),
            model: model.to_string(),
            usage,
        });
        self.events.emit(Event::RequestCompleted {
            id: request_id.to_string(),
            finish: None,
        });
    }

    /// Wrap a provider stream so usage/lifecycle events flow out the bus, and so provider-reported
    /// usage is priced from the bundled dataset (a lookup + multiply — never a fetch). The core does
    /// not count tokens; it relays what the provider emits (principle 5).
    fn instrument(&self, d: Dispatched<ResponseStream>) -> ResponseStream {
        let Dispatched {
            value: inner,
            request_id,
            key,
            logical_model: model,
            provider,
            upstream_model,
            rate,
            guard,
        } = d;
        let events = self.events.clone();
        let pricing = self.pricing.clone();
        Box::pin(stream! {
            // Hold the concurrency permit for the life of the stream; the generator dropping (normal end,
            // error, or client disconnect) releases it.
            let _guard = guard;
            let mut inner = inner;
            let mut finish: Option<FinishReason> = None;
            while let Some(item) = inner.next().await {
                match item {
                    Ok(StreamChunk::Usage(u)) => {
                        let priced = pricing.price(&model, u);
                        // Debit observed tokens against this provider/model's tokens/min bucket (the cost
                        // was unknown at admission).
                        rate.debit_tokens(&provider, &upstream_model, priced.total_tokens, Instant::now());
                        events.emit(Event::Usage {
                            id: request_id.clone(),
                            key: key.clone(),
                            model: model.clone(),
                            usage: priced,
                        });
                        yield Ok(StreamChunk::Usage(priced));
                    }
                    Ok(StreamChunk::Finish { index, reason }) => {
                        finish = Some(reason);
                        yield Ok(StreamChunk::Finish { index, reason });
                    }
                    Ok(other) => yield Ok(other),
                    Err(e) => {
                        events.emit(Event::RequestFailed {
                            id: request_id.clone(),
                            error: e.to_string(),
                        });
                        yield Err(e);
                        return;
                    }
                }
            }
            events.emit(Event::RequestCompleted { id: request_id.clone(), finish });
        })
    }

    /// The audio analogue of [`Self::instrument`]: wrap a provider [`AudioStream`] so usage is priced
    /// and usage/lifecycle events flow as audio bytes pass. The core does not measure audio; it relays
    /// what the provider emits (principle 5).
    fn instrument_audio(&self, d: Dispatched<AudioStream>) -> AudioStream {
        let Dispatched {
            value: inner,
            request_id,
            key,
            logical_model: model,
            provider,
            upstream_model,
            rate,
            guard,
        } = d;
        let events = self.events.clone();
        let pricing = self.pricing.clone();
        Box::pin(stream! {
            let _guard = guard;
            let mut inner = inner;
            while let Some(item) = inner.next().await {
                match item {
                    Ok(AudioChunk::Usage(u)) => {
                        let priced = pricing.price(&model, u);
                        rate.debit_tokens(&provider, &upstream_model, priced.total_tokens, Instant::now());
                        events.emit(Event::Usage {
                            id: request_id.clone(),
                            key: key.clone(),
                            model: model.clone(),
                            usage: priced,
                        });
                        yield Ok(AudioChunk::Usage(priced));
                    }
                    Ok(other) => yield Ok(other),
                    Err(e) => {
                        events.emit(Event::RequestFailed {
                            id: request_id.clone(),
                            error: e.to_string(),
                        });
                        yield Err(e);
                        return;
                    }
                }
            }
            events.emit(Event::RequestCompleted { id: request_id.clone(), finish: None });
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{RateLimitConfig, RouteConfig, ServerConfig, Target};
    use async_trait::async_trait;
    use llmleaf_model::{collect, Message, Role, StreamChunk, Usage};
    use std::sync::atomic::{AtomicBool, Ordering};

    /// A chat provider that fails to connect whenever its `down` flag is set — the analogue of a
    /// transient upstream blip — and optionally advertises a native realtime session. `chat` echoes the
    /// upstream model id back as `resp.model`, so a test can tell *which* target on a chain served.
    struct FlakyProvider {
        name: String,
        down: AtomicBool,
        realtime: bool,
    }

    #[async_trait]
    impl Provider for FlakyProvider {
        fn name(&self) -> &str {
            &self.name
        }
        fn supports_realtime(&self) -> bool {
            self.realtime
        }
        async fn chat(
            &self,
            req: ChatRequest,
            _cx: &ProviderCx,
        ) -> Result<ResponseStream, ModelError> {
            if self.down.load(Ordering::SeqCst) {
                return Err(ModelError::Unavailable("upstream blip".into()));
            }
            let chunks: Vec<Result<StreamChunk, ModelError>> = vec![
                Ok(StreamChunk::Start {
                    id: "x".into(),
                    model: req.model.clone(),
                }),
                Ok(StreamChunk::Finish {
                    index: 0,
                    reason: FinishReason::Stop,
                }),
                Ok(StreamChunk::Usage(Usage {
                    prompt_tokens: 1,
                    completion_tokens: 1,
                    total_tokens: 2,
                    cost_usd: None,
                    cache_read_tokens: 0,
                    cache_creation_tokens: 0,
                })),
            ];
            Ok(Box::pin(futures::stream::iter(chunks)))
        }
    }

    /// Build an engine routing `solo-model` to the given providers, in order (the order *is* the
    /// fallback chain). Each tuple is `(name, supports_realtime, starts_down)`. Every target's upstream
    /// model is set to the provider name, so `resp.model` identifies which target served.
    fn engine_with(providers: &[(&str, bool, bool)]) -> Engine {
        let mut registry = ProviderRegistry::new();
        for &(name, realtime, down) in providers {
            registry.register(
                name,
                Arc::new(FlakyProvider {
                    name: name.into(),
                    down: AtomicBool::new(down),
                    realtime,
                }),
            );
        }
        let config = Config {
            server: ServerConfig {
                fallback_cooldown_secs: 30,
                ..Default::default()
            },
            providers: providers
                .iter()
                .map(|&(name, _, _)| ProviderConfig {
                    name: name.into(),
                    kind: "test".into(),
                    endpoint: None,
                    credential: None,
                    prefix: None,
                    settings: Default::default(),
                    limits: None,
                    model_limits: Default::default(),
                })
                .collect(),
            routes: vec![RouteConfig {
                model: "solo-model".into(),
                targets: providers
                    .iter()
                    .map(|&(name, _, _)| Target {
                        provider: name.into(),
                        model: Some(name.into()),
                    })
                    .collect(),
            }],
            ..Default::default()
        };
        Engine::new(
            &config,
            Arc::new(registry),
            EventBus::new(16),
            Arc::new(Pricing::bundled().expect("bundled prices")),
            None,
        )
    }

    fn chat_req() -> ChatRequest {
        chat_req_for("solo-model")
    }

    fn chat_req_for(model: &str) -> ChatRequest {
        ChatRequest {
            model: model.into(),
            messages: vec![Message::text(Role::User, "ping")],
            max_tokens: None,
            temperature: None,
            top_p: None,
            stop: Vec::new(),
            stream: false,
            tools: Vec::new(),
            tool_choice: None,
            thinking: None,
            extra: Default::default(),
        }
    }

    // The only provider on a route is in cooldown but actually healthy: dispatch must NOT black it out
    // with "no eligible targets" — health is a preference, not a veto (principle 8: fail toward
    // availability). With no fresher target to prefer, the cooled-down one is tried anyway and serves.
    #[tokio::test]
    async fn sole_provider_in_cooldown_is_still_tried() {
        let engine = engine_with(&[("p", false, false)]);
        // Stale, node-local cooldown on the only provider we have.
        engine.health().penalize("p", 100, 30);
        assert!(engine.health().is_down("p", 110));

        let stream = engine
            .run(chat_req(), "k".into(), "r1".into(), 110)
            .await
            .expect("sole cooled-down-but-healthy provider must be tried, not skipped");
        let resp = collect(stream).await.expect("stream completes");
        assert_eq!(resp.model, "p", "the sole target served");
        // A successful call clears the (stale) penalty.
        assert!(!engine.health().is_down("p", 110));
    }

    // A multi-target chain still prefers a fresh target over a cooled-down one (the fail-open pass only
    // engages when *every* target is down), so the existing switchover behavior is unchanged. Both
    // targets are healthy here, so a wrongly-attempted `bad` would succeed and serve — the assertions
    // on `resp.model` and `bad`'s untouched cooldown pin down that it was skipped, not merely retried.
    #[tokio::test]
    async fn cooldown_still_skipped_when_a_fresh_target_remains() {
        let engine = engine_with(&[("bad", false, false), ("good", false, false)]);
        // `bad` is cooling down; `good` is fresh — dispatch must reach `good` without touching `bad`.
        engine.health().penalize("bad", 100, 30);
        let stream = engine
            .run(chat_req(), "k".into(), "r1".into(), 110)
            .await
            .expect("fresh target serves");
        let resp = collect(stream).await.expect("stream completes");
        assert_eq!(
            resp.model, "good",
            "the fresh target served, not the cooled one"
        );
        // `bad` was never attempted, so its cooldown is untouched (not re-penalized, not cleared).
        assert!(engine.health().is_down("bad", 110));
    }

    // Realtime mirror of the dispatch fix: the only native-realtime provider is cooling down but
    // healthy. `realtime_target` must still return it rather than drop the session to the chat bridge
    // over a stale node-local cooldown (principle 8).
    #[tokio::test]
    async fn sole_realtime_provider_in_cooldown_is_still_returned() {
        let engine = engine_with(&[("p", true, false)]);
        engine.health().penalize("p", 100, 30);
        assert!(engine.health().is_down("p", 110));

        let target = engine
            .realtime_target("solo-model", "r1", 110)
            .expect("sole cooled-down native provider must be returned, not dropped to the bridge");
        assert_eq!(target.provider_name, "p");
    }

    // With a fresh native-realtime target available, the cooled-down one is not preferred — the
    // fail-open pass only engages when none is fresh.
    #[tokio::test]
    async fn realtime_prefers_fresh_target_over_cooled() {
        let engine = engine_with(&[("cooled", true, false), ("fresh", true, false)]);
        engine.health().penalize("cooled", 100, 30);

        let target = engine
            .realtime_target("solo-model", "r1", 110)
            .expect("a fresh native target is available");
        assert_eq!(target.provider_name, "fresh");
    }

    // A provider that does not advertise native realtime is never returned — even under the fail-open
    // pass — so the session correctly falls through to the chat bridge (capability gap, not a health
    // signal). The two-pass loop must not mistake "no realtime-capable target" for "all cooled down".
    #[tokio::test]
    async fn realtime_none_when_no_native_capable_target() {
        let engine = engine_with(&[("p", false, false)]);
        assert!(engine.realtime_target("solo-model", "r1", 110).is_none());
    }

    // ---- Rate limiting ----------------------------------------------------------------------------

    fn rl(rpm: Option<u64>, tpm: Option<u64>, conc: Option<u32>) -> RateLimitConfig {
        RateLimitConfig {
            requests_per_min: rpm,
            tokens_per_min: tpm,
            max_concurrent: conc,
        }
    }

    /// Build an engine routing `solo-model` through the given targets in order, each with optional node-
    /// local limits and every provider healthy. `max_wait_ms` is the bounded hot-path wait for capacity.
    /// Each target's upstream model equals its name, so `resp.model` identifies which target served.
    fn rate_limited_engine(
        targets: &[(&str, Option<RateLimitConfig>)],
        max_wait_ms: u64,
    ) -> Engine {
        let mut registry = ProviderRegistry::new();
        for (name, _) in targets {
            registry.register(
                *name,
                Arc::new(FlakyProvider {
                    name: (*name).into(),
                    down: AtomicBool::new(false),
                    realtime: false,
                }),
            );
        }
        let config = Config {
            server: ServerConfig {
                fallback_cooldown_secs: 30,
                rate_limit_max_wait_ms: max_wait_ms,
                ..Default::default()
            },
            providers: targets
                .iter()
                .map(|(name, limits)| ProviderConfig {
                    name: (*name).into(),
                    kind: "test".into(),
                    endpoint: None,
                    credential: None,
                    prefix: None,
                    settings: Default::default(),
                    limits: limits.clone(),
                    model_limits: Default::default(),
                })
                .collect(),
            routes: vec![RouteConfig {
                model: "solo-model".into(),
                targets: targets
                    .iter()
                    .map(|(name, _)| Target {
                        provider: (*name).into(),
                        model: Some((*name).into()),
                    })
                    .collect(),
            }],
            ..Default::default()
        };
        Engine::new(
            &config,
            Arc::new(registry),
            EventBus::new(16),
            Arc::new(Pricing::bundled().expect("bundled prices")),
            None,
        )
    }

    // An over-limit target is skipped exactly like a cooled-down one: a permanently saturated primary
    // (max_concurrent = 0 ⇒ no permit ever) falls through to the next, healthy target (principle 8).
    #[tokio::test]
    async fn rate_limited_target_falls_through_to_next() {
        let engine = rate_limited_engine(
            &[("limited", Some(rl(None, None, Some(0)))), ("good", None)],
            5000,
        );
        let stream = engine
            .run(chat_req(), "k".into(), "r1".into(), 0)
            .await
            .expect("falls through to the healthy target");
        let resp = collect(stream).await.expect("stream completes");
        assert_eq!(resp.model, "good", "the rate-limited primary was skipped");
    }

    // When the *only* target is saturated and the wait budget is zero, the consumer gets a `RateLimited`
    // (429) rather than a hang or a 502 — fall through, then reject.
    #[tokio::test]
    async fn all_saturated_with_zero_wait_yields_rate_limited() {
        let engine = rate_limited_engine(&[("limited", Some(rl(None, None, Some(0))))], 0);
        let err = match engine.run(chat_req(), "k".into(), "r1".into(), 0).await {
            Ok(_) => panic!("a fully saturated chain with no wait budget must reject"),
            Err(e) => e,
        };
        assert!(
            matches!(err, EngineError::RateLimited { retry_after_secs } if retry_after_secs >= 1),
            "got {err:?}"
        );
    }

    // With a positive wait budget the engine waits for token-bucket capacity to refill, then serves —
    // the over-limit behavior the user chose. Paused virtual time auto-advances across the engine's
    // internal `sleep`, so this is deterministic.
    #[tokio::test(start_paused = true)]
    async fn waits_for_capacity_then_serves() {
        // 60 requests/min = 1/sec, burst 60. Drain the burst, then a fresh request must wait ~1s.
        let engine = rate_limited_engine(&[("p", Some(rl(Some(60), None, None)))], 5000);
        for _ in 0..60 {
            engine
                .rate()
                .try_admit("p", "p", Instant::now())
                .expect("drain the burst");
        }
        // The bucket is empty; this request waits for one token to refill (~1s < 5s budget) then serves.
        let stream = engine
            .run(chat_req(), "k".into(), "r1".into(), 0)
            .await
            .expect("waited for capacity, then served");
        let resp = collect(stream).await.expect("stream completes");
        assert_eq!(resp.model, "p");
    }

    // ---- Dynamic topology (the pulled [control.topology] layer) -------------------------------------

    /// A [`ProviderFactory`] that builds a healthy [`FlakyProvider`] for kind `"test"` (counting how
    /// often it was asked) and knows no other kind — the test double for the binary's factory.
    struct TestFactory {
        built: std::sync::atomic::AtomicUsize,
    }

    impl TestFactory {
        fn new() -> Self {
            TestFactory {
                built: std::sync::atomic::AtomicUsize::new(0),
            }
        }
    }

    impl ProviderFactory for TestFactory {
        fn build(&self, kind: &str) -> Option<Arc<dyn Provider>> {
            if kind != "test" {
                return None;
            }
            self.built.fetch_add(1, Ordering::SeqCst);
            Some(Arc::new(FlakyProvider {
                name: "dyn".into(),
                down: AtomicBool::new(false),
                realtime: false,
            }))
        }
    }

    fn dyn_provider(name: &str, kind: &str) -> ProviderConfig {
        ProviderConfig {
            name: name.into(),
            kind: kind.into(),
            endpoint: None,
            credential: None,
            prefix: None,
            settings: Default::default(),
            limits: None,
            model_limits: Default::default(),
        }
    }

    fn dyn_route(model: &str, provider: &str) -> RouteConfig {
        RouteConfig {
            model: model.into(),
            targets: vec![Target {
                provider: provider.into(),
                model: Some(provider.into()),
            }],
        }
    }

    // A pulled provider + route become servable; a later pull without them cleans both up (the route
    // unresolves, the instance is gone, the stale cooldown entry is dropped).
    #[tokio::test]
    async fn install_topology_adds_then_removes_dynamic_resources() {
        let engine = engine_with(&[("p", false, false)]);
        let factory = TestFactory::new();

        let diff = engine.install_topology(
            vec![dyn_provider("extra", "test")],
            vec![dyn_route("extra-model", "extra")],
            &factory,
        );
        assert_eq!(diff.providers_added, 1);
        assert_eq!(diff.routes_added, 1);
        assert!(diff.skipped.is_empty(), "{:?}", diff.skipped);

        let stream = engine
            .run(chat_req_for("extra-model"), "k".into(), "r1".into(), 0)
            .await
            .expect("the pulled route serves through the pulled provider");
        let resp = collect(stream).await.expect("stream completes");
        assert_eq!(resp.model, "extra", "the dynamic target served");

        // Leave a cooldown entry behind, then pull a topology without the provider: everything about it
        // must be cleaned up, while the file base keeps serving untouched.
        engine.health().penalize("extra", 100, 30);
        let diff = engine.install_topology(Vec::new(), Vec::new(), &factory);
        assert_eq!(diff.providers_removed, 1);
        assert_eq!(diff.routes_removed, 1);
        assert!(matches!(
            engine
                .run(chat_req_for("extra-model"), "k".into(), "r2".into(), 0)
                .await,
            Err(EngineError::NoRoute(_))
        ));
        assert!(
            engine.health().snapshot(110).is_empty(),
            "the removed provider's cooldown entry was dropped"
        );
        let stream = engine
            .run(chat_req(), "k".into(), "r3".into(), 0)
            .await
            .expect("the file base still serves");
        collect(stream).await.expect("stream completes");
    }

    // The file base always wins (principle 6): a pulled provider/route colliding with a config-file
    // name is skipped — reported, not installed — and the base behavior is unchanged.
    #[tokio::test]
    async fn install_topology_never_shadows_the_config_base() {
        let engine = engine_with(&[("p", false, false)]);
        let factory = TestFactory::new();

        let diff = engine.install_topology(
            vec![dyn_provider("p", "test")],
            vec![dyn_route("solo-model", "p")],
            &factory,
        );
        assert!(diff.is_noop());
        assert_eq!(diff.skipped.len(), 2, "{:?}", diff.skipped);
        assert_eq!(factory.built.load(Ordering::SeqCst), 0, "nothing was built");

        let stream = engine
            .run(chat_req(), "k".into(), "r1".into(), 0)
            .await
            .expect("base route untouched");
        let resp = collect(stream).await.expect("stream completes");
        assert_eq!(resp.model, "p", "the base target still serves");
    }

    // Re-pulling an identical topology is a no-op: the snapshot is not even swapped, so nothing is
    // rebuilt and no node-local state can churn. An *updated* provider config is applied without
    // rebuilding the instance (only a `kind` change goes back through the factory).
    #[tokio::test]
    async fn install_topology_diffs_updates_and_reuses_instances() {
        let engine = engine_with(&[("p", false, false)]);
        let factory = TestFactory::new();

        let providers = vec![dyn_provider("extra", "test")];
        let routes = vec![dyn_route("extra-model", "extra")];
        engine.install_topology(providers.clone(), routes.clone(), &factory);
        assert_eq!(factory.built.load(Ordering::SeqCst), 1);

        // Identical pull ⇒ no-op, same snapshot object.
        let before = engine.topology();
        let diff = engine.install_topology(providers.clone(), routes.clone(), &factory);
        assert!(diff.is_noop(), "{diff:?}");
        assert!(Arc::ptr_eq(&before, &engine.topology()));
        assert_eq!(factory.built.load(Ordering::SeqCst), 1);

        // A settings-only change is an update that reuses the instance (config flows per-call via
        // ProviderCx, so there is nothing to rebuild).
        let mut updated = providers.clone();
        updated[0]
            .settings
            .insert("organization".into(), serde_json::json!("org-x"));
        let diff = engine.install_topology(updated, routes.clone(), &factory);
        assert_eq!(diff.providers_updated, 1);
        assert_eq!(factory.built.load(Ordering::SeqCst), 1, "instance reused");

        // A kind change rebuilds through the factory.
        let mut rekinded = providers.clone();
        rekinded[0].kind = "unknown-kind".into();
        let diff = engine.install_topology(rekinded, routes.clone(), &factory);
        // The unknown kind cannot be built: the provider is skipped (and, having existed before, it
        // counts as removed) — a bad pull entry never poisons the rest.
        assert_eq!(diff.providers_removed, 1);
        assert_eq!(diff.skipped.len(), 1, "{:?}", diff.skipped);
        let diff = engine.install_topology(providers, routes, &factory);
        assert_eq!(diff.providers_added, 1);
        assert_eq!(
            factory.built.load(Ordering::SeqCst),
            2,
            "rebuilt after re-add"
        );
    }

    // A topology swap must not reset rate-limit state for providers whose limits did not change: the
    // drained bucket stays drained across an unrelated install, and only a limits *change* on the
    // provider itself gets fresh state.
    #[tokio::test(start_paused = true)]
    async fn install_topology_preserves_unchanged_rate_state() {
        let engine = rate_limited_engine(&[("p", Some(rl(Some(1), None, None)))], 0);
        let factory = TestFactory::new();
        let t0 = Instant::now();

        // Drain the base provider's single-token burst.
        engine
            .rate()
            .try_admit("p", "p", t0)
            .expect("the one token");
        assert!(engine.rate().try_admit("p", "p", t0).is_err());

        // A dynamic provider with its own limits, drained too.
        let mut dyn_p = dyn_provider("extra", "test");
        dyn_p.limits = Some(rl(Some(1), None, None));
        engine.install_topology(
            vec![dyn_p.clone()],
            vec![dyn_route("extra-model", "extra")],
            &factory,
        );
        engine
            .rate()
            .try_admit("extra", "extra", t0)
            .expect("the one dynamic token");

        // An unrelated change (a new route) swaps the snapshot — both drained buckets must survive.
        engine.install_topology(
            vec![dyn_p.clone()],
            vec![
                dyn_route("extra-model", "extra"),
                dyn_route("extra-model-2", "extra"),
            ],
            &factory,
        );
        assert!(
            engine.rate().try_admit("p", "p", t0).is_err(),
            "base state kept"
        );
        assert!(
            engine.rate().try_admit("extra", "extra", t0).is_err(),
            "dynamic state kept"
        );

        // Changing the dynamic provider's limits rebuilds *its* state only.
        dyn_p.limits = Some(rl(Some(2), None, None));
        engine.install_topology(
            vec![dyn_p],
            vec![dyn_route("extra-model", "extra")],
            &factory,
        );
        assert!(
            engine.rate().try_admit("extra", "extra", t0).is_ok(),
            "fresh bucket"
        );
        assert!(
            engine.rate().try_admit("p", "p", t0).is_err(),
            "base still kept"
        );
    }

    // A concurrency permit is held for the life of the response stream and released when it ends, so a
    // second request is admitted only after the first stream is consumed.
    #[tokio::test]
    async fn concurrency_permit_released_after_stream_ends() {
        let engine = rate_limited_engine(&[("p", Some(rl(None, None, Some(1))))], 0);
        let s1 = engine
            .run(chat_req(), "k".into(), "r1".into(), 0)
            .await
            .expect("first request takes the only permit");
        // While the first stream is live, the single permit is in use — a fresh admission is refused.
        assert!(
            engine.rate().try_admit("p", "p", Instant::now()).is_err(),
            "the only permit is held by the in-flight stream"
        );
        // Draining the stream to completion drops its guard and releases the permit.
        collect(s1).await.expect("stream completes");
        assert!(
            engine.rate().try_admit("p", "p", Instant::now()).is_ok(),
            "the permit was released when the stream ended"
        );
    }
}
