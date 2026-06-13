//! Simulation of the hot path's hard logic: ordered fallback, node-local health cooldown, and the
//! lifecycle event stream, over *simulated time*.
//!
//! The engine is a pure function of its inputs and the caller-supplied `now` (SOUL.md principle 9:
//! "the core holds no clock for decisions"). Driving [`llmleaf_core::Engine::run`] directly lets the
//! scenario control simulated time exactly — penalties both hold and expire on a schedule the test
//! dictates — which is impossible through the wall-clock-driven HTTP edge.
//!
//! The reference model ([`predict`]) is the spec (SOUL.md principles 8 & 9) written plainly and
//! independently of `engine.rs`/`route.rs`, so a divergence in either is caught.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use llmleaf_core::config::{Config, ProviderConfig, RouteConfig, ServerConfig, Target};
use llmleaf_core::{build_state, EngineError, Event};
use llmleaf_model::{
    ChatRequest, FinishReason, Message, ModelError, ResponseStream, Role, StreamChunk, Usage,
};
use llmleaf_provider::{Provider, ProviderCx, ProviderRegistry};

use crate::ensure;
use crate::rng::Rng;

/// A provider whose success/failure is driven by a shared flag the scenario toggles. When "up" it
/// emits a real, ordered canonical stream (echoing the upstream model it was handed); when "down" it
/// fails to connect — exactly the failure the engine penalizes and falls past (SOUL.md principle 8).
struct SimProvider {
    name: String,
    down: Arc<AtomicBool>,
}

#[async_trait]
impl Provider for SimProvider {
    fn name(&self) -> &str {
        &self.name
    }

    async fn chat(&self, req: ChatRequest, _cx: &ProviderCx) -> Result<ResponseStream, ModelError> {
        if self.down.load(Ordering::SeqCst) {
            return Err(ModelError::Unavailable(format!("{} is down", self.name)));
        }
        // Synthetic upstream model names (`um-*` / a logical id) are intentionally absent from the
        // pricing dataset, so the token counts asserted below are never perturbed by cost lookup.
        let chunks: Vec<Result<StreamChunk, ModelError>> = vec![
            Ok(StreamChunk::Start {
                id: format!("sim-{}", req.model),
                model: req.model.clone(),
            }),
            Ok(StreamChunk::Content {
                index: 0,
                delta: format!("reply {}", req.model),
            }),
            Ok(StreamChunk::Finish {
                index: 0,
                reason: FinishReason::Stop,
            }),
            Ok(StreamChunk::Usage(Usage {
                prompt_tokens: 3,
                completion_tokens: 2,
                total_tokens: 5,
                cost_usd: None,
            })),
        ];
        Ok(Box::pin(futures::stream::iter(chunks)))
    }
}

/// A routing target as generated for a scenario (mirrors [`Target`] but owned for the reference model).
#[derive(Clone)]
pub(crate) struct SimTarget {
    pub(crate) provider: String,
    pub(crate) upstream: Option<String>,
}

/// One generated scenario topology: providers (each with a live fault flag), routes, the set of
/// logical models a request may ask for, and the node's fallback cooldown.
pub(crate) struct Topology {
    /// Registered provider names in a *stable, generation order*. Iterating the `registered`/`faults`
    /// collections would expose HashMap ordering — randomized per process — and break the guarantee
    /// that a seed replays the identical scenario. Any RNG choice over providers uses this vector.
    pub(crate) providers: Vec<String>,
    /// Names of registered providers (the reference model treats any other name as "unregistered").
    pub(crate) registered: HashSet<String>,
    /// Live fault flags shared with the [`SimProvider`]s — the single source of truth for "is down".
    pub(crate) faults: HashMap<String, Arc<AtomicBool>>,
    /// Logical model -> ordered fallback chain.
    pub(crate) routes: HashMap<String, Vec<SimTarget>>,
    /// Models a request may name: routed models, an empty-target route, and an unrouted name.
    pub(crate) requestable: Vec<String>,
    pub(crate) cooldown_secs: u64,
}

const UNROUTED_MODEL: &str = "model-with-no-route";
const EMPTY_ROUTE_MODEL: &str = "model-with-empty-chain";

pub(crate) fn generate_topology(rng: &mut Rng) -> Topology {
    let nproviders = rng.range(2, 4);
    let provider_names: Vec<String> = (0..nproviders).map(|i| format!("p{i}")).collect();
    let registered: HashSet<String> = provider_names.iter().cloned().collect();
    let faults: HashMap<String, Arc<AtomicBool>> = provider_names
        .iter()
        .map(|n| (n.clone(), Arc::new(AtomicBool::new(false))))
        .collect();

    // A "ghost" provider name that routes may reference but which is never registered — exercises the
    // engine's unregistered-target branch (records an error, emits no health event, falls through).
    let use_ghost = rng.chance(1, 3);
    let mut target_pool = provider_names.clone();
    if use_ghost {
        target_pool.push("ghost".to_string());
    }

    let nroutes = rng.range(2, 4);
    let mut routes: HashMap<String, Vec<SimTarget>> = HashMap::new();
    let mut requestable: Vec<String> = Vec::new();
    for r in 0..nroutes {
        let model = format!("m{r}");
        let chain_len = rng.range(1, 4);
        let chain: Vec<SimTarget> = (0..chain_len)
            .map(|i| {
                let provider = rng.pick(&target_pool).clone();
                let upstream = if rng.bool() {
                    Some(format!("um-{model}-{i}"))
                } else {
                    None
                };
                SimTarget { provider, upstream }
            })
            .collect();
        routes.insert(model.clone(), chain);
        requestable.push(model);
    }

    // An empty-target route (NoTargets) and an unrouted model (NoRoute) round out the edge cases.
    routes.insert(EMPTY_ROUTE_MODEL.to_string(), Vec::new());
    requestable.push(EMPTY_ROUTE_MODEL.to_string());
    requestable.push(UNROUTED_MODEL.to_string());

    Topology {
        providers: provider_names,
        registered,
        faults,
        routes,
        requestable,
        cooldown_secs: rng.range(3, 20),
    }
}

fn build_config(topo: &Topology) -> Config {
    let providers: Vec<ProviderConfig> = topo
        .registered
        .iter()
        .map(|name| ProviderConfig {
            name: name.clone(),
            kind: "sim".to_string(),
            endpoint: None,
            credential: None,
            prefix: None,
            settings: Default::default(),
        })
        .collect();

    let routes: Vec<RouteConfig> = topo
        .routes
        .iter()
        .map(|(model, chain)| RouteConfig {
            model: model.clone(),
            targets: chain
                .iter()
                .map(|t| Target {
                    provider: t.provider.clone(),
                    model: t.upstream.clone(),
                })
                .collect(),
        })
        .collect();

    Config {
        server: ServerConfig {
            listen: "127.0.0.1:0".to_string(),
            admin_token: None,
            event_buffer: 8192,
            include_payloads: false,
            fallback_cooldown_secs: topo.cooldown_secs,
        },
        providers,
        routes,
        keys: Vec::new(),
        control: Default::default(),
        oauth: Default::default(),
    }
}

pub(crate) fn build_registry(topo: &Topology) -> ProviderRegistry {
    let mut registry = ProviderRegistry::new();
    for name in &topo.registered {
        let provider = SimProvider {
            name: name.clone(),
            down: topo.faults[name].clone(),
        };
        registry.register(name.clone(), Arc::new(provider));
    }
    registry
}

/// What the reference model predicts a single request resolves to.
#[derive(Debug, PartialEq)]
pub(crate) enum Outcome {
    /// Served by `provider`, upstream model `upstream`.
    Served {
        provider: String,
        upstream: String,
    },
    /// Every eligible target failed or was unregistered.
    AllFailed,
    NoRoute,
    NoTargets,
}

/// Predict the outcome and the exact emitted-event sequence for one request, given the current health
/// state. Returns the prediction plus the health state *after* the request (committed only once the
/// real engine is confirmed to agree).
pub(crate) fn predict(
    topo: &Topology,
    cooldown_until: &HashMap<String, u64>,
    model: &str,
    now: u64,
) -> (Outcome, Vec<String>, HashMap<String, u64>) {
    let Some(chain) = topo.routes.get(model) else {
        return (Outcome::NoRoute, Vec::new(), cooldown_until.clone());
    };
    if chain.is_empty() {
        return (Outcome::NoTargets, Vec::new(), cooldown_until.clone());
    }

    let mut cooldown = cooldown_until.clone();
    let mut events = vec![format!("started:{model}")];

    // Health is a preference, not a veto (engine `dispatch`): a cooled-down target is skipped only
    // while a fresher one remains; if every target is down we still try the chain rather than black
    // out the only provider on a stale cooldown. The engine expresses this as a two-pass loop
    // (cooldown-honoring, then fail-open); this `any_eligible` predicate is its lock-free mirror.
    let any_eligible = chain.iter().any(|target| {
        cooldown
            .get(&target.provider)
            .is_none_or(|&until| now >= until)
    });

    for target in chain {
        // Node-local skip: a comparison against `now`, never arithmetic (route.rs `is_down`).
        if any_eligible
            && cooldown
                .get(&target.provider)
                .is_some_and(|&until| now < until)
        {
            continue;
        }
        // Unregistered provider: the engine records an error and falls through, emitting nothing and
        // touching no health state.
        if !topo.registered.contains(&target.provider) {
            continue;
        }
        if topo.faults[&target.provider].load(Ordering::SeqCst) {
            // Connect failure: penalize (skip until now + cooldown) and emit a health observation.
            cooldown.insert(target.provider.clone(), now + topo.cooldown_secs);
            events.push(format!("health:{}:degraded", target.provider));
            continue;
        }
        // Success: the chosen target clears its penalty and serves.
        cooldown.remove(&target.provider);
        let upstream = target.upstream.clone().unwrap_or_else(|| model.to_string());
        events.push(format!("routed:{}:{}", target.provider, upstream));
        events.push(format!("usage:{model}:3:2:5"));
        events.push("completed:Some(Stop)".to_string());
        return (
            Outcome::Served {
                provider: target.provider.clone(),
                upstream,
            },
            events,
            cooldown,
        );
    }

    events.push("failed".to_string());
    (Outcome::AllFailed, events, cooldown)
}

/// Project an [`Event`] onto the comparable summary the reference model also produces. Volatile fields
/// (request id, key, timestamps) are dropped; the behaviorally-meaningful fields remain.
fn event_tag(event: &Event) -> String {
    match event {
        Event::RequestStarted { model, .. } => format!("started:{model}"),
        Event::RequestRouted {
            provider,
            upstream_model,
            ..
        } => {
            format!("routed:{provider}:{upstream_model}")
        }
        Event::Usage { model, usage, .. } => format!(
            "usage:{model}:{}:{}:{}",
            usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
        ),
        Event::RequestCompleted { finish, .. } => format!("completed:{finish:?}"),
        Event::RequestFailed { .. } => "failed".to_string(),
        Event::ProviderHealth { provider, status } => format!("health:{provider}:{status}"),
    }
}

fn describe_result(result: &Result<ResponseStream, EngineError>) -> String {
    match result {
        Ok(_) => "Ok(stream)".to_string(),
        Err(e) => format!("Err({e})"),
    }
}

/// Run one full engine scenario for `seed`. Returns a seed-stamped failure report on any divergence
/// from the reference model, or `Ok(())` if the engine behaved exactly as specified throughout.
pub async fn run_scenario(seed: u64) -> Result<(), String> {
    let mut rng = Rng::new(seed);
    let topo = generate_topology(&mut rng);
    let config = build_config(&topo);
    let registry = build_registry(&topo);
    let state = build_state(&config, Arc::new(registry))
        .map_err(|e| format!("seed={seed}: build_state failed: {e}"))?;
    let engine = state.engine.clone();
    let mut rx = state.events.subscribe();

    // The reference model's mirror of the engine's node-local health table.
    let mut cooldown_until: HashMap<String, u64> = HashMap::new();
    let mut now = rng.range(1_000, 100_000);

    let nops = rng.range(40, 80);
    for step in 0..nops {
        // Time advances monotonically; deltas straddle the cooldown window so penalties both hold and
        // expire across the timeline.
        now += rng.range(0, topo.cooldown_secs * 2);

        // Churn upstream health: occasionally flip one provider's fault flag (the shared source of
        // truth both the engine and the reference read at call time). Selection is over the stably
        // ordered `providers` vector — never a HashMap iteration — so the seed is reproducible.
        if rng.chance(1, 3) {
            let name = rng.pick(&topo.providers).clone();
            topo.faults[&name].store(rng.bool(), Ordering::SeqCst);
        }

        let model = rng.pick(&topo.requestable).clone();
        let (expected_outcome, expected_events, next_cooldown) =
            predict(&topo, &cooldown_until, &model, now);

        let req = ChatRequest {
            model: model.clone(),
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
        };

        let result = engine
            .run(req, "sim-key".to_string(), format!("r{step}"), now)
            .await;

        // For a successful run the lifecycle's tail events (Usage, Completed) are emitted only as the
        // stream is consumed — collecting it drives them out and also lets us verify the mapped body.
        match (&expected_outcome, result) {
            (Outcome::Served { upstream, .. }, Ok(stream)) => {
                let resp = llmleaf_model::collect(stream)
                    .await
                    .map_err(|e| format!("seed={seed} step={step}: collect failed: {e}"))?;
                ensure!(
                    &resp.model == upstream,
                    "seed={seed} step={step}: served model {:?} != upstream {:?}",
                    resp.model,
                    upstream
                );
                ensure!(
                    resp.choices.len() == 1,
                    "seed={seed} step={step}: expected 1 choice, got {}",
                    resp.choices.len()
                );
                ensure!(
                    resp.choices[0].text == format!("reply {upstream}"),
                    "seed={seed} step={step}: echoed body was {:?}",
                    resp.choices[0].text
                );
                ensure!(
                    resp.choices[0].finish_reason == Some(FinishReason::Stop),
                    "seed={seed} step={step}: finish reason was {:?}",
                    resp.choices[0].finish_reason
                );
                let tokens = (
                    resp.usage.prompt_tokens,
                    resp.usage.completion_tokens,
                    resp.usage.total_tokens,
                );
                ensure!(
                    tokens == (3, 2, 5),
                    "seed={seed} step={step}: relayed token counts were {tokens:?}"
                );
            }
            (Outcome::AllFailed, Err(EngineError::AllTargetsFailed(_))) => {}
            (Outcome::NoRoute, Err(EngineError::NoRoute(m))) => {
                ensure!(
                    m == model,
                    "seed={seed} step={step}: NoRoute model {m:?} != {model:?}"
                );
            }
            (Outcome::NoTargets, Err(EngineError::NoTargets(m))) => {
                ensure!(
                    m == model,
                    "seed={seed} step={step}: NoTargets model {m:?} != {model:?}"
                );
            }
            (expected, actual) => {
                return Err(format!(
                    "seed={seed} step={step}: model={model} now={now}\n  expected outcome: {expected:?}\n  actual result:   {}",
                    describe_result(&actual)
                ));
            }
        }

        // Drain and compare the exact emitted-event sequence.
        let mut actual_events = Vec::new();
        loop {
            match rx.try_recv() {
                Ok(env) => actual_events.push(event_tag(&env.event)),
                Err(tokio::sync::broadcast::error::TryRecvError::Empty) => break,
                Err(tokio::sync::broadcast::error::TryRecvError::Lagged(n)) => {
                    return Err(format!(
                        "seed={seed} step={step}: event buffer lagged by {n} — raise event_buffer"
                    ));
                }
                Err(tokio::sync::broadcast::error::TryRecvError::Closed) => break,
            }
        }
        ensure!(
            actual_events == expected_events,
            "seed={seed} step={step}: model={model} now={now}: event sequence mismatch\n  actual:   {actual_events:?}\n  expected: {expected_events:?}"
        );

        // The engine agreed with the spec this step — commit the predicted health transition.
        cooldown_until = next_cooldown;
    }

    Ok(())
}
