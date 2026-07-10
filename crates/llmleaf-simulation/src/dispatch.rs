//! Simulation of the engine's *generic* dispatch across all four modalities (chat, embeddings, speech,
//! transcription) — the one `Engine::dispatch` skeleton SOUL.md principle 1 demands be "written once
//! for every modality".
//!
//! Where [`crate::engine`] exercises a single modality (chat) deeply, this exercises the modality-
//! agnostic spine: the same ordered fallback + node-local cooldown, plus the rule the four-modality
//! work added — a provider that *lacks* a modality returns [`ModelError::Unsupported`], which the
//! engine falls past **without** a health penalty (it is not a failure; the provider is healthy, it
//! just doesn't offer that modality). A transient fault, by contrast, still penalizes and falls past
//! exactly as for chat (principle 8).
//!
//! As in [`crate::engine`], the reference model ([`predict`]) is the spec written independently of
//! `engine.rs`, so a divergence in either is caught. The guardrail (see this crate's task brief): the
//! engine is already validated — any mismatch here is a bug in *this* reference or [`SimProvider`].

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use llmleaf_core::config::{Config, ProviderConfig, RouteConfig, ServerConfig, Target};
use llmleaf_core::{build_state, EngineError, Event};
use llmleaf_model::{
    AudioChunk, AudioStream, ChatRequest, Embedding, EmbeddingRequest, EmbeddingResponse,
    FinishReason, Message, ModelError, RerankDocument, RerankRequest, RerankResponse, RerankResult,
    ResponseStream, Role, SpeechRequest, StreamChunk, TranscriptionRequest, TranscriptionResponse,
    Usage,
};
use llmleaf_provider::{Provider, ProviderCx, ProviderRegistry};

use crate::ensure;
use crate::rng::Rng;

/// The four modalities the generic dispatch fans out over. Chat is mandatory for every provider; the
/// other three are opt-in (the trait's defaults return [`ModelError::Unsupported`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Modality {
    Chat,
    Embed,
    Speech,
    Transcribe,
    Rerank,
}

/// A stable, generation-ordered list of all modalities. Used wherever the RNG picks a modality so the
/// choice never depends on HashMap/HashSet iteration order (determinism is load-bearing).
const ALL_MODALITIES: [Modality; 5] = [
    Modality::Chat,
    Modality::Embed,
    Modality::Speech,
    Modality::Transcribe,
    Modality::Rerank,
];

/// A provider with a per-modality capability set plus a shared fault flag the scenario toggles. For a
/// given modality the precedence — mirrored exactly by the reference [`predict`] — is:
///   1. fault flag set -> [`ModelError::Unavailable`] (a transient connect failure, penalized);
///   2. else modality not in the capability set -> [`ModelError::Unsupported`] (skipped, no penalty);
///   3. else success, with the modality's canonical payload + provider-reported usage.
///
/// The fault check comes *first*: a down provider connects to nothing, so it cannot report that it
/// lacks a modality — it just fails to connect, exactly like the chat sim's down provider.
struct SimProvider {
    name: String,
    down: Arc<AtomicBool>,
    caps: HashSet<Modality>,
}

impl SimProvider {
    /// The transient failure a down provider returns for any modality (what the engine penalizes).
    fn fault(&self) -> ModelError {
        ModelError::Unavailable(format!("{} is down", self.name))
    }

    /// The "this provider lacks the modality" signal the engine falls past without a penalty.
    fn unsupported(&self, what: &str) -> ModelError {
        ModelError::Unsupported(format!("{} does not support {what}", self.name))
    }
}

#[async_trait]
impl Provider for SimProvider {
    fn name(&self) -> &str {
        &self.name
    }

    async fn chat(&self, req: ChatRequest, _cx: &ProviderCx) -> Result<ResponseStream, ModelError> {
        if self.down.load(Ordering::SeqCst) {
            return Err(self.fault());
        }
        if !self.caps.contains(&Modality::Chat) {
            // Chat is always granted in this scenario, so this branch is unreachable; kept for symmetry
            // and so a future capability change can't silently serve an "unsupported" chat.
            return Err(self.unsupported("chat"));
        }
        // Synthetic upstream model names (`um-*`) are intentionally absent from the pricing dataset, so
        // the token counts asserted below are never perturbed by cost lookup.
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
                cache_read_tokens: 0,
                cache_creation_tokens: 0,
            })),
        ];
        Ok(Box::pin(futures::stream::iter(chunks)))
    }

    async fn embed(
        &self,
        req: EmbeddingRequest,
        _cx: &ProviderCx,
    ) -> Result<EmbeddingResponse, ModelError> {
        if self.down.load(Ordering::SeqCst) {
            return Err(self.fault());
        }
        if !self.caps.contains(&Modality::Embed) {
            return Err(self.unsupported("embeddings"));
        }
        Ok(EmbeddingResponse {
            model: req.model,
            embeddings: vec![Embedding {
                index: 0,
                vector: vec![0.0],
            }],
            usage: Usage {
                prompt_tokens: 7,
                total_tokens: 7,
                ..Default::default()
            },
        })
    }

    async fn speech(
        &self,
        _req: SpeechRequest,
        _cx: &ProviderCx,
    ) -> Result<AudioStream, ModelError> {
        if self.down.load(Ordering::SeqCst) {
            return Err(self.fault());
        }
        if !self.caps.contains(&Modality::Speech) {
            return Err(self.unsupported("speech synthesis"));
        }
        let chunks: Vec<Result<AudioChunk, ModelError>> = vec![
            Ok(AudioChunk::Start {
                content_type: "audio/mpeg".to_string(),
            }),
            Ok(AudioChunk::Data(Bytes::from_static(b"x"))),
            Ok(AudioChunk::Usage(Usage {
                prompt_tokens: 2,
                total_tokens: 2,
                ..Default::default()
            })),
            Ok(AudioChunk::Finish),
        ];
        Ok(Box::pin(futures::stream::iter(chunks)))
    }

    async fn transcribe(
        &self,
        req: TranscriptionRequest,
        _cx: &ProviderCx,
    ) -> Result<TranscriptionResponse, ModelError> {
        if self.down.load(Ordering::SeqCst) {
            return Err(self.fault());
        }
        if !self.caps.contains(&Modality::Transcribe) {
            return Err(self.unsupported("transcription"));
        }
        Ok(TranscriptionResponse {
            text: format!("t {}", req.model),
            language: None,
            duration: None,
            usage: Usage {
                prompt_tokens: 6,
                total_tokens: 6,
                ..Default::default()
            },
        })
    }

    async fn rerank(
        &self,
        req: RerankRequest,
        _cx: &ProviderCx,
    ) -> Result<RerankResponse, ModelError> {
        if self.down.load(Ordering::SeqCst) {
            return Err(self.fault());
        }
        if !self.caps.contains(&Modality::Rerank) {
            return Err(self.unsupported("rerank"));
        }
        Ok(RerankResponse {
            model: req.model,
            results: vec![RerankResult {
                index: 0,
                relevance_score: 1.0,
                document: None,
            }],
            usage: Usage {
                prompt_tokens: 7,
                total_tokens: 7,
                ..Default::default()
            },
        })
    }
}

/// A routing target as generated for a scenario (mirrors [`Target`] but owned for the reference model).
#[derive(Clone)]
struct SimTarget {
    provider: String,
    upstream: Option<String>,
}

/// One generated scenario topology: providers (each with a capability set and a live fault flag),
/// routes, the requestable logical models, and the node's fallback cooldown.
struct Topology {
    /// Registered provider names in a *stable, generation order*. Iterating `registered`/`faults`/
    /// `caps` would expose HashMap/HashSet ordering — randomized per process — and break the guarantee
    /// that a seed replays the identical scenario. Any RNG choice over providers uses this vector.
    providers: Vec<String>,
    /// Names of registered providers (the reference model treats any other name as "unregistered").
    registered: HashSet<String>,
    /// Live fault flags shared with the [`SimProvider`]s — the single source of truth for "is down".
    faults: HashMap<String, Arc<AtomicBool>>,
    /// Per-provider capability set, shared (by value clone) with each [`SimProvider`]. Capabilities are
    /// fixed for the scenario: a provider either offers a modality or it never does.
    caps: HashMap<String, HashSet<Modality>>,
    /// Logical model -> ordered fallback chain.
    routes: HashMap<String, Vec<SimTarget>>,
    /// Models a request may name: routed models, an empty-target route, and an unrouted name.
    requestable: Vec<String>,
    cooldown_secs: u64,
}

const UNROUTED_MODEL: &str = "model-with-no-route";
const EMPTY_ROUTE_MODEL: &str = "model-with-empty-chain";

fn generate_topology(rng: &mut Rng) -> Topology {
    let nproviders = rng.range(2, 4);
    let provider_names: Vec<String> = (0..nproviders).map(|i| format!("p{i}")).collect();
    let registered: HashSet<String> = provider_names.iter().cloned().collect();
    let faults: HashMap<String, Arc<AtomicBool>> = provider_names
        .iter()
        .map(|n| (n.clone(), Arc::new(AtomicBool::new(false))))
        .collect();

    // Each provider's capability set: chat is always granted; the other three modalities are independent
    // coin flips. This is what makes the chain land on the next *capable* provider per modality.
    let caps: HashMap<String, HashSet<Modality>> = provider_names
        .iter()
        .map(|name| {
            let mut set = HashSet::new();
            set.insert(Modality::Chat);
            if rng.bool() {
                set.insert(Modality::Embed);
            }
            if rng.bool() {
                set.insert(Modality::Speech);
            }
            if rng.bool() {
                set.insert(Modality::Transcribe);
            }
            if rng.bool() {
                set.insert(Modality::Rerank);
            }
            (name.clone(), set)
        })
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
        caps,
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
            limits: None,
            model_limits: Default::default(),
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
            ..Default::default()
        },
        providers,
        routes,
        keys: Vec::new(),
        control: Default::default(),
        oauth: Default::default(),
    }
}

fn build_registry(topo: &Topology) -> ProviderRegistry {
    let mut registry = ProviderRegistry::new();
    for name in &topo.registered {
        let provider = SimProvider {
            name: name.clone(),
            down: topo.faults[name].clone(),
            caps: topo.caps[name].clone(),
        };
        registry.register(name.clone(), Arc::new(provider));
    }
    registry
}

/// What the reference model predicts a single request resolves to.
#[derive(Debug, PartialEq)]
enum Outcome {
    /// Served by `provider`, upstream model `upstream`.
    Served {
        provider: String,
        upstream: String,
    },
    /// Every eligible target failed, was unregistered, or lacked the modality.
    AllFailed,
    NoRoute,
    NoTargets,
}

/// The per-modality usage the [`SimProvider`] reports and the `finish` tag the lifecycle ends on.
/// Centralized so the reference's `usage:`/`completed:` predictions and the SimProvider's payloads can
/// never drift apart.
fn modality_profile(modality: Modality) -> ((u64, u64, u64), &'static str) {
    match modality {
        // (prompt, completion, total), finish-debug
        Modality::Chat => ((3, 2, 5), "Some(Stop)"),
        Modality::Embed => ((7, 0, 7), "None"),
        Modality::Speech => ((2, 0, 2), "None"),
        Modality::Transcribe => ((6, 0, 6), "None"),
        // Rerank is a batch modality with no completion side (like Embed): prompt == total, no stream.
        Modality::Rerank => ((7, 0, 7), "None"),
    }
}

/// Predict the outcome and the exact emitted-event sequence for one request of `modality`, given the
/// current health state. Returns the prediction plus the health state *after* the request (committed
/// only once the real engine is confirmed to agree). Walks the route chain exactly like the engine.
fn predict(
    topo: &Topology,
    cooldown_until: &HashMap<String, u64>,
    model: &str,
    modality: Modality,
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
    let ((p, c, t), finish) = modality_profile(modality);

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
        // Fault wins over capability: a down provider fails to *connect*, so it returns Unavailable for
        // any modality (the SimProvider checks `down` first). Penalize + emit a health observation.
        if topo.faults[&target.provider].load(Ordering::SeqCst) {
            cooldown.insert(target.provider.clone(), now + topo.cooldown_secs);
            events.push(format!("health:{}:degraded", target.provider));
            continue;
        }
        // The new rule: a healthy provider that lacks this modality returns Unsupported. The engine
        // falls past it with NO penalty and NO health event — the chain keeps looking for a capable one.
        if !topo.caps[&target.provider].contains(&modality) {
            continue;
        }
        // Success: the chosen target clears its penalty and serves.
        cooldown.remove(&target.provider);
        let upstream = target.upstream.clone().unwrap_or_else(|| model.to_string());
        events.push(format!("routed:{}:{}", target.provider, upstream));
        events.push(format!("usage:{model}:{p}:{c}:{t}"));
        events.push(format!("completed:{finish}"));
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
/// (request id, key, timestamps) are dropped; the behaviorally-meaningful fields remain. Identical in
/// shape to [`crate::engine::event_tag`] — the lifecycle event stream is modality-agnostic.
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

/// Drive the engine for one operation of `modality` against `model`, returning the prediction-comparable
/// outcome. Consuming streaming modalities (chat, speech) is what drives their tail events out the bus —
/// the same reason [`crate::engine`] collects the chat stream.
async fn drive(
    engine: &llmleaf_core::Engine,
    modality: Modality,
    model: &str,
    request_id: String,
    now: u64,
) -> Result<Outcome, String> {
    let key = "sim-key".to_string();
    match modality {
        Modality::Chat => {
            let req = ChatRequest {
                model: model.to_string(),
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
            match engine.run(req, key, request_id, now).await {
                Ok(stream) => {
                    let resp = llmleaf_model::collect(stream)
                        .await
                        .map_err(|e| format!("chat collect failed: {e}"))?;
                    // `resp.model` is the upstream id the provider echoed back.
                    Ok(Outcome::Served {
                        provider: String::new(),
                        upstream: resp.model,
                    })
                }
                Err(e) => Ok(engine_err_outcome(e, model)),
            }
        }
        Modality::Embed => {
            let req = EmbeddingRequest {
                model: model.to_string(),
                input: vec!["ping".to_string()],
                dimensions: None,
                encoding_format: None,
                extra: Default::default(),
            };
            match engine.embed(req, key, request_id, now).await {
                Ok(resp) => Ok(Outcome::Served {
                    provider: String::new(),
                    upstream: resp.model,
                }),
                Err(e) => Ok(engine_err_outcome(e, model)),
            }
        }
        Modality::Speech => {
            let req = SpeechRequest {
                model: model.to_string(),
                input: "ping".to_string(),
                voice: "alloy".to_string(),
                response_format: None,
                speed: None,
                extra: Default::default(),
            };
            match engine.speech(req, key, request_id, now).await {
                Ok(stream) => {
                    // Collecting drives the tail (Usage, Completed) out the bus, like the chat stream.
                    let audio = llmleaf_model::collect_audio(stream)
                        .await
                        .map_err(|e| format!("speech collect failed: {e}"))?;
                    // Speech carries no upstream model id; the request's upstream is unobservable from
                    // the collected audio, so the outcome only records that *some* target served.
                    let _ = audio;
                    Ok(Outcome::Served {
                        provider: String::new(),
                        upstream: String::new(),
                    })
                }
                Err(e) => Ok(engine_err_outcome(e, model)),
            }
        }
        Modality::Transcribe => {
            let req = TranscriptionRequest {
                model: model.to_string(),
                audio: Bytes::from_static(b"audio"),
                filename: "a.wav".to_string(),
                language: None,
                prompt: None,
                response_format: None,
                temperature: None,
                extra: Default::default(),
            };
            match engine.transcribe(req, key, request_id, now).await {
                Ok(resp) => {
                    // `resp.text` is `"t {upstream}"`; recover the upstream id the provider was handed.
                    let upstream = resp
                        .text
                        .strip_prefix("t ")
                        .unwrap_or(&resp.text)
                        .to_string();
                    Ok(Outcome::Served {
                        provider: String::new(),
                        upstream,
                    })
                }
                Err(e) => Ok(engine_err_outcome(e, model)),
            }
        }
        Modality::Rerank => {
            let req = RerankRequest {
                model: model.to_string(),
                query: "ping".to_string(),
                documents: vec![RerankDocument::Text("doc".to_string())],
                top_n: None,
                return_documents: None,
                extra: Default::default(),
            };
            match engine.rerank(req, key, request_id, now).await {
                Ok(resp) => Ok(Outcome::Served {
                    provider: String::new(),
                    upstream: resp.model,
                }),
                Err(e) => Ok(engine_err_outcome(e, model)),
            }
        }
    }
}

/// Map an [`EngineError`] onto the reference [`Outcome`] variants (shared across all four modalities —
/// dispatch returns the same errors regardless of modality).
fn engine_err_outcome(err: EngineError, _model: &str) -> Outcome {
    match err {
        EngineError::AllTargetsFailed(_) => Outcome::AllFailed,
        EngineError::NoRoute(_) => Outcome::NoRoute,
        EngineError::NoTargets(_) => Outcome::NoTargets,
        // The simulation builds state without an interceptor, so the hot path never blocks.
        EngineError::Blocked(reason) => {
            unreachable!("no interceptor configured, got Blocked: {reason}")
        }
        // The dispatch topology configures no provider rate limits, so admission never throttles.
        EngineError::RateLimited { retry_after_secs } => {
            unreachable!(
                "no rate limits configured, got RateLimited: retry after {retry_after_secs}s"
            )
        }
        // These dispatch scenarios cover chat/embed/speech/voices/transcribe; they never submit a
        // batch, so the batch-only errors cannot arise here.
        EngineError::EmptyBatch
        | EngineError::MixedProviderBatch
        | EngineError::BatchNotFound(_) => {
            unreachable!("dispatch scenarios do not submit batches")
        }
    }
}

/// Run one full multi-modality dispatch scenario for `seed`. Returns a seed-stamped failure report on
/// any divergence from the reference model, or `Ok(())` if the engine behaved exactly as specified.
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

        // Churn upstream health: occasionally flip one provider's fault flag (the shared source of truth
        // both the engine and the reference read at call time). Selection is over the stably ordered
        // `providers` vector — never a HashMap iteration — so the seed is reproducible.
        if rng.chance(1, 3) {
            let name = rng.pick(&topo.providers).clone();
            topo.faults[&name].store(rng.bool(), Ordering::SeqCst);
        }

        // Pick a modality, then a requestable model. Both picks iterate stable Vecs.
        let modality = *rng.pick(&ALL_MODALITIES);
        let model = rng.pick(&topo.requestable).clone();
        let (expected_outcome, expected_events, next_cooldown) =
            predict(&topo, &cooldown_until, &model, modality, now);

        let actual_outcome = drive(&engine, modality, &model, format!("r{step}"), now)
            .await
            .map_err(|e| {
                format!("seed={seed} step={step}: model={model} modality={modality:?}: {e}")
            })?;

        // Compare outcome variants. For a served request the engine rewrites the provider request's
        // model to the upstream id, so the body the provider echoes back must equal the expected
        // upstream (where the modality carries one — speech does not).
        match (&expected_outcome, &actual_outcome) {
            (
                Outcome::Served {
                    upstream: exp_upstream,
                    ..
                },
                Outcome::Served {
                    upstream: got_upstream,
                    ..
                },
            ) => {
                let body_carries_upstream = matches!(
                    modality,
                    Modality::Chat | Modality::Embed | Modality::Transcribe | Modality::Rerank
                );
                if body_carries_upstream {
                    ensure!(
                        got_upstream == exp_upstream,
                        "seed={seed} step={step}: model={model} modality={modality:?}: served upstream {got_upstream:?} != expected {exp_upstream:?}"
                    );
                }
            }
            (Outcome::AllFailed, Outcome::AllFailed)
            | (Outcome::NoRoute, Outcome::NoRoute)
            | (Outcome::NoTargets, Outcome::NoTargets) => {}
            (expected, actual) => {
                return Err(format!(
                    "seed={seed} step={step}: model={model} modality={modality:?} now={now}\n  expected outcome: {expected:?}\n  actual outcome:   {actual:?}"
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
            "seed={seed} step={step}: model={model} modality={modality:?} now={now}: event sequence mismatch\n  actual:   {actual_events:?}\n  expected: {expected_events:?}"
        );

        // The engine agreed with the spec this step — commit the predicted health transition.
        cooldown_until = next_cooldown;
    }

    Ok(())
}
