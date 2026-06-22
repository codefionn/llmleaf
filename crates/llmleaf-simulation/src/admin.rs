//! Simulation of the read-only admin / observability surface: the token guard and the routes / health
//! / keys views (SOUL.md "Architecture soul": observability only, node-local, no mutation path).
//!
//! The surface used to live inside the axum handlers, reachable only through a real listener — so the
//! seeded harness could never reach it. With the logic lifted onto the transport-agnostic
//! [`llmleaf_core::Observability`] trait, this family drives it **in-process, with no networking**, the
//! same way the other families drive [`llmleaf_core::Engine`] directly. `health(now)` takes the
//! caller's `now`, so the cooldown timeline is checked over *simulated time* — impossible through the
//! wall-clock-driven HTTP edge.
//!
//! Each view is cross-checked against an independent reference model. The health reference is the engine
//! family's own [`crate::engine::predict`]/cooldown table (reused, not re-derived, so it cannot drift);
//! the others are pure projections of the state this family installs. Every assertion that touches a
//! core-side `HashMap`/`HashSet` (every view does) compares as a set/map, never an ordered `Vec` — the
//! core builds these views by iterating maps, whose order is randomized per process.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::atomic::Ordering;
use std::sync::Arc;

use llmleaf_core::config::{
    Config, KeyConfig, ProviderConfig, RouteConfig, Secret, ServerConfig, Target,
};
use llmleaf_core::{build_state, AdminAccess, HealthView, KeyView, Observability, Verdict};
use llmleaf_model::{ChatRequest, Message, Role};

use crate::engine::{self, Topology};
use crate::ensure;
use crate::rng::Rng;

/// The reference mirror of a single key's installed state (config base + verdict overlay), as the
/// secret-free [`KeyView`] should reflect it. `None` means "no restriction at that layer".
#[derive(Clone, Default)]
struct RefKey {
    name: Option<String>,
    config_allowed: Option<HashSet<String>>,
    verdict: RefVerdict,
}

#[derive(Clone, Default)]
struct RefVerdict {
    blocked: bool,
    suspended_until: Option<u64>,
    allowed_models: Option<HashSet<String>>,
}

/// An order-independent normal form for one key, so the roster can be compared as a map regardless of
/// the `HashSet`→`Vec` round-trip and HashMap iteration order the snapshot introduces.
#[derive(Debug, PartialEq, Eq)]
struct NormKey {
    id: String,
    config_allowed: Option<BTreeSet<String>>,
    blocked: bool,
    suspended_until: Option<u64>,
    allowed_models: Option<BTreeSet<String>>,
}

fn norm_view(kv: &KeyView) -> NormKey {
    NormKey {
        id: kv.id.clone(),
        config_allowed: kv
            .config_allowed
            .as_ref()
            .map(|v| v.iter().cloned().collect()),
        blocked: kv.verdict.blocked,
        suspended_until: kv.verdict.suspended_until,
        allowed_models: kv
            .verdict
            .allowed_models
            .as_ref()
            .map(|s| s.iter().cloned().collect()),
    }
}

fn norm_ref(key_id: &str, rk: &RefKey) -> NormKey {
    NormKey {
        id: rk.name.clone().unwrap_or_else(|| key_id.to_string()),
        config_allowed: rk
            .config_allowed
            .as_ref()
            .map(|s| s.iter().cloned().collect()),
        blocked: rk.verdict.blocked,
        suspended_until: rk.verdict.suspended_until,
        allowed_models: rk
            .verdict
            .allowed_models
            .as_ref()
            .map(|s| s.iter().cloned().collect()),
    }
}

/// A possibly-empty random subset of `universe` (a key may be allowed no models — a valid, if useless,
/// restriction). Only the seeded [`Rng`] drives membership, so it replays identically.
fn random_subset(rng: &mut Rng, universe: &[String]) -> HashSet<String> {
    universe.iter().filter(|_| rng.bool()).cloned().collect()
}

/// The `/admin/health` view as a comparable map: provider → "down at `now`".
fn health_map(hv: &HealthView) -> HashMap<String, bool> {
    hv.providers
        .iter()
        .map(|p| (p.provider.clone(), p.down))
        .collect()
}

/// The reference health projection at `now`: a provider appears iff it has a cooldown entry (penalized
/// at least once and not cleared by a later success), and is `down` while `now < until`.
fn reference_health(cooldown: &HashMap<String, u64>, now: u64) -> HashMap<String, bool> {
    cooldown
        .iter()
        .map(|(p, &until)| (p.clone(), now < until))
        .collect()
}

/// The minimal chat request used to drive the engine so health side effects (penalize / clear) occur.
fn chat_request(model: &str) -> ChatRequest {
    ChatRequest {
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
    }
}

/// Build a config from a topology, attaching prefix namespaces, the admin token, and base keys. Unlike
/// the engine family's `build_config`, this is the variant that exercises the admin surface — so it sets
/// what that family hard-codes off (`admin_token`, `[[keys]]`, per-provider `prefix`). Providers are
/// iterated over the stable `topo.providers` vector, never the `registered` HashSet, so config order is
/// reproducible.
fn build_config(
    topo: &Topology,
    prefixes: &HashMap<String, String>,
    admin_token: Option<&str>,
    keys: Vec<KeyConfig>,
) -> Config {
    let providers: Vec<ProviderConfig> = topo
        .providers
        .iter()
        .map(|name| ProviderConfig {
            name: name.clone(),
            kind: "sim".to_string(),
            endpoint: None,
            credential: None,
            prefix: prefixes.get(name).cloned(),
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
            admin_token: admin_token.map(|t| Secret::new(t.to_string())),
            event_buffer: 8192,
            include_payloads: false,
            fallback_cooldown_secs: topo.cooldown_secs,
            ..Default::default()
        },
        providers,
        routes,
        keys,
        control: Default::default(),
        oauth: Default::default(),
    }
}

/// Assert the `/admin/keys` view matches the reference roster (compared as an order-independent map).
fn check_keys_view(
    keys_view: &[KeyView],
    reference: &HashMap<String, RefKey>,
    seed: u64,
    ctx: &str,
) -> Result<(), String> {
    let actual: HashMap<String, NormKey> = keys_view
        .iter()
        .map(|kv| (kv.key_id.clone(), norm_view(kv)))
        .collect();
    let expected: HashMap<String, NormKey> = reference
        .iter()
        .map(|(id, rk)| (id.clone(), norm_ref(id, rk)))
        .collect();
    ensure!(
        actual == expected,
        "seed={seed}: keys view mismatch ({ctx})\n  actual:   {actual:?}\n  expected: {expected:?}"
    );
    Ok(())
}

/// Run one full admin-surface scenario for `seed`: token guard, routes view, keys view, health view —
/// all driven through the [`Observability`] trait in-process, each checked against a reference model.
pub async fn run_scenario(seed: u64) -> Result<(), String> {
    let mut rng = Rng::new(seed);
    let topo = engine::generate_topology(&mut rng);

    // Prefix namespaces: assign some providers a `<prefix>` so `routes()` exposes a non-trivial prefix
    // table. Distinct lengths (`x`, `xx`, …) let the longest-first contract be checked for real. Iterate
    // the stable `providers` vector so the assignment is reproducible.
    let mut prefix_pairs: Vec<(String, String)> = Vec::new();
    let mut prefix_of: HashMap<String, String> = HashMap::new();
    for (i, provider) in topo.providers.iter().enumerate() {
        if rng.chance(1, 2) {
            let prefix = "x".repeat(i + 1);
            prefix_pairs.push((prefix.clone(), provider.clone()));
            prefix_of.insert(provider.clone(), prefix);
        }
    }

    // Admin token: absent ~1/3 of the time (the disabled surface), else a generated token.
    let admin_token: Option<String> = if rng.chance(1, 3) {
        None
    } else {
        Some(format!("adm-{}", rng.next_u64()))
    };

    // Base keys ("config"): id, an optional friendly name, and an optional config allow-list. The
    // password is irrelevant to the secret-free view — a constant placeholder hash avoids the KDF (the
    // view never verifies it; key authorization is covered by the `keys` family).
    let key_models: Vec<String> = (0..rng.range(2, 4)).map(|i| format!("km{i}")).collect();
    let known_ids: Vec<String> = (0..rng.range(1, 4)).map(|i| format!("key-{i}")).collect();
    let mut config_keys: Vec<KeyConfig> = Vec::new();
    let mut reference: HashMap<String, RefKey> = HashMap::new();
    for id in &known_ids {
        let config_allowed = if rng.chance(1, 2) {
            Some(random_subset(&mut rng, &key_models))
        } else {
            None
        };
        let name = if rng.chance(2, 3) {
            Some(format!("name-of-{id}"))
        } else {
            None
        };
        config_keys.push(KeyConfig {
            id: id.clone(),
            pw_hash: Secret::new("$1$unused$placeholderhashunusedxxxxxx".to_string()),
            name: name.clone(),
            allowed_models: config_allowed.as_ref().map(|s| s.iter().cloned().collect()),
        });
        reference.insert(
            id.clone(),
            RefKey {
                name,
                config_allowed,
                verdict: RefVerdict::default(),
            },
        );
    }

    let config = build_config(&topo, &prefix_of, admin_token.as_deref(), config_keys);
    let registry = engine::build_registry(&topo);
    let state = build_state(&config, Arc::new(registry))
        .map_err(|e| format!("seed={seed}: build_state failed: {e}"))?;

    // --- 1) Token guard + admin-view bool (pure; no timeline). -------------------------------------
    let configured = admin_token.as_deref();
    for _ in 0..rng.range(8, 20) {
        let presented: Option<String> = match rng.below(3) {
            0 => admin_token.clone(), // correct (or None)
            1 => Some(format!("{}x", admin_token.clone().unwrap_or_default())), // derived wrong token
            _ => None,                                                          // absent
        };
        let expected = match (configured, presented.as_deref()) {
            (None, _) => AdminAccess::Disabled,
            (Some(_), None) => AdminAccess::Unauthorized,
            (Some(exp), Some(got)) if got == exp => AdminAccess::Authorized,
            (Some(_), Some(_)) => AdminAccess::Unauthorized,
        };
        let actual = state.admin_access(presented.as_deref());
        ensure!(
            actual == expected,
            "seed={seed}: admin_access(configured={configured:?}, presented={presented:?})\n  actual:   {actual:?}\n  expected: {expected:?}"
        );
        let view_expected = expected == AdminAccess::Authorized;
        ensure!(
            state.admin_view(presented.as_deref()) == view_expected,
            "seed={seed}: admin_view(presented={presented:?}) != {view_expected}"
        );
    }

    // --- 2) Routes view: models (set) + prefixes (set) + longest-first contract. --------------------
    let routes = state.routes();
    let actual_models: BTreeSet<&str> = routes.models.iter().map(String::as_str).collect();
    let expected_models: BTreeSet<&str> = topo.routes.keys().map(String::as_str).collect();
    ensure!(
        actual_models == expected_models,
        "seed={seed}: routes.models mismatch\n  actual:   {actual_models:?}\n  expected: {expected_models:?}"
    );
    let actual_prefixes: BTreeSet<(String, String)> = routes
        .prefixes
        .iter()
        .map(|p| (p.prefix.clone(), p.provider.clone()))
        .collect();
    let expected_prefixes: BTreeSet<(String, String)> = prefix_pairs.iter().cloned().collect();
    ensure!(
        actual_prefixes == expected_prefixes,
        "seed={seed}: routes.prefixes mismatch\n  actual:   {actual_prefixes:?}\n  expected: {expected_prefixes:?}"
    );
    let lens: Vec<usize> = routes.prefixes.iter().map(|p| p.prefix.len()).collect();
    ensure!(
        lens.windows(2).all(|w| w[0] >= w[1]),
        "seed={seed}: prefixes not longest-first: {lens:?}"
    );

    // --- 3) Keys view: config base, then an optional verdict overlay (the limits pull). -------------
    check_keys_view(&state.keys().keys, &reference, seed, "config base")?;
    if rng.chance(1, 2) {
        // A verdict install replaces EVERY key's verdict at once; a key absent from the map gets the
        // default (no restriction) — exactly the snapshot-swap the limits refresher performs.
        let mut verdicts: HashMap<String, Verdict> = HashMap::new();
        let mut ref_verdicts: HashMap<String, RefVerdict> = HashMap::new();
        for id in &known_ids {
            if rng.bool() {
                continue; // this key carries no verdict this round
            }
            let blocked = rng.chance(1, 4);
            let suspended_until = if rng.chance(1, 3) {
                Some(rng.range(1, 1_000_000))
            } else {
                None
            };
            let allowed_models = if rng.chance(1, 3) {
                Some(random_subset(&mut rng, &key_models))
            } else {
                None
            };
            verdicts.insert(
                id.clone(),
                Verdict {
                    blocked,
                    suspended_until,
                    allowed_models: allowed_models.clone(),
                },
            );
            ref_verdicts.insert(
                id.clone(),
                RefVerdict {
                    blocked,
                    suspended_until,
                    allowed_models,
                },
            );
        }
        state.keys.install_verdicts(verdicts);
        for (id, rk) in reference.iter_mut() {
            rk.verdict = ref_verdicts.remove(id).unwrap_or_default();
        }
        check_keys_view(
            &state.keys().keys,
            &reference,
            seed,
            "after verdict install",
        )?;
    }

    // --- 4) Health view: drive the engine over simulated time, checking the projection each step. ----
    let engine = state.engine.clone();
    let mut cooldown_until: HashMap<String, u64> = HashMap::new();
    let mut now = rng.range(1_000, 100_000);
    ensure!(
        health_map(&state.health(now)) == reference_health(&cooldown_until, now),
        "seed={seed}: initial health view should be empty"
    );

    let nops = rng.range(40, 80);
    for step in 0..nops {
        // Time advances monotonically, deltas straddling the cooldown window so penalties both hold and
        // expire across the timeline.
        now += rng.range(0, topo.cooldown_secs * 2);

        // Churn upstream health (the shared fault flag both the engine and the reference read at call
        // time). Selection is over the stable `providers` vector, never a HashMap iteration.
        if rng.chance(1, 3) {
            let name = rng.pick(&topo.providers).clone();
            topo.faults[&name].store(rng.bool(), Ordering::SeqCst);
        }

        let model = rng.pick(&topo.requestable).clone();
        // Reuse the validated engine reference for the health transition (lock-step, never re-derived).
        let (_, _, next) = engine::predict(&topo, &cooldown_until, &model, now);

        // Drive the real engine: a connect failure penalizes, a success clears — the side effects the
        // health view then reports. Collecting a served stream drives its full lifecycle (as the engine
        // family does); a routing error (NoRoute/NoTargets/AllFailed) touches no health state.
        if let Ok(stream) = engine
            .run(
                chat_request(&model),
                "sim-key".to_string(),
                format!("r{step}"),
                now,
            )
            .await
        {
            let _ = llmleaf_model::collect(stream).await;
        }
        cooldown_until = next;

        let actual = health_map(&state.health(now));
        let expected = reference_health(&cooldown_until, now);
        ensure!(
            actual == expected,
            "seed={seed} step={step} now={now}: health view mismatch\n  actual:   {actual:?}\n  expected: {expected:?}"
        );
    }

    // Far past every cooldown window: penalized-then-expired providers must read `down:false` (their
    // entries linger), exercising the expiry path of the snapshot's `now < until` comparison.
    let far = now + topo.cooldown_secs * 4 + 100;
    ensure!(
        health_map(&state.health(far)) == reference_health(&cooldown_until, far),
        "seed={seed}: health view at far-future now={far} mismatch (expiry path)"
    );

    Ok(())
}
