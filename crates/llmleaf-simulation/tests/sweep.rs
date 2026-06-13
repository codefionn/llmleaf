//! The `cargo test` entry point for the simulation harness: sweep a fixed band of seeds (see
//! [`llmleaf_simulation::seeds`]) and fail with the offending seed if any scenario diverges.
//!
//! Reproduce a reported failure with `LLMLEAF_SIM_SEED=<n> cargo test -p llmleaf-simulation`, or
//! widen the band with `LLMLEAF_SIM_ITERS=<count>`. For an open-ended search use the `llmleaf-sim`
//! binary instead (`cargo run -p llmleaf-simulation --release -- 5m`).

use llmleaf_simulation::{admin, collect, dispatch, engine, keys, provider, seeds};

#[tokio::test]
async fn engine_routing_simulation() {
    for seed in seeds() {
        if let Err(report) = engine::run_scenario(seed).await {
            panic!("{report}");
        }
    }
}

#[tokio::test]
async fn admin_simulation() {
    for seed in seeds() {
        if let Err(report) = admin::run_scenario(seed).await {
            panic!("{report}");
        }
    }
}

#[tokio::test]
async fn engine_dispatch_simulation() {
    for seed in seeds() {
        if let Err(report) = dispatch::run_scenario(seed).await {
            panic!("{report}");
        }
    }
}

#[tokio::test]
async fn provider_simulation() {
    for seed in seeds() {
        if let Err(report) = provider::run_scenario(seed).await {
            panic!("{report}");
        }
    }
}

#[test]
fn key_verdict_simulation() {
    for seed in seeds() {
        if let Err(report) = keys::run_scenario(seed) {
            panic!("{report}");
        }
    }
}

#[test]
fn collect_folding_simulation() {
    for seed in seeds() {
        if let Err(report) = collect::run_scenario(seed) {
            panic!("{report}");
        }
    }
}
