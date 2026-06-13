//! The llmleaf seeded simulation harness.
//!
//! This crate is a QA tool, not part of llmleaf itself: it depends on the core and the canonical model
//! but is **never** a dependency of any production crate (the data plane stays oblivious to its tests).
//!
//! It is *model-based* simulation testing. A single `u64` seed deterministically generates an entire
//! scenario, the real code is driven through it, and an independent, spec-derived reference model
//! predicts every outcome. Any divergence is a bug — and because the scenario is a pure function of
//! the seed, every failure is reproducible forever from its seed alone.
//!
//! Determinism is load-bearing, so the PRNG is a hand-rolled SplitMix64 ([`rng`]) rather than `rand`
//! or `proptest`: a recorded seed must replay the *identical* scenario on any machine and any future
//! build, which external RNG/strategy crates cannot guarantee across versions. Zero extra dependencies
//! also keeps faith with SOUL.md's minimalism.
//!
//! Two front ends share the same scenario logic:
//!   - `tests/sweep.rs` — the `cargo test` entry point: sweeps a fixed band of seeds.
//!   - `src/main.rs` (`llmleaf-sim`) — the soak runner: hammers seeds until a time budget elapses.
//!
//! Each scenario runner returns `Result<(), String>` (the `Err` is a fully-formed, seed-stamped
//! failure report) instead of panicking, so the soak runner can present a clean reproducer.

pub mod admin;
pub mod collect;
pub mod dispatch;
pub mod engine;
pub mod keys;
pub mod provider;
pub mod rng;

pub use rng::{seeds, Rng};

/// Run every scenario family for one seed. The soak runner calls this; the test sweep calls the
/// per-family runners directly so each gets its own named test.
pub async fn run_seed(seed: u64) -> Result<(), String> {
    engine::run_scenario(seed).await?;
    dispatch::run_scenario(seed).await?;
    admin::run_scenario(seed).await?;
    provider::run_scenario(seed).await?;
    keys::run_scenario(seed)?;
    collect::run_scenario(seed)?;
    Ok(())
}

/// `return Err(format!(...))` unless the condition holds — the workhorse of the reference-model
/// checks, keeping each assertion a single readable line that still produces a seed-stamped report.
#[macro_export]
macro_rules! ensure {
    ($cond:expr, $($arg:tt)*) => {
        if !($cond) {
            return Err(format!($($arg)*));
        }
    };
}
