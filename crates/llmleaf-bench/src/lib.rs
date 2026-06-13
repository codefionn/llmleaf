//! The llmleaf micro-benchmark harness.
//!
//! Like `llmleaf-simulation`, this crate is a QA tool: it depends on the core and the canonical model
//! but is **never** a dependency of any production crate. Where the simulation harness checks that the
//! hot path is *correct*, this one checks that it is *cheap* — the two figures SOUL.md keeps returning
//! to: wall-clock **time** and heap **allocations** ("the hot path is sacred… justify every allocation").
//!
//! It is deliberately dependency-free in the same spirit as the simulation harness's hand-rolled RNG:
//! no `criterion`, no `dhat`. Time comes from `Instant`; allocations come from a counting global
//! allocator ([`alloc`]) that a binary or test installs. That keeps the workspace lean and the
//! measurement transparent (principle 7) — there is no third-party machinery between the code and the
//! number.
//!
//! Two front ends share the same cases ([`cases`]):
//!   - `src/main.rs` (`llmleaf-bench`) — prints the full time + allocation table.
//!   - `tests/ceilings.rs` — the `cargo test` guard: asserts each hot operation stays under an
//!     allocation ceiling, so a change that smuggles in extra allocations fails CI.
//!
//! Both install [`alloc::CountingAllocator`] as the global allocator; the library itself installs
//! nothing, so merely linking it changes no one's allocator.

pub mod alloc;
pub mod cases;
pub mod harness;

pub use harness::{Row, Suite};

/// Build and run the full suite (time + allocations), honouring an optional substring `filter`.
pub fn run(filter: Option<String>) -> Suite {
    let mut suite = Suite::new(filter);
    cases::register(&mut suite);
    suite
}

/// Build and run the suite measuring allocations only (fast; skips the timing passes). Used by the
/// ceiling regression test, which asserts on allocation counts and has no use for timings.
pub fn run_alloc_only(filter: Option<String>) -> Suite {
    let mut suite = Suite::alloc_only(filter);
    cases::register(&mut suite);
    suite
}
