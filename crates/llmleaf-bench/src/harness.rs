//! The measurement loop and result table — the part that is independent of *what* is benchmarked.
//!
//! Every case is two closures: `prepare` builds a fresh input, `run` is the measured operation that
//! consumes it. Splitting them is what lets the harness charge `run` and only `run`:
//!
//! - **Allocations** (exact, deterministic): build one input with `prepare` *outside* the measured
//!   window, then [`measure_allocs`] a single `run`. The input's own allocations never count.
//! - **Time** (subtracted): time `iters` rounds of `prepare(); run()`, time the same `iters` rounds
//!   of `prepare()` alone, and report the difference per iter. So an operation whose input is
//!   expensive to build (cloning a JSON value, boxing a stream) is still timed for itself, not its
//!   setup. The trade-off is noise when `prepare` dominates `run`; the allocation figure, which is
//!   exact, is the one to trust in that regime.
//!
//! Timing is calibrated per case to a wall-clock target so cheap and expensive operations both get a
//! stable sample without a hand-tuned iteration count. Inputs are built one at a time (never pooled),
//! so even a multi-million-iteration calibration holds only a single input live.

use std::time::{Duration, Instant};

use crate::alloc::{measure_allocs, AllocStats};

/// Rounds of `prepare(); run()` run before measuring, to settle one-time lazy initialisation.
const WARMUP: u64 = 16;
/// Iterations of the quick probe used to estimate per-iteration cost during calibration.
const PROBE_ITERS: u64 = 64;
/// Calibration floor/ceiling on the timed iteration count.
const MIN_ITERS: u64 = 200;
const MAX_ITERS: u64 = 4_000_000;

/// One benchmarked case's result.
pub struct Row {
    pub name: String,
    /// Iterations the timing sample averaged over (`0` when timing was skipped).
    pub iters: u64,
    /// Estimated nanoseconds per `run`, with `prepare` subtracted out. Distinguish the two ways this
    /// is `0.0`: timing was skipped (the suite's [`Suite::alloc_only`] mode — check that, not this
    /// field), or — for a case whose `prepare` dwarfs `run` — the subtraction underflowed measurement
    /// noise and was clamped. In the latter regime trust the exact allocation count, not the timing.
    pub ns_per_iter: f64,
    /// Exact allocations made by a single `run`.
    pub alloc: AllocStats,
}

/// A collection of cases. Built once, run as cases are registered, then reported.
pub struct Suite {
    rows: Vec<Row>,
    /// Substring filter: a case whose name does not contain it is skipped entirely.
    filter: Option<String>,
    /// Per-case wall-clock target the timed sample is calibrated toward.
    target: Duration,
    /// When false, only allocations are measured (the timing passes are skipped). The ceiling test
    /// uses this: it asserts on allocation counts and has no use for slow, machine-dependent timings.
    measure_time: bool,
}

impl Suite {
    /// A suite that measures both time and allocations.
    pub fn new(filter: Option<String>) -> Self {
        Suite {
            rows: Vec::new(),
            filter,
            target: Duration::from_millis(80),
            measure_time: true,
        }
    }

    /// A suite that measures allocations only (fast; for the regression test).
    pub fn alloc_only(filter: Option<String>) -> Self {
        Suite {
            measure_time: false,
            ..Suite::new(filter)
        }
    }

    fn selected(&self, name: &str) -> bool {
        self.filter.as_deref().is_none_or(|f| name.contains(f))
    }

    /// Register and immediately run one case. `prepare` builds a fresh input (excluded from both the
    /// reported time and the reported allocations); `run` is the measured operation, which consumes it.
    pub fn bench<In, Out>(
        &mut self,
        name: &str,
        mut prepare: impl FnMut() -> In,
        mut run: impl FnMut(In) -> Out,
    ) {
        if !self.selected(name) {
            return;
        }

        // Settle any one-time initialisation before either measurement. This must run the *real*
        // `run` closure on *this* thread: some dependencies allocate once, lazily, per thread on first
        // use (notably `futures::executor::block_on`, which the async cases call). Warming up with a
        // cheaper proxy, or on another thread, would let that one-time allocation leak into the first
        // measured window and inflate the count by one.
        for _ in 0..WARMUP {
            let input = std::hint::black_box(prepare());
            std::hint::black_box(run(input));
        }

        // Allocations: one prepared input, charge only the run.
        let input = prepare();
        let (_out, alloc) = measure_allocs(|| run(input));

        let (iters, ns_per_iter) = if self.measure_time {
            let iters = self.calibrate(&mut prepare, &mut run);
            let total = time_pass(iters, &mut prepare, &mut run, true);
            let prep = time_pass(iters, &mut prepare, &mut run, false);
            let ns = (total.as_nanos() as f64 - prep.as_nanos() as f64).max(0.0) / iters as f64;
            (iters, ns)
        } else {
            (0, 0.0)
        };

        self.rows.push(Row {
            name: name.into(),
            iters,
            ns_per_iter,
            alloc,
        });
    }

    /// Pick a timed iteration count that lands near [`Suite::target`], from a quick probe.
    ///
    /// The probe estimates `run`'s *own* cost (the same `total - prepare` subtraction the real pass
    /// reports), not the combined `prepare + run`. Otherwise a case whose `prepare` dominates would be
    /// charged for its setup and given too few iterations — starving the subtraction of samples
    /// exactly where its noise is worst.
    fn calibrate<In, Out>(
        &self,
        prepare: &mut impl FnMut() -> In,
        run: &mut impl FnMut(In) -> Out,
    ) -> u64 {
        let total = time_pass(PROBE_ITERS, prepare, run, true);
        let prep = time_pass(PROBE_ITERS, prepare, run, false);
        let run_ns = (total.as_nanos() as f64 - prep.as_nanos() as f64) / PROBE_ITERS as f64;
        // Below ~1ns the run-only estimate is in the noise (sub-resolution lookups); fall back to the
        // combined cost so `want` stays finite and the clamp settles it to MAX_ITERS.
        let per_iter = if run_ns >= 1.0 {
            run_ns
        } else {
            (total.as_nanos() as f64 / PROBE_ITERS as f64).max(1.0)
        };
        let want = (self.target.as_nanos() as f64 / per_iter) as u64;
        want.clamp(MIN_ITERS, MAX_ITERS)
    }

    pub fn rows(&self) -> &[Row] {
        &self.rows
    }

    /// Look up a finished case by exact name.
    pub fn row(&self, name: &str) -> Option<&Row> {
        self.rows.iter().find(|r| r.name == name)
    }

    /// A human-readable, aligned table (the runner's default output).
    pub fn table(&self) -> String {
        let name_w = self
            .rows
            .iter()
            .map(|r| r.name.len())
            .max()
            .unwrap_or(4)
            .max(4);
        let mut out = String::new();
        out.push_str(&format!(
            "{:<name_w$}  {:>12}  {:>8}  {:>8}  {:>12}\n",
            "case",
            "time/iter",
            "allocs",
            "reallocs",
            "bytes",
            name_w = name_w,
        ));
        out.push_str(&format!(
            "{}\n",
            "-".repeat(name_w + 2 + 12 + 2 + 8 + 2 + 8 + 2 + 12)
        ));
        for r in &self.rows {
            let time = if self.measure_time {
                fmt_ns(r.ns_per_iter)
            } else {
                "—".to_string()
            };
            out.push_str(&format!(
                "{:<name_w$}  {:>12}  {:>8}  {:>8}  {:>12}\n",
                r.name,
                time,
                r.alloc.allocs,
                r.alloc.reallocs,
                r.alloc.bytes,
                name_w = name_w,
            ));
        }
        out
    }

    /// Machine-readable output for tracking numbers over time.
    pub fn csv(&self) -> String {
        let mut out = String::from("case,ns_per_iter,allocs,reallocs,bytes,iters\n");
        for r in &self.rows {
            out.push_str(&format!(
                "{},{:.1},{},{},{},{}\n",
                r.name, r.ns_per_iter, r.alloc.allocs, r.alloc.reallocs, r.alloc.bytes, r.iters,
            ));
        }
        out
    }
}

/// One timing pass. When `include_run` is false, only `prepare` runs (its input is dropped), which is
/// the baseline subtracted from the full pass so the reported time charges `run` alone.
fn time_pass<In, Out>(
    iters: u64,
    prepare: &mut impl FnMut() -> In,
    run: &mut impl FnMut(In) -> Out,
    include_run: bool,
) -> Duration {
    let start = Instant::now();
    for _ in 0..iters {
        let input = std::hint::black_box(prepare());
        if include_run {
            std::hint::black_box(run(input));
        } else {
            std::hint::black_box(input);
        }
    }
    start.elapsed()
}

/// Render a nanosecond figure with a sensible unit.
fn fmt_ns(ns: f64) -> String {
    if ns < 1_000.0 {
        format!("{ns:.1} ns")
    } else if ns < 1_000_000.0 {
        format!("{:.2} µs", ns / 1_000.0)
    } else {
        format!("{:.2} ms", ns / 1_000_000.0)
    }
}
