//! A counting global allocator and the single-threaded window that measures against it.
//!
//! SOUL.md is emphatic that the hot path must "justify every allocation". To hold that line we need
//! to *count* allocations, not just time them — a change that doubles a request's allocation count
//! while staying the same speed is still a regression. This module is how the harness sees those
//! counts: a thin [`GlobalAlloc`] wrapper over the system allocator that tallies every `alloc` /
//! `realloc` and the bytes they request.
//!
//! Counting is process-global (three atomics), so a measurement is only meaningful when **nothing
//! else is allocating concurrently**. The harness honours this by running strictly single-threaded
//! and driving async work with `futures::executor::block_on` on the measuring thread — no runtime
//! worker pool, no background threads to pollute the tally. [`measure_allocs`] snapshots the counters
//! around one call and reports the delta.
//!
//! The allocator is only counted when a binary/test installs it as *the* global allocator:
//!
//! ```ignore
//! #[global_allocator]
//! static GLOBAL: llmleaf_bench::alloc::CountingAllocator = llmleaf_bench::alloc::CountingAllocator;
//! ```
//!
//! Without that line the wrapper is never the active allocator, the counters stay at zero, and
//! [`is_active`] returns `false` — measurements would be vacuous, so the runner and the ceiling test
//! both assert it up front.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};

/// Number of `alloc` (and `alloc_zeroed`, which routes through `alloc`) calls observed.
static ALLOCS: AtomicU64 = AtomicU64::new(0);
/// Number of `realloc` calls observed (a grow/shrink in place, counted separately from fresh allocs).
static REALLOCS: AtomicU64 = AtomicU64::new(0);
/// Total bytes requested: every `alloc`'s size, plus the growth portion of each `realloc`.
static BYTES: AtomicU64 = AtomicU64::new(0);

/// A `#[global_allocator]`-installable wrapper over [`System`] that tallies every allocation.
///
/// It only forwards to the system allocator and bumps a counter — zero behavioural difference, a
/// couple of relaxed atomic adds of overhead. Install it in a binary or test root (see module docs);
/// the library never installs it itself, so importing this crate has no effect on anyone's allocator.
pub struct CountingAllocator;

// SAFETY: every method forwards verbatim to the global `System` allocator with the same arguments;
// the only added work is incrementing counters, which cannot affect allocation validity.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        System.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // Deallocations are not part of "how many allocations did this operation make"; leave them
        // uncounted so the reported figure is gross allocations, the number SOUL.md cares about.
        System.dealloc(ptr, layout)
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        REALLOCS.fetch_add(1, Ordering::Relaxed);
        if new_size > layout.size() {
            BYTES.fetch_add((new_size - layout.size()) as u64, Ordering::Relaxed);
        }
        System.realloc(ptr, layout, new_size)
    }
}

/// The allocation tally for one measured operation. All counts are *gross* (deallocations ignored).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AllocStats {
    /// Fresh allocations (`alloc` + `alloc_zeroed`).
    pub allocs: u64,
    /// In-place grow/shrink calls (`realloc`) — e.g. a `Vec`/`String` outgrowing its capacity.
    pub reallocs: u64,
    /// Bytes requested across those allocations (and realloc growth).
    pub bytes: u64,
}

/// Run `f` and report what it allocated.
///
/// Correctness depends on the caller's contract: [`CountingAllocator`] must be the installed global
/// allocator and **no other thread may allocate** for the duration of the call (the counters are
/// process-global). The harness guarantees both. The result of `f` is returned alongside the stats so
/// the operation's output cannot be optimised away.
pub fn measure_allocs<T>(f: impl FnOnce() -> T) -> (T, AllocStats) {
    let a0 = ALLOCS.load(Ordering::Relaxed);
    let r0 = REALLOCS.load(Ordering::Relaxed);
    let b0 = BYTES.load(Ordering::Relaxed);

    let out = std::hint::black_box(f());

    let stats = AllocStats {
        allocs: ALLOCS.load(Ordering::Relaxed) - a0,
        reallocs: REALLOCS.load(Ordering::Relaxed) - r0,
        bytes: BYTES.load(Ordering::Relaxed) - b0,
    };
    (out, stats)
}

/// Is [`CountingAllocator`] actually the installed global allocator?
///
/// A probe: force one allocation and check the counter moved. The runner and the ceiling test assert
/// this so a misconfiguration (forgetting the `#[global_allocator]` line) fails loudly instead of
/// silently reporting zero allocations everywhere.
pub fn is_active() -> bool {
    let before = ALLOCS.load(Ordering::Relaxed);
    let probe = std::hint::black_box(vec![0u8; 64]);
    std::hint::black_box(&probe);
    ALLOCS.load(Ordering::Relaxed) > before
}
