//! A deterministic PRNG and the seed-band helper shared by every scenario.
//!
//! SplitMix64 — a fixed, well-distributed algorithm with no hidden state. A seed reproduces the exact
//! same stream of values forever, on any platform, which is the whole point: a recorded failing seed
//! must replay the identical scenario.

/// A SplitMix64 generator. Cheap to create per scenario; `Rng::new(seed)` fully determines its output.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, n)`. `n` must be non-zero.
    pub fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }

    /// Uniform in `[lo, hi]` inclusive.
    pub fn range(&mut self, lo: u64, hi: u64) -> u64 {
        lo + self.below(hi - lo + 1)
    }

    pub fn bool(&mut self) -> bool {
        self.next_u64() & 1 == 1
    }

    /// True with probability `num/den`.
    pub fn chance(&mut self, num: u64, den: u64) -> bool {
        self.below(den) < num
    }

    pub fn pick<'a, T>(&mut self, xs: &'a [T]) -> &'a T {
        &xs[self.below(xs.len() as u64) as usize]
    }
}

/// The band of seeds the `cargo test` sweep covers. `LLMLEAF_SIM_SEED` pins exactly one seed (to
/// reproduce a reported failure); `LLMLEAF_SIM_ITERS` widens the band (default 256 — plenty for a
/// regular `cargo test`, raise it for a nightly sweep). The soak binary ignores this and runs to a
/// time budget instead.
pub fn seeds() -> Vec<u64> {
    if let Ok(s) = std::env::var("LLMLEAF_SIM_SEED") {
        return vec![s.parse().expect("LLMLEAF_SIM_SEED must be a u64")];
    }
    let iters = std::env::var("LLMLEAF_SIM_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(256);
    (0..iters).collect()
}
