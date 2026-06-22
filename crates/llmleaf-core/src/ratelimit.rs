//! Node-local provider rate limiting (SOUL.md principles 1, 8, and 9).
//!
//! The sibling of [`crate::route::HealthTable`]: where the health table routes *away* from a provider
//! that already failed, this throttles flow *toward* a provider so it is not pushed past its published
//! request/token/concurrency limits in the first place. Both are node-local HA flow control — derived,
//! droppable state, a fast local decision (principle 9), never cross-node coordination and never a
//! network round-trip. This is deliberately NOT per-key usage accounting (principle 5, owned by the
//! pulled `[control.limits]` verdicts): it counts in-flight flow for switchover, not usage for billing.
//!
//! Limits compose in two scopes: a provider-global bucket and an optional per-(provider, model) bucket.
//! A request must pass *both*; the stricter binds. Each scope carries up to three independent dimensions
//! — requests/min and tokens/min (token buckets) and max-concurrent (a semaphore).
//!
//! Admission ([`RateLimiter::try_admit`]) is non-blocking and allocation-light: a borrowed-key map
//! lookup, a couple of per-entry mutex critical sections, and `try_acquire_owned` on a semaphore (an
//! `Arc` clone, no heap allocation). It returns either a [`RateGuard`] (holding the concurrency permits,
//! released on drop) or a `Duration` estimate of how long until this scope could admit — which the
//! engine uses to wait, bounded, when *every* target on a chain is saturated (principle 1: the wait is
//! the operator's opted-in latency, capped by `server.rate_limit_max_wait_ms`, never unbounded).

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio::time::Instant;

use crate::config::{ProviderConfig, RateLimitConfig};

/// How long to wait before re-polling when the *only* binding constraint is concurrency: a permit frees
/// when an in-flight request finishes, which has no deterministic refill time, so the engine re-polls.
/// Token-bucket waits are exact; this is the "approx" granularity for the semaphore case.
const CONCURRENCY_POLL: Duration = Duration::from_millis(50);

/// A sentinel "effectively never" wait for a degenerate `0`/min limit (refill rate zero). The engine
/// caps every wait at `rate_limit_max_wait_ms` anyway, so a 0-limit scope just falls through then 429s.
const NEVER: Duration = Duration::from_secs(86_400);

/// Node-local rate limiter. Built once from config and then read-only at the map level (the set of
/// scopes is fixed at load), so a hot-path lookup borrows `&str` keys with no lock and no allocation;
/// only the per-entry bucket/semaphore state mutates, each behind its own mutex/atomic.
#[derive(Default)]
pub struct RateLimiter {
    providers: HashMap<String, ProviderLimits>,
}

struct ProviderLimits {
    global: Limit,
    per_model: HashMap<String, Limit>,
}

/// The limits for one scope (a provider, or one of its models). Each dimension is independently optional.
#[derive(Default)]
struct Limit {
    rpm: Option<TokenBucket>,
    tpm: Option<TokenBucket>,
    sem: Option<Arc<Semaphore>>,
}

impl Limit {
    fn from_config(cfg: &RateLimitConfig) -> Self {
        Limit {
            rpm: cfg.requests_per_min.map(TokenBucket::per_minute),
            tpm: cfg.tokens_per_min.map(TokenBucket::per_minute),
            sem: cfg
                .max_concurrent
                .map(|n| Arc::new(Semaphore::new(n as usize))),
        }
    }

    /// Whether this scope declares any limit at all (so an all-`None` scope can be skipped cheaply).
    fn is_unlimited(&self) -> bool {
        self.rpm.is_none() && self.tpm.is_none() && self.sem.is_none()
    }
}

impl RateLimiter {
    /// Build the limiter from provider config. A provider with neither global `limits` nor any
    /// `model_limits` contributes no entry, so it is unlimited and its admission is a single missing-key
    /// lookup that returns immediately.
    pub fn new(providers: &[ProviderConfig]) -> Self {
        let mut map = HashMap::new();
        for p in providers {
            let global = p
                .limits
                .as_ref()
                .map(Limit::from_config)
                .unwrap_or_default();
            let per_model: HashMap<String, Limit> = p
                .model_limits
                .iter()
                .map(|(model, cfg)| (model.clone(), Limit::from_config(cfg)))
                .collect();
            if global.is_unlimited() && per_model.is_empty() {
                continue;
            }
            map.insert(p.name.clone(), ProviderLimits { global, per_model });
        }
        RateLimiter { providers: map }
    }

    /// Try to admit one request to `(provider, model)` at `now`. On success returns a [`RateGuard`]
    /// holding the concurrency permits (released when the guard drops, i.e. when the request's stream
    /// ends). On failure returns the soonest `Duration` after which this scope might admit — the engine
    /// either falls through to the next target or, when all are saturated, waits this long (bounded).
    ///
    /// Atomicity: a request must satisfy *every* declared dimension of *both* the global and the
    /// per-model scope simultaneously. Resources are acquired in order and any partial acquisition is
    /// undone before returning `Err`, so a rejected request leaves no token consumed and no permit held.
    pub fn try_admit(
        &self,
        provider: &str,
        model: &str,
        now: Instant,
    ) -> Result<RateGuard, Duration> {
        let Some(p) = self.providers.get(provider) else {
            return Ok(RateGuard::EMPTY);
        };
        let model_limit = p.per_model.get(model);

        // 1) Concurrency permits first — each is released automatically if a later step returns `Err`
        //    (the `OwnedSemaphorePermit` local drops at the early return). Global then per-model.
        let global_permit = match &p.global.sem {
            Some(sem) => Some(
                sem.clone()
                    .try_acquire_owned()
                    .map_err(|_| CONCURRENCY_POLL)?,
            ),
            None => None,
        };
        let model_permit = match model_limit.and_then(|l| l.sem.as_ref()) {
            Some(sem) => Some(
                sem.clone()
                    .try_acquire_owned()
                    .map_err(|_| CONCURRENCY_POLL)?,
            ),
            None => None,
        };

        // 2) Token-per-minute floors (a read, never a take — the actual token cost is unknown until the
        //    response reports usage, so admission only checks the bucket is not already exhausted).
        if let Some(tpm) = &p.global.tpm {
            tpm.has_capacity(now)?;
        }
        if let Some(tpm) = model_limit.and_then(|l| l.tpm.as_ref()) {
            tpm.has_capacity(now)?;
        }

        // 3) Requests-per-minute takes. The global take is the only mutation that may need rolling back
        //    (if the per-model take then fails); the permits and the read-only TPM checks above need no
        //    undo.
        let took_global_rpm = match &p.global.rpm {
            Some(rpm) => {
                rpm.try_take(1.0, now)?;
                true
            }
            None => false,
        };
        if let Some(rpm) = model_limit.and_then(|l| l.rpm.as_ref()) {
            if let Err(wait) = rpm.try_take(1.0, now) {
                if took_global_rpm {
                    p.global.rpm.as_ref().unwrap().refund(1.0, now);
                }
                return Err(wait);
            }
        }

        Ok(RateGuard {
            _global: global_permit,
            _model: model_permit,
        })
    }

    /// Debit provider-reported `tokens` from the tokens/min buckets of `(provider, model)`. Called when
    /// usage is observed on the stream (the cost is not known at admission), so a burst may drive a
    /// bucket negative and then throttle subsequent admissions until it refills. A no-op when neither
    /// scope declares a tokens/min limit.
    pub fn debit_tokens(&self, provider: &str, model: &str, tokens: u64, now: Instant) {
        let Some(p) = self.providers.get(provider) else {
            return;
        };
        let n = tokens as f64;
        if let Some(tpm) = &p.global.tpm {
            tpm.debit(n, now);
        }
        if let Some(tpm) = p.per_model.get(model).and_then(|l| l.tpm.as_ref()) {
            tpm.debit(n, now);
        }
    }
}

/// Holds the concurrency permits a request acquired at admission. Dropping it releases them (returning
/// in-flight capacity), so the engine keeps it alive for the life of the response stream and lets it
/// drop when the stream ends — including on error or client disconnect, since a dropped stream drops the
/// guard. Carries no requests/min state: that token is consumed at admission and not refunded on a
/// dispatch failure (the call was made), matching the health table's "a failed attempt still happened".
#[derive(Debug)]
pub struct RateGuard {
    _global: Option<OwnedSemaphorePermit>,
    _model: Option<OwnedSemaphorePermit>,
}

impl RateGuard {
    /// The guard for an unlimited scope: holds nothing, releases nothing.
    const EMPTY: RateGuard = RateGuard {
        _global: None,
        _model: None,
    };
}

/// A classic token bucket: `capacity` tokens of burst, refilling at `refill_per_sec`, lazily topped up
/// against a monotonic clock on each access. Per-entry `Mutex` (not a shared `RwLock`) so contention is
/// sharded per scope. Time is `tokio::time::Instant`, supplied by the caller — monotonic (no clock-skew
/// handling), and controllable under `tokio::time::pause()` so the wait path is deterministically
/// testable alongside the engine's `sleep`.
struct TokenBucket {
    capacity: f64,
    refill_per_sec: f64,
    state: Mutex<BucketState>,
}

struct BucketState {
    tokens: f64,
    last: Instant,
}

impl TokenBucket {
    /// A bucket whose burst is one minute's worth of `per_min` and which refills at `per_min/60` per
    /// second, starting full.
    fn per_minute(per_min: u64) -> Self {
        let capacity = per_min as f64;
        TokenBucket {
            capacity,
            refill_per_sec: capacity / 60.0,
            state: Mutex::new(BucketState {
                tokens: capacity,
                last: Instant::now(),
            }),
        }
    }

    /// Lazily add the tokens accrued since the last access, clamped to `capacity`. Monotonic time means
    /// `now >= last` always, but the saturating compare keeps it safe regardless.
    fn refill(&self, state: &mut BucketState, now: Instant) {
        let elapsed = now.saturating_duration_since(state.last).as_secs_f64();
        if elapsed > 0.0 {
            state.tokens = (state.tokens + elapsed * self.refill_per_sec).min(self.capacity);
            state.last = now;
        }
    }

    /// Take `n` tokens if available; otherwise report how long until `n` will have accrued.
    fn try_take(&self, n: f64, now: Instant) -> Result<(), Duration> {
        let mut state = self.state.lock().unwrap();
        self.refill(&mut state, now);
        if state.tokens >= n {
            state.tokens -= n;
            Ok(())
        } else {
            Err(self.wait_for(n - state.tokens))
        }
    }

    /// Return `n` tokens previously taken (used to undo a partial admission), never exceeding capacity.
    fn refund(&self, n: f64, now: Instant) {
        let mut state = self.state.lock().unwrap();
        self.refill(&mut state, now);
        state.tokens = (state.tokens + n).min(self.capacity);
    }

    /// Admit iff the bucket is not exhausted (tokens above zero). Does not take — the real token cost is
    /// debited later from observed usage. When exhausted (possibly negative from overshoot), report how
    /// long until it climbs back above zero.
    fn has_capacity(&self, now: Instant) -> Result<(), Duration> {
        let mut state = self.state.lock().unwrap();
        self.refill(&mut state, now);
        if state.tokens > 0.0 {
            Ok(())
        } else {
            Err(self.wait_for(-state.tokens + f64::EPSILON))
        }
    }

    /// Debit `n` tokens; may drive the bucket negative (overshoot), which throttles later admissions.
    fn debit(&self, n: f64, now: Instant) {
        let mut state = self.state.lock().unwrap();
        self.refill(&mut state, now);
        state.tokens -= n;
    }

    /// Time to accrue `deficit` tokens at the refill rate. Guards a zero/negative rate (a `0`/min limit)
    /// and non-finite math so `Duration::from_secs_f64` never panics.
    fn wait_for(&self, deficit: f64) -> Duration {
        if self.refill_per_sec <= 0.0 {
            return NEVER;
        }
        let secs = deficit.max(0.0) / self.refill_per_sec;
        if secs.is_finite() {
            Duration::from_secs_f64(secs)
        } else {
            NEVER
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(rpm: Option<u64>, tpm: Option<u64>, conc: Option<u32>) -> RateLimitConfig {
        RateLimitConfig {
            requests_per_min: rpm,
            tokens_per_min: tpm,
            max_concurrent: conc,
        }
    }

    fn provider(name: &str, global: Option<RateLimitConfig>) -> ProviderConfig {
        ProviderConfig {
            name: name.into(),
            kind: "test".into(),
            endpoint: None,
            credential: None,
            prefix: None,
            settings: Default::default(),
            limits: global,
            model_limits: Default::default(),
        }
    }

    // A provider with no limits configured contributes no entry: admission is an immediate pass.
    #[tokio::test]
    async fn unlimited_provider_always_admits() {
        let rl = RateLimiter::new(&[provider("p", None)]);
        for _ in 0..1000 {
            assert!(rl.try_admit("p", "m", Instant::now()).is_ok());
        }
        // An unknown provider is likewise unlimited (no entry).
        assert!(rl.try_admit("unknown", "m", Instant::now()).is_ok());
    }

    // RPM is a token bucket: a burst up to capacity is admitted, the next is refused with a wait, and
    // after enough virtual time one slot has refilled.
    #[tokio::test(start_paused = true)]
    async fn rpm_bucket_admits_burst_then_refuses_then_refills() {
        let rl = RateLimiter::new(&[provider("p", Some(cfg(Some(60), None, None)))]); // 60/min = 1/sec
        let t0 = Instant::now();
        // Drain the full 60-token burst.
        for _ in 0..60 {
            rl.try_admit("p", "m", t0).expect("within burst");
        }
        // 61st is refused; the wait is ~1s (one token at 1/sec).
        let wait = rl.try_admit("p", "m", t0).expect_err("burst exhausted");
        assert!(
            wait <= Duration::from_secs(1) + Duration::from_millis(50),
            "{wait:?}"
        );
        // After 1s one token has refilled → exactly one more admit, then refused again.
        let t1 = t0 + Duration::from_secs(1);
        rl.try_admit("p", "m", t1).expect("one refilled");
        assert!(rl.try_admit("p", "m", t1).is_err());
    }

    // Max-concurrent is a semaphore: permits are held by the guards and only freeing a guard re-admits.
    #[tokio::test]
    async fn concurrency_permits_gate_inflight() {
        let rl = RateLimiter::new(&[provider("p", Some(cfg(None, None, Some(2))))]);
        let g1 = rl.try_admit("p", "m", Instant::now()).expect("1st");
        let _g2 = rl.try_admit("p", "m", Instant::now()).expect("2nd");
        // Third is refused while two are in flight, with the concurrency poll interval as the wait.
        assert_eq!(
            rl.try_admit("p", "m", Instant::now()).unwrap_err(),
            CONCURRENCY_POLL
        );
        // Releasing one permit re-opens a slot.
        drop(g1);
        assert!(rl.try_admit("p", "m", Instant::now()).is_ok());
    }

    // Global and per-model limits compose: the stricter binds, and a per-model rejection refunds the
    // global RPM token it had already taken (so the global bucket is not silently drained).
    #[tokio::test(start_paused = true)]
    async fn global_and_per_model_compose_and_rollback() {
        let mut p = provider("p", Some(cfg(Some(600), None, None))); // generous global
        p.model_limits
            .insert("tight".into(), cfg(Some(60), None, None)); // strict model: 1/sec
        let rl = RateLimiter::new(&[p]);
        let t0 = Instant::now();
        // Drain the tight model's 60-token burst (each also takes one global token).
        for _ in 0..60 {
            rl.try_admit("p", "tight", t0).expect("within model burst");
        }
        // The next "tight" request is refused by the per-model bucket...
        assert!(rl.try_admit("p", "tight", t0).is_err());
        // ...and the global bucket still has its remaining capacity: a different, unlimited-per-model
        // model on the same provider keeps admitting (the rejected request refunded its global token).
        for _ in 0..540 {
            rl.try_admit("p", "other", t0)
                .expect("global still has room");
        }
        // Now the global 600-token burst is exhausted too.
        assert!(rl.try_admit("p", "other", t0).is_err());
    }

    // TPM admits while positive, is driven negative by an observed-usage debit (overshoot), then refuses
    // until virtual time refills it back above zero.
    #[tokio::test(start_paused = true)]
    async fn tpm_floor_admits_until_debited_negative_then_recovers() {
        let rl = RateLimiter::new(&[provider("p", Some(cfg(None, Some(600), None)))]); // 600 tok/min = 10/sec
        let t0 = Instant::now();
        assert!(rl.try_admit("p", "m", t0).is_ok(), "starts with capacity");
        // A big request reports 1000 tokens → bucket goes to 600 - 1000 = -400.
        rl.debit_tokens("p", "m", 1000, t0);
        let wait = rl
            .try_admit("p", "m", t0)
            .expect_err("exhausted by overshoot");
        // ~40s to climb from -400 back above 0 at 10 tok/sec.
        assert!(
            wait >= Duration::from_secs(39) && wait <= Duration::from_secs(41),
            "{wait:?}"
        );
        // After 41s it has refilled above zero.
        assert!(rl.try_admit("p", "m", t0 + Duration::from_secs(41)).is_ok());
    }
}
