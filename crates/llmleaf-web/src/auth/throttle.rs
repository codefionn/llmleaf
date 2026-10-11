//! A tiny in-memory failed-attempt throttle for the operator login endpoints.
//!
//! The master password is a single shared secret, so online guessing is the attack that matters.
//! Nothing else in the app limits authentication attempts (the usage `limiter` caps consumer keys,
//! not logins). This module keeps a per-client-IP failure counter in a sliding window and refuses
//! further attempts once the budget is spent — deliberately small and dependency-free.
//!
//! State is process-local and resets on restart; that is acceptable for throttling (it is not an
//! audit log). Behind a reverse proxy, pass the real client address via `ConnectInfo` (the server is
//! started with `into_make_service_with_connect_info`); unknown clients share one bucket.

use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

/// Failed attempts allowed per client within [`WINDOW`] before attempts are refused.
const MAX_FAILURES: u32 = 10;
/// Sliding window length.
const WINDOW: Duration = Duration::from_secs(5 * 60);

struct Entry {
    failures: u32,
    window_start: Instant,
}

fn entries() -> &'static Mutex<HashMap<IpAddr, Entry>> {
    static ENTRIES: OnceLock<Mutex<HashMap<IpAddr, Entry>>> = OnceLock::new();
    ENTRIES.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Whether `ip` may attempt a login right now (`Ok`) or is locked out (`Err(retry_after_secs)`).
pub fn check(ip: IpAddr) -> Result<(), u64> {
    let map = entries().lock().unwrap();
    match map.get(&ip) {
        Some(e) if e.failures >= MAX_FAILURES && e.window_start.elapsed() < WINDOW => {
            Err(remaining_secs(e))
        }
        _ => Ok(()),
    }
}

/// Record a failed attempt for `ip` (starting/refreshing the window only from the first failure).
pub fn record_failure(ip: IpAddr) {
    let mut map = entries().lock().unwrap();
    let now = Instant::now();
    let e = map.entry(ip).or_insert(Entry {
        failures: 0,
        window_start: now,
    });
    if e.window_start.elapsed() >= WINDOW {
        e.failures = 0;
        e.window_start = now;
    }
    e.failures += 1;
    if map.len() > 10_000 {
        // Bounded memory: drop anything whose window has fully elapsed.
        map.retain(|_, e| e.window_start.elapsed() < WINDOW);
    }
}

/// Clear the slate for `ip` after a successful login (also clears lockouts).
pub fn reset(ip: IpAddr) {
    entries().lock().unwrap().remove(&ip);
}

fn remaining_secs(e: &Entry) -> u64 {
    WINDOW
        .saturating_sub(e.window_start.elapsed())
        .as_secs()
        .max(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::Ipv4Addr;

    #[test]
    fn locks_out_after_max_failures_and_resets() {
        let ip = IpAddr::V4(Ipv4Addr::new(10, 255, 255, 1));
        assert!(check(ip).is_ok());
        for _ in 0..MAX_FAILURES {
            record_failure(ip);
        }
        assert!(check(ip).is_err());
        reset(ip);
        assert!(check(ip).is_ok());
    }

    #[test]
    fn buckets_are_independent() {
        let a = IpAddr::V4(Ipv4Addr::new(10, 255, 255, 2));
        let b = IpAddr::V4(Ipv4Addr::new(10, 255, 255, 3));
        for _ in 0..MAX_FAILURES {
            record_failure(a);
        }
        assert!(check(a).is_err());
        assert!(check(b).is_ok());
    }
}
