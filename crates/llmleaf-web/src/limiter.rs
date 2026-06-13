//! The automated limiter — the canonical bolt-on (SOUL.md: "Gatekeeper / limiter (a role, not a
//! component)"). A background loop turns the pushed usage into verdicts the core then PULLS. It also
//! does housekeeping (expired sessions / abandoned OIDC flows). Disabled ⇒ verdicts are operator-set
//! only; the loop still does housekeeping.
//!
//! Policy is deliberately simple and node-local: a rolling 30-day per-key cost cap and a 24h per-key
//! request cap. When a key trips a cap the limiter *suspends* it (never blocks — a block is an operator
//! decision); once the key falls back under the cap a later pass lifts the limiter-set suspension. The
//! limiter only ever touches suspensions it set (`verdict_source='limiter'`), never an operator's.

use std::sync::Arc;
use std::time::Duration;

use tokio::task::JoinHandle;
use tokio::time::{interval, MissedTickBehavior};
use tokio_util::sync::CancellationToken;

use crate::config::WebConfig;
use crate::db::{self, now_secs, usage::since_secs_ago, Db};

const DAY_SECS: u64 = 86_400;
const COST_WINDOW_SECS: u64 = 30 * DAY_SECS;
/// Events older than this are pruned each pass (bounded storage; the dashboards cover the recent past).
const EVENT_RETENTION_SECS: u64 = 90 * DAY_SECS;

/// Spawn the limiter + housekeeping loop. Ticks every `limiter.interval_secs`.
pub fn spawn(config: Arc<WebConfig>, db: Db, shutdown: CancellationToken) -> JoinHandle<()> {
    tokio::spawn(async move {
        let period = Duration::from_secs(config.limiter.interval_secs.max(1));
        let mut tick = interval(period);
        tick.set_missed_tick_behavior(MissedTickBehavior::Delay);
        loop {
            tokio::select! {
                _ = shutdown.cancelled() => break,
                _ = tick.tick() => {
                    if let Err(e) = pass(&config, &db).await {
                        tracing::warn!(error = %e, "limiter pass failed");
                    }
                }
            }
        }
        tracing::info!("limiter loop stopped");
    })
}

/// One pass: housekeeping always, then the cap policy when enabled.
async fn pass(config: &WebConfig, db: &Db) -> Result<(), sqlx::Error> {
    let _ = db::session::prune_sessions(db).await;
    let _ = db::session::prune_oidc_flows(db, DAY_SECS * 1000).await;
    let _ = db::usage::prune_before(db, since_secs_ago(EVENT_RETENTION_SECS)).await;

    if !config.limiter.enabled {
        return Ok(());
    }
    let cost_cap = config.limiter.monthly_cost_cap;
    let req_cap = config.limiter.daily_request_cap;
    if cost_cap <= 0.0 && req_cap == 0 {
        return Ok(()); // enabled but no caps set — nothing to enforce
    }

    let now = now_secs();
    let cost_window = since_secs_ago(COST_WINDOW_SECS);
    let req_window = since_secs_ago(DAY_SECS);

    // Scan keys with recent activity AND keys the limiter currently has suspended — the latter so a
    // quiet key whose spend has aged out still gets its auto-suspension lifted.
    let mut scope: std::collections::HashSet<String> = db::usage::active_keys(db, cost_window)
        .await?
        .into_iter()
        .collect();
    scope.extend(db::keys::limiter_suspended_ids(db).await?);

    for key in scope {
        let (cost30, _) = db::usage::key_window(db, &key, cost_window).await?;
        let (_, reqs24) = db::usage::key_window(db, &key, req_window).await?;

        let mut suspend_until: Option<u64> = None;
        if cost_cap > 0.0 && cost30 >= cost_cap {
            // Re-evaluated each pass; stays suspended while over, lifted once back under.
            suspend_until = Some(now + DAY_SECS);
        }
        if req_cap > 0 && reqs24 >= req_cap {
            let midnight = ((now / DAY_SECS) + 1) * DAY_SECS;
            suspend_until = Some(suspend_until.map_or(midnight, |u| u.max(midnight)));
        }

        match suspend_until {
            Some(until) => {
                db::keys::limiter_suspend(db, &key, until).await?;
                tracing::info!(key = %key, until, cost30, reqs24, "limiter suspended key");
            }
            None => db::keys::limiter_clear(db, &key).await?,
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{LimiterSection, WebConfig};
    use crate::wire::{Envelope, Event, Usage};

    fn cfg(cost_cap: f64, req_cap: u64) -> WebConfig {
        let mut c = WebConfig::from_toml_str("").unwrap();
        c.limiter = LimiterSection {
            enabled: true,
            interval_secs: 30,
            monthly_cost_cap: cost_cap,
            daily_request_cap: req_cap,
        };
        c
    }

    async fn mem() -> Db {
        db::connect("sqlite::memory:").await.unwrap()
    }

    #[tokio::test]
    async fn cost_cap_suspends_then_clears() {
        let db = mem().await;
        db::keys::create(&db, "spend", "$2y$12$h", None, &[])
            .await
            .unwrap();
        // One expensive usage event "now".
        let now_ms = db::now_ms();
        db::usage::ingest(
            &db,
            &[
                Envelope {
                    ts_ms: now_ms,
                    event: Event::RequestStarted {
                        id: "r".into(),
                        key: "spend".into(),
                        model: "m".into(),
                        request: None,
                    },
                },
                Envelope {
                    ts_ms: now_ms,
                    event: Event::Usage {
                        id: "r".into(),
                        key: "spend".into(),
                        model: "m".into(),
                        usage: Usage {
                            prompt_tokens: 1,
                            completion_tokens: 1,
                            total_tokens: 2,
                            cost_usd: Some(10.0),
                        },
                    },
                },
            ],
        )
        .await
        .unwrap();

        // Cap of $5 — the $10 spend trips it.
        pass(&cfg(5.0, 0), &db).await.unwrap();
        let v = db::keys::verdicts(&db).await.unwrap();
        assert!(v.get("spend").and_then(|x| x.suspended_until).is_some());

        // Drop the spend below the cap by pruning the events, then a pass clears the suspension.
        db::usage::prune_before(&db, now_ms + 1).await.unwrap();
        pass(&cfg(5.0, 0), &db).await.unwrap();
        let v = db::keys::verdicts(&db).await.unwrap();
        assert!(v.get("spend").and_then(|x| x.suspended_until).is_none());
    }
}
