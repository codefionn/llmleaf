//! Ingest of the pushed usage/lifecycle events, and the aggregate queries the dashboards run over them.
//! The web app is the downstream "others account" half of SOUL.md principle 5: the core observes and
//! forgets; here we store and aggregate.

use sqlx::Row;

use super::{now_secs, Db};
use crate::dto::{EventRow, UsageBy, UsagePoint, UsageTotals};
use crate::wire::{Envelope, Event};

/// Insert a pushed batch of events in one transaction. Unknown variants are stored as `kind='unknown'`
/// so nothing is silently dropped. Returns the number of rows written.
pub async fn ingest(db: &Db, events: &[Envelope]) -> Result<u64, sqlx::Error> {
    let mut tx = db.begin().await?;
    let mut n = 0u64;
    for env in events {
        let ts = env.ts_ms as i64;
        let (kind, request_id, key_id, model, provider, pt, ct, tt, cost, detail) =
            decompose(&env.event);
        sqlx::query(
            "INSERT INTO events
               (ts_ms, kind, request_id, key_id, model, provider,
                prompt_tokens, completion_tokens, total_tokens, cost, detail)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(ts)
        .bind(kind)
        .bind(request_id)
        .bind(key_id)
        .bind(model)
        .bind(provider)
        .bind(pt)
        .bind(ct)
        .bind(tt)
        .bind(cost)
        .bind(detail)
        .execute(&mut *tx)
        .await?;
        n += 1;
    }
    tx.commit().await?;
    Ok(n)
}

type Decomposed = (
    &'static str,   // kind
    Option<String>, // request_id
    Option<String>, // key_id
    Option<String>, // model
    Option<String>, // provider
    i64,            // prompt_tokens
    i64,            // completion_tokens
    i64,            // total_tokens
    f64,            // cost
    Option<String>, // detail
);

fn decompose(event: &Event) -> Decomposed {
    match event {
        Event::RequestStarted { id, key, model, .. } => (
            "request_started",
            Some(id.clone()),
            Some(key.clone()),
            Some(model.clone()),
            None,
            0,
            0,
            0,
            0.0,
            None,
        ),
        Event::RequestRouted {
            id,
            provider,
            upstream_model,
        } => (
            "request_routed",
            Some(id.clone()),
            None,
            None,
            Some(provider.clone()),
            0,
            0,
            0,
            0.0,
            Some(upstream_model.clone()),
        ),
        Event::Usage {
            id,
            key,
            model,
            usage,
        } => (
            "usage",
            Some(id.clone()),
            Some(key.clone()),
            Some(model.clone()),
            None,
            usage.prompt_tokens as i64,
            usage.completion_tokens as i64,
            usage.total_tokens as i64,
            usage.cost_usd.unwrap_or(0.0),
            Some(format!("{} tok", usage.total_tokens)),
        ),
        Event::RequestCompleted { id, finish } => (
            "request_completed",
            Some(id.clone()),
            None,
            None,
            None,
            0,
            0,
            0,
            0.0,
            finish.clone(),
        ),
        Event::RequestFailed { id, error } => (
            "request_failed",
            Some(id.clone()),
            None,
            None,
            None,
            0,
            0,
            0,
            0.0,
            Some(error.clone()),
        ),
        Event::ProviderHealth { provider, status } => (
            "provider_health",
            None,
            None,
            None,
            Some(provider.clone()),
            0,
            0,
            0,
            0.0,
            Some(status.clone()),
        ),
        Event::Unknown => ("unknown", None, None, None, None, 0, 0, 0, 0.0, None),
    }
}

/// `requests` counts `request_started`; token/cost sums come from `usage` rows. Combined into one pass.
const TOTALS_SELECT: &str = "
    COALESCE(SUM(CASE WHEN kind='request_started' THEN 1 ELSE 0 END), 0) AS requests,
    COALESCE(SUM(CASE WHEN kind='usage' THEN prompt_tokens ELSE 0 END), 0) AS pt,
    COALESCE(SUM(CASE WHEN kind='usage' THEN completion_tokens ELSE 0 END), 0) AS ct,
    COALESCE(SUM(CASE WHEN kind='usage' THEN total_tokens ELSE 0 END), 0) AS tt,
    COALESCE(SUM(CASE WHEN kind='usage' THEN cost ELSE 0 END), 0.0) AS cost
";

fn row_to_totals(r: &sqlx::sqlite::SqliteRow) -> UsageTotals {
    UsageTotals {
        requests: r.get::<i64, _>("requests") as u64,
        prompt_tokens: r.get::<i64, _>("pt") as u64,
        completion_tokens: r.get::<i64, _>("ct") as u64,
        total_tokens: r.get::<i64, _>("tt") as u64,
        cost: r.get::<f64, _>("cost"),
    }
}

/// Totals over events with `ts_ms >= since_ms` (pass 0 for all-time).
pub async fn totals_since(db: &Db, since_ms: u64) -> Result<UsageTotals, sqlx::Error> {
    let sql = format!("SELECT {TOTALS_SELECT} FROM events WHERE ts_ms >= ?");
    let row = sqlx::query(&sql)
        .bind(since_ms as i64)
        .fetch_one(db)
        .await?;
    Ok(row_to_totals(&row))
}

/// A time series bucketed at `bucket_ms`, from `since_ms` to now.
pub async fn series(
    db: &Db,
    since_ms: u64,
    bucket_ms: u64,
) -> Result<Vec<UsagePoint>, sqlx::Error> {
    let bucket = bucket_ms.max(1) as i64;
    let sql = "
        SELECT (ts_ms / ?) * ? AS bucket,
               COALESCE(SUM(CASE WHEN kind='request_started' THEN 1 ELSE 0 END), 0) AS requests,
               COALESCE(SUM(CASE WHEN kind='usage' THEN total_tokens ELSE 0 END), 0) AS tokens,
               COALESCE(SUM(CASE WHEN kind='usage' THEN cost ELSE 0 END), 0.0) AS cost
        FROM events WHERE ts_ms >= ?
        GROUP BY bucket ORDER BY bucket";
    let rows = sqlx::query(sql)
        .bind(bucket)
        .bind(bucket)
        .bind(since_ms as i64)
        .fetch_all(db)
        .await?;
    Ok(rows
        .iter()
        .map(|r| UsagePoint {
            bucket_ms: r.get::<i64, _>("bucket") as u64,
            requests: r.get::<i64, _>("requests") as u64,
            total_tokens: r.get::<i64, _>("tokens") as u64,
            cost: r.get::<f64, _>("cost"),
        })
        .collect())
}

/// Usage grouped by a column (`model` or `key_id`), highest-cost first, limited.
async fn by_column(
    db: &Db,
    column: &str,
    since_ms: u64,
    limit: u32,
) -> Result<Vec<UsageBy>, sqlx::Error> {
    // `column` is a fixed internal literal ("model" | "key_id"), never user input — safe to interpolate.
    let sql = format!(
        "SELECT COALESCE({column}, '(unattributed)') AS label, {TOTALS_SELECT}
         FROM events WHERE ts_ms >= ? AND (kind='usage' OR kind='request_started')
         GROUP BY {column} ORDER BY cost DESC, tt DESC LIMIT ?"
    );
    let rows = sqlx::query(&sql)
        .bind(since_ms as i64)
        .bind(limit as i64)
        .fetch_all(db)
        .await?;
    Ok(rows
        .iter()
        .map(|r| UsageBy {
            label: r.get::<String, _>("label"),
            totals: row_to_totals(r),
        })
        .collect())
}

pub async fn by_model(db: &Db, since_ms: u64, limit: u32) -> Result<Vec<UsageBy>, sqlx::Error> {
    by_column(db, "model", since_ms, limit).await
}

pub async fn by_key(db: &Db, since_ms: u64, limit: u32) -> Result<Vec<UsageBy>, sqlx::Error> {
    by_column(db, "key_id", since_ms, limit).await
}

/// The most recent `limit` events, newest first, for the event-log table.
pub async fn recent(db: &Db, limit: u32) -> Result<Vec<EventRow>, sqlx::Error> {
    let rows = sqlx::query(
        "SELECT ts_ms, kind, request_id, key_id, model, provider, total_tokens, detail
         FROM events ORDER BY id DESC LIMIT ?",
    )
    .bind(limit as i64)
    .fetch_all(db)
    .await?;
    Ok(rows
        .iter()
        .map(|r| {
            let kind: String = r.get("kind");
            let tokens: i64 = r.get("total_tokens");
            let mut detail: String = r.get::<Option<String>, _>("detail").unwrap_or_default();
            if kind == "usage" && detail.is_empty() {
                detail = format!("{tokens} tok");
            }
            EventRow {
                ts_ms: r.get::<i64, _>("ts_ms") as u64,
                kind,
                request_id: r.get::<Option<String>, _>("request_id").unwrap_or_default(),
                key: r.get::<Option<String>, _>("key_id"),
                model: r.get::<Option<String>, _>("model"),
                provider: r.get::<Option<String>, _>("provider"),
                detail,
            }
        })
        .collect())
}

/// Limiter input: a key's `usage` cost and `request_started` count since `since_ms`.
pub async fn key_window(db: &Db, key_id: &str, since_ms: u64) -> Result<(f64, u64), sqlx::Error> {
    let row = sqlx::query(
        "SELECT COALESCE(SUM(CASE WHEN kind='usage' THEN cost ELSE 0 END), 0.0) AS cost,
                COALESCE(SUM(CASE WHEN kind='request_started' THEN 1 ELSE 0 END), 0) AS requests
         FROM events WHERE key_id = ? AND ts_ms >= ?",
    )
    .bind(key_id)
    .bind(since_ms as i64)
    .fetch_one(db)
    .await?;
    Ok((
        row.get::<f64, _>("cost"),
        row.get::<i64, _>("requests") as u64,
    ))
}

/// Distinct key ids seen on the event stream since `since_ms` (limiter scan scope).
pub async fn active_keys(db: &Db, since_ms: u64) -> Result<Vec<String>, sqlx::Error> {
    let rows =
        sqlx::query("SELECT DISTINCT key_id FROM events WHERE key_id IS NOT NULL AND ts_ms >= ?")
            .bind(since_ms as i64)
            .fetch_all(db)
            .await?;
    Ok(rows.iter().map(|r| r.get::<String, _>("key_id")).collect())
}

/// Delete events older than `ts_ms`. Optional retention housekeeping; returns rows removed.
pub async fn prune_before(db: &Db, ts_ms: u64) -> Result<u64, sqlx::Error> {
    let res = sqlx::query("DELETE FROM events WHERE ts_ms < ?")
        .bind(ts_ms as i64)
        .execute(db)
        .await?;
    Ok(res.rows_affected())
}

/// Convenience: window start `secs` seconds before now, in ms (saturating).
pub fn since_secs_ago(secs: u64) -> u64 {
    now_secs().saturating_sub(secs) * 1000
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wire::Usage;

    async fn mem() -> Db {
        super::super::connect("sqlite::memory:").await.unwrap()
    }

    fn ev(ts_ms: u64, event: Event) -> Envelope {
        Envelope { ts_ms, event }
    }

    #[tokio::test]
    async fn ingest_and_totals() {
        let db = mem().await;
        let batch = vec![
            ev(
                1000,
                Event::RequestStarted {
                    id: "r1".into(),
                    key: "k1".into(),
                    model: "gpt-4o".into(),
                    request: None,
                },
            ),
            ev(
                1100,
                Event::Usage {
                    id: "r1".into(),
                    key: "k1".into(),
                    model: "gpt-4o".into(),
                    usage: Usage {
                        prompt_tokens: 10,
                        completion_tokens: 20,
                        total_tokens: 30,
                        cost_usd: Some(0.5),
                    },
                },
            ),
            ev(
                1200,
                Event::RequestCompleted {
                    id: "r1".into(),
                    finish: Some("stop".into()),
                },
            ),
            ev(
                2000,
                Event::RequestStarted {
                    id: "r2".into(),
                    key: "k2".into(),
                    model: "demo".into(),
                    request: None,
                },
            ),
            ev(
                2100,
                Event::Usage {
                    id: "r2".into(),
                    key: "k2".into(),
                    model: "demo".into(),
                    usage: Usage {
                        prompt_tokens: 5,
                        completion_tokens: 5,
                        total_tokens: 10,
                        cost_usd: Some(0.1),
                    },
                },
            ),
        ];
        assert_eq!(ingest(&db, &batch).await.unwrap(), 5);

        let t = totals_since(&db, 0).await.unwrap();
        assert_eq!(t.requests, 2);
        assert_eq!(t.total_tokens, 40);
        assert!((t.cost - 0.6).abs() < 1e-9);

        let by_model = by_model(&db, 0, 10).await.unwrap();
        assert_eq!(by_model.len(), 2);
        assert_eq!(by_model[0].label, "gpt-4o"); // higher cost first

        let (cost, reqs) = key_window(&db, "k1", 0).await.unwrap();
        assert!((cost - 0.5).abs() < 1e-9);
        assert_eq!(reqs, 1);

        let recent = recent(&db, 10).await.unwrap();
        assert_eq!(recent.len(), 5);
        assert_eq!(recent[0].request_id, "r2"); // newest first
    }
}
