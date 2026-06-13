//! The consumer-key roster + verdict overlay: what the core PULLS, and what the operator/limiter set.

use std::collections::HashMap;

use sqlx::Row;

use super::{models_to_json, now_ms, parse_models, Db};
use crate::dto::KeyRow;
use crate::wire::{KeyDto, Verdict};

/// Full row as stored. Internal — projected to [`KeyDto`] (secrets, for the core) or [`KeyRow`] (safe,
/// for the UI) at the boundary.
#[derive(sqlx::FromRow)]
struct KeyRecord {
    id: String,
    pw_hash: String,
    name: Option<String>,
    allowed_models: Option<String>,
    blocked: i64,
    suspended_until: Option<i64>,
    verdict_models: Option<String>,
    #[allow(dead_code)]
    verdict_source: Option<String>,
    created_at_ms: i64,
    #[allow(dead_code)]
    updated_at_ms: i64,
}

impl KeyRecord {
    fn to_dto(&self) -> KeyDto {
        let allowed = parse_models(self.allowed_models.clone());
        KeyDto {
            id: self.id.clone(),
            pw_hash: self.pw_hash.clone(),
            name: self.name.clone(),
            allowed_models: (!allowed.is_empty()).then_some(allowed),
        }
    }

    /// The verdict overlay, or `None` when the key is unrestricted (so the verdict pull stays minimal).
    fn to_verdict(&self) -> Option<Verdict> {
        let blocked = self.blocked != 0;
        let verdict_models = parse_models(self.verdict_models.clone());
        let has_models = !verdict_models.is_empty();
        if !blocked && self.suspended_until.is_none() && !has_models {
            return None;
        }
        Some(Verdict {
            blocked,
            suspended_until: self.suspended_until.map(|v| v as u64),
            allowed_models: has_models.then_some(verdict_models),
        })
    }

    fn to_row(&self, last_seen: Option<u64>) -> KeyRow {
        KeyRow {
            id: self.id.clone(),
            name: self.name.clone(),
            allowed_models: parse_models(self.allowed_models.clone()),
            blocked: self.blocked != 0,
            suspended_until: self.suspended_until.map(|v| v as u64),
            verdict_allowed_models: parse_models(self.verdict_models.clone()),
            created_at_ms: self.created_at_ms as u64,
            last_seen_ms: last_seen,
        }
    }
}

async fn all(db: &Db) -> Result<Vec<KeyRecord>, sqlx::Error> {
    sqlx::query_as::<_, KeyRecord>("SELECT * FROM keys ORDER BY id")
        .fetch_all(db)
        .await
}

/// The identity roster the core's `IdentityRefresher` pulls: `{ "keys": [KeyDto] }`.
pub async fn roster(db: &Db) -> Result<Vec<KeyDto>, sqlx::Error> {
    Ok(all(db).await?.iter().map(KeyRecord::to_dto).collect())
}

/// The verdict overlay the core's `VerdictRefresher` pulls: `{ "verdicts": { id: Verdict } }`. Only
/// restricted keys are emitted.
pub async fn verdicts(db: &Db) -> Result<HashMap<String, Verdict>, sqlx::Error> {
    let mut out = HashMap::new();
    for rec in all(db).await? {
        if let Some(v) = rec.to_verdict() {
            out.insert(rec.id.clone(), v);
        }
    }
    Ok(out)
}

/// Per-key last-seen instants (max event ts), for the UI roster.
async fn last_seen_map(db: &Db) -> Result<HashMap<String, u64>, sqlx::Error> {
    let rows = sqlx::query(
        "SELECT key_id, MAX(ts_ms) AS last FROM events WHERE key_id IS NOT NULL GROUP BY key_id",
    )
    .fetch_all(db)
    .await?;
    let mut map = HashMap::new();
    for r in rows {
        let key: String = r.get("key_id");
        let last: i64 = r.get("last");
        map.insert(key, last as u64);
    }
    Ok(map)
}

/// The UI roster: safe [`KeyRow`]s (no `pw_hash`) with last-seen filled in.
pub async fn list_rows(db: &Db) -> Result<Vec<KeyRow>, sqlx::Error> {
    let seen = last_seen_map(db).await?;
    Ok(all(db)
        .await?
        .iter()
        .map(|rec| rec.to_row(seen.get(&rec.id).copied()))
        .collect())
}

/// Insert a new key. Errors (unique violation) if the id already exists.
pub async fn create(
    db: &Db,
    id: &str,
    pw_hash: &str,
    name: Option<&str>,
    allowed_models: &[String],
) -> Result<(), sqlx::Error> {
    let now = now_ms() as i64;
    sqlx::query(
        "INSERT INTO keys (id, pw_hash, name, allowed_models, blocked, created_at_ms, updated_at_ms)
         VALUES (?, ?, ?, ?, 0, ?, ?)",
    )
    .bind(id)
    .bind(pw_hash)
    .bind(name)
    .bind(models_to_json(allowed_models))
    .bind(now)
    .bind(now)
    .execute(db)
    .await?;
    Ok(())
}

pub async fn delete(db: &Db, id: &str) -> Result<(), sqlx::Error> {
    sqlx::query("DELETE FROM keys WHERE id = ?")
        .bind(id)
        .execute(db)
        .await?;
    Ok(())
}

/// Set the full verdict overlay for a key. `source` is 'manual' or 'limiter'. Passing all-clear values
/// (not blocked, no suspension, no models) effectively clears the verdict.
pub async fn set_verdict(
    db: &Db,
    key_id: &str,
    blocked: bool,
    suspended_until: Option<u64>,
    verdict_models: &[String],
    source: &str,
) -> Result<(), sqlx::Error> {
    let now = now_ms() as i64;
    let cleared = !blocked && suspended_until.is_none() && verdict_models.is_empty();
    sqlx::query(
        "UPDATE keys SET blocked = ?, suspended_until = ?, verdict_models = ?, verdict_source = ?,
         updated_at_ms = ? WHERE id = ?",
    )
    .bind(blocked as i64)
    .bind(suspended_until.map(|v| v as i64))
    .bind(models_to_json(verdict_models))
    .bind(if cleared { None } else { Some(source) })
    .bind(now)
    .bind(key_id)
    .execute(db)
    .await?;
    Ok(())
}

/// Operator toggle: set/clear the hard block, marking the verdict operator-owned. Leaves any suspension
/// or model narrowing untouched.
pub async fn set_blocked(db: &Db, id: &str, blocked: bool) -> Result<(), sqlx::Error> {
    sqlx::query(
        "UPDATE keys SET blocked = ?, verdict_source = 'manual', updated_at_ms = ? WHERE id = ?",
    )
    .bind(blocked as i64)
    .bind(now_ms() as i64)
    .bind(id)
    .execute(db)
    .await?;
    Ok(())
}

/// Operator suspend: suspend until `until` (unix secs), operator-owned. `until = None` lifts it.
pub async fn set_suspended(db: &Db, id: &str, until: Option<u64>) -> Result<(), sqlx::Error> {
    sqlx::query(
        "UPDATE keys SET suspended_until = ?, verdict_source = 'manual', updated_at_ms = ? WHERE id = ?",
    )
    .bind(until.map(|v| v as i64))
    .bind(now_ms() as i64)
    .bind(id)
    .execute(db)
    .await?;
    Ok(())
}

/// Operator model narrowing: set/clear the runtime `allowed_models` verdict overlay, operator-owned.
pub async fn set_narrow(db: &Db, id: &str, models: &[String]) -> Result<(), sqlx::Error> {
    sqlx::query(
        "UPDATE keys SET verdict_models = ?, verdict_source = 'manual', updated_at_ms = ? WHERE id = ?",
    )
    .bind(models_to_json(models))
    .bind(now_ms() as i64)
    .bind(id)
    .execute(db)
    .await?;
    Ok(())
}

/// Limiter helper: suspend a key until `until` (unix secs) with source='limiter', WITHOUT touching a
/// block or model narrowing an operator may have set. No-op if already suspended at least that long.
pub async fn limiter_suspend(db: &Db, key_id: &str, until: u64) -> Result<(), sqlx::Error> {
    let now = now_ms() as i64;
    sqlx::query(
        "UPDATE keys SET suspended_until = ?, verdict_source = 'limiter', updated_at_ms = ?
         WHERE id = ? AND (suspended_until IS NULL OR suspended_until < ?)",
    )
    .bind(until as i64)
    .bind(now)
    .bind(key_id)
    .bind(until as i64)
    .execute(db)
    .await?;
    Ok(())
}

/// Ids of keys the limiter currently has suspended. The limiter must re-scan these even when a key has
/// gone quiet (no recent events), so an expired/no-longer-applicable auto-suspension still gets lifted.
pub async fn limiter_suspended_ids(db: &Db) -> Result<Vec<String>, sqlx::Error> {
    let rows = sqlx::query(
        "SELECT id FROM keys WHERE verdict_source = 'limiter' AND suspended_until IS NOT NULL",
    )
    .fetch_all(db)
    .await?;
    Ok(rows.iter().map(|r| r.get::<String, _>("id")).collect())
}

/// Limiter helper: lift a suspension that the limiter itself set and that has now expired or no longer
/// applies. Never clears an operator ('manual') suspension.
pub async fn limiter_clear(db: &Db, key_id: &str) -> Result<(), sqlx::Error> {
    let now = now_ms() as i64;
    sqlx::query(
        "UPDATE keys SET suspended_until = NULL, verdict_source = NULL, updated_at_ms = ?
         WHERE id = ? AND verdict_source = 'limiter'",
    )
    .bind(now)
    .bind(key_id)
    .execute(db)
    .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn mem() -> Db {
        super::super::connect("sqlite::memory:").await.unwrap()
    }

    #[tokio::test]
    async fn roster_and_verdicts_roundtrip() {
        let db = mem().await;
        create(
            &db,
            "team-a",
            "$2y$12$hash",
            Some("Team A"),
            &["gpt-4o".into()],
        )
        .await
        .unwrap();
        create(&db, "team-b", "$6$hash", None, &[]).await.unwrap();

        let roster = roster(&db).await.unwrap();
        assert_eq!(roster.len(), 2);
        let a = roster.iter().find(|k| k.id == "team-a").unwrap();
        assert_eq!(a.pw_hash, "$2y$12$hash");
        assert_eq!(a.allowed_models.as_deref().unwrap(), ["gpt-4o"]);
        let b = roster.iter().find(|k| k.id == "team-b").unwrap();
        assert_eq!(b.allowed_models, None);

        // No verdicts until one is set.
        assert!(verdicts(&db).await.unwrap().is_empty());

        set_verdict(&db, "team-b", true, None, &[], "manual")
            .await
            .unwrap();
        let v = verdicts(&db).await.unwrap();
        assert_eq!(v.len(), 1);
        assert!(v["team-b"].blocked);
    }

    #[tokio::test]
    async fn limiter_suspend_does_not_clobber_manual_block() {
        let db = mem().await;
        create(&db, "k", "$2y$12$h", None, &[]).await.unwrap();
        set_verdict(&db, "k", true, None, &[], "manual")
            .await
            .unwrap();
        limiter_suspend(&db, "k", 9_999_999_999).await.unwrap();
        let v = verdicts(&db).await.unwrap();
        assert!(v["k"].blocked); // block survives
        assert_eq!(v["k"].suspended_until, Some(9_999_999_999));
        // limiter_clear lifts only the suspension, not the manual block.
        limiter_clear(&db, "k").await.unwrap();
        let v = verdicts(&db).await.unwrap();
        assert!(v["k"].blocked);
        assert_eq!(v["k"].suspended_until, None);
    }
}
