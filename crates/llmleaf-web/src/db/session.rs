//! Operator sessions and transient OIDC-flow state. Sessions are opaque random tokens stored here; the
//! browser cookie carries only the token, so logout and expiry are server-authoritative.

use base64::Engine;
use rand::RngCore;
use sqlx::Row;

use super::{now_ms, Db};
use crate::dto::Session;

/// 256 bits of CSPRNG, URL-safe base64 (no padding). Used for session tokens and OIDC state/nonce.
pub fn random_token() -> String {
    let mut buf = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut buf);
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(buf)
}

/// Create a session and return its token (the cookie value).
pub async fn create_session(
    db: &Db,
    subject: &str,
    method: &str,
    ttl_secs: u64,
) -> Result<String, sqlx::Error> {
    let token = random_token();
    let now = now_ms();
    let expires = now + ttl_secs.saturating_mul(1000);
    sqlx::query(
        "INSERT INTO sessions (token, subject, method, created_ms, expires_ms) VALUES (?, ?, ?, ?, ?)",
    )
    .bind(&token)
    .bind(subject)
    .bind(method)
    .bind(now as i64)
    .bind(expires as i64)
    .execute(db)
    .await?;
    Ok(token)
}

/// Resolve a session token to its identity, or `None` if unknown/expired. Expired rows are swept lazily.
pub async fn lookup_session(db: &Db, token: &str) -> Result<Option<Session>, sqlx::Error> {
    let row = sqlx::query("SELECT subject, method, expires_ms FROM sessions WHERE token = ?")
        .bind(token)
        .fetch_optional(db)
        .await?;
    let Some(row) = row else { return Ok(None) };
    let expires: i64 = row.get("expires_ms");
    if (expires as u64) <= now_ms() {
        // Expired — delete and treat as absent.
        let _ = delete_session(db, token).await;
        return Ok(None);
    }
    Ok(Some(Session {
        subject: row.get("subject"),
        method: row.get("method"),
    }))
}

pub async fn delete_session(db: &Db, token: &str) -> Result<(), sqlx::Error> {
    sqlx::query("DELETE FROM sessions WHERE token = ?")
        .bind(token)
        .execute(db)
        .await?;
    Ok(())
}

/// Delete all expired sessions (periodic housekeeping).
pub async fn prune_sessions(db: &Db) -> Result<u64, sqlx::Error> {
    let res = sqlx::query("DELETE FROM sessions WHERE expires_ms <= ?")
        .bind(now_ms() as i64)
        .execute(db)
        .await?;
    Ok(res.rows_affected())
}

// --- OIDC flow state ---------------------------------------------------------------------------

pub async fn create_oidc_flow(
    db: &Db,
    state: &str,
    code_verifier: &str,
    nonce: &str,
    redirect_to: Option<&str>,
) -> Result<(), sqlx::Error> {
    sqlx::query(
        "INSERT INTO oidc_flows (state, code_verifier, nonce, redirect_to, created_ms)
         VALUES (?, ?, ?, ?, ?)",
    )
    .bind(state)
    .bind(code_verifier)
    .bind(nonce)
    .bind(redirect_to)
    .bind(now_ms() as i64)
    .execute(db)
    .await?;
    Ok(())
}

/// Take (and delete) a pending OIDC flow by `state`. Single-use by construction — the row is gone after
/// this, so a replayed callback finds nothing.
pub async fn take_oidc_flow(
    db: &Db,
    state: &str,
) -> Result<Option<(String, String, Option<String>)>, sqlx::Error> {
    let row = sqlx::query(
        "DELETE FROM oidc_flows WHERE state = ? RETURNING code_verifier, nonce, redirect_to",
    )
    .bind(state)
    .fetch_optional(db)
    .await?;
    Ok(row.map(|r| {
        (
            r.get::<String, _>("code_verifier"),
            r.get::<String, _>("nonce"),
            r.get::<Option<String>, _>("redirect_to"),
        )
    }))
}

/// Drop flow rows older than `max_age_ms` (abandoned logins).
pub async fn prune_oidc_flows(db: &Db, max_age_ms: u64) -> Result<u64, sqlx::Error> {
    let cutoff = now_ms().saturating_sub(max_age_ms) as i64;
    let res = sqlx::query("DELETE FROM oidc_flows WHERE created_ms < ?")
        .bind(cutoff)
        .execute(db)
        .await?;
    Ok(res.rows_affected())
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn mem() -> Db {
        super::super::connect("sqlite::memory:").await.unwrap()
    }

    #[tokio::test]
    async fn session_lifecycle() {
        let db = mem().await;
        let token = create_session(&db, "alice", "password", 3600)
            .await
            .unwrap();
        let s = lookup_session(&db, &token).await.unwrap().unwrap();
        assert_eq!(s.subject, "alice");
        assert_eq!(s.method, "password");
        delete_session(&db, &token).await.unwrap();
        assert!(lookup_session(&db, &token).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn expired_session_is_swept() {
        let db = mem().await;
        let token = create_session(&db, "bob", "password", 0).await.unwrap();
        // ttl 0 => already expired.
        assert!(lookup_session(&db, &token).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn oidc_flow_is_single_use() {
        let db = mem().await;
        create_oidc_flow(&db, "state123", "verifier", "nonce", Some("/keys"))
            .await
            .unwrap();
        let taken = take_oidc_flow(&db, "state123").await.unwrap().unwrap();
        assert_eq!(taken.0, "verifier");
        assert_eq!(taken.2.as_deref(), Some("/keys"));
        // Second take finds nothing.
        assert!(take_oidc_flow(&db, "state123").await.unwrap().is_none());
    }
}
