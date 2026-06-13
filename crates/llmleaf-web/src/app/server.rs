//! Server functions — the typed RPC the browser UI calls. On the server each body runs against the DB
//! (after an auth check); on the client each is a `fetch` to `/api/...`. Bodies are compiled only under
//! `ssr`, so the server-only imports live inside each function.

use leptos::prelude::*;

use crate::dto::{
    Accounting, CoreStatus, EventRow, KeyRow, NewKey, NewKeyResult, Overview, Session, VerdictInput,
};

/// Who is logged in (drives the nav + the client-side auth gate). `Ok(None)` when not logged in.
#[server(name = Whoami, prefix = "/api")]
pub async fn whoami() -> Result<Option<Session>, ServerFnError> {
    use crate::auth;
    Ok(auth::current_session().await)
}

#[server(name = GetOverview, prefix = "/api")]
pub async fn get_overview() -> Result<Overview, ServerFnError> {
    use crate::{auth, db, state::AppState};
    auth::require_session().await?;
    let st = expect_context::<AppState>();
    let db = &st.db;
    let day = db::usage::since_secs_ago(86_400);
    const HOUR_MS: u64 = 3_600_000; // dashboard series bucket

    let totals_24h = db::usage::totals_since(db, day).await.map_err(to_err)?;
    let totals_all = db::usage::totals_since(db, 0).await.map_err(to_err)?;
    let series_24h = db::usage::series(db, day, HOUR_MS).await.map_err(to_err)?;
    let top_models = db::usage::by_model(db, day, 8).await.map_err(to_err)?;
    let keys = db::keys::list_rows(db).await.map_err(to_err)?;

    Ok(Overview {
        totals_24h,
        totals_all,
        active_keys: keys.len() as u64,
        blocked_keys: keys.iter().filter(|k| k.blocked).count() as u64,
        series_24h,
        top_models,
    })
}

#[server(name = ListKeys, prefix = "/api")]
pub async fn list_keys() -> Result<Vec<KeyRow>, ServerFnError> {
    use crate::{auth, db, state::AppState};
    auth::require_session().await?;
    let st = expect_context::<AppState>();
    db::keys::list_rows(&st.db).await.map_err(to_err)
}

#[server(name = CreateKey, prefix = "/api")]
pub async fn create_key(input: NewKey) -> Result<NewKeyResult, ServerFnError> {
    use base64::Engine;

    use crate::{auth, db, state::AppState};
    auth::require_session().await?;
    let st = expect_context::<AppState>();

    let id = input.id.trim().to_string();
    if id.is_empty() {
        return Err(ServerFnError::new("key id is required"));
    }
    // Generate a strong password when none was supplied; it is shown to the operator exactly once.
    let password = if input.password.trim().is_empty() {
        db::session::random_token()
    } else {
        input.password.trim().to_string()
    };
    let hash = auth::password::hash_consumer_password(&password)
        .map_err(|e| ServerFnError::new(format!("hashing failed: {e:?}")))?;
    let models = parse_models(&input.allowed_models);
    let name = input
        .name
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty());

    db::keys::create(&st.db, &id, &hash, name, &models)
        .await
        .map_err(|e| ServerFnError::new(format!("could not create key '{id}': {e}")))?;

    // The consumer presents `Authorization: Bearer base64("id:password")` (the core's token format).
    let bearer = base64::engine::general_purpose::STANDARD.encode(format!("{id}:{password}"));
    Ok(NewKeyResult {
        id,
        bearer_token: bearer,
        password,
    })
}

#[server(name = DeleteKey, prefix = "/api")]
pub async fn delete_key(id: String) -> Result<(), ServerFnError> {
    use crate::{auth, db, state::AppState};
    auth::require_session().await?;
    let st = expect_context::<AppState>();
    db::keys::delete(&st.db, &id).await.map_err(to_err)
}

#[server(name = SetVerdict, prefix = "/api")]
pub async fn set_verdict(input: VerdictInput) -> Result<(), ServerFnError> {
    use crate::{auth, db, state::AppState};
    auth::require_session().await?;
    let st = expect_context::<AppState>();
    let models = parse_models(&input.allowed_models);
    let suspended = input.suspended_until.filter(|&u| u > 0);
    db::keys::set_verdict(
        &st.db,
        &input.key_id,
        input.blocked,
        suspended,
        &models,
        "manual",
    )
    .await
    .map_err(to_err)
}

#[server(name = ClearVerdict, prefix = "/api")]
pub async fn clear_verdict(id: String) -> Result<(), ServerFnError> {
    use crate::{auth, db, state::AppState};
    auth::require_session().await?;
    let st = expect_context::<AppState>();
    db::keys::set_verdict(&st.db, &id, false, None, &[], "manual")
        .await
        .map_err(to_err)
}

#[server(name = BlockKey, prefix = "/api")]
pub async fn block_key(id: String, blocked: bool) -> Result<(), ServerFnError> {
    use crate::{auth, db, state::AppState};
    auth::require_session().await?;
    let st = expect_context::<AppState>();
    db::keys::set_blocked(&st.db, &id, blocked)
        .await
        .map_err(to_err)
}

/// Suspend a key for `secs` from now (server clock). `secs = 0` lifts the suspension.
#[server(name = SuspendKey, prefix = "/api")]
pub async fn suspend_key(id: String, secs: u64) -> Result<(), ServerFnError> {
    use crate::{auth, db, state::AppState};
    auth::require_session().await?;
    let st = expect_context::<AppState>();
    let until = (secs > 0).then(|| db::now_secs() + secs);
    db::keys::set_suspended(&st.db, &id, until)
        .await
        .map_err(to_err)
}

#[server(name = NarrowKey, prefix = "/api")]
pub async fn narrow_key(id: String, models: String) -> Result<(), ServerFnError> {
    use crate::{auth, db, state::AppState};
    auth::require_session().await?;
    let st = expect_context::<AppState>();
    db::keys::set_narrow(&st.db, &id, &parse_models(&models))
        .await
        .map_err(to_err)
}

#[server(name = RecentEvents, prefix = "/api")]
pub async fn recent_events(limit: u32) -> Result<Vec<EventRow>, ServerFnError> {
    use crate::{auth, db, state::AppState};
    auth::require_session().await?;
    let st = expect_context::<AppState>();
    db::usage::recent(&st.db, limit.clamp(1, 500))
        .await
        .map_err(to_err)
}

#[server(name = GetAccounting, prefix = "/api")]
pub async fn get_accounting(days: u32) -> Result<Accounting, ServerFnError> {
    use crate::{auth, db, state::AppState};
    auth::require_session().await?;
    let st = expect_context::<AppState>();
    let db = &st.db;
    let days = days.clamp(1, 365);
    let since = db::usage::since_secs_ago(days as u64 * 86_400);
    Ok(Accounting {
        window_days: days,
        totals: db::usage::totals_since(db, since).await.map_err(to_err)?,
        by_key: db::usage::by_key(db, since, 100).await.map_err(to_err)?,
        by_model: db::usage::by_model(db, since, 100).await.map_err(to_err)?,
    })
}

#[server(name = GetCoreStatus, prefix = "/api")]
pub async fn get_core_status() -> Result<CoreStatus, ServerFnError> {
    use crate::{admin, auth, state::AppState};
    auth::require_session().await?;
    let st = expect_context::<AppState>();
    Ok(admin::fetch_status(&st.http, st.config.core.as_ref()).await)
}

/// sqlx error → opaque server error (don't leak SQL to the client).
#[cfg(feature = "ssr")]
fn to_err(e: sqlx::Error) -> ServerFnError {
    tracing::error!(error = %e, "server function db error");
    ServerFnError::new("internal error")
}

/// Split a comma/space/newline-separated model list into trimmed, non-empty ids. Shared by create/verdict.
#[cfg(feature = "ssr")]
fn parse_models(raw: &str) -> Vec<String> {
    raw.split([',', ' ', '\n', '\t'])
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect()
}
