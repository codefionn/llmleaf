//! The SQLite persistence layer (ssr-only). All queries are runtime (`sqlx::query*`), so a build needs
//! no live database (no `DATABASE_URL` at compile time). Migrations are embedded from `./migrations`.

use std::str::FromStr;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use sqlx::sqlite::{SqliteConnectOptions, SqlitePool, SqlitePoolOptions};

pub mod keys;
pub mod session;
pub mod usage;

/// The shared handle threaded through the app (in `AppState`, into server-fn context).
pub type Db = SqlitePool;

/// Open the pool (creating the file if missing) and run migrations to current.
pub async fn connect(url: &str) -> Result<Db, sqlx::Error> {
    let opts = SqliteConnectOptions::from_str(url)?
        .create_if_missing(true)
        .busy_timeout(Duration::from_secs(5))
        .foreign_keys(true);
    let pool = SqlitePoolOptions::new()
        .max_connections(5)
        .connect_with(opts)
        .await?;
    sqlx::migrate!("./migrations").run(&pool).await?;
    Ok(pool)
}

pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

pub fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Parse a JSON-array-of-strings column (or NULL) into a Vec, tolerating malformed data as empty.
pub(crate) fn parse_models(raw: Option<String>) -> Vec<String> {
    raw.and_then(|s| serde_json::from_str::<Vec<String>>(&s).ok())
        .unwrap_or_default()
}

/// Serialize a model list back to a column value: `None` (SQL NULL) for an empty list so "no
/// restriction" is represented uniformly.
pub(crate) fn models_to_json(models: &[String]) -> Option<String> {
    if models.is_empty() {
        None
    } else {
        serde_json::to_string(models).ok()
    }
}
