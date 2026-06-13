//! The shared server state, threaded into axum handlers (`State<AppState>`) and provided into the Leptos
//! reactive context so server functions can reach the DB, config, and outbound HTTP client.

use std::sync::Arc;

use axum::extract::FromRef;
use leptos::prelude::LeptosOptions;

use crate::config::WebConfig;
use crate::db::Db;

#[derive(Clone)]
pub struct AppState {
    pub leptos_options: LeptosOptions,
    pub db: Db,
    pub config: Arc<WebConfig>,
    /// Outbound HTTP for OIDC + the core admin GETs (rustls; the workspace TLS backend).
    pub http: reqwest::Client,
    /// The resolved operator master-password hash (config or DEV fallback), verified at login.
    pub master_hash: Arc<Option<String>>,
}

// `leptos_routes`/`with_state` need `LeptosOptions: FromRef<AppState>`.
impl FromRef<AppState> for LeptosOptions {
    fn from_ref(state: &AppState) -> Self {
        state.leptos_options.clone()
    }
}
