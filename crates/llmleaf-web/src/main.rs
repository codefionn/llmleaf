//! The llmleaf-web server binary. Wiring only:
//!   1. load config, open + migrate the SQLite store,
//!   2. build `AppState` (DB, config, HTTP client, resolved master-password hash),
//!   3. assemble the axum app — Leptos UI (with state in context) + server-function endpoint +
//!      machine control endpoints + auth handlers, all behind the session gate,
//!   4. spawn the limiter/housekeeping loop and serve with graceful shutdown.

#[cfg(feature = "ssr")]
#[tokio::main]
async fn main() {
    use std::sync::Arc;

    use axum::routing::get;
    use axum::Router;
    use leptos::logging::log;
    use leptos::prelude::*;
    use leptos_axum::{generate_route_list, LeptosRoutes};
    use llmleaf_web::app::{shell, App};
    use llmleaf_web::state::AppState;
    use llmleaf_web::{auth, config::WebConfig, control, db, limiter};
    use tokio_util::sync::CancellationToken;

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let config = Arc::new(WebConfig::load());

    let db = match db::connect(&config.server.database_url).await {
        Ok(db) => db,
        Err(e) => {
            tracing::error!(error = %e, url = %config.server.database_url, "failed to open database");
            std::process::exit(1);
        }
    };
    let master_hash = Arc::new(config.auth.resolved_password_hash());
    if config.control.token.is_none() {
        tracing::warn!(
            "[control].token is unset — the identity/verdicts/usage endpoints are OPEN. Set it in production."
        );
    }

    // Leptos config: site address, output paths, etc., from cargo-leptos env / Cargo.toml metadata.
    let conf = get_configuration(None).expect("failed to read leptos configuration");
    let leptos_options = conf.leptos_options;
    let addr = leptos_options.site_addr;
    let routes = generate_route_list(App);

    let state = AppState {
        leptos_options: leptos_options.clone(),
        db: db.clone(),
        config: config.clone(),
        http: reqwest::Client::new(),
        master_hash,
    };

    let shutdown = CancellationToken::new();
    let limiter_handle = limiter::spawn(config.clone(), db.clone(), shutdown.clone());

    // Per-request context: make `AppState` available to server functions and route rendering.
    let app = Router::<AppState>::new()
        .route(
            "/api/{*fn_name}",
            get(server_fn_handler).post(server_fn_handler),
        )
        .merge(control::router())
        .merge(auth::router())
        .route("/healthz", get(|| async { "ok" }))
        .leptos_routes_with_context(
            &state,
            routes,
            {
                let state = state.clone();
                move || provide_context(state.clone())
            },
            {
                let opts = leptos_options.clone();
                move || shell(opts.clone())
            },
        )
        .fallback(leptos_axum::file_and_error_handler::<AppState, _>(shell))
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            auth::gate,
        ))
        .with_state(state);

    log!("llmleaf-web listening on http://{}", &addr);
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .expect("failed to bind listener");

    let server = axum::serve(listener, app.into_make_service());
    let graceful = server.with_graceful_shutdown(async move {
        wait_for_shutdown().await;
        tracing::info!("shutdown signal received; draining");
        shutdown.cancel();
    });
    if let Err(e) = graceful.await {
        tracing::error!(error = %e, "server error");
    }
    let _ = limiter_handle.await;
}

/// Dispatch a server-function call with `AppState` provided into context (so the fn can reach the DB).
#[cfg(feature = "ssr")]
async fn server_fn_handler(
    axum::extract::State(state): axum::extract::State<llmleaf_web::state::AppState>,
    request: axum::extract::Request,
) -> impl axum::response::IntoResponse {
    use leptos::prelude::provide_context;
    leptos_axum::handle_server_fns_with_context(move || provide_context(state.clone()), request)
        .await
}

/// Resolve when to begin shutdown: ctrl-c anywhere, plus SIGTERM on unix.
#[cfg(feature = "ssr")]
async fn wait_for_shutdown() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};
        match signal(SignalKind::terminate()) {
            Ok(mut term) => {
                tokio::select! {
                    _ = ctrl_c => {}
                    _ = term.recv() => {}
                }
            }
            Err(_) => ctrl_c.await,
        }
    }
    #[cfg(not(unix))]
    {
        ctrl_c.await;
    }
}

#[cfg(not(feature = "ssr"))]
pub fn main() {
    // Pure client-side builds hydrate via `lib::hydrate`; there is no client `main`.
}
