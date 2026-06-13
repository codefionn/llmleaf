//! The llmleaf binary.
//!
//! Its whole job is wiring (SOUL.md: "The binary — load config, build the registry, spawn control
//! tasks, serve"):
//!   1. load the config file (the immutable base — principle 6),
//!   2. build the provider registry from `config.providers` using the first-party factory,
//!   3. build the core state, spawn the outbound control tasks (`llmleaf-control`), and serve.
//!
//! The core itself never names a provider and does no outbound control HTTP; this is the one place
//! `kind` strings become instances and the one place the control client is wired in.

use std::process::ExitCode;
use std::sync::Arc;

use llmleaf_core::Config;
use llmleaf_provider::ProviderRegistry;
use tokio_util::sync::CancellationToken;

/// Embedded dev config used only when no config file is present, so `cargo run` works out of the box.
/// Loud warning at startup makes clear this is not for production.
const DEV_CONFIG: &str = r#"
[server]
listen = "127.0.0.1:8080"
admin_token = "dev-admin"

[[providers]]
name = "echo"
kind = "echo"

[[routes]]
model = "echo"
targets = [{ provider = "echo" }]

# Dev key. `pw_hash` is bcrypt of "s3cret" (cost 4); a consumer authenticates with
#   Authorization: Bearer bG9jYWwtZGV2OnMzY3JldA==   # = base64("local-dev:s3cret")
[[keys]]
id = "local-dev"
pw_hash = "$2y$04$IcVq6nhz5Tf85lBpWclgKeDjWxWMHlIXLE696.T7m9Eg12HekWFJO"
name = "local-dev"
"#;

#[tokio::main]
async fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    match run().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            tracing::error!("fatal: {e}");
            ExitCode::FAILURE
        }
    }
}

async fn run() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let config = load_config()?;

    // Build the registry from config. The binary owns the `kind` → implementation mapping, and wires
    // the production transports (reqwest for HTTP, tungstenite for realtime) into every provider — the
    // single place a real network client is injected (tests/benches/sim inject fakes instead).
    let transports = llmleaf_providers::Transports::real();
    let mut registry = ProviderRegistry::new();
    for p in &config.providers {
        let provider = llmleaf_providers::build(&p.kind, &transports).ok_or_else(|| {
            format!(
                "unknown provider kind '{}' for '{}'. known kinds: {:?}",
                p.kind,
                p.name,
                llmleaf_providers::known_kinds()
            )
        })?;
        tracing::info!(name = %p.name, kind = %p.kind, "registered provider");
        registry.register(p.name.clone(), provider);
    }

    // Build the optional sync interceptor (reqwest-backed) and the core state. The interceptor must be
    // wired into the engine at construction, so it is built before the state.
    let interceptor = llmleaf_control::build_interceptor(&config.control);
    let state = llmleaf_core::build_state_with(&config, Arc::new(registry), interceptor)?;
    let listen = config.server.listen.clone();

    // Spawn the outbound control tasks (only those configured). Identity priming runs synchronously
    // inside `start`, before we serve, so a cold node never opens on an empty identity cache.
    let shutdown = CancellationToken::new();
    let control = llmleaf_control::start(
        &config.control,
        state.keys.clone(),
        &state.events,
        shutdown.clone(),
    )
    .await;

    // OAuth2 resource server (`[oauth]`): prime the JWKS pull and wire RFC 7662 introspection before
    // serving, so a cold node fails closed on an empty key set (mirrors identity priming inside `start`).
    let mut oauth_handles = Vec::new();
    if let Some(oauth_cfg) = &config.oauth {
        if let Some(verifier) = state.oauth.clone() {
            oauth_handles =
                llmleaf_control::start_oauth(oauth_cfg, verifier, shutdown.clone()).await;
        }
    }

    // Serve with graceful shutdown wired to ctrl-c / SIGTERM, which also cancels the control tasks.
    let shutdown_signal = {
        let token = shutdown.clone();
        async move {
            wait_for_shutdown().await;
            tracing::info!("shutdown signal received; draining");
            token.cancel();
        }
    };
    let result = llmleaf_core::serve_with_state(state, &listen, shutdown_signal).await;

    // The listener has stopped; make sure the background tasks are cancelled and drained.
    shutdown.cancel();
    control.join().await;
    for h in oauth_handles {
        let _ = h.await;
    }
    result
}

/// Resolve when the process should begin shutting down: ctrl-c on any platform, plus SIGTERM on unix.
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

/// Resolve the config: an explicit path argument, else `LLMLEAF_CONFIG`, else `./llmleaf.toml`,
/// else the embedded dev config.
fn load_config() -> Result<Config, Box<dyn std::error::Error + Send + Sync>> {
    let explicit = std::env::args()
        .nth(1)
        .or_else(|| std::env::var("LLMLEAF_CONFIG").ok());

    if let Some(path) = explicit {
        tracing::info!(%path, "loading config");
        return Ok(Config::from_path(&path)?);
    }

    let default_path = "llmleaf.toml";
    if std::path::Path::new(default_path).exists() {
        tracing::info!(path = default_path, "loading config");
        return Ok(Config::from_path(default_path)?);
    }

    tracing::warn!("no config file found — starting with the embedded DEV config (echo provider, key 'sk-local'). NOT for production.");
    Ok(Config::from_toml_str(DEV_CONFIG)?)
}
