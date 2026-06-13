//! llmleaf data plane — the proxy core.
//!
//! This crate is "the core" of SOUL.md: compat surfaces on the front, the extension boundary on the
//! back (it depends on [`llmleaf_provider`], never on a concrete provider), and
//! routing/fallback/key-enforcement/event-emission in between. Small enough to hold in your head.
//!
//! It holds no durable, inbound-mutated state: config is the immutable base, the key store is a
//! node-local cache refreshed by *pulling* identity/verdicts from configured control endpoints, and
//! everything else is *pushed* out the event bus to a configured sink. There is no inbound mutation
//! surface. That is what makes multi-node trivial (principle 9) — each node pulls independently, and
//! killing the control plane leaves the proxy proxying from its last-good cache (config alone).

pub mod admin;
pub mod batch_id;
pub mod compat;
pub mod config;
pub mod engine;
pub mod events;
pub mod keys;
#[cfg(feature = "oauth")]
pub mod oauth;
pub mod route;
pub mod server;

pub use admin::{
    AdminAccess, HealthView, KeysView, Observability, PrefixView, ProviderHealth, RoutesView,
};
pub use config::{
    AuthConfig, AuthScheme, Config, ConfigError, ControlConfig, IdentitySource, InterceptHook,
    InterceptPhase, IntrospectionConfig, LimitsSource, OAuthConfig, OnError, ResolvedAuth, Secret,
    UsageSink,
};
pub use engine::{Engine, EngineError, InterceptOutcome, Interceptor};
pub use events::{Envelope, Event, EventBus};
pub use keys::{AuthError, IdentityInput, KeyStore, KeyView, Verdict};
#[cfg(feature = "oauth")]
pub use oauth::{OAuthVerifier, TokenIntrospector};
pub use route::{HealthTable, Router};
pub use server::{build_router, build_state, build_state_with, serve, serve_with_state, AppState};
