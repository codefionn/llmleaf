//! The read-only admin / observability surface, lifted off the HTTP transport.
//!
//! SOUL.md "Architecture soul": the admin GETs (`/admin/routes|health|keys`) are observability only —
//! there is no mutation path (P5/P9: node-local, read-only, pulled-not-pushed). Historically this logic
//! lived inside the axum handlers, reachable only through a real listener. Here it is a transport-
//! agnostic trait on [`AppState`]: production drives it from the axum handlers (real networking); the
//! zero-dependency simulation drives it directly, with no axum/HTTP/networking, and cross-checks every
//! view against an independent reference model.
//!
//! Every method is a pure read of node-local state (config base + last-pulled cache). It is entirely
//! OFF the consumer hot path (chat/embeddings/…), so trait dispatch here adds no hot-path cost (P1) —
//! and it is implemented on the concrete [`AppState`], so even the admin handlers dispatch statically
//! (no `Box`/`Arc<dyn>` is introduced on any request path).

use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::keys::KeyView;
use crate::server::AppState;

/// Digest an admin token for storage/comparison. The configured token is hashed at startup and the
/// plaintext dropped, so memory (state dumps, debug output, core dumps) never holds it; each
/// presented token is digested the same way and the digests compared in constant time.
pub(crate) fn token_digest(token: &str) -> [u8; 32] {
    Sha256::digest(token.as_bytes()).into()
}

/// Constant-time equality over two digests — no early exit on a prefix match.
fn digest_eq(a: &[u8; 32], b: &[u8; 32]) -> bool {
    a.iter()
        .zip(b.iter())
        .fold(0u8, |acc, (x, y)| acc | (x ^ y))
        == 0
}

/// The outcome of the admin token check — transport-agnostic. The HTTP edge maps this to status codes
/// (`Disabled` → 404, `Unauthorized` → 401); the simulation matches on it directly. There is no
/// `HeaderMap` and no `Response` here: callers pass the *presented* token value (`x-admin-token`),
/// already extracted from whatever transport they speak.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdminAccess {
    /// No admin token is configured ⇒ the read-only admin surface is closed (maps to 404).
    Disabled,
    /// A token is configured but the presented one is missing or wrong (maps to 401).
    Unauthorized,
    /// Configured and matched ⇒ proceed.
    Authorized,
}

/// `GET /admin/routes` body. Field order matches the historical `json!` literal so the serialized bytes
/// are identical: `{"models":[…],"prefixes":[{"prefix":…,"provider":…}]}`.
#[derive(Debug, Clone, Serialize)]
pub struct RoutesView {
    pub models: Vec<String>,
    pub prefixes: Vec<PrefixView>,
}

/// One prefix namespace ⇒ the provider it routes to.
#[derive(Debug, Clone, Serialize)]
pub struct PrefixView {
    pub prefix: String,
    pub provider: String,
}

/// `GET /admin/health` body: `{"providers":[{"provider":…,"down":…}]}`.
#[derive(Debug, Clone, Serialize)]
pub struct HealthView {
    pub providers: Vec<ProviderHealth>,
}

/// One provider's node-local cooldown state at the queried instant. `down` is this node's view only
/// (P9), never a cluster verdict.
#[derive(Debug, Clone, Serialize)]
pub struct ProviderHealth {
    pub provider: String,
    pub down: bool,
}

/// `GET /admin/keys` body: `{"keys":[ <KeyView>… ]}`. [`KeyView`] is the existing secret-free redacted
/// view (no token, no password hash).
#[derive(Debug, Clone, Serialize)]
pub struct KeysView {
    pub keys: Vec<KeyView>,
}

/// The read-only admin / observability surface, decoupled from the HTTP transport. See the module docs
/// for the SOUL rationale. Implemented on the concrete [`AppState`]; never dispatched as `dyn` on a
/// request path.
pub trait Observability {
    /// Decide admin access from the *presented* token (the value of `x-admin-token`, already extracted
    /// from any header by the caller). `None` ⇒ no token presented. The single source of truth for both
    /// the `/admin/*` guard and the `/v1/models` admin-view selector.
    fn admin_access(&self, presented: Option<&str>) -> AdminAccess;

    /// Convenience predicate for `GET /v1/models`: the admin-enhanced view is shown iff
    /// `admin_access(presented) == Authorized`. Never an error — a public consumer with a wrong or
    /// absent admin token still gets the public catalog (the endpoint is public-by-default).
    fn admin_view(&self, presented: Option<&str>) -> bool {
        matches!(self.admin_access(presented), AdminAccess::Authorized)
    }

    /// `/admin/routes` — the explicitly-routed logical models + prefix namespaces.
    fn routes(&self) -> RoutesView;

    /// `/admin/health` — the node-local provider cooldown snapshot at `now` (unix seconds). `now` is a
    /// parameter, never an internal clock read: that is what lets the simulation drive this over
    /// controlled time (it mirrors `KeyStore::authorize(.., now)` and `HealthTable::snapshot(now)`).
    fn health(&self, now: u64) -> HealthView;

    /// `/admin/keys` — the secret-free key roster + verdict overlay snapshot.
    fn keys(&self) -> KeysView;
}

impl Observability for AppState {
    fn admin_access(&self, presented: Option<&str>) -> AdminAccess {
        let Some(expected) = &self.admin_token else {
            return AdminAccess::Disabled;
        };
        match presented {
            // Compare digests, never the plaintext token, and in constant time.
            Some(t) if digest_eq(&token_digest(t), expected.as_ref()) => AdminAccess::Authorized,
            _ => AdminAccess::Unauthorized,
        }
    }

    fn routes(&self) -> RoutesView {
        let topology = self.engine.topology();
        let router = topology.router();
        RoutesView {
            models: router.models().map(str::to_string).collect(),
            prefixes: router
                .prefixes()
                .map(|(prefix, provider)| PrefixView {
                    prefix: prefix.to_string(),
                    provider: provider.to_string(),
                })
                .collect(),
        }
    }

    fn health(&self, now: u64) -> HealthView {
        HealthView {
            providers: self
                .engine
                .health()
                .snapshot(now)
                .into_iter()
                .map(|(provider, down)| ProviderHealth { provider, down })
                .collect(),
        }
    }

    fn keys(&self) -> KeysView {
        KeysView {
            keys: self.keys.snapshot(),
        }
    }
}
