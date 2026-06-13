//! View-model DTOs exchanged between Leptos server functions and the browser. These compile on BOTH
//! sides (wasm + server), so they stay pure-serde and carry NO secrets — a key's `pw_hash` never
//! crosses this boundary. Persistence types live in `db` (ssr-only); these are their safe projections.

use serde::{Deserialize, Serialize};

/// A consumer key as shown in the UI — identity + the operator-set verdict overlay. No `pw_hash`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct KeyRow {
    pub id: String,
    pub name: Option<String>,
    /// Static model allow-list (the identity base). Empty ⇒ all routed models.
    pub allowed_models: Vec<String>,
    /// Verdict overlay ↓
    pub blocked: bool,
    pub suspended_until: Option<u64>,
    /// Runtime narrowing on top of `allowed_models`. Empty ⇒ no extra narrowing.
    pub verdict_allowed_models: Vec<String>,
    pub created_at_ms: u64,
    /// Last time an event referenced this key, if ever seen on the usage stream.
    pub last_seen_ms: Option<u64>,
}

/// Aggregate counters over some window or slice.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq)]
pub struct UsageTotals {
    pub requests: u64,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
    pub cost: f64,
}

/// One point of a time series (a fixed-width bucket, labelled by its start in unix ms).
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq)]
pub struct UsagePoint {
    pub bucket_ms: u64,
    pub requests: u64,
    pub total_tokens: u64,
    pub cost: f64,
}

/// Usage rolled up by an arbitrary string dimension (a key id/name, or a model id).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct UsageBy {
    pub label: String,
    pub totals: UsageTotals,
}

/// A recent lifecycle event, flattened for the event-log table.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct EventRow {
    pub ts_ms: u64,
    /// `request_started` | `request_routed` | `usage` | `request_completed` | `request_failed` | `provider_health`
    pub kind: String,
    pub request_id: String,
    pub key: Option<String>,
    pub model: Option<String>,
    pub provider: Option<String>,
    /// A human-readable extra column (error text, finish reason, token count, health status, …).
    pub detail: String,
}

/// The dashboard's at-a-glance summary.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Overview {
    pub totals_24h: UsageTotals,
    pub totals_all: UsageTotals,
    pub active_keys: u64,
    pub blocked_keys: u64,
    pub series_24h: Vec<UsagePoint>,
    pub top_models: Vec<UsageBy>,
}

/// Read-only mirror of the core's `GET /admin/routes` (`{ models, prefixes:[{prefix,provider}] }`).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct RoutesView {
    pub models: Vec<String>,
    pub prefixes: Vec<PrefixInfo>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct PrefixInfo {
    pub prefix: String,
    pub provider: String,
}

/// Read-only mirror of the core's `GET /admin/health` (`{ providers:[{provider,down}] }`).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct HealthRow {
    pub provider: String,
    pub down: bool,
}

/// Bundles whatever the core admin GETs returned, plus whether the core was reachable at all.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct CoreStatus {
    pub reachable: bool,
    pub routes: RoutesView,
    pub health: Vec<HealthRow>,
    /// Populated when `reachable` is false: why the admin probe failed.
    pub error: Option<String>,
}

// ---------------------------------------------------------------------------------------------
// Mutation inputs (server-function arguments)
// ---------------------------------------------------------------------------------------------

/// Create a new consumer key. The server hashes `password` (bcrypt) and stores only the hash; the
/// plaintext is shown to the operator exactly once in the response and never persisted.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NewKey {
    pub id: String,
    pub name: Option<String>,
    /// Comma/space-separated model ids; empty ⇒ all routed models. Parsed server-side.
    pub allowed_models: String,
    /// If empty, the server generates a strong password and returns it once.
    pub password: String,
}

/// What `create_key` hands back: the ready-to-use bearer token, shown ONCE.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NewKeyResult {
    pub id: String,
    /// `base64("id:password")` — exactly the `Authorization: Bearer <token>` the consumer presents.
    pub bearer_token: String,
    /// The generated (or supplied) plaintext password, echoed once for the operator to copy.
    pub password: String,
}

/// Set/clear the verdict overlay for a key (the limiter role, done by hand from the UI).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VerdictInput {
    pub key_id: String,
    pub blocked: bool,
    /// Unix seconds; `None`/0 clears the suspension.
    pub suspended_until: Option<u64>,
    /// Comma/space-separated; empty ⇒ clear the runtime narrowing.
    pub allowed_models: String,
}

/// Accounting rollup over a window: grand totals plus per-key and per-model breakdowns.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Accounting {
    pub window_days: u32,
    pub totals: UsageTotals,
    pub by_key: Vec<UsageBy>,
    pub by_model: Vec<UsageBy>,
}

/// The identity of the logged-in operator (for the nav bar / auth gate).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Session {
    pub subject: String,
    /// "password" or "oidc".
    pub method: String,
}
