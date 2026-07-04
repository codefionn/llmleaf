//! Configuration — the immutable base of the core (SOUL.md principle 6: "Config is the base; the
//! pulled control plane is a layer"). Everything the core needs to run is defined here: providers,
//! routes, compat surfaces, base keys, and *which* control endpoints to pull from and push to. The
//! core must be fully operable from a config file alone; runtime state (the pulled key roster and
//! verdict overlay) is refreshed into a node-local cache, but the config only ever *names* those
//! endpoints — it never depends on them being up. Omit `[control]` and the file `[[keys]]` are the
//! whole roster.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("reading config {path}: {source}")]
    Io {
        path: String,
        source: std::io::Error,
    },
    #[error("parsing config: {0}")]
    Parse(#[from] toml::de::Error),
    /// A post-parse integrity check failed (e.g. a `[control.*]` sub-table names an `auth` id that no
    /// `[[control.auth]]` defines, or two auth entries share an id). Caught at load so a typo'd
    /// reference fails fast rather than silently sending no credential.
    #[error("invalid config: {0}")]
    Invalid(String),
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub server: ServerConfig,
    /// Provider *instances*. The core stores these opaquely; the binary's factory turns each into a
    /// concrete `Provider` keyed on [`ProviderConfig::kind`]. The core never matches on `kind`.
    pub providers: Vec<ProviderConfig>,
    /// Logical-model routes with ordered fallback targets.
    pub routes: Vec<RouteConfig>,
    /// Base consumer keys (principle 6: config is the base). When `[control.identity]` is configured
    /// the pulled roster layers on top; with no control plane these are the entire roster.
    pub keys: Vec<KeyConfig>,
    /// Outbound control-plane integrations. Omitted entirely ⇒ pure data-plane from config alone.
    #[serde(default)]
    pub control: ControlConfig,
    /// Optional OAuth2 resource-server auth: validate consumer-presented JWT access tokens as a second,
    /// additive scheme beside `[[keys]]`. Omitted ⇒ only static keys authenticate. Parsed even without
    /// the `oauth` build feature (so config files stay portable); a build lacking the feature refuses to
    /// start when this is set, rather than silently ignoring it.
    #[serde(default)]
    pub oauth: Option<OAuthConfig>,
}

impl Config {
    pub fn from_toml_str(s: &str) -> Result<Self, ConfigError> {
        let cfg: Config = toml::from_str(s)?;
        cfg.control.validate()?;
        Ok(cfg)
    }

    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|source| ConfigError::Io {
            path: path.display().to_string(),
            source,
        })?;
        Self::from_toml_str(&text)
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ServerConfig {
    /// Socket address the consumer + admin + event surfaces bind to.
    pub listen: String,
    /// Token guarding the admin API and event stream. `None` ⇒ those surfaces are disabled
    /// (config-only operation is still fully possible — principle 6).
    pub admin_token: Option<Secret>,
    /// Capacity of the in-memory event broadcast ring. Bounded on purpose: the core never stores
    /// events, it relays them; slow consumers lose the oldest, they never back-pressure the hot path.
    pub event_buffer: usize,
    /// Include full canonical payloads in lifecycle events (principle: "configurably including full
    /// payloads so bolt-ons have everything they need"). Off by default — opt-in for archival/replay.
    pub include_payloads: bool,
    /// How long (seconds) a node skips a provider after it fails a request. Node-local, principle 9.
    pub fallback_cooldown_secs: u64,
    /// Upper bound (milliseconds) on how long the hot path may *wait* for rate-limit capacity when
    /// every target on a chain is saturated (see [`RateLimitConfig`]). The engine first falls through
    /// the fallback chain; only if all targets are over their limit does it sleep until the soonest one
    /// frees up, capped here. `0` ⇒ never wait (fall through, then reject with `429` immediately). This
    /// is latency the operator opts into knowingly (principle 1: the hot path is sacred — waiting is
    /// bounded and off by default-ish, never unbounded queueing).
    pub rate_limit_max_wait_ms: u64,
}

impl Default for ServerConfig {
    fn default() -> Self {
        ServerConfig {
            listen: "127.0.0.1:8080".to_string(),
            admin_token: None,
            event_buffer: 1024,
            include_payloads: false,
            fallback_cooldown_secs: 15,
            rate_limit_max_wait_ms: 5000,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderConfig {
    /// Instance name; routing targets refer to this.
    pub name: String,
    /// Which provider implementation to instantiate. Interpreted by the binary's factory, never by
    /// the core (principle 2).
    pub kind: String,
    #[serde(default)]
    pub endpoint: Option<String>,
    #[serde(default)]
    pub credential: Option<Secret>,
    /// Optional namespace prefix. When set, any request for `<prefix>/<model>` that no explicit
    /// route already claims resolves to this provider, with the upstream model id being everything
    /// after the `<prefix>/` segment. Lets a provider expose its whole catalog without a route per
    /// model. Explicit routes always win; among providers the longest matching prefix wins.
    #[serde(default)]
    pub prefix: Option<String>,
    /// Free-form provider-specific settings passed through to the extension.
    #[serde(default)]
    pub settings: Map<String, Value>,
    /// Provider-global rate limits (node-local flow control — principle 9). Applies to *every* request
    /// routed to this instance, across all its models. Omitted ⇒ no global limit. See [`RateLimitConfig`].
    #[serde(default)]
    pub limits: Option<RateLimitConfig>,
    /// Per-model rate limits, keyed by the **upstream** model id (the id this provider sees, i.e. a
    /// route target's `model`, not the consumer's logical model). A request must pass *both* the
    /// provider-global [`Self::limits`] *and* the matching per-model entry (if any) — they compose, the
    /// stricter binds. A model absent from this map is governed by the global limit alone. Empty ⇒ no
    /// per-model limits.
    #[serde(default)]
    pub model_limits: HashMap<String, RateLimitConfig>,
}

/// Node-local rate limits for one scope (a provider instance, or one of its models). Each dimension is
/// independently optional; an omitted dimension is unlimited. This is *flow control toward the upstream*
/// — the same node-local HA family as `fallback_cooldown_secs` (principles 8 and 9), NOT per-key usage
/// accounting (principle 5, which the pulled `[control.limits]` verdicts own). Enforcement is a fast,
/// allocation-free local check on the hot path (principle 1); it never counts usage for billing.
///
/// **Multi-node (principle 9):** every node enforces its *own* slice with no cross-node coordination, so
/// a cluster-wide cap should be divided by the node count (or treated as approximate) — exactly like the
/// node-local cooldown.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RateLimitConfig {
    /// Max requests per minute (a token bucket whose burst capacity is this value). Omitted ⇒ unlimited.
    #[serde(default)]
    pub requests_per_min: Option<u64>,
    /// Max provider-reported tokens per minute (a token bucket). Debited when usage is observed — token
    /// cost is not known at admission — so a burst may overshoot and then throttle until the bucket
    /// refills. Omitted ⇒ unlimited.
    #[serde(default)]
    pub tokens_per_min: Option<u64>,
    /// Max simultaneous in-flight requests (a semaphore). A permit is held for the life of the request's
    /// stream and released when it ends. Omitted ⇒ unlimited.
    #[serde(default)]
    pub max_concurrent: Option<u32>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RouteConfig {
    /// The logical model id consumers request.
    pub model: String,
    /// Ordered list of provider targets. Order *is* the fallback chain (principle 8).
    pub targets: Vec<Target>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Target {
    /// Provider instance name (matches a [`ProviderConfig::name`]).
    pub provider: String,
    /// Upstream model id. Defaults to the route's logical model when omitted.
    #[serde(default)]
    pub model: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KeyConfig {
    /// Stable, non-secret key identifier — the `key-id` half of the consumer token and the log-safe id
    /// the core stamps onto every event for this key. Consumers present `Authorization: Bearer
    /// <base64(id:password)>`; the core decodes it, looks the key up by `id`, and verifies the password
    /// against [`KeyConfig::pw_hash`]. (HTTP-Basic shape: the split is on the first `:`, so `id` cannot
    /// contain a colon but the password may.)
    pub id: String,
    /// The key's password, **hashed** — never the plaintext. A standard Unix/crypt(3) MCF string, e.g.
    /// bcrypt from `htpasswd -bnBC 12 <id> <password>` (`$2y$…`), or a `$1$`/`$5$`/`$6$` shadow hash.
    /// Supports `env:VAR` indirection; a key whose `env:` var is unset is skipped at load.
    pub pw_hash: Secret,
    /// Optional human-friendly label. Defaults to [`KeyConfig::id`] when omitted.
    #[serde(default)]
    pub name: Option<String>,
    /// Static model allow-list. `None` ⇒ all models this node routes; a list containing a bare
    /// `"*"` means the same (the OAuth role-mapping convention). Entries may use `*` as a wildcard
    /// matching any run of characters — `"gpt-*"`, `"openrouter/openai/*"` — alongside exact ids.
    /// A gatekeeper verdict can further restrict at runtime, but never widen past this base.
    #[serde(default)]
    pub allowed_models: Option<Vec<String>>,
}

/// OAuth2 resource-server config: validate consumer-presented JWT access tokens against an identity
/// provider's signing keys (JWKS), authorize models by mapping a roles/groups claim to model sets, and
/// optionally confirm liveness via RFC 7662 introspection. A second, *additive* consumer-auth scheme
/// beside `[[keys]]` — a bearer that is a JWT (three dot-separated segments) is validated here;
/// anything else falls through to the static key store.
///
/// Like keys, this is base authentication config (principle 6): an inline `jwks` makes the JWT path
/// fully operable from the config file alone; naming a `jwks_uri` lets the control plane refresh the
/// signing keys on an interval (analogous to `[[keys]]` seeding the roster that `[control.identity]`
/// refreshes).
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OAuthConfig {
    /// Expected `iss` claim; also the OIDC discovery base (`<issuer>/.well-known/openid-configuration`)
    /// the control plane uses to find `jwks_uri`/introspection when those are not given explicitly.
    pub issuer: String,
    /// Accepted `aud` claim values. A token whose audience matches none is rejected.
    pub audience: Vec<String>,
    /// Explicit JWKS endpoint. `None` ⇒ the control plane discovers it from `issuer`.
    #[serde(default)]
    pub jwks_uri: Option<String>,
    /// Inline JWKS (RFC 7517 `{"keys":[...]}`): the config-base signing keys, so the JWT path works with
    /// no control plane (principle 6). A configured `jwks_uri` pull replaces this set at runtime.
    #[serde(default)]
    pub jwks: Option<Value>,
    /// Allowed signing algorithms (allow-list). Defaults to `["RS256","ES256"]`. `none`/HS* are refused.
    #[serde(default = "default_oauth_algorithms")]
    pub algorithms: Vec<String>,
    /// Claim whose value becomes the log-safe identity (the `KeyId` stamped on events). Default `sub`.
    #[serde(default = "default_identity_claim")]
    pub identity_claim: String,
    /// Claim holding the caller's roles/groups (a JSON array of strings, or a space-delimited string).
    /// Default `roles`.
    #[serde(default = "default_roles_claim")]
    pub roles_claim: String,
    /// Role/group → allowed models. A token's allowed set is the union over its roles. A role mapped to
    /// `["*"]` grants every model the node routes; entries may use `*` as a wildcard (`"gpt-*"`,
    /// `"openrouter/openai/*"`); a token bearing no mapped role can use no model.
    #[serde(default)]
    pub role_models: HashMap<String, Vec<String>>,
    /// JWKS refresh interval (seconds) for the control-plane pull. The first pull primes before the
    /// listener opens (fail-closed cold start).
    #[serde(default = "default_jwks_refresh_secs")]
    pub jwks_refresh_secs: u64,
    /// Per JWKS-fetch timeout, ms.
    #[serde(default = "default_pull_timeout_ms")]
    pub timeout_ms: u64,
    /// Cold-start posture when the JWKS pull is unreachable and no inline `jwks` seeded the cache: like
    /// identity, FAILS CLOSED by default (an empty key set validates nothing, so it admits no one).
    #[serde(default = "default_on_error_deny")]
    pub on_error: OnError,
    /// Optional RFC 7662 token introspection (revocation). Omitted ⇒ liveness rests on the JWT `exp`.
    #[serde(default)]
    pub introspection: Option<IntrospectionConfig>,
}

/// RFC 7662 token introspection: POST the access token to the IdP and trust its `active` answer, cached
/// per token for `cache_ttl_secs` so the hot path is not a network round-trip per request.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IntrospectionConfig {
    /// Introspection endpoint. Required.
    pub url: String,
    /// Credential for the introspection call (the resource server's client secret / bearer). `env:`
    /// indirection supported.
    #[serde(default)]
    pub credential: Option<Secret>,
    /// How long an `active` result is trusted before re-introspecting, seconds.
    #[serde(default = "default_introspect_cache_ttl_secs")]
    pub cache_ttl_secs: u64,
    /// Per-call timeout, ms.
    #[serde(default = "default_pull_timeout_ms")]
    pub timeout_ms: u64,
}

/// Outbound control-plane integration (SOUL.md principle 5, inverted model). Every sub-table is
/// independently optional: omit one ⇒ that hook is disabled ⇒ the core falls back to the config base
/// ([`Config::keys`]) alone, so config-only operation stays fully possible (principle 6). The core
/// PULLS identity and limits on an interval and caches them node-locally (principle 9: each node holds
/// its own derived state, dropped freely, rebuilt on demand); it PUSHES usage out async and never
/// waits on it; it optionally calls one SYNC intercept hook in-flight (the single sanctioned hot-path
/// insertion, principle 1).
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ControlConfig {
    /// Named, reusable outbound auth schemes. Each entry carries a stable `id`; any sub-table below
    /// attaches one with `auth = "<id>"`, so a credential is defined *once* here and shared across
    /// identity/limits/usage/intercept rather than repeated inline. Two schemes (see [`AuthScheme`]):
    /// a standard HTTP `bearer` (`Authorization: Bearer <token>`) or a custom `header`
    /// (`<name>: <value>`). Empty ⇒ sub-tables fall back to their inline `credential` (a bare bearer).
    /// A sub-table may set `auth` *or* `credential`, never both (rejected at load).
    pub auth: Vec<AuthConfig>,
    /// Key identity source (who exists, password hash, base allow-list). Disabled when omitted ⇒ keys
    /// come only from file `[[keys]]`. This is *authentication*: a control-plane blip would admit
    /// unknown callers, so a cold cache FAILS CLOSED by default (`on_error = "deny"`), unlike limits.
    pub identity: Option<IdentitySource>,
    /// Verdict/limits source (the mutable overlay: block, suspend, narrow models). Disabled when
    /// omitted ⇒ every authenticated key runs with an empty verdict. This is *policy on an
    /// already-authenticated key*, so it FAILS OPEN by default (`on_error = "allow"`): a blip keeps the
    /// last-good verdicts serving (principle 8).
    pub limits: Option<LimitsSource>,
    /// Usage/lifecycle push sink. Disabled when omitted ⇒ events are emitted to the in-process bus and
    /// dropped with no subscriber, exactly as today. Async, batched, never back-pressures the hot path.
    pub usage: Option<UsageSink>,
    /// In-flight sync intercept hook. Disabled when omitted (or with no `phases`) ⇒ no payload is ever
    /// shipped off-node on the hot path. Opt-in per route/key; the operator pays its latency knowingly.
    pub intercept: Option<InterceptHook>,
}

impl ControlConfig {
    /// Post-parse integrity check ([`Config::from_toml_str`] runs it): `[[control.auth]]` ids are
    /// unique, and every sub-table's `auth = "<id>"` reference names a defined id and does not also set
    /// an inline `credential` (the two are mutually exclusive — pick the central reference or the
    /// inline shorthand, not both).
    fn validate(&self) -> Result<(), ConfigError> {
        let mut seen = HashSet::with_capacity(self.auth.len());
        for a in &self.auth {
            if a.id.trim().is_empty() {
                return Err(ConfigError::Invalid(
                    "a [[control.auth]] entry has an empty `id`".to_string(),
                ));
            }
            if !seen.insert(a.id.as_str()) {
                return Err(ConfigError::Invalid(format!(
                    "duplicate [[control.auth]] id `{}`",
                    a.id
                )));
            }
            // A custom header name must be a legal HTTP field-name; otherwise reqwest rejects it only at
            // send time, so every outbound control call would fail silently. Catch the typo at load.
            if let AuthScheme::Header { header, .. } = &a.scheme {
                if !is_valid_header_name(header) {
                    return Err(ConfigError::Invalid(format!(
                        "[[control.auth]] id `{}` has invalid header name `{header}` (must be a \
                         non-empty HTTP token: letters, digits, and any of !#$%&'*+-.^_`|~)",
                        a.id
                    )));
                }
            }
        }
        // (sub-table label, its `auth` reference, whether it also set an inline `credential`).
        let refs = [
            (
                "control.identity",
                self.identity.as_ref().and_then(|s| s.auth.as_deref()),
                self.identity
                    .as_ref()
                    .is_some_and(|s| s.credential.is_some()),
            ),
            (
                "control.limits",
                self.limits.as_ref().and_then(|s| s.auth.as_deref()),
                self.limits.as_ref().is_some_and(|s| s.credential.is_some()),
            ),
            (
                "control.usage",
                self.usage.as_ref().and_then(|s| s.auth.as_deref()),
                self.usage.as_ref().is_some_and(|s| s.credential.is_some()),
            ),
            (
                "control.intercept",
                self.intercept.as_ref().and_then(|s| s.auth.as_deref()),
                self.intercept
                    .as_ref()
                    .is_some_and(|s| s.credential.is_some()),
            ),
        ];
        for (label, reference, has_credential) in refs {
            let Some(id) = reference else { continue };
            if has_credential {
                return Err(ConfigError::Invalid(format!(
                    "[{label}] sets both `auth` and `credential`; use one"
                )));
            }
            if !self.auth.iter().any(|a| a.id == id) {
                return Err(ConfigError::Invalid(format!(
                    "[{label}] references unknown auth id `{id}` — define it in [[control.auth]]"
                )));
            }
        }
        Ok(())
    }

    /// The resolved auth a sub-table should attach to its outbound calls: the scheme named by its
    /// `auth = "<id>"` reference (looked up in [`ControlConfig::auth`]) when set, otherwise its inline
    /// `credential` as a bare bearer. `env:VAR` indirection is applied. `None` ⇒ no auth header.
    ///
    /// Callers (in `llmleaf-control`) attach the result to the HTTP request — the core itself stays
    /// HTTP-client-free (principle 2). References are assumed valid: [`ControlConfig::validate`] (run at
    /// load) already rejected dangling ids and the `auth`+`credential` both-set case, so a `None` from a
    /// set reference means only that an `env:` var resolved to nothing.
    pub fn resolve_auth(
        &self,
        reference: Option<&str>,
        fallback: Option<&Secret>,
    ) -> Option<ResolvedAuth> {
        match reference {
            Some(id) => self
                .auth
                .iter()
                .find(|a| a.id == id)
                .and_then(|a| a.scheme.resolve()),
            None => fallback.and_then(Secret::resolve).map(ResolvedAuth::Bearer),
        }
    }
}

/// A named, reusable control-plane auth scheme. Defined once in `[[control.auth]]`, referenced by
/// `auth = "<id>"` from any `[control.*]` sub-table so the same credential is shared, not repeated.
#[derive(Debug, Clone, Deserialize)]
pub struct AuthConfig {
    /// Stable identifier a sub-table references via `auth = "<id>"`. Unique within `[[control.auth]]`.
    pub id: String,
    /// How the credential is presented on the wire (`kind = "bearer"` | `"header"`).
    #[serde(flatten)]
    pub scheme: AuthScheme,
}

/// The wire form of a control-plane credential. `kind = "bearer"` ⇒ `Authorization: Bearer <token>`
/// (the conventional HTTP bearer); `kind = "header"` ⇒ an arbitrary `<header>: <value>` for control
/// planes that authenticate on a custom header (e.g. `X-API-Key`). Both resolve `env:VAR` indirection.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase", deny_unknown_fields)]
pub enum AuthScheme {
    /// Standard HTTP bearer auth.
    Bearer {
        /// The bearer token. `env:VAR` indirection supported.
        token: Secret,
    },
    /// A custom header carrying the credential, e.g. `X-API-Key: <value>`.
    Header {
        /// The header name (e.g. `X-API-Key`).
        header: String,
        /// The header value. `env:VAR` indirection supported.
        value: Secret,
    },
}

impl AuthScheme {
    /// Resolve `env:VAR` indirection into a concrete, ready-to-attach [`ResolvedAuth`]. `None` when the
    /// underlying secret's `env:` var is unset (the credential is simply absent, like an unset key).
    fn resolve(&self) -> Option<ResolvedAuth> {
        match self {
            AuthScheme::Bearer { token } => token.resolve().map(ResolvedAuth::Bearer),
            AuthScheme::Header { header, value } => {
                value.resolve().map(|value| ResolvedAuth::Header {
                    name: header.clone(),
                    value,
                })
            }
        }
    }
}

/// A control-plane auth resolved to concrete strings (env indirection applied), ready for the outbound
/// HTTP client to attach. Produced by [`ControlConfig::resolve_auth`]; attached in `llmleaf-control`
/// (the core stays HTTP-client-free — principle 2).
#[derive(Clone, PartialEq, Eq)]
pub enum ResolvedAuth {
    /// `Authorization: Bearer <token>`.
    Bearer(String),
    /// A custom header: `<name>: <value>`.
    Header { name: String, value: String },
}

// Never print a resolved credential's contents — mirrors [`Secret`]'s redacting `Debug`. The header
// *name* is not secret (it's a fixed identifier like `X-API-Key`); the bearer token and header value
// are. Guards the secret-redaction invariant against a future `?`-format of any struct holding this.
impl std::fmt::Debug for ResolvedAuth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResolvedAuth::Bearer(_) => f.write_str("Bearer(****)"),
            ResolvedAuth::Header { name, .. } => {
                write!(f, "Header {{ name: {name:?}, value: **** }}")
            }
        }
    }
}

/// Whether a hook lets traffic through or refuses it when the control API is unreachable or errors.
/// Per-hook, with security-appropriate defaults (identity ⇒ [`OnError::Deny`], everything else ⇒
/// [`OnError::Allow`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OnError {
    /// Fail toward availability (principle 8): keep the last-good cache and admit. The default for
    /// limits and intercept.
    Allow,
    /// Fail closed: refuse. The default for identity — a cold cache here is an unauthenticated request.
    Deny,
}

/// PULL: key identity. Polled every `refresh_secs`; the response replaces the node-local identity
/// cache (a snapshot swap, principle 9). A failed poll retains the last-good cache.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IdentitySource {
    /// Endpoint the core GETs the key roster from. Required.
    pub url: String,
    /// Inline bearer credential sent as `Authorization: Bearer <secret>`. `env:VAR` indirection
    /// supported. The shorthand for the common case; for a shared or custom-header scheme set `auth`
    /// instead (the two are mutually exclusive).
    #[serde(default)]
    pub credential: Option<Secret>,
    /// Reference a central `[[control.auth]]` entry by its `id`, instead of an inline `credential`.
    /// Lets several sub-tables share one bearer/custom-header scheme. Mutually exclusive with
    /// `credential`; an unknown id is rejected at load.
    #[serde(default)]
    pub auth: Option<String>,
    /// Poll interval, seconds. The first poll runs at startup, before the listener accepts traffic, so a
    /// cold node never serves on an empty identity cache.
    #[serde(default = "default_identity_refresh_secs")]
    pub refresh_secs: u64,
    /// Per-request timeout, ms. A poll that exceeds it counts as a failure (last-good cache retained).
    #[serde(default = "default_pull_timeout_ms")]
    pub timeout_ms: u64,
    /// What an unreachable control API means on *cold start* (no last-good cache yet). FAILS CLOSED by
    /// default: an empty identity cache must reject, never admit. A warm node always serves last-good.
    #[serde(default = "default_on_error_deny")]
    pub on_error: OnError,
}

/// PULL: verdicts/limits. Same shape as identity but fails OPEN by default. Polled every `refresh_secs`;
/// the response replaces the node-local verdict overlay. A key absent from the response carries an empty
/// verdict (no restriction); a verdict only narrows, never widens past the config/identity base.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LimitsSource {
    /// Endpoint the core GETs the verdict roster from. Required. May equal `identity.url`.
    pub url: String,
    #[serde(default)]
    pub credential: Option<Secret>,
    /// Reference a central `[[control.auth]]` entry by `id` instead of an inline `credential` (bearer
    /// or custom-header). Mutually exclusive with `credential`.
    #[serde(default)]
    pub auth: Option<String>,
    #[serde(default = "default_limits_refresh_secs")]
    pub refresh_secs: u64,
    #[serde(default = "default_pull_timeout_ms")]
    pub timeout_ms: u64,
    /// Policy on an authenticated key, so FAILS OPEN by default: a blip keeps last-good verdicts (or no
    /// verdict on cold start) rather than blocking paying traffic (principle 8).
    #[serde(default = "default_on_error_allow")]
    pub on_error: OnError,
}

/// PUSH: usage/lifecycle events. The core batches and POSTs; it never blocks the hot path on the sink.
/// Buffering and payload inclusion are governed by the in-process bus (`server.event_buffer`,
/// `server.include_payloads`) the reporter subscribes to — a full ring drops oldest for a slow sink.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UsageSink {
    /// Endpoint the core POSTs event batches to. Required.
    pub url: String,
    #[serde(default)]
    pub credential: Option<Secret>,
    /// Reference a central `[[control.auth]]` entry by `id` instead of an inline `credential` (bearer
    /// or custom-header). Mutually exclusive with `credential`.
    #[serde(default)]
    pub auth: Option<String>,
    /// Flush a batch at most this often, ms (whichever of `batch_ms`/`batch_max` trips first).
    #[serde(default = "default_batch_ms")]
    pub batch_ms: u64,
    /// Flush early once this many events accrue. Bounds batch size and tail latency to the sink.
    #[serde(default = "default_batch_max")]
    pub batch_max: usize,
    /// Per-flush timeout, ms.
    #[serde(default = "default_push_timeout_ms")]
    pub timeout_ms: u64,
}

/// SYNC: the one configurable hot-path insertion (principle 1, "the single sanctioned insertion is a
/// sync interceptor"). The core POSTs `{phase,key,model,payload}` and waits for `pass|block|rewrite`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InterceptHook {
    /// Endpoint the core POSTs the in-flight payload to. Required.
    pub url: String,
    #[serde(default)]
    pub credential: Option<Secret>,
    /// Reference a central `[[control.auth]]` entry by `id` instead of an inline `credential` (bearer
    /// or custom-header). Mutually exclusive with `credential`.
    #[serde(default)]
    pub auth: Option<String>,
    /// Which phases to intercept. Empty ⇒ the hook never fires (disabled by data). Opt-in is explicit.
    #[serde(default)]
    pub phases: Vec<InterceptPhase>,
    /// Restrict to these key-ids. `None` ⇒ every key (still gated by `phases`). Per-key opt-in.
    #[serde(default)]
    pub keys: Option<Vec<String>>,
    /// Restrict to these logical models. `None` ⇒ every model. Per-route opt-in.
    #[serde(default)]
    pub models: Option<Vec<String>>,
    /// Per-call timeout, ms. The operator pays this latency knowingly. Kept tight by default.
    #[serde(default = "default_intercept_timeout_ms")]
    pub timeout_ms: u64,
    /// A timed-out/unreachable hook FAILS OPEN by default (pass): availability over screening. Set
    /// `deny` for a hard security gate that must never be bypassed by an outage.
    #[serde(default = "default_on_error_allow")]
    pub on_error: OnError,
}

/// When the sync intercept hook fires. Mirrors the lifecycle boundaries the event stream already names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InterceptPhase {
    /// Before the request is routed — block or rewrite the inbound payload.
    Request,
    /// After the response collects — block or rewrite the outbound payload.
    Response,
}

fn default_identity_refresh_secs() -> u64 {
    30
}
fn default_limits_refresh_secs() -> u64 {
    5
}
fn default_pull_timeout_ms() -> u64 {
    2000
}
fn default_push_timeout_ms() -> u64 {
    5000
}
fn default_intercept_timeout_ms() -> u64 {
    1000
}
fn default_batch_ms() -> u64 {
    1000
}
fn default_batch_max() -> usize {
    256
}
fn default_on_error_allow() -> OnError {
    OnError::Allow
}
fn default_on_error_deny() -> OnError {
    OnError::Deny
}
fn default_oauth_algorithms() -> Vec<String> {
    vec!["RS256".to_string(), "ES256".to_string()]
}
fn default_identity_claim() -> String {
    "sub".to_string()
}
fn default_roles_claim() -> String {
    "roles".to_string()
}
fn default_jwks_refresh_secs() -> u64 {
    3600
}
fn default_introspect_cache_ttl_secs() -> u64 {
    30
}

/// Whether `name` is a valid HTTP field-name (RFC 7230 token): non-empty and every byte a `tchar`.
/// Used to validate a custom-header auth scheme at config load, so a malformed name fails fast rather
/// than erroring on every outbound control request at send time.
fn is_valid_header_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"!#$%&'*+-.^_`|~".contains(&b))
}

/// A secret value that is either a literal or an `env:VAR` reference resolved at load time. Keeps
/// real credentials out of the config file when desired.
#[derive(Clone, Deserialize)]
#[serde(transparent)]
pub struct Secret(String);

impl Secret {
    pub fn new(s: impl Into<String>) -> Self {
        Secret(s.into())
    }

    /// Resolve to the concrete secret. `env:VAR` reads the environment; anything else is literal.
    pub fn resolve(&self) -> Option<String> {
        match self.0.strip_prefix("env:") {
            Some(var) => std::env::var(var).ok(),
            None => Some(self.0.clone()),
        }
    }
}

// Never print a secret's contents.
impl std::fmt::Debug for Secret {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Secret(****)")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimal_config_uses_defaults() {
        let cfg = Config::from_toml_str("").unwrap();
        assert_eq!(cfg.server.listen, "127.0.0.1:8080");
        assert!(cfg.providers.is_empty());
    }

    #[test]
    fn full_config_round_trips() {
        let toml = r#"
            [server]
            listen = "0.0.0.0:9000"
            admin_token = "env:LLMLEAF_ADMIN"

            [[providers]]
            name = "openai-main"
            kind = "openai"
            endpoint = "https://api.openai.com/v1"
            credential = "env:OPENAI_API_KEY"
            settings = { organization = "org-x" }

            [[routes]]
            model = "gpt-4o"
            targets = [
              { provider = "openai-main", model = "gpt-4o" },
              { provider = "echo" },
            ]

            [[keys]]
            id = "ci"
            pw_hash = "$2y$04$IcVq6nhz5Tf85lBpWclgKeDjWxWMHlIXLE696.T7m9Eg12HekWFJO"
            name = "ci-team"
            allowed_models = ["gpt-4o"]
        "#;
        let cfg = Config::from_toml_str(toml).unwrap();
        assert_eq!(cfg.server.listen, "0.0.0.0:9000");
        assert_eq!(cfg.providers[0].kind, "openai");
        assert_eq!(cfg.routes[0].targets.len(), 2);
        assert_eq!(cfg.routes[0].targets[1].model, None);
        assert_eq!(cfg.keys[0].id, "ci");
        assert_eq!(cfg.keys[0].name.as_deref(), Some("ci-team"));
        assert_eq!(
            cfg.keys[0].pw_hash.resolve().as_deref(),
            Some("$2y$04$IcVq6nhz5Tf85lBpWclgKeDjWxWMHlIXLE696.T7m9Eg12HekWFJO")
        );
    }

    #[test]
    fn provider_rate_limits_parse_with_defaults_and_per_model() {
        let toml = r#"
            [server]
            rate_limit_max_wait_ms = 2000

            [[providers]]
            name = "openai-main"
            kind = "openai"

            [providers.limits]
            requests_per_min = 10000
            tokens_per_min = 2000000
            max_concurrent = 200

            [providers.model_limits."gpt-4o"]
            requests_per_min = 5000
            max_concurrent = 100
        "#;
        let cfg = Config::from_toml_str(toml).unwrap();
        assert_eq!(cfg.server.rate_limit_max_wait_ms, 2000);
        let p = &cfg.providers[0];
        let g = p.limits.as_ref().unwrap();
        assert_eq!(g.requests_per_min, Some(10000));
        assert_eq!(g.tokens_per_min, Some(2000000));
        assert_eq!(g.max_concurrent, Some(200));
        let m = &p.model_limits["gpt-4o"];
        assert_eq!(m.requests_per_min, Some(5000));
        assert_eq!(m.tokens_per_min, None); // omitted dimension ⇒ unlimited
        assert_eq!(m.max_concurrent, Some(100));
    }

    #[test]
    fn provider_without_limits_defaults_to_none() {
        let cfg = Config::from_toml_str(
            r#"
            [[providers]]
            name = "echo"
            kind = "echo"
        "#,
        )
        .unwrap();
        assert!(cfg.providers[0].limits.is_none());
        assert!(cfg.providers[0].model_limits.is_empty());
        // The wait cap has a sane non-zero default (waiting is the chosen over-limit behavior).
        assert_eq!(cfg.server.rate_limit_max_wait_ms, 5000);
    }

    #[test]
    fn unknown_field_in_rate_limit_is_rejected() {
        let err = Config::from_toml_str(
            r#"
            [[providers]]
            name = "p"
            kind = "echo"
            [providers.limits]
            requests_per_min = 1
            bogus = 2
        "#,
        )
        .unwrap_err();
        assert!(matches!(err, ConfigError::Parse(_)), "got {err:?}");
    }

    #[test]
    fn no_control_section_is_default() {
        let cfg = Config::from_toml_str("").unwrap();
        assert!(cfg.control.identity.is_none());
        assert!(cfg.control.limits.is_none());
        assert!(cfg.control.usage.is_none());
        assert!(cfg.control.intercept.is_none());
    }

    #[test]
    fn no_oauth_section_is_default() {
        assert!(Config::from_toml_str("").unwrap().oauth.is_none());
    }

    #[test]
    fn oauth_section_parses_with_defaults_and_overrides() {
        let toml = r#"
            [oauth]
            issuer = "https://idp.example.com/"
            audience = ["llmleaf"]
            jwks_uri = "https://idp.example.com/jwks"

            [oauth.role_models]
            "llmleaf-power" = ["gpt-4o", "claude-opus-4"]
            "llmleaf-basic" = ["gpt-4o-mini"]

            [oauth.introspection]
            url = "https://idp.example.com/introspect"
            credential = "env:LLMLEAF_INTROSPECT_SECRET"
            cache_ttl_secs = 15
        "#;
        let cfg = Config::from_toml_str(toml).unwrap();
        let oauth = cfg.oauth.expect("oauth parses");
        assert_eq!(oauth.issuer, "https://idp.example.com/");
        assert_eq!(oauth.audience, vec!["llmleaf".to_string()]);
        assert_eq!(
            oauth.jwks_uri.as_deref(),
            Some("https://idp.example.com/jwks")
        );
        // Defaults.
        assert_eq!(
            oauth.algorithms,
            vec!["RS256".to_string(), "ES256".to_string()]
        );
        assert_eq!(oauth.identity_claim, "sub");
        assert_eq!(oauth.roles_claim, "roles");
        assert_eq!(oauth.jwks_refresh_secs, 3600);
        assert_eq!(oauth.on_error, OnError::Deny); // authentication fails closed by default
                                                   // Role mapping + introspection override.
        assert_eq!(
            oauth.role_models["llmleaf-power"],
            vec!["gpt-4o", "claude-opus-4"]
        );
        let intro = oauth.introspection.expect("introspection parses");
        assert_eq!(intro.url, "https://idp.example.com/introspect");
        assert_eq!(intro.cache_ttl_secs, 15);
        assert_eq!(intro.timeout_ms, 2000); // default
    }

    #[test]
    fn control_section_parses_with_defaults_and_overrides() {
        let toml = r#"
            [control.identity]
            url = "https://ctl/keys"
            credential = "env:LLMLEAF_CONTROL_TOKEN"

            [control.limits]
            url = "https://ctl/verdicts"
            refresh_secs = 3
            on_error = "deny"

            [control.usage]
            url = "https://ctl/usage"
            batch_max = 64

            [control.intercept]
            url = "https://ctl/screen"
            phases = ["request"]
            keys = ["demo-team"]
        "#;
        let cfg = Config::from_toml_str(toml).unwrap();

        let identity = cfg.control.identity.unwrap();
        assert_eq!(identity.url, "https://ctl/keys");
        assert_eq!(identity.refresh_secs, 30); // default
        assert_eq!(identity.on_error, OnError::Deny); // identity default fails closed
        assert_eq!(
            identity.credential.unwrap().resolve(),
            std::env::var("LLMLEAF_CONTROL_TOKEN").ok()
        );

        let limits = cfg.control.limits.unwrap();
        assert_eq!(limits.refresh_secs, 3); // override
        assert_eq!(limits.on_error, OnError::Deny); // override (default would be Allow)

        let usage = cfg.control.usage.unwrap();
        assert_eq!(usage.batch_max, 64);
        assert_eq!(usage.batch_ms, 1000); // default

        let intercept = cfg.control.intercept.unwrap();
        assert_eq!(intercept.phases, vec![InterceptPhase::Request]);
        assert_eq!(
            intercept.keys.as_deref(),
            Some(["demo-team".to_string()].as_slice())
        );
        assert_eq!(intercept.on_error, OnError::Allow); // intercept default fails open
    }

    #[test]
    fn control_auth_registry_parses_both_schemes() {
        let toml = r#"
            [[control.auth]]
            id = "ctl-bearer"
            kind = "bearer"
            token = "s3cret-token"

            [[control.auth]]
            id = "ctl-header"
            kind = "header"
            header = "X-API-Key"
            value = "s3cret-value"
        "#;
        let cfg = Config::from_toml_str(toml).unwrap();
        assert_eq!(cfg.control.auth.len(), 2);
        assert!(matches!(
            cfg.control.auth[0].scheme,
            AuthScheme::Bearer { .. }
        ));
        assert!(matches!(
            cfg.control.auth[1].scheme,
            AuthScheme::Header { .. }
        ));
    }

    #[test]
    fn control_sub_table_resolves_referenced_auth() {
        let toml = r#"
            [[control.auth]]
            id = "ctl-bearer"
            kind = "bearer"
            token = "tok"

            [[control.auth]]
            id = "ctl-header"
            kind = "header"
            header = "X-API-Key"
            value = "val"

            [control.identity]
            url = "https://ctl/keys"
            auth = "ctl-bearer"

            [control.usage]
            url = "https://ctl/usage"
            auth = "ctl-header"
        "#;
        let cfg = Config::from_toml_str(toml).unwrap();
        let ctl = &cfg.control;

        let identity = ctl.identity.as_ref().unwrap();
        assert_eq!(
            ctl.resolve_auth(identity.auth.as_deref(), identity.credential.as_ref()),
            Some(ResolvedAuth::Bearer("tok".to_string()))
        );

        let usage = ctl.usage.as_ref().unwrap();
        assert_eq!(
            ctl.resolve_auth(usage.auth.as_deref(), usage.credential.as_ref()),
            Some(ResolvedAuth::Header {
                name: "X-API-Key".to_string(),
                value: "val".to_string(),
            })
        );
    }

    #[test]
    fn inline_credential_resolves_as_bearer_without_reference() {
        let cfg = Config::from_toml_str(
            r#"
            [control.limits]
            url = "https://ctl/verdicts"
            credential = "inline-tok"
        "#,
        )
        .unwrap();
        let limits = cfg.control.limits.as_ref().unwrap();
        assert_eq!(
            cfg.control
                .resolve_auth(limits.auth.as_deref(), limits.credential.as_ref()),
            Some(ResolvedAuth::Bearer("inline-tok".to_string()))
        );
    }

    #[test]
    fn dangling_auth_reference_is_rejected() {
        let err = Config::from_toml_str(
            r#"
            [control.identity]
            url = "https://ctl/keys"
            auth = "missing"
        "#,
        )
        .unwrap_err();
        assert!(matches!(err, ConfigError::Invalid(_)), "got {err:?}");
    }

    #[test]
    fn auth_and_credential_both_set_is_rejected() {
        let err = Config::from_toml_str(
            r#"
            [[control.auth]]
            id = "a"
            kind = "bearer"
            token = "t"

            [control.identity]
            url = "https://ctl/keys"
            auth = "a"
            credential = "also-this"
        "#,
        )
        .unwrap_err();
        assert!(matches!(err, ConfigError::Invalid(_)), "got {err:?}");
    }

    #[test]
    fn invalid_header_name_is_rejected() {
        let err = Config::from_toml_str(
            r#"
            [[control.auth]]
            id = "h"
            kind = "header"
            header = "X Api Key"
            value = "v"
        "#,
        )
        .unwrap_err();
        assert!(matches!(err, ConfigError::Invalid(_)), "got {err:?}");
        // A legal token header name passes.
        assert!(Config::from_toml_str(
            r#"
            [[control.auth]]
            id = "h"
            kind = "header"
            header = "X-Api-Key"
            value = "v"
        "#,
        )
        .is_ok());
    }

    #[test]
    fn empty_auth_id_is_rejected() {
        let err = Config::from_toml_str(
            r#"
            [[control.auth]]
            id = ""
            kind = "bearer"
            token = "t"
        "#,
        )
        .unwrap_err();
        assert!(matches!(err, ConfigError::Invalid(_)), "got {err:?}");
    }

    #[test]
    fn unknown_field_in_auth_entry_is_rejected() {
        let err = Config::from_toml_str(
            r#"
            [[control.auth]]
            id = "a"
            kind = "bearer"
            token = "t"
            bogus = "x"
        "#,
        )
        .unwrap_err();
        assert!(matches!(err, ConfigError::Parse(_)), "got {err:?}");
    }

    #[test]
    fn resolved_auth_debug_redacts_secrets() {
        let bearer = format!("{:?}", ResolvedAuth::Bearer("s3cret-token".to_string()));
        assert!(!bearer.contains("s3cret-token"), "{bearer}");
        let header = format!(
            "{:?}",
            ResolvedAuth::Header {
                name: "X-Api-Key".to_string(),
                value: "s3cret-value".to_string(),
            }
        );
        assert!(!header.contains("s3cret-value"), "{header}");
        // The (non-secret) header name is still visible for diagnostics.
        assert!(header.contains("X-Api-Key"), "{header}");
    }

    #[test]
    fn duplicate_auth_id_is_rejected() {
        let err = Config::from_toml_str(
            r#"
            [[control.auth]]
            id = "dup"
            kind = "bearer"
            token = "t1"

            [[control.auth]]
            id = "dup"
            kind = "bearer"
            token = "t2"
        "#,
        )
        .unwrap_err();
        assert!(matches!(err, ConfigError::Invalid(_)), "got {err:?}");
    }
}
