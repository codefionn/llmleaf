//! Web-app configuration (the base layer, mirroring the core's config-first ethos). Loaded from a TOML
//! file; every secret may be given inline, or as `env:VAR` / `file:/path` indirection so deployments
//! keep secrets out of the file. With no file present a loud DEV fallback lets `cargo leptos watch`
//! run out of the box — never for production.

use std::path::Path;

use serde::Deserialize;

/// A secret that resolves from the environment, a file, or an inline literal — `env:VAR`, `file:/path`,
/// or the value itself. Mirrors the core's `Secret`.
#[derive(Debug, Clone, Deserialize)]
#[serde(transparent)]
pub struct Secret(String);

impl Secret {
    /// Resolve to the underlying string, reading the env var / file as directed. `None` if the source
    /// is missing or empty.
    pub fn resolve(&self) -> Option<String> {
        let raw = self.0.trim();
        let resolved = if let Some(var) = raw.strip_prefix("env:") {
            std::env::var(var).ok()?
        } else if let Some(path) = raw.strip_prefix("file:") {
            std::fs::read_to_string(path).ok()?.trim().to_string()
        } else {
            raw.to_string()
        };
        (!resolved.is_empty()).then_some(resolved)
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct WebConfig {
    #[serde(default)]
    pub server: ServerSection,
    #[serde(default)]
    pub control: ControlSection,
    #[serde(default)]
    pub core: Option<CoreSection>,
    #[serde(default)]
    pub auth: AuthSection,
    #[serde(default)]
    pub oidc: Option<OidcSection>,
    #[serde(default)]
    pub limiter: LimiterSection,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ServerSection {
    /// sqlx connection string. `?mode=rwc` creates the file if absent.
    #[serde(default = "default_database_url")]
    pub database_url: String,
}

impl Default for ServerSection {
    fn default() -> Self {
        ServerSection {
            database_url: default_database_url(),
        }
    }
}

/// Guards the machine-facing control endpoints (identity/verdicts/usage). The core presents this as
/// `Authorization: Bearer <token>`. Omit ⇒ the endpoints are open (DEV only; logged loudly).
#[derive(Debug, Clone, Default, Deserialize)]
pub struct ControlSection {
    #[serde(default)]
    pub token: Option<Secret>,
}

/// A core node's read-only admin surface, for the dashboard's routes/health mirror. Optional — omit and
/// the UI simply shows the core as "not configured".
#[derive(Debug, Clone, Deserialize)]
pub struct CoreSection {
    /// Base URL, e.g. `http://127.0.0.1:8080`. The admin GETs are appended.
    pub admin_url: String,
    #[serde(default)]
    pub admin_token: Option<Secret>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AuthSection {
    /// crypt(3) MCF hash (bcrypt `$2*$`) of the operator master password. `htpasswd -bnBC 12 x <pw>`.
    #[serde(default)]
    pub master_password_hash: Option<Secret>,
    #[serde(default = "default_session_ttl")]
    pub session_ttl_secs: u64,
}

// Manual `Default` (NOT derived): the derived impl would zero `session_ttl_secs`, expiring every session
// instantly when `[auth]` is omitted. `Default` must agree with the serde field defaults.
impl Default for AuthSection {
    fn default() -> Self {
        AuthSection {
            master_password_hash: None,
            session_ttl_secs: default_session_ttl(),
        }
    }
}

/// OIDC SSO login (optional second operator-auth scheme). Standard authorization-code flow with PKCE;
/// the id_token is verified against the issuer's JWKS.
#[derive(Debug, Clone, Deserialize)]
pub struct OidcSection {
    /// Issuer URL; OIDC discovery is `{issuer}/.well-known/openid-configuration`.
    pub issuer: String,
    pub client_id: String,
    #[serde(default)]
    pub client_secret: Option<Secret>,
    /// Must exactly match a redirect URI registered at the IdP, e.g. `https://host/auth/oidc/callback`.
    pub redirect_url: String,
    #[serde(default = "default_oidc_scopes")]
    pub scopes: Vec<String>,
    /// If non-empty, only these `sub`/`email` values may log in. Empty ⇒ any IdP-authenticated user.
    #[serde(default)]
    pub allowed_subjects: Vec<String>,
}

/// The optional automated limiter (the canonical bolt-on). When enabled, a background loop turns the
/// pushed usage into verdicts (e.g. suspend a key that blew its monthly cost cap), which the core then
/// pulls. Disabled ⇒ verdicts are operator-set only.
#[derive(Debug, Clone, Deserialize)]
pub struct LimiterSection {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_limiter_interval")]
    pub interval_secs: u64,
    /// Per-key rolling 30-day spend cap (USD). Exceeding it suspends the key until the window clears.
    /// `0` ⇒ no cap.
    #[serde(default)]
    pub monthly_cost_cap: f64,
    /// Per-key 24h request cap. Exceeding it suspends the key for the rest of the day. `0` ⇒ no cap.
    #[serde(default)]
    pub daily_request_cap: u64,
}

impl Default for LimiterSection {
    fn default() -> Self {
        LimiterSection {
            enabled: false,
            interval_secs: default_limiter_interval(),
            monthly_cost_cap: 0.0,
            daily_request_cap: 0,
        }
    }
}

fn default_database_url() -> String {
    "sqlite://llmleaf-web.db?mode=rwc".to_string()
}
fn default_session_ttl() -> u64 {
    86_400
}
fn default_oidc_scopes() -> Vec<String> {
    vec!["openid".into(), "email".into(), "profile".into()]
}
fn default_limiter_interval() -> u64 {
    30
}

impl WebConfig {
    pub fn from_toml_str(s: &str) -> Result<Self, toml::de::Error> {
        toml::from_str(s)
    }

    pub fn from_path(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let text = std::fs::read_to_string(path)?;
        Self::from_toml_str(&text)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))
    }

    /// Resolve config the way the binary does: explicit `LLMLEAF_WEB_CONFIG`, else `./llmleaf-web.toml`,
    /// else a DEV default (in-file SQLite, dev master password, open control endpoints) with a warning.
    pub fn load() -> Self {
        if let Ok(path) = std::env::var("LLMLEAF_WEB_CONFIG") {
            match Self::from_path(&path) {
                Ok(c) => {
                    tracing::info!(%path, "loaded llmleaf-web config");
                    return c;
                }
                Err(e) => {
                    tracing::error!(%path, error = %e, "failed to load config; using DEV defaults")
                }
            }
        }
        let default_path = "llmleaf-web.toml";
        if Path::new(default_path).exists() {
            match Self::from_path(default_path) {
                Ok(c) => {
                    tracing::info!(path = default_path, "loaded llmleaf-web config");
                    return c;
                }
                Err(e) => {
                    tracing::error!(path = default_path, error = %e, "failed to load config; using DEV defaults")
                }
            }
        }
        tracing::warn!(
            "no llmleaf-web config found — DEV defaults: SQLite ./llmleaf-web.db, OPEN control endpoints, \
             master password 'llmleaf-dev'. NOT for production."
        );
        Self::dev_default()
    }

    /// DEV fallback. Master password defaults to `llmleaf-dev` (see [`AuthSection::resolved_password_hash`]).
    fn dev_default() -> Self {
        WebConfig {
            server: ServerSection::default(),
            control: ControlSection::default(),
            core: None,
            auth: AuthSection::default(),
            oidc: None,
            limiter: LimiterSection::default(),
        }
    }
}

impl AuthSection {
    /// The bcrypt hash to verify operator logins against. Configured hash if present; otherwise a
    /// freshly-computed hash of the DEV password `llmleaf-dev` so a config-less dev run can still log in.
    pub fn resolved_password_hash(&self) -> Option<String> {
        if let Some(h) = self.master_password_hash.as_ref().and_then(Secret::resolve) {
            return Some(h);
        }
        // DEV fallback: hash the well-known dev password at startup (default cost/variant).
        pwhash::bcrypt::hash("llmleaf-dev").ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_full_config() {
        let cfg = WebConfig::from_toml_str(
            r#"
            [server]
            database_url = "sqlite::memory:"

            [control]
            token = "shared-secret"

            [core]
            admin_url = "http://127.0.0.1:8080"
            admin_token = "admintok"

            [auth]
            master_password_hash = "$2y$12$abc"
            session_ttl_secs = 3600

            [oidc]
            issuer = "https://idp.example.com"
            client_id = "llmleaf-web"
            client_secret = "shh"
            redirect_url = "https://host/auth/oidc/callback"

            [limiter]
            enabled = true
            monthly_cost_cap = 25.0
            daily_request_cap = 10000
        "#,
        )
        .unwrap();
        assert_eq!(cfg.server.database_url, "sqlite::memory:");
        assert_eq!(
            cfg.control.token.unwrap().resolve().unwrap(),
            "shared-secret"
        );
        assert_eq!(cfg.core.unwrap().admin_url, "http://127.0.0.1:8080");
        assert_eq!(cfg.auth.session_ttl_secs, 3600);
        let oidc = cfg.oidc.unwrap();
        assert_eq!(oidc.client_id, "llmleaf-web");
        assert_eq!(oidc.scopes, vec!["openid", "email", "profile"]); // default
        assert!(cfg.limiter.enabled);
        assert_eq!(cfg.limiter.monthly_cost_cap, 25.0);
    }

    #[test]
    fn empty_config_is_all_defaults() {
        let cfg = WebConfig::from_toml_str("").unwrap();
        assert_eq!(cfg.server.database_url, default_database_url());
        assert!(cfg.control.token.is_none());
        assert!(cfg.core.is_none());
        assert!(!cfg.limiter.enabled);
        // DEV password hash is always available so a config-less run can log in.
        assert!(cfg.auth.resolved_password_hash().is_some());
        // Regression: an omitted [auth] section must still carry the real session TTL, not 0 (which
        // would expire every session immediately). The struct Default must match the serde field default.
        assert_eq!(cfg.auth.session_ttl_secs, default_session_ttl());
        assert_eq!(AuthSection::default().session_ttl_secs, 86_400);
    }

    #[test]
    fn secret_resolves_env() {
        std::env::set_var("LLMLEAF_WEB_TEST_SECRET", "resolved-value");
        let s: Secret = Secret("env:LLMLEAF_WEB_TEST_SECRET".to_string());
        assert_eq!(s.resolve().unwrap(), "resolved-value");
    }
}
