//! Shared connection plumbing for the llmleaf examples (`chat`, `tts`, `stt`).
//!
//! Each example is a *consumer* of the proxy's OpenAI-compatible surfaces. This module turns the
//! server's own `llmleaf.toml` into the things a client needs — where to connect, what bearer to
//! present, and which logical model to ask for — so each example stays focused on its modality.
//!
//! It is included into every example with `#[path = "common/mod.rs"] mod common;` (a subdirectory so
//! Cargo does not treat it as an example of its own).
//!
//! ## Authentication
//!
//! A consumer presents `Authorization: Bearer <base64(key-id:password)>` — the HTTP-Basic shape, so
//! the bearer is just `base64("id:password")`. The config stores only the password *hash*, so a
//! client cannot recover its bearer from the config alone; it needs the plaintext. For a zero-setup
//! local demo these default to the dedicated `example-cli` key shipped in `llmleaf.toml`, whose dev
//! password is published there on purpose (a local-only credential — never do this in production).
//! Override for real use, highest precedence first:
//!   - `LLMLEAF_API_KEY` — a full `base64(id:password)` bearer, sent verbatim;
//!   - `LLMLEAF_API_ID` + `LLMLEAF_API_PASSWORD` — id and password the client base64-encodes.

// Each example uses only a subset of these helpers; that's expected for shared example glue.
#![allow(dead_code)]

use llmleaf_core::Config;

pub type BoxError = Box<dyn std::error::Error + Send + Sync>;

/// The dedicated dev key shipped in `llmleaf.toml`. The plaintext lives in the repo on purpose: it is
/// a local example credential with no reach beyond a dev echo route. NEVER ship a plaintext password.
pub const DEV_ID: &str = "example-cli";
pub const DEV_PASSWORD: &str = "llmleaf-dev";

/// Everything an example needs to talk to the server, derived from `llmleaf.toml` + the environment.
pub struct Conn {
    pub config: Config,
    pub base_url: String,
    /// The `base64(id:password)` token to put in `Authorization: Bearer`.
    pub bearer: String,
    /// A log-safe label for the credential (the key id, or `(env bearer)`).
    pub id_label: String,
}

/// Load `llmleaf.toml` (overridable with `LLMLEAF_CONFIG`) and derive connection settings.
pub fn connect() -> Result<Conn, BoxError> {
    let config_path =
        std::env::var("LLMLEAF_CONFIG").unwrap_or_else(|_| "llmleaf.toml".to_string());
    let config = Config::from_path(&config_path).map_err(|e| {
        format!(
            "loading {config_path}: {e}\n\
             hint: copy llmleaf.example.toml to llmleaf.toml (it is gitignored), then start the \
             server with `cargo run -p llmleaf`."
        )
    })?;
    let base_url = base_url(&config);
    let (id_label, bearer) = credentials();
    Ok(Conn {
        config,
        base_url,
        bearer,
        id_label,
    })
}

/// Build the bearer token. `LLMLEAF_API_KEY` (a full bearer) wins; otherwise id+password from the
/// environment or the dev defaults are HTTP-Basic-encoded into `base64(id:password)`.
pub fn credentials() -> (String, String) {
    if let Ok(bearer) = std::env::var("LLMLEAF_API_KEY") {
        return ("(env bearer)".to_string(), bearer);
    }
    let id = std::env::var("LLMLEAF_API_ID").unwrap_or_else(|_| DEV_ID.to_string());
    let password =
        std::env::var("LLMLEAF_API_PASSWORD").unwrap_or_else(|_| DEV_PASSWORD.to_string());
    let bearer = base64_encode(format!("{id}:{password}").as_bytes());
    (id, bearer)
}

/// Derive the client base URL from the server's listen address. A wildcard bind (`0.0.0.0`, `[::]`)
/// is not dialable, so rewrite *only an exact wildcard host* to loopback (never a substring, so an
/// address like `10.0.0.0:8080` is left intact). `LLMLEAF_BASE_URL` overrides entirely.
pub fn base_url(config: &Config) -> String {
    if let Ok(url) = std::env::var("LLMLEAF_BASE_URL") {
        return url.trim_end_matches('/').to_string();
    }
    let listen = config.server.listen.trim();

    // A whole-address wildcard (no port) is not dialable — map it straight to loopback.
    match listen {
        "0.0.0.0" => return "http://127.0.0.1".to_string(),
        "[::]" | "::" => return "http://[::1]".to_string(),
        _ => {}
    }

    // Split host:port from the right, but never on a `:` *inside* an IPv6 literal's `[..]` brackets
    // (a port-less `[2001:db8::1]` has its last `:` between the brackets, not before a port).
    let (host, port) = match listen.rsplit_once(':') {
        Some((h, _)) if h.starts_with('[') && !h.ends_with(']') => (listen, ""),
        Some((h, p)) => (h, p),
        None => (listen, ""),
    };
    let host = match host {
        "0.0.0.0" => "127.0.0.1",
        "[::]" | "::" => "[::1]",
        h => h,
    };
    match port {
        "" => format!("http://{host}"),
        p => format!("http://{host}:{p}"),
    }
}

/// The logical model to use: an explicit choice, else `LLMLEAF_MODEL`, else the first configured route.
pub fn pick_model(config: &Config, explicit: Option<String>) -> Option<String> {
    explicit
        .or_else(|| std::env::var("LLMLEAF_MODEL").ok())
        .or_else(|| config.routes.first().map(|r| r.model.clone()))
}

/// Pull `error.message` out of an error response body, falling back to the raw body.
pub fn error_detail(body: &str) -> String {
    serde_json::from_str::<serde_json::Value>(body)
        .ok()
        .and_then(|v| {
            v.get("error")
                .and_then(|e| e.get("message"))
                .and_then(|m| m.as_str())
                .map(str::to_string)
        })
        .unwrap_or_else(|| body.to_string())
}

/// Minimal standard-alphabet base64 with `=` padding (RFC 4648). Kept inline so the examples pull no
/// extra crate for one tiny encode; the core's token decoder accepts this exact shape.
pub fn base64_encode(input: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(input.len().div_ceil(3) * 4);
    for chunk in input.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = *chunk.get(1).unwrap_or(&0) as u32;
        let b2 = *chunk.get(2).unwrap_or(&0) as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(ALPHABET[((n >> 18) & 63) as usize] as char);
        out.push(ALPHABET[((n >> 12) & 63) as usize] as char);
        out.push(if chunk.len() > 1 {
            ALPHABET[((n >> 6) & 63) as usize] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            ALPHABET[(n & 63) as usize] as char
        } else {
            '='
        });
    }
    out
}
