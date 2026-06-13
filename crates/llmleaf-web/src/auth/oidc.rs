//! OIDC SSO login — the standard authorization-code flow with PKCE, by hand over reqwest (rustls) so
//! the workspace pulls no native-tls and reuses the same `jsonwebtoken` the core uses for crypto.
//!
//! Flow: discover endpoints → redirect to the IdP with `state`/`nonce`/PKCE → on callback, exchange the
//! code for an `id_token` → verify its signature against the issuer JWKS and its `iss`/`aud`/`exp`/
//! `nonce` → the `sub` (or `email`) becomes the operator identity. The PKCE verifier + nonce live in the
//! `oidc_flows` table between the two legs (single-use).

use base64::Engine;
use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::config::OidcSection;
use crate::db;
use crate::state::AppState;

#[derive(Deserialize)]
struct Discovery {
    authorization_endpoint: String,
    token_endpoint: String,
    jwks_uri: String,
    issuer: String,
}

#[derive(Deserialize)]
struct TokenResponse {
    id_token: String,
}

#[derive(Deserialize)]
struct IdClaims {
    sub: String,
    #[serde(default)]
    email: Option<String>,
    #[serde(default)]
    nonce: Option<String>,
}

fn b64url(bytes: &[u8]) -> String {
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes)
}

async fn discover(http: &reqwest::Client, issuer: &str) -> Result<Discovery, String> {
    let url = format!(
        "{}/.well-known/openid-configuration",
        issuer.trim_end_matches('/')
    );
    let resp = http
        .get(&url)
        .timeout(std::time::Duration::from_secs(10))
        .send()
        .await
        .map_err(|e| format!("discovery request failed: {e}"))?
        .error_for_status()
        .map_err(|e| format!("discovery returned error: {e}"))?;
    resp.json::<Discovery>()
        .await
        .map_err(|e| format!("discovery parse failed: {e}"))
}

/// Begin login: build the PKCE pair + state/nonce, persist the flow, and return the IdP redirect URL.
pub async fn begin_login(
    state: &AppState,
    oidc: &OidcSection,
    redirect_to: Option<&str>,
) -> Result<String, String> {
    let disc = discover(&state.http, &oidc.issuer).await?;

    let verifier = db::session::random_token(); // 256-bit, URL-safe — a valid PKCE verifier
    let challenge = b64url(&Sha256::digest(verifier.as_bytes()));
    let csrf_state = db::session::random_token();
    let nonce = db::session::random_token();

    db::session::create_oidc_flow(&state.db, &csrf_state, &verifier, &nonce, redirect_to)
        .await
        .map_err(|e| format!("could not persist login flow: {e}"))?;

    let scope = oidc.scopes.join(" ");
    let mut url = reqwest::Url::parse(&disc.authorization_endpoint)
        .map_err(|e| format!("bad authorization_endpoint: {e}"))?;
    url.query_pairs_mut()
        .append_pair("response_type", "code")
        .append_pair("client_id", &oidc.client_id)
        .append_pair("redirect_uri", &oidc.redirect_url)
        .append_pair("scope", &scope)
        .append_pair("state", &csrf_state)
        .append_pair("nonce", &nonce)
        .append_pair("code_challenge", &challenge)
        .append_pair("code_challenge_method", "S256");
    Ok(url.into())
}

/// Complete login from the callback. Returns `(subject, redirect_to)` on success. Verifies the code,
/// the id_token signature/claims, the nonce, and the optional subject allow-list.
pub async fn complete_login(
    state: &AppState,
    oidc: &OidcSection,
    code: &str,
    state_param: &str,
) -> Result<(String, Option<String>), String> {
    // Single-use: taking the flow deletes it, so a replayed callback finds nothing.
    let (verifier, nonce, redirect_to) = db::session::take_oidc_flow(&state.db, state_param)
        .await
        .map_err(|e| format!("flow lookup failed: {e}"))?
        .ok_or("unknown or expired login state")?;

    let disc = discover(&state.http, &oidc.issuer).await?;

    // Exchange the code for tokens (PKCE; client_secret_post when a secret is configured).
    let secret = oidc.client_secret.as_ref().and_then(|s| s.resolve());
    let mut form = vec![
        ("grant_type", "authorization_code"),
        ("code", code),
        ("redirect_uri", oidc.redirect_url.as_str()),
        ("client_id", oidc.client_id.as_str()),
        ("code_verifier", verifier.as_str()),
    ];
    if let Some(sec) = secret.as_deref() {
        form.push(("client_secret", sec));
    }
    let token: TokenResponse = state
        .http
        .post(&disc.token_endpoint)
        .timeout(std::time::Duration::from_secs(10))
        .form(&form)
        .send()
        .await
        .map_err(|e| format!("token request failed: {e}"))?
        .error_for_status()
        .map_err(|e| format!("token endpoint returned error: {e}"))?
        .json()
        .await
        .map_err(|e| format!("token parse failed: {e}"))?;

    let claims = verify_id_token(&state.http, oidc, &disc, &token.id_token).await?;

    // Replay/binding check: the id_token nonce must equal the one we sent.
    match claims.nonce.as_deref() {
        Some(n) if n == nonce => {}
        _ => return Err("id_token nonce mismatch".into()),
    }

    let subject = claims.email.clone().unwrap_or_else(|| claims.sub.clone());
    if !oidc.allowed_subjects.is_empty()
        && !oidc
            .allowed_subjects
            .iter()
            .any(|s| s == &subject || s == &claims.sub)
    {
        return Err(format!("subject '{subject}' is not permitted"));
    }
    Ok((subject, redirect_to))
}

/// Verify the id_token signature against the issuer JWKS and its registered claims.
async fn verify_id_token(
    http: &reqwest::Client,
    oidc: &OidcSection,
    disc: &Discovery,
    id_token: &str,
) -> Result<IdClaims, String> {
    use jsonwebtoken::{decode, decode_header, jwk::JwkSet, Algorithm, DecodingKey, Validation};

    let header = decode_header(id_token).map_err(|e| format!("bad id_token header: {e}"))?;
    // Public-key algorithms only — never accept HS*/none for an IdP-signed token.
    if !matches!(
        header.alg,
        Algorithm::RS256
            | Algorithm::RS384
            | Algorithm::RS512
            | Algorithm::ES256
            | Algorithm::ES384
    ) {
        return Err(format!("disallowed id_token alg {:?}", header.alg));
    }
    let kid = header.kid.ok_or("id_token has no kid")?;

    let jwks: JwkSet = http
        .get(&disc.jwks_uri)
        .timeout(std::time::Duration::from_secs(10))
        .send()
        .await
        .map_err(|e| format!("jwks fetch failed: {e}"))?
        .error_for_status()
        .map_err(|e| format!("jwks returned error: {e}"))?
        .json()
        .await
        .map_err(|e| format!("jwks parse failed: {e}"))?;

    let jwk = jwks
        .keys
        .iter()
        .find(|k| k.common.key_id.as_deref() == Some(kid.as_str()))
        .ok_or("no JWKS key matches the id_token kid")?;
    let key = DecodingKey::from_jwk(jwk).map_err(|e| format!("bad jwks key: {e}"))?;

    let mut validation = Validation::new(header.alg);
    validation.set_audience(&[&oidc.client_id]);
    validation.set_issuer(&[disc.issuer.as_str()]);
    validation.validate_exp = true;

    let data = decode::<IdClaims>(id_token, &key, &validation)
        .map_err(|e| format!("id_token verification failed: {e}"))?;
    Ok(data.claims)
}
