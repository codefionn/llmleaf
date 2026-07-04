//! OAuth2 resource-server consumer auth: validate consumer-presented JWT access tokens.
//!
//! A second, *additive* scheme beside the static [`KeyStore`](crate::keys::KeyStore). A bearer that
//! looks like a JWT (three dot-separated segments) is routed here; anything else falls through to the
//! key store. The contract is identical — `authorize(token, model, now) -> Result<KeyId, AuthError>` —
//! so the rest of the pipeline (engine, events, accounting) never learns which scheme authenticated a
//! request; an OAuth identity flows as the same log-safe `KeyId`.
//!
//! ## What lives here vs. in the control plane (principle 2)
//!
//! The core is HTTP-client-free. This module does **crypto only**: it verifies a JWT signature against
//! a node-local **JWKS snapshot** and checks claims. The signing keys are *fed in* wholesale via
//! [`OAuthVerifier::install_jwks`] — the exact analogue of [`KeyStore::install_identities`] — by the
//! `JwksRefresher` in `llmleaf-control`, which owns the outbound fetch. An inline `[oauth].jwks` seeds
//! the snapshot at construction so the JWT path is fully operable from config alone (principle 6).
//! Per-request liveness (RFC 7662 introspection) is delegated through the HTTP-free
//! [`TokenIntrospector`] trait, whose reqwest implementation also lives in the control crate — the same
//! shape as the engine's [`Interceptor`](crate::engine::Interceptor).
//!
//! ## Keeping the hot path sacred (principle 1)
//!
//! Signature verification is real asymmetric crypto, so a proven token is cached node-locally
//! (`token -> {identity, allowed-models, exp}`), mirroring the key store's verification cache. The warm
//! path is then a lookup and an `exp` comparison — no crypto, and (with introspection off) no network.

use std::collections::{HashMap, HashSet};
use std::str::FromStr;
use std::sync::{Arc, RwLock};

use jsonwebtoken::jwk::JwkSet;
use jsonwebtoken::{decode, decode_header, Algorithm, DecodingKey, Validation};
use serde_json::Value;

use crate::config::OAuthConfig;
use crate::keys::{AuthError, KeyId};

/// Bound on the proven-token cache. Unlike the key store (one valid token per key), distinct JWTs are
/// unbounded, so we prune expired entries once the map grows past this — live tokens are then bounded
/// by issuance-rate × token-lifetime, not by total traffic.
const MAX_VERIFIED: usize = 16_384;

/// The models a token may use: every routed model (a role mapped to a bare `"*"`), or an explicit
/// set whose entries may be `*` wildcard patterns (`"gpt-*"`, `"openrouter/openai/*"`) — the same
/// matching the key store's allow-lists use ([`crate::keys::allow_set_permits`]).
#[derive(Clone)]
enum AllowSet {
    All,
    Only(HashSet<String>),
}

impl AllowSet {
    fn permits(&self, model: &str) -> bool {
        match self {
            AllowSet::All => true,
            AllowSet::Only(s) => crate::keys::allow_set_permits(s, model),
        }
    }
}

/// Per-request token liveness check (RFC 7662), delegated off-core. HTTP-free trait in the core
/// (principle 2); the reqwest implementation lives in `llmleaf-control` and is wired in by the binary.
/// Mirrors [`crate::engine::Interceptor`].
#[async_trait::async_trait]
pub trait TokenIntrospector: Send + Sync {
    /// `Ok(true)` if the IdP reports the token `active`, `Ok(false)` if it is inactive/revoked. A
    /// network/timeout failure is `Err` so the verifier can apply its fail-closed posture.
    async fn introspect(&self, token: &str) -> Result<bool, String>;
}

/// An atomically-swappable set of ready-to-use decoding keys, keyed by `kid`. Built once from the
/// inline config JWKS (the base) and replaced wholesale by the JWKS pull — never mutated in place.
struct JwksSnapshot {
    /// `kid` → decoding key. A key with no `kid` is stored under the empty string and used as the
    /// fallback when a token header carries no `kid` (or when the set holds exactly one key).
    keys: HashMap<String, DecodingKey>,
}

/// A proven token's cache entry. Valid until its own `exp`; introspection (when enabled) is re-checked
/// once `introspect_ok_until` passes.
struct VerifiedEntry {
    key_id: KeyId,
    allowed: AllowSet,
    /// Token expiry (unix seconds). The entry is dead once `now >= exp` — a comparison, not a clock.
    exp: u64,
    /// While `now < introspect_ok_until` the last `active` introspection is trusted; `0` ⇒ introspection
    /// disabled (the entry then rests purely on `exp`).
    introspect_ok_until: u64,
}

/// Validates consumer JWT access tokens against a node-local JWKS, authorizes models from a roles/groups
/// claim, and (optionally) confirms liveness via introspection. Cheap to share behind an `Arc`.
pub struct OAuthVerifier {
    /// Prebuilt validation (algorithms allow-list, issuer, audience, exp/nbf), passed by reference to
    /// every `decode` call (jsonwebtoken borrows it — no per-decode clone).
    validation: Validation,
    identity_claim: String,
    roles_claim: String,
    /// Role/group → models. A role whose config list contains `"*"` becomes [`AllowSet::All`].
    role_models: HashMap<String, AllowSet>,
    /// `Some` ⇒ introspection is required for this verifier; the trait object is installed at startup.
    introspect_ttl: Option<u64>,
    jwks: RwLock<Arc<JwksSnapshot>>,
    verified: RwLock<HashMap<String, VerifiedEntry>>,
    /// The reqwest-backed introspector, installed by the control plane after construction (mirrors how
    /// the JWKS roster is installed). `None` while unset; if introspection is required and this is still
    /// `None`, the verifier fails closed.
    introspector: RwLock<Option<Arc<dyn TokenIntrospector>>>,
}

impl OAuthVerifier {
    /// Build the verifier from config (principle 6: config is the base). An inline `jwks` seeds the
    /// signing-key snapshot so the JWT path works with no control plane; otherwise the snapshot starts
    /// empty (fail-closed: nothing validates until the JWKS pull primes it). Returns an error only on a
    /// misconfiguration that can never authenticate anything (no usable algorithm, unparseable inline
    /// JWKS).
    pub fn from_config(cfg: &OAuthConfig) -> Result<Self, String> {
        // Allow-list of signing algorithms. This is a *public-key* resource server, so HMAC (HS*) is
        // rejected outright — accepting it alongside RSA keys is the classic alg-confusion footgun.
        let mut algorithms = Vec::new();
        for a in &cfg.algorithms {
            match Algorithm::from_str(a) {
                Ok(alg) if !is_hmac(alg) => algorithms.push(alg),
                Ok(_) => {
                    tracing::warn!(alg = %a, "ignoring HMAC algorithm in [oauth].algorithms (public-key resource server only)")
                }
                Err(_) => return Err(format!("unknown [oauth] algorithm '{a}'")),
            }
        }
        if algorithms.is_empty() {
            return Err("[oauth].algorithms has no usable (non-HMAC) algorithm".to_string());
        }

        let mut validation = Validation::new(algorithms[0]);
        validation.algorithms = algorithms;
        validation.set_issuer(&[&cfg.issuer]);
        validation.set_audience(&cfg.audience);
        validation.validate_nbf = true;
        // A token must carry the claims we validate; otherwise it could omit `aud`/`iss` to dodge the
        // checks. `exp` is required by default — we add issuer and audience.
        validation.set_required_spec_claims(&["exp", "iss", "aud"]);

        let role_models = cfg
            .role_models
            .iter()
            .map(|(role, models)| {
                let set = if models.iter().any(|m| m == "*") {
                    AllowSet::All
                } else {
                    AllowSet::Only(models.iter().cloned().collect())
                };
                (role.clone(), set)
            })
            .collect();

        // Seed the snapshot from the inline JWKS, if any.
        let keys = match &cfg.jwks {
            Some(value) => parse_jwks(value)?,
            None => HashMap::new(),
        };

        Ok(OAuthVerifier {
            validation,
            identity_claim: cfg.identity_claim.clone(),
            roles_claim: cfg.roles_claim.clone(),
            role_models,
            introspect_ttl: cfg.introspection.as_ref().map(|i| i.cache_ttl_secs),
            jwks: RwLock::new(Arc::new(JwksSnapshot { keys })),
            verified: RwLock::new(HashMap::new()),
            introspector: RwLock::new(None),
        })
    }

    /// Install a freshly pulled JWKS, replacing the signing-key set wholesale. Called only by the
    /// control plane's refresher, never on the hot path (mirrors [`KeyStore::install_identities`]).
    /// Already-proven, unexpired tokens stay valid (their signature was checked once; `exp` bounds
    /// them); new tokens validate against the new keys. Returns the number of usable keys installed.
    pub fn install_jwks(&self, jwks_json: &Value) -> Result<usize, String> {
        let keys = parse_jwks(jwks_json)?;
        let n = keys.len();
        *self.jwks.write().unwrap() = Arc::new(JwksSnapshot { keys });
        Ok(n)
    }

    /// Wire in the introspector implementation (the control plane's reqwest client). Set once at
    /// startup, before traffic — mirrors how the JWKS roster is installed from off-core.
    pub fn install_introspector(&self, introspector: Arc<dyn TokenIntrospector>) {
        *self.introspector.write().unwrap() = Some(introspector);
    }

    /// The auth check, mirroring [`KeyStore::authorize`]'s contract. Warm path: a lookup and an `exp`
    /// comparison (plus, if introspection is enabled, a TTL comparison). Cold path: verify the JWT
    /// signature against the JWKS, map claims → identity + allowed models, optionally introspect, cache.
    /// Every failure collapses to [`AuthError::Unknown`] (no oracle) except a model the token's roles do
    /// not grant, which is [`AuthError::ModelNotAllowed`].
    pub async fn authorize(&self, token: &str, model: &str, now: u64) -> Result<KeyId, AuthError> {
        self.authorize_inner(token, Some(model), now).await
    }

    /// Identity-only counterpart, mirroring [`KeyStore::authorize_identity`]: prove the JWT is valid and
    /// live without gating on any model. For the batch retrieve/cancel/results surfaces, which route by
    /// an opaque id and carry no model.
    pub async fn authorize_identity(&self, token: &str, now: u64) -> Result<KeyId, AuthError> {
        self.authorize_inner(token, None, now).await
    }

    async fn authorize_inner(
        &self,
        token: &str,
        model: Option<&str>,
        now: u64,
    ) -> Result<KeyId, AuthError> {
        // Warm path.
        {
            let verified = self.verified.read().unwrap();
            if let Some(entry) = verified.get(token) {
                if now < entry.exp {
                    let introspection_fresh =
                        self.introspect_ttl.is_none() || now < entry.introspect_ok_until;
                    if introspection_fresh {
                        return if model.is_none_or(|m| entry.allowed.permits(m)) {
                            Ok(entry.key_id.clone())
                        } else {
                            Err(AuthError::ModelNotAllowed)
                        };
                    }
                    // Else: token still unexpired but introspection cache is stale — fall through to
                    // re-introspect (the signature need not be re-checked, but keep it simple and
                    // re-validate on the cold path; introspection TTLs are short).
                }
            }
        }

        // Cold path: verify the signature against the current JWKS snapshot.
        let snap = self.jwks.read().unwrap().clone();
        let header = decode_header(token).map_err(|_| AuthError::Unknown)?;
        let kid = header.kid.unwrap_or_default();
        let key = snap
            .keys
            .get(&kid)
            .or_else(|| {
                (snap.keys.len() == 1)
                    .then(|| snap.keys.values().next())
                    .flatten()
            })
            .ok_or(AuthError::Unknown)?;

        let data = decode::<Value>(token, key, &self.validation).map_err(|_| AuthError::Unknown)?;
        let claims = data.claims;

        let key_id = claims
            .get(&self.identity_claim)
            .and_then(|v| v.as_str())
            .ok_or(AuthError::Unknown)?
            .to_string();
        let exp = claims
            .get("exp")
            .and_then(|v| v.as_u64())
            .ok_or(AuthError::Unknown)?;
        let allowed = self.allowed_models(&claims);

        // Liveness (RFC 7662), fail-closed: an inactive token, or an introspector that errors or was
        // never wired, denies. The result is trusted for `cache_ttl_secs`.
        let introspect_ok_until = match self.introspect_ttl {
            None => 0,
            Some(ttl) => {
                let introspector = self.introspector.read().unwrap().clone();
                match introspector {
                    Some(i) => match i.introspect(token).await {
                        Ok(true) => now.saturating_add(ttl),
                        Ok(false) => return Err(AuthError::Unknown),
                        Err(e) => {
                            tracing::warn!(error = %e, "token introspection failed; denying (fail closed)");
                            return Err(AuthError::Unknown);
                        }
                    },
                    None => {
                        tracing::warn!("introspection configured but no introspector wired; denying (fail closed)");
                        return Err(AuthError::Unknown);
                    }
                }
            }
        };

        // Cache the proven token (it verified; the model check is read live against `allowed`).
        {
            let mut verified = self.verified.write().unwrap();
            if verified.len() >= MAX_VERIFIED {
                verified.retain(|_t, e| now < e.exp);
            }
            verified.insert(
                token.to_string(),
                VerifiedEntry {
                    key_id: key_id.clone(),
                    allowed: allowed.clone(),
                    exp,
                    introspect_ok_until,
                },
            );
        }

        if model.is_none_or(|m| allowed.permits(m)) {
            Ok(key_id)
        } else {
            Err(AuthError::ModelNotAllowed)
        }
    }

    /// Union the model sets of every role the token carries. A token bearing no mapped role can use no
    /// model (fail-closed) — `[oauth].role_models` must grant the roles you intend to serve.
    fn allowed_models(&self, claims: &Value) -> AllowSet {
        let mut acc: HashSet<String> = HashSet::new();
        for role in self.token_roles(claims) {
            match self.role_models.get(&role) {
                Some(AllowSet::All) => return AllowSet::All,
                Some(AllowSet::Only(set)) => acc.extend(set.iter().cloned()),
                None => {}
            }
        }
        AllowSet::Only(acc)
    }

    /// Read the roles/groups claim: a JSON array of strings, or a single space-delimited string.
    fn token_roles(&self, claims: &Value) -> Vec<String> {
        match claims.get(&self.roles_claim) {
            Some(Value::Array(arr)) => arr
                .iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect(),
            Some(Value::String(s)) => s.split_whitespace().map(str::to_string).collect(),
            _ => Vec::new(),
        }
    }
}

/// `true` for the HMAC family, which a public-key resource server must not accept.
fn is_hmac(alg: Algorithm) -> bool {
    matches!(alg, Algorithm::HS256 | Algorithm::HS384 | Algorithm::HS512)
}

/// Parse an RFC 7517 JWKS JSON value into ready decoding keys, keyed by `kid`. A key that cannot be
/// turned into a decoding key is skipped (warned), not fatal; the call errors only if the JSON is not a
/// JWKS at all, so a single bad key never takes down the whole set.
fn parse_jwks(value: &Value) -> Result<HashMap<String, DecodingKey>, String> {
    let set: JwkSet =
        serde_json::from_value(value.clone()).map_err(|e| format!("invalid JWKS: {e}"))?;
    let mut keys = HashMap::with_capacity(set.keys.len());
    for jwk in &set.keys {
        let kid = jwk.common.key_id.clone().unwrap_or_default();
        match DecodingKey::from_jwk(jwk) {
            Ok(key) => {
                keys.insert(kid, key);
            }
            Err(e) => tracing::warn!(kid = %kid, error = %e, "skipping unusable JWK"),
        }
    }
    Ok(keys)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{IntrospectionConfig, OAuthConfig};
    use jsonwebtoken::{encode, EncodingKey, Header};
    use serde_json::json;
    use std::time::{SystemTime, UNIX_EPOCH};

    // A throwaway RSA-2048 keypair generated for these tests only (never a real key). The verifier
    // validates against the public half (the JWK `n`/`e` below); the tests sign with the private PEM.
    const PRIV_PEM: &str = "-----BEGIN PRIVATE KEY-----
MIIEvAIBADANBgkqhkiG9w0BAQEFAASCBKYwggSiAgEAAoIBAQDHmL3shlks3m7A
izZUdfJ8r5jru5BTcUGkPRRzmqN85Kc1ja2IUCi/GOWsGskyMoFvfTxfF+wkP1tE
6YFmHr6JzFLR7j0cowJDnUC/sPrFuFoYo4pwQ6Nfj+KdmS0YldT5MxGMrFC5+XhD
kBtRfyAZB1yDCRxYc6Qjdm64e/AnxPMebeOeXOpkoif+t98kYPkU/ciSCrXk+89i
U7tAD9w2rixmAKgE/xjPtIDCJCXx/8yBr4SE+5FgIp6gedWwM8jfMpEnQ153EotA
fC4UY6iydY/ZABcH6eAjN5tcc3GCJwrAeNNyFUsa3hRSs5YYt/Cg1fbNdas1/foB
rdp87DF/AgMBAAECggEAD5NS8tOoaXtxb2i4VHlLHcFthgEXf04dvg10J0dLIxJh
cqjmgKDSD9jcsgRPF1yLRDd6JWjjqQ/L9Vc2x/t13qFJWeTPnl32wxmmNVnqtSYw
ha6xaBb8d0HDqze0+1pUickbQVErEDAwWy6yKwxoD/vxPgUmfJ0y6FTeyVpb6xvB
3Jdf9timhMTcLKrWX2lZOzumcAMdEB07cKa5i0O7VmaIi18MHHtMAxDPFYmH3zRj
GHTEiG9CgX0PeegUJJ6UosIsYHS3ffPaFOLHdxKeUI9pNd22KS6f/hYw5JV0mlWj
V2vNeLZox0lFWr2LpZvwtnYoOilo6FtRxntBJDgM9QKBgQDyv0Q9mE5fQsv1AD24
iFKQm8+6OwF6aj+7x7Fh8iFmaMhy1QS4iGnsSKG+gpOuv4Y2VnAWkvl5lkbNqR2Q
UidkDTEzrK+pSSL8RK//VoBkWPHdD0KSeA5jpq6OkPNOGXJPv9L6qlzyodha9Stn
5QrLT9frBkRT6YZUINXGNUqtnQKBgQDSfmLtfFX5KVwUXboL7vTSYbFBW04XIo2D
MgG5BbgAWOD7BUueavechTrp6NWy9BdGuDcYdFmyGhaiksjmjKgxILiiRHPKZ0rR
spzi2rdn7mf4TatlVOjEhk3RHPnXt3NHyxaDjIibuzQg1So4gYCYeBoNQ4tSyd+3
4BvACyK+ywKBgDidfVf0TssO9Cy3Gq4QNVl+zYr8uhKH90dfgrJ6W2ncJGwL3KcB
QdsW8YpSQAlwp+fCbzAEpWTDNM5AlCGV15QRM5l+vBV6dktp5bZiwV0mFXreVUN9
9rUtKeccTXK4op2Q+q71zunT73b/WVlI5SSYAZ7R/lunymmSoukwgrNNAoGAFX5a
U62yibKzeFDpwnYwAZuAIhBUtQAOOytMCbyPwVlR/8/WnOKHTjA8h84CpDdQeSQu
XAzla85GKWuboA8UzY9tywP6XtyVQzxd0LxMpgwuCajEq6Az9H1STNwMlv5ggyA1
uFENUfmXpRKfMQ3oN6lr35baDyYV9gyoSL/aWGkCgYBeyJpvZQJZybauY4KSC47g
3WpAjf75h2tGOR0GWU9kubEmxR98YI00e6rw/8aKpdPm2w1jt6wveFheWRmMRzBp
NRETGoq5v+WZ/mZUu2THNlwrJylp/G14jsq3pkmsRExNfNAhE79psN5giVlQyg4v
ojjQDhJdmvFlgMzpZBqBxg==
-----END PRIVATE KEY-----";
    const JWK_N: &str = "x5i97IZZLN5uwIs2VHXyfK-Y67uQU3FBpD0Uc5qjfOSnNY2tiFAovxjlrBrJMjKBb308XxfsJD9bROmBZh6-icxS0e49HKMCQ51Av7D6xbhaGKOKcEOjX4_inZktGJXU-TMRjKxQufl4Q5AbUX8gGQdcgwkcWHOkI3ZuuHvwJ8TzHm3jnlzqZKIn_rffJGD5FP3Ikgq15PvPYlO7QA_cNq4sZgCoBP8Yz7SAwiQl8f_Mga-EhPuRYCKeoHnVsDPI3zKRJ0NedxKLQHwuFGOosnWP2QAXB-ngIzebXHNxgicKwHjTchVLGt4UUrOWGLfwoNX2zXWrNf36Aa3afOwxfw";
    const JWK_E: &str = "AQAB";
    const KID: &str = "test-key";

    fn now() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }

    fn jwks() -> Value {
        json!({ "keys": [
            { "kty": "RSA", "use": "sig", "alg": "RS256", "kid": KID, "n": JWK_N, "e": JWK_E }
        ] })
    }

    /// A verifier whose roles map `power → {gpt-4o, claude-opus-4}`, `basic → {gpt-4o-mini}`,
    /// `admin → {*}`, `router → {openrouter/openai/*}`, seeded with the inline test JWKS.
    fn verifier() -> OAuthVerifier {
        let cfg = OAuthConfig {
            issuer: "https://idp.test/".into(),
            audience: vec!["llmleaf".into()],
            jwks_uri: None,
            jwks: Some(jwks()),
            algorithms: vec!["RS256".into()],
            identity_claim: "sub".into(),
            roles_claim: "roles".into(),
            role_models: HashMap::from([
                (
                    "power".into(),
                    vec!["gpt-4o".into(), "claude-opus-4".into()],
                ),
                ("basic".into(), vec!["gpt-4o-mini".into()]),
                ("admin".into(), vec!["*".into()]),
                ("router".into(), vec!["openrouter/openai/*".into()]),
            ]),
            jwks_refresh_secs: 3600,
            timeout_ms: 2000,
            on_error: crate::config::OnError::Deny,
            introspection: None,
        };
        OAuthVerifier::from_config(&cfg).expect("verifier builds")
    }

    /// Mint a signed JWT with the given roles, audience, issuer and expiry offset.
    fn token(roles: Value, aud: &str, iss: &str, exp: i64) -> String {
        let mut header = Header::new(Algorithm::RS256);
        header.kid = Some(KID.into());
        let claims = json!({
            "sub": "user-123",
            "aud": aud,
            "iss": iss,
            "exp": exp,
            "roles": roles,
        });
        let key = EncodingKey::from_rsa_pem(PRIV_PEM.as_bytes()).expect("priv key");
        encode(&header, &claims, &key).expect("encode")
    }

    fn good(roles: Value) -> String {
        token(roles, "llmleaf", "https://idp.test/", now() as i64 + 3600)
    }

    #[tokio::test]
    async fn valid_token_authorizes_a_granted_model() {
        let v = verifier();
        let t = good(json!(["power"]));
        assert_eq!(
            v.authorize(&t, "gpt-4o", now()).await,
            Ok("user-123".to_string())
        );
        assert_eq!(
            v.authorize(&t, "claude-opus-4", now()).await,
            Ok("user-123".to_string())
        );
    }

    #[tokio::test]
    async fn granted_role_does_not_grant_other_models() {
        let v = verifier();
        let t = good(json!(["power"]));
        // `power` does not include the mini model, which `basic` would.
        assert_eq!(
            v.authorize(&t, "gpt-4o-mini", now()).await,
            Err(AuthError::ModelNotAllowed)
        );
    }

    #[tokio::test]
    async fn wildcard_role_grants_every_model() {
        let v = verifier();
        let t = good(json!(["admin"]));
        assert_eq!(
            v.authorize(&t, "anything-at-all", now()).await,
            Ok("user-123".to_string())
        );
    }

    #[tokio::test]
    async fn wildcard_pattern_role_grants_by_prefix() {
        let v = verifier();
        let t = good(json!(["router"]));
        assert_eq!(
            v.authorize(&t, "openrouter/openai/gpt-4o", now()).await,
            Ok("user-123".to_string())
        );
        assert_eq!(
            v.authorize(&t, "openrouter/mistral/large", now()).await,
            Err(AuthError::ModelNotAllowed)
        );
    }

    #[tokio::test]
    async fn unmapped_role_grants_nothing() {
        let v = verifier();
        let t = good(json!(["ghost"]));
        assert_eq!(
            v.authorize(&t, "gpt-4o", now()).await,
            Err(AuthError::ModelNotAllowed)
        );
    }

    #[tokio::test]
    async fn union_of_roles() {
        let v = verifier();
        let t = good(json!(["power", "basic"]));
        assert_eq!(
            v.authorize(&t, "gpt-4o", now()).await,
            Ok("user-123".to_string())
        );
        assert_eq!(
            v.authorize(&t, "gpt-4o-mini", now()).await,
            Ok("user-123".to_string())
        );
    }

    #[tokio::test]
    async fn expired_wrong_aud_wrong_iss_and_tampered_all_unknown() {
        let v = verifier();
        let n = now() as i64;
        let expired = token(json!(["power"]), "llmleaf", "https://idp.test/", n - 3600);
        let wrong_aud = token(
            json!(["power"]),
            "someone-else",
            "https://idp.test/",
            n + 3600,
        );
        let wrong_iss = token(json!(["power"]), "llmleaf", "https://evil.test/", n + 3600);
        assert_eq!(
            v.authorize(&expired, "gpt-4o", now()).await,
            Err(AuthError::Unknown)
        );
        assert_eq!(
            v.authorize(&wrong_aud, "gpt-4o", now()).await,
            Err(AuthError::Unknown)
        );
        assert_eq!(
            v.authorize(&wrong_iss, "gpt-4o", now()).await,
            Err(AuthError::Unknown)
        );

        // Flip the last char of the signature → bad signature → Unknown (no oracle).
        let mut tampered = good(json!(["power"]));
        let last = tampered.pop().unwrap();
        tampered.push(if last == 'A' { 'B' } else { 'A' });
        assert_eq!(
            v.authorize(&tampered, "gpt-4o", now()).await,
            Err(AuthError::Unknown)
        );
    }

    #[tokio::test]
    async fn garbage_and_non_jwt_are_unknown() {
        let v = verifier();
        assert_eq!(
            v.authorize("not.a.jwt", "gpt-4o", now()).await,
            Err(AuthError::Unknown)
        );
        assert_eq!(
            v.authorize("", "gpt-4o", now()).await,
            Err(AuthError::Unknown)
        );
    }

    #[tokio::test]
    async fn warm_cache_returns_same_verdict() {
        let v = verifier();
        let t = good(json!(["power"]));
        assert_eq!(
            v.authorize(&t, "gpt-4o", now()).await,
            Ok("user-123".to_string())
        );
        // Second call hits the verified cache (no re-decode); same answers.
        assert_eq!(
            v.authorize(&t, "gpt-4o", now()).await,
            Ok("user-123".to_string())
        );
        assert_eq!(
            v.authorize(&t, "gpt-4o-mini", now()).await,
            Err(AuthError::ModelNotAllowed)
        );
    }

    #[tokio::test]
    async fn no_jwks_fails_closed() {
        let v = verifier();
        // Swap in an empty key set (cold start, no inline keys yet): nothing validates.
        v.install_jwks(&json!({ "keys": [] })).unwrap();
        let t = good(json!(["power"]));
        assert_eq!(
            v.authorize(&t, "gpt-4o", now()).await,
            Err(AuthError::Unknown)
        );
    }

    struct StubIntrospector {
        active: bool,
    }
    #[async_trait::async_trait]
    impl TokenIntrospector for StubIntrospector {
        async fn introspect(&self, _token: &str) -> Result<bool, String> {
            Ok(self.active)
        }
    }

    #[tokio::test]
    async fn introspection_inactive_denies() {
        let cfg = OAuthConfig {
            issuer: "https://idp.test/".into(),
            audience: vec!["llmleaf".into()],
            jwks_uri: None,
            jwks: Some(jwks()),
            algorithms: vec!["RS256".into()],
            identity_claim: "sub".into(),
            roles_claim: "roles".into(),
            role_models: HashMap::from([("admin".into(), vec!["*".into()])]),
            jwks_refresh_secs: 3600,
            timeout_ms: 2000,
            on_error: crate::config::OnError::Deny,
            introspection: Some(IntrospectionConfig {
                url: "https://idp.test/introspect".into(),
                credential: None,
                cache_ttl_secs: 30,
                timeout_ms: 2000,
            }),
        };
        let v = OAuthVerifier::from_config(&cfg).unwrap();
        let t = good(json!(["admin"]));

        // No introspector wired yet → fail closed even though the JWT is valid.
        assert_eq!(
            v.authorize(&t, "gpt-4o", now()).await,
            Err(AuthError::Unknown)
        );

        // Inactive introspection → denied.
        v.install_introspector(Arc::new(StubIntrospector { active: false }));
        assert_eq!(
            v.authorize(&t, "gpt-4o", now()).await,
            Err(AuthError::Unknown)
        );
    }

    // --- Server-level dispatch: a JWT bearer travels the real router; static keys still work. ---

    use async_trait::async_trait;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use llmleaf_model::{
        ChatRequest, FinishReason, ModelError, ResponseStream, StreamChunk, Usage,
    };
    use llmleaf_provider::{Provider, ProviderCx, ProviderRegistry};
    use tower::ServiceExt;

    struct OneLineProvider;
    #[async_trait]
    impl Provider for OneLineProvider {
        fn name(&self) -> &str {
            "mock"
        }
        async fn chat(
            &self,
            req: ChatRequest,
            _cx: &ProviderCx,
        ) -> Result<ResponseStream, ModelError> {
            let chunks: Vec<Result<StreamChunk, ModelError>> = vec![
                Ok(StreamChunk::Start {
                    id: "m1".into(),
                    model: req.model.clone(),
                }),
                Ok(StreamChunk::Content {
                    index: 0,
                    delta: "hi".into(),
                }),
                Ok(StreamChunk::Finish {
                    index: 0,
                    reason: FinishReason::Stop,
                }),
                Ok(StreamChunk::Usage(Usage {
                    prompt_tokens: 1,
                    completion_tokens: 1,
                    total_tokens: 2,
                    cost_usd: None,
                    cache_read_tokens: 0,
                    cache_creation_tokens: 0,
                })),
            ];
            Ok(Box::pin(futures::stream::iter(chunks)))
        }
    }

    // Static key "local:s3cret" (bcrypt cost 4) → token base64("local:s3cret").
    const STATIC_HASH: &str = "$2y$04$IcVq6nhz5Tf85lBpWclgKeDjWxWMHlIXLE696.T7m9Eg12HekWFJO";
    const STATIC_TOKEN: &str = "bG9jYWw6czNjcmV0";

    fn oauth_router() -> axum::Router {
        // Inline JWKS in config (config-operable, no control plane), role `admin → *`, plus a static key
        // alongside — both schemes mounted on one router.
        let toml = format!(
            r#"
            [[providers]]
            name = "mock"
            kind = "mock"

            [[routes]]
            model = "gpt-4o"
            targets = [{{ provider = "mock" }}]

            [[keys]]
            id = "local"
            pw_hash = "{STATIC_HASH}"

            [oauth]
            issuer = "https://idp.test/"
            audience = ["llmleaf"]
            algorithms = ["RS256"]

            [oauth.role_models]
            admin = ["*"]

            [[oauth.jwks.keys]]
            kty = "RSA"
            use = "sig"
            alg = "RS256"
            kid = "{KID}"
            n = "{JWK_N}"
            e = "{JWK_E}"
            "#
        );
        let config = crate::config::Config::from_toml_str(&toml).unwrap();
        let mut registry = ProviderRegistry::new();
        registry.register("mock", Arc::new(OneLineProvider));
        let state = crate::server::build_state(&config, Arc::new(registry)).unwrap();
        crate::server::build_router(state)
    }

    fn chat_req(bearer: &str) -> Request<Body> {
        Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", format!("Bearer {bearer}"))
            .header("content-type", "application/json")
            .body(Body::from(
                serde_json::to_vec(&json!({
                    "model": "gpt-4o",
                    "stream": false,
                    "messages": [{ "role": "user", "content": "hello" }]
                }))
                .unwrap(),
            ))
            .unwrap()
    }

    #[tokio::test]
    async fn router_accepts_jwt_keeps_static_keys_and_rejects_garbage() {
        let app = oauth_router();
        let jwt = good(json!(["admin"]));

        // A JWT bearer is dispatched to the OAuth verifier → 200.
        let resp = app.clone().oneshot(chat_req(&jwt)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK, "JWT should authenticate");

        // A static key on the same router still works (additive).
        let resp = app.clone().oneshot(chat_req(STATIC_TOKEN)).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "static key should still authenticate"
        );

        // A garbage JWT-shaped bearer → 401.
        let resp = app.clone().oneshot(chat_req("aaa.bbb.ccc")).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::UNAUTHORIZED,
            "bad JWT should be rejected"
        );
    }

    #[tokio::test]
    async fn introspection_active_allows() {
        let cfg = OAuthConfig {
            issuer: "https://idp.test/".into(),
            audience: vec!["llmleaf".into()],
            jwks_uri: None,
            jwks: Some(jwks()),
            algorithms: vec!["RS256".into()],
            identity_claim: "sub".into(),
            roles_claim: "roles".into(),
            role_models: HashMap::from([("admin".into(), vec!["*".into()])]),
            jwks_refresh_secs: 3600,
            timeout_ms: 2000,
            on_error: crate::config::OnError::Deny,
            introspection: Some(IntrospectionConfig {
                url: "https://idp.test/introspect".into(),
                credential: None,
                cache_ttl_secs: 30,
                timeout_ms: 2000,
            }),
        };
        let v = OAuthVerifier::from_config(&cfg).unwrap();
        v.install_introspector(Arc::new(StubIntrospector { active: true }));
        let t = good(json!(["admin"]));
        assert_eq!(
            v.authorize(&t, "gpt-4o", now()).await,
            Ok("user-123".to_string())
        );
    }
}
