//! Virtual key store and verdict enforcement (SOUL.md principle 5, inverted control plane).
//!
//! The core *observes; others account — and gatekeep*. This store never counts usage and never does
//! arithmetic to decide access. Every request is a **lookup**: present a token, get the current
//! verdict, compare. Identity and verdicts are *pulled* from configured control endpoints on an
//! interval and installed here as a whole new snapshot ([`KeyStore::install_identities`],
//! [`KeyStore::install_verdicts`]); there is no inbound mutation surface. A failed pull leaves the
//! last-good snapshot in place, so the proxy keeps serving (principle 8).
//!
//! Two layers stack here (principle 6): the config base (a key's identity + static `allowed_models`,
//! seeded from the config file and optionally refreshed by the identity pull) and the verdict overlay
//! (refreshed by the limits pull). A verdict may further restrict, suspend, or block — never widen.
//!
//! ## Token format
//!
//! A consumer presents `Authorization: Bearer <base64(key-id:password)>` — the HTTP-Basic shape, so
//! the split is on the *first* `:` (the id holds no colon; the password may). Config and the admin
//! API never see the plaintext password: they store only its **hash**, a standard Unix/crypt(3) MCF
//! string (bcrypt `$2y$…` from `htpasswd -bnBC 12`, or a `$1$`/`$5$`/`$6$` shadow hash). On each
//! request the core decodes the token, finds the key by id, and verifies the password against the
//! stored hash.
//!
//! ## Keeping the hot path sacred (principle 1)
//!
//! A crypt(3) hash is a *deliberately slow* KDF — running it per request would gut the hot path. So a
//! proven token is cached id-ward in a node-local map: the first request for a key pays the KDF once,
//! every later one is the principle-5 "lookup and a comparison". The cache holds only tokens that
//! verified, so it is bounded by the number of keys (one valid token each) — a wrong password is
//! never cached and cannot grow it. It is pure node-local state derived from observations (principle
//! 9): no coordination, dropped freely, rebuilt on demand.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};

use serde::{Deserialize, Serialize};

use crate::config::KeyConfig;

/// A stable, log-safe identifier for a key (its name, or its key-id). Never a secret.
pub type KeyId = String;

/// Upper bound on a decoded token (`key-id:password`). Decoding lands in a stack buffer this size, so
/// an unknown or garbled token is rejected without touching the heap; a longer token is treated as
/// unknown. Comfortably covers a long id plus bcrypt's 72-byte password ceiling.
const MAX_TOKEN_DECODED: usize = 512;

/// The per-key verdict. Pulled from the configured limiter and cached node-locally; enforced by
/// lookup only.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Verdict {
    /// Hard block, regardless of model or time.
    #[serde(default)]
    pub blocked: bool,
    /// Suspended while `now < suspended_until` (unix seconds). A comparison, not a countdown.
    #[serde(default)]
    pub suspended_until: Option<u64>,
    /// Runtime model restriction layered on top of the config allow-list. `None` ⇒ no extra
    /// restriction. Entries may be `*` wildcard patterns (`"gpt-*"`, `"openrouter/openai/*"`),
    /// matched like the config allow-list (see [`allow_set_permits`]).
    #[serde(default)]
    pub allowed_models: Option<HashSet<String>>,
}

/// Why an authorization lookup failed. Surfaced to the consumer edge as the right status code.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthError {
    /// Token not recognised. Deliberately conflates every "no": a malformed token, an unknown key-id,
    /// and a wrong password all collapse to one answer so the edge reveals no oracle.
    Unknown,
    /// A verdict blocks this key outright.
    Blocked,
    /// Suspended by verdict until the given unix time.
    Suspended { until: u64 },
    /// The requested model is outside this key's allow-list (config base and/or verdict).
    ModelNotAllowed,
}

#[derive(Clone)]
struct KeyState {
    name: Option<String>,
    /// The key's password hash (a crypt(3) MCF string). Verified against, never reversed.
    pw_hash: String,
    /// Base restriction from the identity layer (config file or identity pull).
    config_allowed: Option<HashSet<String>>,
    /// Verdict overlay from the limits pull.
    verdict: Verdict,
}

impl KeyState {
    /// The log-safe display id: the friendly name if set, else the key-id itself.
    fn display_id(&self, id: &str) -> KeyId {
        self.name.clone().unwrap_or_else(|| id.to_string())
    }
}

/// One key's identity as installed by the control plane's identity pull — the resolved form of a
/// [`KeyConfig`]. `pw_hash` is a crypt(3) MCF string, never plaintext.
pub struct IdentityInput {
    pub id: String,
    pub pw_hash: String,
    pub name: Option<String>,
    pub allowed_models: Option<Vec<String>>,
}

/// An atomically-swappable view of all key state: the identity layer plus the verdict overlay. Built
/// once from config (the base), then replaced wholesale by the pull refreshers — never mutated in
/// place. The hot path clones the `Arc` and drops the lock before the KDF, so the slow hash never runs
/// under a lock.
struct KeySnapshot {
    keys: HashMap<KeyId, KeyState>,
    /// Bumped only when the *identity* layer is replaced (hashes may have rotated). A verdict-only swap
    /// keeps it, so the warm-path fast check (see [`KeyStore::authorize`]) stays valid across verdict
    /// refreshes while still reading the new verdict live.
    identity_gen: u64,
}

/// A proven token's cache entry. Tagged with the identity generation and hash it verified against so a
/// later identity swap can be detected with a string compare — never another KDF.
struct VerifiedEntry {
    id: KeyId,
    identity_gen: u64,
    pw_hash: String,
}

/// Thread-safe key store. The hot path clones the current snapshot `Arc` under a short read lock,
/// releases the lock, and never `.await`s while held — the KDF runs lock-free against the clone.
pub struct KeyStore {
    /// The current snapshot: the config base plus whatever the control plane last installed. Swapped
    /// wholesale by the pull refreshers; never mutated in place.
    snapshot: RwLock<Arc<KeySnapshot>>,
    /// Verification cache: a proven token → the key it belongs to (plus the identity it verified
    /// against). Lets the warm hot path skip the password KDF (see module docs). Node-local, bounded by
    /// the number of keys.
    verified: RwLock<HashMap<String, VerifiedEntry>>,
}

impl KeyStore {
    /// Build the base store from config keys (principle 6: config is the base). A key whose `pw_hash`
    /// is an `env:` reference to an unset var is skipped (mirrors the old token behaviour). This is the
    /// generation-0 identity seed; with no control plane it is the permanent static store.
    pub fn from_config(keys: &[KeyConfig]) -> Self {
        let mut map = HashMap::new();
        for k in keys {
            let Some(pw_hash) = k.pw_hash.resolve() else {
                continue;
            };
            map.insert(
                k.id.clone(),
                KeyState {
                    name: k.name.clone(),
                    pw_hash,
                    config_allowed: allow_list(k.allowed_models.as_deref()),
                    verdict: Verdict::default(),
                },
            );
        }
        KeyStore {
            snapshot: RwLock::new(Arc::new(KeySnapshot {
                keys: map,
                identity_gen: 0,
            })),
            verified: RwLock::new(HashMap::new()),
        }
    }

    /// The hot-path check (principle 5): a lookup and a comparison, no arithmetic, no counting.
    /// `now` (unix seconds) is supplied by the caller — the store holds no clock.
    ///
    /// `token` is the raw bearer string (`base64(key-id:password)`). The warm path is a cache hit and
    /// a verdict comparison; only the first request for a key pays the password KDF (see module docs).
    pub fn authorize(&self, token: &str, model: &str, now: u64) -> Result<KeyId, AuthError> {
        self.authorize_inner(token, Some(model), now)
    }

    /// Authenticate identity only — prove the token maps to a live, non-blocked, non-suspended key —
    /// without gating on any model. For surfaces that carry no model in the request: the batch
    /// retrieve/cancel/results endpoints route by an opaque id, not a model, so there is nothing to gate.
    pub fn authorize_identity(&self, token: &str, now: u64) -> Result<KeyId, AuthError> {
        self.authorize_inner(token, None, now)
    }

    fn authorize_inner(
        &self,
        token: &str,
        model: Option<&str>,
        now: u64,
    ) -> Result<KeyId, AuthError> {
        // Take a cheap snapshot handle and release the lock immediately — the slow KDF never runs under
        // a lock, and a concurrent swap just installs a new Arc while this request finishes on its own.
        let snap = self.snapshot.read().unwrap().clone();

        // Warm path: a token proven before maps straight to its key, skipping the KDF entirely.
        {
            let verified = self.verified.read().unwrap();
            if let Some(entry) = verified.get(token) {
                if let Some(state) = snap.keys.get(&entry.id) {
                    // If the identity layer hasn't been swapped since we cached this token, or the
                    // key's hash is byte-identical (a verdict-only swap, or an identity refresh that
                    // left this key untouched), the identity is unchanged — trust the cache and read
                    // the live verdict. Otherwise the password rotated: fall through and re-verify.
                    if entry.identity_gen == snap.identity_gen || entry.pw_hash == state.pw_hash {
                        return enforce(state, &entry.id, model, now);
                    }
                }
            }
        }

        // Cold path: decode `key-id:password`, find the key by id, verify the password against its
        // hash. Decoding lands in a stack buffer so an unknown/garbled token never touches the heap.
        let mut buf = [0u8; MAX_TOKEN_DECODED];
        let len = b64_decode_into(token, &mut buf).ok_or(AuthError::Unknown)?;
        // `password` is the bytes verbatim after the first ':' — NOT trimmed. A trailing newline
        // from a credential base64'd with `echo … | base64` (echo appends '\n') stays in it and
        // fails `verify` below, surfacing as a spurious `unknown api key` even when the configured
        // hash is correct. Clients must encode with `printf`/`base64 -w0`. (See README / example.toml.)
        let (id, password) = split_credential(&buf[..len]).ok_or(AuthError::Unknown)?;

        let (outcome, pw_hash) = {
            let state = snap.keys.get(id).ok_or(AuthError::Unknown)?;
            if !pwhash::unix::verify(password, &state.pw_hash) {
                return Err(AuthError::Unknown);
            }
            // Verification passed: a genuine token for `id`. The verdict is decided here against the
            // snapshot clone; the password is never reached again for this token.
            (enforce(state, id, model, now), state.pw_hash.clone())
        };

        // The password verified (any other path returned `Unknown` above), so cache the proven token.
        // Stored even when the verdict currently blocks: the token is genuine; the verdict is re-read
        // live on the warm path next time.
        self.verified.write().unwrap().insert(
            token.to_string(),
            VerifiedEntry {
                id: id.to_string(),
                identity_gen: snap.identity_gen,
                pw_hash,
            },
        );
        outcome
    }

    /// Install a freshly pulled identity roster, replacing the identity layer wholesale (principle 6:
    /// the config file is the seed; the identity pull refreshes it). Existing verdicts are carried over
    /// for keys that survive. Bumps the identity generation so warm cache entries re-check their hash
    /// (a string compare, not a KDF) on next use, and prunes cached tokens for keys that vanished.
    /// Called only by the control plane's refresher, never on the hot path.
    pub fn install_identities(&self, roster: Vec<IdentityInput>) {
        let mut guard = self.snapshot.write().unwrap();
        let prev = &guard.keys;
        let mut keys = HashMap::with_capacity(roster.len());
        for k in roster {
            // Carry over the current verdict overlay for a surviving key; reset for a new one.
            let verdict = prev
                .get(&k.id)
                .map(|s| s.verdict.clone())
                .unwrap_or_default();
            keys.insert(
                k.id.clone(),
                KeyState {
                    name: k.name,
                    pw_hash: k.pw_hash,
                    config_allowed: allow_list(k.allowed_models.as_deref()),
                    verdict,
                },
            );
        }
        let identity_gen = guard.identity_gen.wrapping_add(1);
        // Prune cached tokens whose key no longer exists; rotated-but-surviving keys are handled lazily
        // on the warm path by the hash compare.
        self.verified
            .write()
            .unwrap()
            .retain(|_t, e| keys.contains_key(&e.id));
        *guard = Arc::new(KeySnapshot { keys, identity_gen });
    }

    /// Install a freshly pulled verdict overlay, replacing every key's verdict (a key absent from
    /// `verdicts` gets the default empty verdict — no restriction). Identities are untouched, so the
    /// identity generation is preserved and the warm cache stays valid; the new verdict is read live.
    /// Called only by the control plane's refresher, never on the hot path.
    pub fn install_verdicts(&self, mut verdicts: HashMap<KeyId, Verdict>) {
        let mut guard = self.snapshot.write().unwrap();
        let mut keys = guard.keys.clone();
        for (id, state) in keys.iter_mut() {
            state.verdict = verdicts.remove(id).unwrap_or_default();
        }
        let identity_gen = guard.identity_gen;
        *guard = Arc::new(KeySnapshot { keys, identity_gen });
    }

    /// The effective model scope of a key, or `None` when the key is unrestricted (or unknown).
    ///
    /// The scope keeps the config allow-list and the verdict overlay as *separate layers* — wildcard
    /// entries (`"gpt-*"`) make their intersection impossible to materialize as a set — and
    /// [`ModelScope::permits`] answers for a model exactly as [`enforce`] would: it must pass every
    /// layer that is present. The model-listing surface uses this to scope what a key can see
    /// (OpenAI/OpenRouter return only the models a key can access).
    ///
    /// `key_id` is the *display* id returned by [`Self::authorize_token_identity`] (the friendly name
    /// if set), while the snapshot is keyed by the raw key-id — so we match on [`KeyState::display_id`].
    /// Display ids are assumed unique (the same assumption the event and admin surfaces already rely
    /// on); two keys sharing a `name` would be indistinguishable here.
    pub fn model_scope(&self, key_id: &KeyId) -> Option<ModelScope> {
        let snap = self.snapshot.read().unwrap();
        let state = snap
            .keys
            .iter()
            .find(|(id, state)| &state.display_id(id) == key_id)
            .map(|(_, state)| state)?;
        match (&state.config_allowed, &state.verdict.allowed_models) {
            (None, None) => None,
            (config, verdict) => Some(ModelScope {
                config: config.clone(),
                verdict: verdict.clone(),
            }),
        }
    }

    /// A secret-free snapshot for the read-only admin listing. The password hash is never exposed.
    pub fn snapshot(&self) -> Vec<KeyView> {
        self.snapshot
            .read()
            .unwrap()
            .keys
            .iter()
            .map(|(id, state)| KeyView {
                id: state.display_id(id),
                key_id: id.clone(),
                config_allowed: state
                    .config_allowed
                    .as_ref()
                    .map(|s| s.iter().cloned().collect()),
                verdict: state.verdict.clone(),
            })
            .collect()
    }
}

/// Materialize a configured model allow-list into the restriction the store enforces. A list
/// containing a bare `"*"` means "every routed model" — the same convention the OAuth role→models
/// mapping uses ([`crate::oauth`]) — so it collapses to `None` (unrestricted). Other entries are
/// kept verbatim; ones containing `*` act as wildcard patterns at match time ([`allow_set_permits`]).
fn allow_list(models: Option<&[String]>) -> Option<HashSet<String>> {
    let models = models?;
    if models.iter().any(|m| m == "*") {
        return None;
    }
    Some(models.iter().cloned().collect())
}

/// A key's effective model restriction, as its two layers (config base, verdict overlay — principle
/// 6) kept apart: wildcard entries make the intersection impossible to materialize as a set, so the
/// scope is *tested*, never enumerated. Built by [`KeyStore::model_scope`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelScope {
    config: Option<HashSet<String>>,
    verdict: Option<HashSet<String>>,
}

impl ModelScope {
    /// `true` when every layer that is present permits `model` — the same answer [`enforce`] gives
    /// on the request path.
    pub fn permits(&self, model: &str) -> bool {
        self.config
            .as_ref()
            .is_none_or(|s| allow_set_permits(s, model))
            && self
                .verdict
                .as_ref()
                .is_none_or(|s| allow_set_permits(s, model))
    }
}

/// `true` when `model` is permitted by an allow-list: an exact entry (one hash lookup — the common
/// case, so a plain list stays O(1)), or any entry containing `*` that it wildcard-matches.
pub(crate) fn allow_set_permits(set: &HashSet<String>, model: &str) -> bool {
    set.contains(model)
        || set
            .iter()
            .any(|p| p.contains('*') && wildcard_match(p, model))
}

/// Match `model` against `pattern`, where each `*` matches any (possibly empty) run of characters
/// and everything else matches literally — `"gpt-*"`, `"openrouter/openai/*"`, `"*-mini"`.
/// Iterative two-pointer glob with single-star backtracking: allocation-free (principle 1), and
/// byte-wise, which is exact for UTF-8 since `*` is ASCII.
fn wildcard_match(pattern: &str, model: &str) -> bool {
    let (p, t) = (pattern.as_bytes(), model.as_bytes());
    let (mut pi, mut ti) = (0, 0);
    // The position after the most recent `*` and the text index it has consumed up to; on a
    // mismatch, back up here and let the star swallow one more byte.
    let mut star: Option<(usize, usize)> = None;
    while ti < t.len() {
        if pi < p.len() && p[pi] == b'*' {
            pi += 1;
            star = Some((pi, ti));
        } else if pi < p.len() && p[pi] == t[ti] {
            pi += 1;
            ti += 1;
        } else if let Some((sp, st)) = star {
            pi = sp;
            ti = st + 1;
            star = Some((sp, st + 1));
        } else {
            return false;
        }
    }
    // Only trailing stars may remain unconsumed.
    while pi < p.len() && p[pi] == b'*' {
        pi += 1;
    }
    pi == p.len()
}

/// Apply the documented precedence once a key is in hand: blocked ▸ suspended ▸ config allow-list ▸
/// verdict allow-list ▸ allowed. Allow-list entries may be `*` wildcard patterns
/// ([`allow_set_permits`]). A lookup and comparisons only — no arithmetic (principle 5).
fn enforce(state: &KeyState, id: &str, model: Option<&str>, now: u64) -> Result<KeyId, AuthError> {
    if state.verdict.blocked {
        return Err(AuthError::Blocked);
    }
    if let Some(until) = state.verdict.suspended_until {
        if now < until {
            return Err(AuthError::Suspended { until });
        }
    }
    // Model gating applies only when the surface carries a model. The batch retrieve/cancel/results
    // surfaces route by an opaque id and carry no model (`None`), so they are gated on identity +
    // blocked/suspended only — there is nothing to check a model allow-list against.
    if let Some(model) = model {
        // Config base restriction.
        if let Some(allowed) = &state.config_allowed {
            if !allow_set_permits(allowed, model) {
                return Err(AuthError::ModelNotAllowed);
            }
        }
        // Verdict overlay restriction.
        if let Some(allowed) = &state.verdict.allowed_models {
            if !allow_set_permits(allowed, model) {
                return Err(AuthError::ModelNotAllowed);
            }
        }
    }
    Ok(state.display_id(id))
}

/// Split a decoded `key-id:password` on its first `:`. `None` if there is no colon or either half is
/// not valid UTF-8. Borrows out of `bytes`, so it allocates nothing.
fn split_credential(bytes: &[u8]) -> Option<(&str, &str)> {
    let colon = bytes.iter().position(|&b| b == b':')?;
    let id = std::str::from_utf8(&bytes[..colon]).ok()?;
    let password = std::str::from_utf8(&bytes[colon + 1..]).ok()?;
    Some((id, password))
}

/// Decode standard-alphabet base64 (RFC 4648, with or without `=` padding) into `out`, returning the
/// number of bytes written, or `None` if the input is not valid base64 or would overflow `out`.
/// Allocation-free — the caller owns `out` (a stack buffer on the hot path), so an unknown or garbled
/// token is rejected without ever touching the heap. Mirrors the hand-rolled encoder in
/// `compat::embeddings`: SOUL keeps base64 dependency-free in the core.
fn b64_decode_into(input: &str, out: &mut [u8]) -> Option<usize> {
    fn sextet(b: u8) -> Option<u32> {
        Some(match b {
            b'A'..=b'Z' => b - b'A',
            b'a'..=b'z' => b - b'a' + 26,
            b'0'..=b'9' => b - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            _ => return None,
        } as u32)
    }

    let bytes = input.as_bytes();
    // Strip up to two trailing `=` (padding is only legal at the very end; a `=` elsewhere is rejected
    // by `sextet` below). Decoding the remaining body bit-by-bit handles padded and unpadded input.
    let mut end = bytes.len();
    let mut pad = 0;
    while end > 0 && bytes[end - 1] == b'=' && pad < 2 {
        end -= 1;
        pad += 1;
    }
    let body = &bytes[..end];
    // A base64 body length ≡ 1 (mod 4) cannot occur — reject it rather than emit a truncated byte.
    if body.len() % 4 == 1 {
        return None;
    }

    let mut acc: u32 = 0;
    let mut nbits: u32 = 0;
    let mut written = 0usize;
    for &b in body {
        acc = (acc << 6) | sextet(b)?;
        nbits += 6;
        if nbits >= 8 {
            nbits -= 8;
            *out.get_mut(written)? = (acc >> nbits) as u8;
            written += 1;
        }
    }
    Some(written)
}

/// A redacted view of a key for the admin listing surface. Carries no secret — neither a token nor a
/// password hash, only the public key-id and the current restrictions.
#[derive(Debug, Clone, Serialize)]
pub struct KeyView {
    /// The log-safe display id (friendly name if set, else the key-id) — matches the id stamped on events.
    pub id: KeyId,
    /// The stable key-id used to address this key in tokens and admin calls.
    pub key_id: String,
    pub config_allowed: Option<Vec<String>>,
    pub verdict: Verdict,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Secret;

    // bcrypt of "s3cret" at cost 4 (`htpasswd -bnBC 4 … s3cret`). Cost 4 keeps the tests fast; real
    // deployments use the htpasswd default of 12.
    const BCRYPT_S3CRET: &str = "$2y$04$IcVq6nhz5Tf85lBpWclgKeDjWxWMHlIXLE696.T7m9Eg12HekWFJO";
    // The narrow key's token: base64("narrow:s3cret").
    const TOK_NARROW: &str = "bmFycm93OnMzY3JldA==";

    fn key(id: &str, hash: &str, allowed: Option<Vec<String>>) -> KeyConfig {
        KeyConfig {
            id: id.into(),
            pw_hash: Secret::new(hash),
            name: Some(format!("{id}-team")),
            allowed_models: allowed,
        }
    }

    fn store() -> KeyStore {
        KeyStore::from_config(&[
            key("open", BCRYPT_S3CRET, None),
            key("narrow", BCRYPT_S3CRET, Some(vec!["gpt-4o".into()])),
        ])
    }

    /// base64("open:s3cret") — computed here so the test is self-evidently the open key's token.
    fn tok_open() -> String {
        b64("open:s3cret")
    }
    fn b64(s: &str) -> String {
        // Tiny standard-alphabet encoder, test-only (the core only ever decodes).
        const A: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut out = String::new();
        for chunk in s.as_bytes().chunks(3) {
            let b = [
                chunk[0],
                *chunk.get(1).unwrap_or(&0),
                *chunk.get(2).unwrap_or(&0),
            ];
            let n = chunk.len();
            out.push(A[(b[0] >> 2) as usize] as char);
            out.push(A[(((b[0] & 0x3) << 4) | (b[1] >> 4)) as usize] as char);
            out.push(if n > 1 {
                A[(((b[1] & 0xf) << 2) | (b[2] >> 6)) as usize] as char
            } else {
                '='
            });
            out.push(if n > 2 {
                A[(b[2] & 0x3f) as usize] as char
            } else {
                '='
            });
        }
        out
    }

    #[test]
    fn model_scope_intersects_config_and_verdict() {
        let s = store();
        // Unrestricted key (no config allow-list, no verdict): None ⇒ sees everything.
        assert_eq!(s.model_scope(&"open-team".to_string()), None);
        // Config-only restriction.
        let scope = s.model_scope(&"narrow-team".to_string()).unwrap();
        assert!(scope.permits("gpt-4o"));
        assert!(!scope.permits("o3-mini"));
        // Verdict-only restriction (on the otherwise-open key).
        let mut verdicts = HashMap::new();
        verdicts.insert(
            "open".to_string(),
            Verdict {
                allowed_models: Some(HashSet::from(["o3-mini".to_string()])),
                ..Default::default()
            },
        );
        // And an intersecting verdict on the config-restricted key.
        verdicts.insert(
            "narrow".to_string(),
            Verdict {
                allowed_models: Some(HashSet::from([
                    "gpt-4o".to_string(),
                    "claude-opus-4".to_string(),
                ])),
                ..Default::default()
            },
        );
        s.install_verdicts(verdicts);
        let scope = s.model_scope(&"open-team".to_string()).unwrap();
        assert!(scope.permits("o3-mini"));
        assert!(!scope.permits("gpt-4o"));
        // config {gpt-4o} ∩ verdict {gpt-4o, claude-opus-4}: only gpt-4o passes both layers.
        let scope = s.model_scope(&"narrow-team".to_string()).unwrap();
        assert!(scope.permits("gpt-4o"));
        assert!(!scope.permits("claude-opus-4"));
    }

    #[test]
    fn wildcard_allow_list_matches_patterns() {
        // Prefix (`openrouter/openai/*`), suffix (`*-mini`), and exact entries mix in one list;
        // matching is exact-first, wildcard-second.
        let s = KeyStore::from_config(&[key(
            "wild",
            BCRYPT_S3CRET,
            Some(vec![
                "openrouter/openai/*".into(),
                "*-mini".into(),
                "claude-opus-4".into(),
            ]),
        )]);
        let tok = b64("wild:s3cret");
        for allowed in [
            "openrouter/openai/gpt-4o",
            "openrouter/openai/o3", // prefix wildcard spans path segments
            "o3-mini",
            "claude-opus-4",
        ] {
            assert_eq!(
                s.authorize(&tok, allowed, 0),
                Ok("wild-team".to_string()),
                "model={allowed}"
            );
        }
        for denied in ["gpt-4o", "openrouter/mistral/large", "o3-mini-high"] {
            assert_eq!(
                s.authorize(&tok, denied, 0),
                Err(AuthError::ModelNotAllowed),
                "model={denied}"
            );
        }
        // The listing scope answers identically.
        let scope = s.model_scope(&"wild-team".to_string()).unwrap();
        assert!(scope.permits("openrouter/openai/gpt-4o"));
        assert!(!scope.permits("openrouter/mistral/large"));
    }

    #[test]
    fn wildcard_verdict_narrows() {
        // A verdict overlay may also carry patterns; both layers must pass. Config
        // `openrouter/*` narrowed by verdict `openrouter/openai/*`.
        let s = KeyStore::from_config(&[key(
            "wild",
            BCRYPT_S3CRET,
            Some(vec!["openrouter/*".into()]),
        )]);
        s.install_verdicts(verdicts(
            "wild",
            Verdict {
                allowed_models: Some(HashSet::from(["openrouter/openai/*".to_string()])),
                ..Default::default()
            },
        ));
        let tok = b64("wild:s3cret");
        assert_eq!(
            s.authorize(&tok, "openrouter/openai/gpt-4o", 0),
            Ok("wild-team".to_string())
        );
        assert_eq!(
            s.authorize(&tok, "openrouter/mistral/large", 0),
            Err(AuthError::ModelNotAllowed)
        );
        let scope = s.model_scope(&"wild-team".to_string()).unwrap();
        assert!(scope.permits("openrouter/openai/gpt-4o"));
        assert!(!scope.permits("openrouter/mistral/large"));
    }

    #[test]
    fn wildcard_match_cases() {
        for (pattern, model, expect) in [
            ("*", "anything", true),
            ("*", "", true),
            ("gpt-*", "gpt-4o", true),
            ("gpt-*", "gpt-", true),
            ("gpt-*", "gpt", false),
            ("openrouter/openai/*", "openrouter/openai/gpt-4o", true),
            ("openrouter/openai/*", "openrouter/openai/", true),
            ("openrouter/openai/*", "openrouter/openai", false),
            ("*-mini", "o3-mini", true),
            ("*-mini", "o3-mini-high", false),
            // Multiple stars require backtracking: the first `*` must not swallow the "4".
            ("gpt-*o*", "gpt-4o", true),
            ("a*b*c", "aXbYc", true),
            ("a*b*c", "aXbY", false),
            ("literal", "literal", true),
            ("literal", "literally", false),
        ] {
            assert_eq!(
                wildcard_match(pattern, model),
                expect,
                "pattern={pattern} model={model}"
            );
        }
    }

    #[test]
    fn star_allow_list_means_unrestricted() {
        // `allowed_models = ["*"]` is the "every routed model" convention (as in the OAuth
        // role mapping) — NOT a literal set entry that matches nothing. It must behave
        // exactly like omitting the list: any model authorizes, and the listing scope is
        // unrestricted (`None`).
        let s = KeyStore::from_config(&[key("star", BCRYPT_S3CRET, Some(vec!["*".to_string()]))]);
        let tok = b64("star:s3cret");
        assert_eq!(s.authorize(&tok, "gpt-4o", 0), Ok("star-team".to_string()));
        assert_eq!(s.model_scope(&"star-team".to_string()), None);
    }

    #[test]
    fn unknown_token_rejected() {
        // Well-formed base64 of an unknown id.
        assert_eq!(
            store().authorize(&b64("ghost:s3cret"), "gpt-4o", 0),
            Err(AuthError::Unknown)
        );
    }

    #[test]
    fn malformed_token_rejected() {
        let s = store();
        assert_eq!(
            s.authorize("not base64!", "gpt-4o", 0),
            Err(AuthError::Unknown)
        ); // bad alphabet
        assert_eq!(
            s.authorize(&b64("no-colon-here"), "gpt-4o", 0),
            Err(AuthError::Unknown)
        ); // no ':'
    }

    #[test]
    fn wrong_password_rejected() {
        // Right id, wrong password — indistinguishable from an unknown key (no oracle).
        assert_eq!(
            store().authorize(&b64("open:wrong"), "gpt-4o", 0),
            Err(AuthError::Unknown)
        );
    }

    #[test]
    fn correct_password_allows_any_model() {
        assert_eq!(
            store().authorize(&tok_open(), "anything", 0),
            Ok("open-team".into())
        );
    }

    #[test]
    fn config_restriction_enforced() {
        let s = store();
        assert_eq!(
            s.authorize(TOK_NARROW, "gpt-4o", 0),
            Ok("narrow-team".into())
        );
        assert_eq!(
            s.authorize(TOK_NARROW, "claude-opus-4", 0),
            Err(AuthError::ModelNotAllowed)
        );
    }

    /// One verdict for `id`, as the limits pull would install it (others reset to empty).
    fn verdicts(id: &str, verdict: Verdict) -> HashMap<KeyId, Verdict> {
        HashMap::from([(id.to_string(), verdict)])
    }

    fn ident(id: &str, hash: &str, allowed: Option<Vec<String>>) -> IdentityInput {
        IdentityInput {
            id: id.into(),
            pw_hash: hash.into(),
            name: Some(format!("{id}-team")),
            allowed_models: allowed,
        }
    }

    #[test]
    fn verdict_blocks_and_suspends() {
        let s = store();
        s.install_verdicts(verdicts(
            "open",
            Verdict {
                blocked: true,
                ..Default::default()
            },
        ));
        assert_eq!(
            s.authorize(&tok_open(), "gpt-4o", 0),
            Err(AuthError::Blocked)
        );

        s.install_verdicts(verdicts(
            "open",
            Verdict {
                suspended_until: Some(100),
                ..Default::default()
            },
        ));
        assert_eq!(
            s.authorize(&tok_open(), "gpt-4o", 50),
            Err(AuthError::Suspended { until: 100 })
        );
        // Past the suspension instant — a comparison, not arithmetic.
        assert_eq!(
            s.authorize(&tok_open(), "gpt-4o", 100),
            Ok("open-team".into())
        );
    }

    #[test]
    fn verdict_narrows_models() {
        let s = store();
        s.install_verdicts(verdicts(
            "open",
            Verdict {
                allowed_models: Some(["gpt-4o".to_string()].into()),
                ..Default::default()
            },
        ));
        assert_eq!(
            s.authorize(&tok_open(), "gpt-4o", 0),
            Ok("open-team".into())
        );
        assert_eq!(
            s.authorize(&tok_open(), "o3-mini", 0),
            Err(AuthError::ModelNotAllowed)
        );
    }

    #[test]
    fn verdict_survives_warm_cache() {
        // A warm cache hit must read the *new* verdict live, not a stale one — install_verdicts keeps
        // the identity generation so the fast path stays valid, but enforce() reads the live verdict.
        let s = store();
        assert_eq!(
            s.authorize(&tok_open(), "gpt-4o", 0),
            Ok("open-team".into())
        ); // warm the cache
        s.install_verdicts(verdicts(
            "open",
            Verdict {
                blocked: true,
                ..Default::default()
            },
        ));
        assert_eq!(
            s.authorize(&tok_open(), "gpt-4o", 0),
            Err(AuthError::Blocked)
        );
    }

    #[test]
    fn verified_cache_invalidated_on_password_rotation() {
        let s = store();
        // Warm the cache with the current token.
        assert_eq!(
            s.authorize(&tok_open(), "gpt-4o", 0),
            Ok("open-team".into())
        );
        // Rotate the password via an identity pull (a fresh bcrypt hash). The cached token must stop
        // working and the new one must start. The roster is installed wholesale, so keep `narrow` too.
        let new_hash = pwhash::bcrypt::hash("rotated").unwrap();
        s.install_identities(vec![
            ident("open", &new_hash, None),
            ident("narrow", BCRYPT_S3CRET, Some(vec!["gpt-4o".into()])),
        ]);
        assert_eq!(
            s.authorize(&tok_open(), "gpt-4o", 0),
            Err(AuthError::Unknown)
        );
        assert_eq!(
            s.authorize(&b64("open:rotated"), "gpt-4o", 0),
            Ok("open-team".into())
        );
        // The carried-over key still works after the identity swap.
        assert_eq!(
            s.authorize(TOK_NARROW, "gpt-4o", 0),
            Ok("narrow-team".into())
        );
    }

    #[test]
    fn removed_key_token_stops_working() {
        let s = store();
        assert_eq!(
            s.authorize(&tok_open(), "gpt-4o", 0),
            Ok("open-team".into())
        );
        // Re-pull the roster without `open` — it is gone from the identity layer.
        s.install_identities(vec![ident(
            "narrow",
            BCRYPT_S3CRET,
            Some(vec!["gpt-4o".into()]),
        )]);
        assert_eq!(
            s.authorize(&tok_open(), "gpt-4o", 0),
            Err(AuthError::Unknown)
        );
        assert_eq!(
            s.authorize(TOK_NARROW, "gpt-4o", 0),
            Ok("narrow-team".into())
        );
    }

    #[test]
    fn verdict_carried_over_across_identity_refresh() {
        // A verdict installed by the limits pull must survive a later identity pull (the layers are
        // independent; install_identities carries verdicts over for surviving keys).
        let s = store();
        s.install_verdicts(verdicts(
            "open",
            Verdict {
                blocked: true,
                ..Default::default()
            },
        ));
        s.install_identities(vec![
            ident("open", BCRYPT_S3CRET, None),
            ident("narrow", BCRYPT_S3CRET, Some(vec!["gpt-4o".into()])),
        ]);
        assert_eq!(
            s.authorize(&tok_open(), "gpt-4o", 0),
            Err(AuthError::Blocked)
        );
    }

    /// htpasswd/shadow compatibility: the formats the README promises must verify. Real fixtures
    /// generated by `htpasswd -bnBC 4` (bcrypt) and `openssl passwd -6` (sha512-crypt), password
    /// "s3cret" throughout.
    #[test]
    fn accepts_standard_crypt_formats() {
        const SHA512: &str = "$6$abcd1234$MiooO./xokT.7mIhy/bj.qbaiNambIgaBn/RoOefGiLVlXt2T5bqDDeYXigee/ANyK16C8amWBmrkG7Atq1Nx/";
        for hash in [BCRYPT_S3CRET, SHA512] {
            let s = KeyStore::from_config(&[key("k", hash, None)]);
            assert_eq!(
                s.authorize(&b64("k:s3cret"), "gpt-4o", 0),
                Ok("k-team".into()),
                "hash={hash}"
            );
            assert_eq!(
                s.authorize(&b64("k:nope"), "gpt-4o", 0),
                Err(AuthError::Unknown),
                "hash={hash}"
            );
        }
    }

    #[test]
    fn base64_decoder_roundtrips() {
        for s in [
            "",
            "a",
            "ab",
            "abc",
            "key-id:p@ss:word",
            "id:s3cret",
            "ünïcödé:pw",
        ] {
            let enc = b64(s);
            let mut buf = [0u8; 256];
            let n = b64_decode_into(&enc, &mut buf).expect("decodes");
            assert_eq!(&buf[..n], s.as_bytes(), "roundtrip {s:?}");
        }
        // Overflow a too-small buffer → None, no panic.
        let mut tiny = [0u8; 2];
        assert_eq!(b64_decode_into(&b64("hello"), &mut tiny), None);
    }
}
