//! Simulation of key authorization: the token format (`base64(key-id:password)` over a hashed
//! password), the config allow-list base (including `*` wildcard entries), and the pulled verdict
//! overlay, enforced over simulated time (SOUL.md principle 5: a lookup and a comparison, never
//! arithmetic; principle 6: config is the base, the pulled control plane is a layer).
//!
//! [`KeyStore::authorize`] takes the caller-supplied `now`, so suspensions can be probed exactly at,
//! before, and after their boundary. The reference model mirrors the documented precedence — including
//! the two "no" cases the token format introduces (an unknown key-id and a wrong password, both
//! `Unknown`) — and is kept in lock-step with the store through the same timeline of *verdict-overlay*
//! installs ([`KeyStore::install_verdicts`]) and *identity-roster* installs
//! ([`KeyStore::install_identities`], which rotate passwords, add, and remove keys wholesale).
//!
//! The store caches a proven token id-ward to keep the password KDF off the warm hot path; presenting
//! the same correct token across ops exercises that cache, while an identity install that rotates a
//! password must invalidate it. A verdict install keeps identities (and the cache) but must be read
//! live. Only the seeded [`Rng`] drives outcomes: the crypt salt inside each generated hash is
//! incidental (verification is salt-agnostic), so results stay reproducible by seed.

use std::collections::{HashMap, HashSet};

use llmleaf_core::config::{KeyConfig, Secret};
use llmleaf_core::{AuthError, IdentityInput, KeyStore, Verdict};

use crate::ensure;
use crate::rng::Rng;

/// The reference mirror of a single key's enforced state. `None` for `config_allowed` /
/// `verdict.allowed_models` means no restriction at that layer.
#[derive(Clone, Default)]
struct RefKey {
    name: Option<String>,
    /// The current correct plaintext password (rotated by upserts).
    password: String,
    config_allowed: Option<HashSet<String>>,
    verdict: RefVerdict,
}

impl RefKey {
    /// The log-safe display id the store returns on success: the friendly name, else the key-id.
    fn display_id(&self, id: &str) -> String {
        self.name.clone().unwrap_or_else(|| id.to_string())
    }
}

#[derive(Clone, Default)]
struct RefVerdict {
    blocked: bool,
    suspended_until: Option<u64>,
    allowed_models: Option<HashSet<String>>,
}

/// Predict an authorization result, applying the documented precedence:
/// unknown id ▸ wrong password ▸ blocked ▸ suspended ▸ config allow-list ▸ verdict allow-list ▸ allowed.
fn predict_auth(
    keys: &HashMap<String, RefKey>,
    id: &str,
    password: &str,
    model: &str,
    now: u64,
) -> Result<String, AuthError> {
    let Some(key) = keys.get(id) else {
        return Err(AuthError::Unknown);
    };
    if password != key.password {
        return Err(AuthError::Unknown);
    }
    if key.verdict.blocked {
        return Err(AuthError::Blocked);
    }
    if let Some(until) = key.verdict.suspended_until {
        if now < until {
            return Err(AuthError::Suspended { until });
        }
    }
    if let Some(allowed) = &key.config_allowed {
        if !ref_permits(allowed, model) {
            return Err(AuthError::ModelNotAllowed);
        }
    }
    if let Some(allowed) = &key.verdict.allowed_models {
        if !ref_permits(allowed, model) {
            return Err(AuthError::ModelNotAllowed);
        }
    }
    Ok(key.display_id(id))
}

/// Reference allow-list matching: an exact entry, or any entry containing `*` treated as a wildcard.
fn ref_permits(set: &HashSet<String>, model: &str) -> bool {
    set.iter()
        .any(|p| p == model || (p.contains('*') && ref_wildcard(p.as_bytes(), model.as_bytes())))
}

/// Independent wildcard matcher for the reference model — deliberately recursive, not the core's
/// iterative backtracking glob, so the two implementations cross-check each other. `*` matches any
/// (possibly empty) run of bytes.
fn ref_wildcard(pattern: &[u8], text: &[u8]) -> bool {
    match pattern.first() {
        None => text.is_empty(),
        Some(b'*') => (0..=text.len()).any(|i| ref_wildcard(&pattern[1..], &text[i..])),
        Some(&c) => text.first() == Some(&c) && ref_wildcard(&pattern[1..], &text[1..]),
    }
}

fn random_subset(rng: &mut Rng, universe: &[String]) -> HashSet<String> {
    // May be empty (a key allowed no models — a valid, if useless, restriction).
    let mut set: HashSet<String> = universe.iter().filter(|_| rng.bool()).cloned().collect();
    // Sometimes mix in `*` wildcard entries: one matching the whole `model-N` universe, one matching
    // a single model by prefix (or nothing, when the index is past the universe), one matching
    // nothing, and occasionally a bare `*` (which the core collapses to "unrestricted" in config
    // allow-lists and matches everything in verdicts — observably identical to the mirror treating
    // it as a match-all pattern).
    if rng.chance(1, 4) {
        set.insert("model-*".to_string());
    }
    if rng.chance(1, 4) {
        set.insert(format!("model-{}*", rng.below(5)));
    }
    if rng.chance(1, 4) {
        set.insert("other-*".to_string());
    }
    if rng.chance(1, 8) {
        set.insert("*".to_string());
    }
    set
}

/// Generate a fresh crypt(3) hash for a password. md5-crypt keeps the sweep fast (bcrypt's KDF would
/// dominate); the core's own tests cover bcrypt/sha512 verification. Salt is incidental to outcomes.
#[allow(deprecated)]
fn hash(password: &str) -> String {
    pwhash::md5_crypt::hash(password).expect("md5_crypt hash")
}

/// The bearer token a consumer presents: `base64(key-id:password)`. The core only ever decodes, so
/// the encoder lives here in the harness.
fn token(id: &str, password: &str) -> String {
    b64_encode(format!("{id}:{password}").as_bytes())
}

/// Standard-alphabet base64 (RFC 4648) with `=` padding — the mirror of the core's decoder.
fn b64_encode(input: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(input.len().div_ceil(3) * 4);
    for chunk in input.chunks(3) {
        let b0 = chunk[0];
        let b1 = *chunk.get(1).unwrap_or(&0);
        let b2 = *chunk.get(2).unwrap_or(&0);
        out.push(ALPHABET[(b0 >> 2) as usize] as char);
        out.push(ALPHABET[(((b0 & 0x03) << 4) | (b1 >> 4)) as usize] as char);
        out.push(if chunk.len() > 1 {
            ALPHABET[(((b1 & 0x0f) << 2) | (b2 >> 6)) as usize] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            ALPHABET[(b2 & 0x3f) as usize] as char
        } else {
            '='
        });
    }
    out
}

/// Run one full key-authorization scenario for `seed`.
pub fn run_scenario(seed: u64) -> Result<(), String> {
    let mut rng = Rng::new(seed);

    // A small universe of models and key-ids. Some ids are never created (to exercise the unknown-id
    // `Unknown`), and requests sometimes name a model outside any allow-list, or a wrong password.
    let all_models: Vec<String> = (0..rng.range(2, 4)).map(|i| format!("model-{i}")).collect();
    let known_ids: Vec<String> = (0..rng.range(1, 4)).map(|i| format!("key-{i}")).collect();
    let ghost_ids: Vec<String> = vec!["key-ghost".to_string()];
    let probe_ids: Vec<String> = known_ids.iter().chain(ghost_ids.iter()).cloned().collect();

    // A monotonic counter so rotated passwords are always distinct from their predecessors.
    let mut pw_seq = 0u64;
    let mut next_password = |rng: &mut Rng| {
        pw_seq += 1;
        format!("pw-{}-{}", rng.range(0, 1_000_000), pw_seq)
    };

    // Base keys from "config": each gets a password (stored only hashed) and an optional allow-list.
    let mut config_keys = Vec::new();
    let mut reference: HashMap<String, RefKey> = HashMap::new();
    for id in &known_ids {
        let config_allowed = if rng.chance(1, 2) {
            Some(random_subset(&mut rng, &all_models))
        } else {
            None
        };
        let name = if rng.chance(2, 3) {
            Some(format!("name-of-{id}"))
        } else {
            None
        };
        let password = next_password(&mut rng);
        config_keys.push(KeyConfig {
            id: id.clone(),
            pw_hash: Secret::new(hash(&password)),
            name: name.clone(),
            allowed_models: config_allowed.as_ref().map(|s| s.iter().cloned().collect()),
        });
        reference.insert(
            id.clone(),
            RefKey {
                name,
                password,
                config_allowed,
                verdict: RefVerdict::default(),
            },
        );
    }

    let store = KeyStore::from_config(&config_keys);

    let mut now = rng.range(1_000, 100_000);
    let nops = rng.range(40, 90);
    for step in 0..nops {
        now += rng.range(0, 50);

        match rng.below(10) {
            // Install a fresh verdict overlay (the limits pull). Replaces EVERY existing key's verdict
            // at once — a key absent from the map gets the default (no restriction). Identities, and so
            // the warm token cache, are untouched; the new verdict must be read live.
            0..=2 => {
                let mut verdicts: HashMap<String, Verdict> = HashMap::new();
                let mut ref_verdicts: HashMap<String, RefVerdict> = HashMap::new();
                for id in &known_ids {
                    if rng.bool() {
                        continue; // this key carries no verdict this round
                    }
                    let blocked = rng.chance(1, 4);
                    let suspended_until = if rng.chance(1, 3) {
                        // Straddle `now`: some suspensions are already expired, some still in force.
                        Some(now.saturating_add(rng.range(0, 40)).saturating_sub(20))
                    } else {
                        None
                    };
                    let allowed_models = if rng.chance(1, 3) {
                        Some(random_subset(&mut rng, &all_models))
                    } else {
                        None
                    };
                    verdicts.insert(
                        id.clone(),
                        Verdict {
                            blocked,
                            suspended_until,
                            allowed_models: allowed_models
                                .as_ref()
                                .map(|s| s.iter().cloned().collect()),
                        },
                    );
                    ref_verdicts.insert(
                        id.clone(),
                        RefVerdict {
                            blocked,
                            suspended_until,
                            allowed_models,
                        },
                    );
                }
                store.install_verdicts(verdicts);
                // Reference mirror: every key currently in the store gets its verdict replaced (default
                // when absent from the map); ids in the map but not present are ignored by both sides.
                for (id, key) in reference.iter_mut() {
                    key.verdict = ref_verdicts.remove(id).unwrap_or_default();
                }
            }
            // Install a fresh identity roster (the identity pull). Replaces the whole identity layer:
            // keys not in the roster are removed; surviving keys keep their verdict; new keys appear;
            // passwords/allow-lists are taken from the roster. A rotated password must invalidate the
            // cached token (the store's warm path re-checks the hash).
            3 => {
                let mut roster: Vec<IdentityInput> = Vec::new();
                let mut new_ref: HashMap<String, RefKey> = HashMap::new();
                for id in &probe_ids {
                    if rng.bool() {
                        continue; // this key is absent from the new roster
                    }
                    let config_allowed = if rng.chance(1, 2) {
                        Some(random_subset(&mut rng, &all_models))
                    } else {
                        None
                    };
                    let name = if rng.chance(2, 3) {
                        Some(format!("ident-{id}"))
                    } else {
                        None
                    };
                    let password = next_password(&mut rng);
                    roster.push(IdentityInput {
                        id: id.clone(),
                        pw_hash: hash(&password),
                        name: name.clone(),
                        allowed_models: config_allowed
                            .as_ref()
                            .map(|s| s.iter().cloned().collect()),
                    });
                    // Carry over the existing verdict for a surviving key (the store does the same).
                    let verdict = reference
                        .get(id)
                        .map(|k| k.verdict.clone())
                        .unwrap_or_default();
                    new_ref.insert(
                        id.clone(),
                        RefKey {
                            name,
                            password,
                            config_allowed,
                            verdict,
                        },
                    );
                }
                store.install_identities(roster);
                reference = new_ref;
            }
            // The hot-path check: authorize and compare against the reference precedence. Sometimes
            // present a deliberately-wrong password to exercise the verification-failure `Unknown`.
            _ => {
                let id = rng.pick(&probe_ids).clone();
                let model = rng.pick(&all_models).clone();
                // The correct password for an existing key (so warm hits and verdicts are exercised),
                // or a wrong one a third of the time. A ghost id has no reference password.
                let use_wrong = rng.chance(1, 3);
                let password = if use_wrong {
                    "definitely-wrong".to_string()
                } else {
                    reference
                        .get(&id)
                        .map(|k| k.password.clone())
                        .unwrap_or_else(|| "x".to_string())
                };
                let tok = token(&id, &password);
                let actual = store.authorize(&tok, &model, now);
                let expected = predict_auth(&reference, &id, &password, &model, now);
                ensure!(
                    actual == expected,
                    "seed={seed} step={step}: authorize(id={id}, wrong_pw={use_wrong}, model={model}, now={now})\n  actual:   {actual:?}\n  expected: {expected:?}"
                );
            }
        }
    }

    Ok(())
}
