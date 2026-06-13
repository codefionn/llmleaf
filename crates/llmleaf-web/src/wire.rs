//! The control-plane wire contract — the exact JSON shapes llmleaf-core PULLS and PUSHES.
//!
//! This crate is a SEPARATE component that speaks the protocol; it deliberately does NOT link
//! `llmleaf-core`. These types mirror the core's contract on the wire (the JSON shapes). A few Rust
//! types are intentionally looser where it is wire-transparent — a model list as `Vec<String>` vs the
//! core's `HashSet<String>`, a `finish` reason as `String` vs the core's `FinishReason` enum (so a new
//! core variant never breaks ingestion). Field *names*, however, must match exactly (see `cost_usd`).
//! The contracts:
//!   - identity  (core PULLs):  `GET  -> { "keys": [ KeyDto ] }`         (llmleaf-control IdentityRefresher)
//!   - verdicts  (core PULLs):  `GET  -> { "verdicts": { id: Verdict } }`(llmleaf-control VerdictRefresher)
//!   - usage     (core PUSHes): `POST <- { "events": [ Envelope ] }`     (llmleaf-control UsageReporter)
//!
//! The tests below pin the shapes against the literal payloads in the core/control test suite. If the
//! core changes the contract, these tests are where it surfaces.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// `skip_serializing_if` helper: omit a `bool` field when it is `false`, keeping the wire object minimal.
fn is_false(b: &bool) -> bool {
    !*b
}

// ---------------------------------------------------------------------------------------------
// Identity pull: { "keys": [ { id, pw_hash, name?, allowed_models? } ] }
// ---------------------------------------------------------------------------------------------

/// One key in the roster the core pulls. `pw_hash` is a crypt(3) MCF string (bcrypt `$2*$`, or
/// `$1$/$5$/$6$`), never plaintext — exactly as the core's `[[keys]]` and identity pull require.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct KeyDto {
    pub id: String,
    pub pw_hash: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allowed_models: Option<Vec<String>>,
}

/// The identity response body the core's `IdentityRefresher` deserializes.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IdentityResponse {
    #[serde(default)]
    pub keys: Vec<KeyDto>,
}

// ---------------------------------------------------------------------------------------------
// Verdict pull: { "verdicts": { id: { blocked, suspended_until?, allowed_models? } } }
// ---------------------------------------------------------------------------------------------

/// A per-key verdict, mirroring `llmleaf_core::keys::Verdict`. Precedence (enforced by the core):
/// blocked ▸ suspended ▸ model allow-list. `allowed_models` here is a runtime *narrowing* layered on
/// top of a key's static `allowed_models`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct Verdict {
    #[serde(default, skip_serializing_if = "is_false")]
    pub blocked: bool,
    /// Suspended while `now < suspended_until` (unix **seconds**). A comparison, not a countdown.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub suspended_until: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allowed_models: Option<Vec<String>>,
}

/// The verdict response body the core's `VerdictRefresher` deserializes.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VerdictResponse {
    #[serde(default)]
    pub verdicts: HashMap<String, Verdict>,
}

// ---------------------------------------------------------------------------------------------
// Usage push: { "events": [ Envelope ] }, Envelope = { ts_ms, event, ...event-fields }
// ---------------------------------------------------------------------------------------------

/// Provider-reported token usage, mirroring `llmleaf_model::Usage`. All fields optional/defaulted so a
/// partial provider report still parses. The JSON key is `cost_usd` (NOT `cost`) — it must match the
/// core's `llmleaf_model::Usage::cost_usd` exactly, or pushed costs deserialize to `None` and all cost
/// accounting + the limiter's cost cap silently see zero.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Usage {
    #[serde(default)]
    pub prompt_tokens: u64,
    #[serde(default)]
    pub completion_tokens: u64,
    #[serde(default)]
    pub total_tokens: u64,
    /// Cost in USD, filled by the core's pricing lookup. Wire key: `cost_usd`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost_usd: Option<f64>,
}

/// A lifecycle/usage event, mirroring `llmleaf_core::events::Event` (internally tagged on `event`,
/// snake_case). Unknown future variants are tolerated via `#[serde(other)]` so a newer core never
/// breaks ingestion.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum Event {
    RequestStarted {
        id: String,
        key: String,
        model: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        request: Option<serde_json::Value>,
    },
    RequestRouted {
        id: String,
        provider: String,
        upstream_model: String,
    },
    Usage {
        id: String,
        key: String,
        model: String,
        usage: Usage,
    },
    RequestCompleted {
        id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        finish: Option<String>,
    },
    RequestFailed {
        id: String,
        error: String,
    },
    ProviderHealth {
        provider: String,
        status: String,
    },
    /// Any event variant this build does not know about. Keeps ingestion forward-compatible.
    #[serde(other)]
    Unknown,
}

/// An event with the wall-clock instant it was emitted, in unix **milliseconds**. The core flattens
/// the tagged `Event` onto this, so the wire object is `{ "ts_ms": .., "event": "..", .. }`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Envelope {
    pub ts_ms: u64,
    #[serde(flatten)]
    pub event: Event,
}

/// The usage batch body the core's `UsageReporter` POSTs.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UsageBatch {
    #[serde(default)]
    pub events: Vec<Envelope>,
}

#[cfg(test)]
mod tests {
    use super::*;

    // The literal bodies below are copied from the core/control test suites; they pin our parsing to
    // the real contract.

    #[test]
    fn identity_roster_parses_contract_shape() {
        let body = r#"{
            "keys": [
                { "id": "demo-team", "pw_hash": "$2y$12$abc", "name": "demo", "allowed_models": ["gpt-4o", "demo"] },
                { "id": "minimal", "pw_hash": "$6$xyz" }
            ]
        }"#;
        let resp: IdentityResponse = serde_json::from_str(body).unwrap();
        assert_eq!(resp.keys.len(), 2);
        assert_eq!(resp.keys[0].id, "demo-team");
        assert_eq!(resp.keys[0].allowed_models.as_deref().unwrap().len(), 2);
        assert_eq!(resp.keys[1].name, None);
        assert_eq!(resp.keys[1].allowed_models, None);
    }

    #[test]
    fn key_dto_roundtrips_without_optional_fields() {
        let k = KeyDto {
            id: "minimal".into(),
            pw_hash: "$6$xyz".into(),
            name: None,
            allowed_models: None,
        };
        let json = serde_json::to_string(&k).unwrap();
        // Optional Nones are omitted so the wire object stays minimal.
        assert_eq!(json, r#"{"id":"minimal","pw_hash":"$6$xyz"}"#);
        let back: KeyDto = serde_json::from_str(&json).unwrap();
        assert_eq!(k, back);
    }

    #[test]
    fn verdict_response_parses_contract_shape() {
        let body = r#"{
            "verdicts": {
                "demo-team":  { "blocked": false, "suspended_until": 1765000000, "allowed_models": ["gpt-4o"] },
                "noisy-team": { "blocked": true }
            }
        }"#;
        let resp: VerdictResponse = serde_json::from_str(body).unwrap();
        assert_eq!(resp.verdicts.len(), 2);
        assert_eq!(resp.verdicts["demo-team"].suspended_until, Some(1765000000));
        assert_eq!(
            resp.verdicts["demo-team"]
                .allowed_models
                .as_deref()
                .unwrap()
                .len(),
            1
        );
        assert!(resp.verdicts["noisy-team"].blocked);
    }

    #[test]
    fn empty_verdict_default_serializes_minimal() {
        // A clean key emits `{}` (all fields defaulted/omitted), which the core reads as unrestricted.
        assert_eq!(serde_json::to_string(&Verdict::default()).unwrap(), "{}");
    }

    #[test]
    fn usage_batch_parses_flattened_tagged_events() {
        let body = r#"{
            "events": [
                { "ts_ms": 1765000000000, "event": "request_started", "id": "r1", "key": "demo-team", "model": "gpt-4o" },
                { "ts_ms": 1765000000100, "event": "request_routed", "id": "r1", "provider": "openai-main", "upstream_model": "gpt-4o" },
                { "ts_ms": 1765000000900, "event": "usage", "id": "r1", "key": "demo-team", "model": "gpt-4o",
                  "usage": { "prompt_tokens": 12, "completion_tokens": 34, "total_tokens": 46, "cost_usd": 0.0009 } },
                { "ts_ms": 1765000001000, "event": "request_completed", "id": "r1", "finish": "stop" }
            ]
        }"#;
        let batch: UsageBatch = serde_json::from_str(body).unwrap();
        assert_eq!(batch.events.len(), 4);
        assert_eq!(batch.events[0].ts_ms, 1765000000000);
        match &batch.events[2].event {
            Event::Usage { usage, key, .. } => {
                assert_eq!(usage.total_tokens, 46);
                // Pins the wire key: the core emits `cost_usd`, not `cost`. With the wrong key this
                // parses to `None` and the whole cost pipeline silently zeroes.
                assert_eq!(usage.cost_usd, Some(0.0009));
                assert_eq!(key, "demo-team");
            }
            other => panic!("expected usage, got {other:?}"),
        }
    }

    #[test]
    fn usage_with_wrong_cost_key_is_none() {
        // Guard against regressing to the `cost` key: a payload using `cost` must NOT populate cost_usd.
        let u: Usage = serde_json::from_str(r#"{ "total_tokens": 5, "cost": 1.23 }"#).unwrap();
        assert_eq!(u.cost_usd, None);
        let u: Usage = serde_json::from_str(r#"{ "total_tokens": 5, "cost_usd": 1.23 }"#).unwrap();
        assert_eq!(u.cost_usd, Some(1.23));
    }

    #[test]
    fn unknown_event_variant_is_tolerated() {
        let body =
            r#"{ "events": [ { "ts_ms": 1, "event": "some_future_event", "whatever": 5 } ] }"#;
        let batch: UsageBatch = serde_json::from_str(body).unwrap();
        assert!(matches!(batch.events[0].event, Event::Unknown));
    }
}
