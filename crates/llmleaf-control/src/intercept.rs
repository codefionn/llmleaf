//! The sync interceptor client (`[control.intercept]`): the reqwest implementation of the core's
//! [`Interceptor`] trait. POSTs `{phase, key, model, payload}` and maps the response `action` to a
//! verdict. The one configurable hot-path insertion (SOUL.md principle 1) — opt-in per key/model, with
//! its own timeout and fail-open/closed policy. The core holds only the trait; this is the HTTP side.

use std::collections::HashSet;
use std::time::Duration;

use llmleaf_core::{
    InterceptHook, InterceptOutcome, InterceptPhase, Interceptor, OnError, ResolvedAuth,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// A reqwest-backed [`Interceptor`]. Built from `[control.intercept]` config.
pub struct HttpInterceptor {
    http: reqwest::Client,
    url: String,
    auth: Option<ResolvedAuth>,
    timeout: Duration,
    on_error: OnError,
    phases: HashSet<InterceptPhase>,
    keys: Option<HashSet<String>>,
    models: Option<HashSet<String>>,
}

#[derive(Serialize)]
struct InterceptRequest<'a> {
    phase: InterceptPhase,
    key: &'a str,
    model: &'a str,
    payload: &'a Value,
}

#[derive(Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
enum InterceptResponse {
    Pass,
    Block {
        #[serde(default)]
        reason: Option<String>,
    },
    Rewrite {
        payload: Value,
    },
}

impl HttpInterceptor {
    pub fn new(http: reqwest::Client, cfg: &InterceptHook, auth: Option<ResolvedAuth>) -> Self {
        HttpInterceptor {
            http,
            url: cfg.url.clone(),
            auth,
            timeout: Duration::from_millis(cfg.timeout_ms.max(1)),
            on_error: cfg.on_error,
            phases: cfg.phases.iter().copied().collect(),
            keys: cfg.keys.as_ref().map(|v| v.iter().cloned().collect()),
            models: cfg.models.as_ref().map(|v| v.iter().cloned().collect()),
        }
    }

    /// Apply the configured failure policy: fail open ⇒ pass, fail closed ⇒ block.
    fn on_error_outcome(&self, reason: &str) -> InterceptOutcome {
        match self.on_error {
            OnError::Allow => InterceptOutcome::Pass,
            OnError::Deny => InterceptOutcome::Block {
                reason: reason.to_string(),
            },
        }
    }
}

#[async_trait::async_trait]
impl Interceptor for HttpInterceptor {
    fn applies(&self, phase: InterceptPhase, key: &str, model: &str) -> bool {
        self.phases.contains(&phase)
            && self.keys.as_ref().is_none_or(|k| k.contains(key))
            && self.models.as_ref().is_none_or(|m| m.contains(model))
    }

    async fn screen(
        &self,
        phase: InterceptPhase,
        key: &str,
        model: &str,
        payload: &Value,
    ) -> InterceptOutcome {
        let body = InterceptRequest {
            phase,
            key,
            model,
            payload,
        };
        let req = crate::apply_auth(
            self.http.post(&self.url).timeout(self.timeout).json(&body),
            self.auth.as_ref(),
        );
        let resp = match req.send().await.and_then(|r| r.error_for_status()) {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!(error = %e, url = %self.url, "interceptor call failed; applying on_error");
                return self.on_error_outcome("interceptor unavailable");
            }
        };
        match resp.json::<InterceptResponse>().await {
            Ok(InterceptResponse::Pass) => InterceptOutcome::Pass,
            Ok(InterceptResponse::Block { reason }) => InterceptOutcome::Block {
                reason: reason.unwrap_or_else(|| "blocked by interceptor".to_string()),
            },
            Ok(InterceptResponse::Rewrite { payload }) => InterceptOutcome::Rewrite(payload),
            Err(e) => {
                tracing::warn!(error = %e, url = %self.url, "interceptor response invalid; applying on_error");
                self.on_error_outcome("interceptor response invalid")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llmleaf_core::Secret;

    fn hook(
        phases: Vec<InterceptPhase>,
        keys: Option<Vec<String>>,
        models: Option<Vec<String>>,
    ) -> InterceptHook {
        InterceptHook {
            url: "http://localhost/screen".into(),
            credential: Some(Secret::new("env:DEFINITELY_UNSET_VAR")),
            auth: None,
            phases,
            keys,
            models,
            timeout_ms: 100,
            on_error: OnError::Allow,
        }
    }

    fn interceptor(h: InterceptHook) -> HttpInterceptor {
        HttpInterceptor::new(reqwest::Client::new(), &h, None)
    }

    #[test]
    fn applies_respects_phase_key_and_model_filters() {
        let ic = interceptor(hook(
            vec![InterceptPhase::Request],
            Some(vec!["k1".into()]),
            Some(vec!["gpt-4o".into()]),
        ));
        assert!(ic.applies(InterceptPhase::Request, "k1", "gpt-4o"));
        // Wrong phase / key / model each disqualify.
        assert!(!ic.applies(InterceptPhase::Response, "k1", "gpt-4o"));
        assert!(!ic.applies(InterceptPhase::Request, "other", "gpt-4o"));
        assert!(!ic.applies(InterceptPhase::Request, "k1", "other"));
    }

    #[test]
    fn applies_unscoped_matches_every_key_and_model() {
        let ic = interceptor(hook(vec![InterceptPhase::Request], None, None));
        assert!(ic.applies(InterceptPhase::Request, "anyone", "anything"));
        assert!(!ic.applies(InterceptPhase::Response, "anyone", "anything"));
    }

    #[test]
    fn on_error_outcome_honors_policy() {
        let allow = interceptor(InterceptHook {
            on_error: OnError::Allow,
            ..hook(vec![], None, None)
        });
        assert!(matches!(
            allow.on_error_outcome("x"),
            InterceptOutcome::Pass
        ));
        let deny = interceptor(InterceptHook {
            on_error: OnError::Deny,
            ..hook(vec![], None, None)
        });
        assert!(matches!(
            deny.on_error_outcome("x"),
            InterceptOutcome::Block { .. }
        ));
    }

    #[test]
    fn response_actions_deserialize() {
        let pass: InterceptResponse = serde_json::from_str(r#"{"action":"pass"}"#).unwrap();
        assert!(matches!(pass, InterceptResponse::Pass));
        let block: InterceptResponse =
            serde_json::from_str(r#"{"action":"block","reason":"nope"}"#).unwrap();
        assert!(matches!(block, InterceptResponse::Block { reason: Some(r) } if r == "nope"));
        let block_bare: InterceptResponse = serde_json::from_str(r#"{"action":"block"}"#).unwrap();
        assert!(matches!(
            block_bare,
            InterceptResponse::Block { reason: None }
        ));
        let rewrite: InterceptResponse =
            serde_json::from_str(r#"{"action":"rewrite","payload":{"model":"x"}}"#).unwrap();
        assert!(matches!(rewrite, InterceptResponse::Rewrite { .. }));
    }
}
