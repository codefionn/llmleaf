//! TypeSafe's native System One Decisions provider.

use std::sync::Arc;

use async_trait::async_trait;
use llmleaf_model::{
    ChatRequest, DecisionsRequest, DecisionsResponse, Modality, ModelError, ModelInfo,
    ResponseStream,
};
use llmleaf_provider::{Provider, ProviderCx};
use serde_json::Value;

use crate::decisions::{decisions_request_to_wire, decisions_response_from_wire};
use crate::http::post_json;
use crate::transport::{HttpRequest, HttpTransport, Transports};

const DEFAULT_ENDPOINT: &str = "https://api.typesafe.ai/v1";

/// Native TypeSafe System One edge. System One is a decisions API, not a chat implementation.
#[derive(Clone)]
pub struct TypeSafeProvider {
    http: Arc<dyn HttpTransport>,
}

impl TypeSafeProvider {
    pub fn new(transports: &Transports) -> Self {
        Self {
            http: transports.http.clone(),
        }
    }

    fn decisions_url(&self, cx: &ProviderCx) -> String {
        let endpoint = cx
            .endpoint
            .as_deref()
            .unwrap_or(DEFAULT_ENDPOINT)
            .trim_end_matches('/');
        format!("{endpoint}/systemone")
    }

    fn models_url(&self, cx: &ProviderCx) -> String {
        let endpoint = cx
            .endpoint
            .as_deref()
            .unwrap_or(DEFAULT_ENDPOINT)
            .trim_end_matches('/');
        format!("{endpoint}/models")
    }

    fn apply_auth(&self, req: HttpRequest, cx: &ProviderCx) -> HttpRequest {
        match &cx.credential {
            Some(credential) => req.bearer(credential),
            None => req,
        }
    }
}

#[async_trait]
impl Provider for TypeSafeProvider {
    fn name(&self) -> &str {
        "typesafe"
    }

    async fn chat(
        &self,
        _req: ChatRequest,
        _cx: &ProviderCx,
    ) -> Result<ResponseStream, ModelError> {
        Err(ModelError::Unsupported(
            "provider 'typesafe' does not support chat".into(),
        ))
    }

    async fn decisions(
        &self,
        req: DecisionsRequest,
        cx: &ProviderCx,
    ) -> Result<DecisionsResponse, ModelError> {
        let url = self.decisions_url(cx);
        let request = self.apply_auth(
            HttpRequest::post(url).json(decisions_request_to_wire(&req)),
            cx,
        );
        let response = post_json(&*self.http, request).await?;
        decisions_response_from_wire(response, &req.model)
    }

    async fn models(&self, cx: &ProviderCx) -> Result<Vec<ModelInfo>, ModelError> {
        let request = self.apply_auth(HttpRequest::get(self.models_url(cx)), cx);
        let response = post_json(&*self.http, request).await?;
        Ok(typesafe_models_to_canonical(response))
    }
}

/// Map TypeSafe's native `GET /v1/models` response into the shared catalog form. Every listed
/// model uses System One, so its output is always a structured decision. The endpoint only lists
/// models enabled for an account, including moving aliases; it may omit accepted pinned versions.
fn typesafe_models_to_canonical(value: Value) -> Vec<ModelInfo> {
    let Value::Object(mut envelope) = value else {
        return Vec::new();
    };
    let Some(Value::Array(models)) = envelope.remove("models") else {
        return Vec::new();
    };
    models
        .into_iter()
        .filter_map(|item| {
            let Value::Object(mut item) = item else {
                return None;
            };
            let name = item.remove("name")?.as_str()?.to_owned();
            if name.is_empty() {
                return None;
            }
            let mut info = ModelInfo::new(name);
            info.modality = Some(Modality::Decisions);
            // Description is a human-readable label for a TypeSafe model, not a separate stable
            // display-name field. Keep it verbatim alongside release_date and any future fields.
            info.extra = item;
            Some(info)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use serde_json::json;

    use super::*;
    use crate::fake::{FakeHttpTransport, FakeRealtimeTransport, FakeResponse};
    use crate::transport::{Method, Transports};

    fn transports(http: FakeHttpTransport) -> Transports {
        Transports {
            http: Arc::new(http),
            realtime: Arc::new(FakeRealtimeTransport::scripted(Vec::new())),
        }
    }

    #[tokio::test]
    async fn models_uses_native_url_auth_and_preserves_type_safe_metadata() {
        let http = FakeHttpTransport::new(|request| {
            assert_eq!(request.method, Method::Get);
            assert_eq!(request.url, "https://api.typesafe.ai/v1/models");
            assert!(request
                .headers
                .iter()
                .any(|(name, value)| { name == "Authorization" && value == "Bearer test-key" }));
            Ok(FakeResponse::ok_json(&json!({ "models": [
                {
                    "name": "jev-latest",
                    "description": "The stable System One release",
                    "release_date": "2026-09-17"
                }
            ] })))
        });
        let provider = TypeSafeProvider::new(&transports(http));
        let cx = ProviderCx {
            credential: Some("test-key".into()),
            ..Default::default()
        };
        let models = provider.models(&cx).await.unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "jev-latest");
        assert_eq!(models[0].modality, Some(Modality::Decisions));
        assert_eq!(
            models[0].extra["description"],
            "The stable System One release"
        );
        assert_eq!(models[0].extra["release_date"], "2026-09-17");
    }

    #[tokio::test]
    async fn models_honors_a_custom_base_url() {
        let http = FakeHttpTransport::new(|request| {
            assert_eq!(request.url, "https://typesafe.example/custom/models");
            Ok(FakeResponse::ok_json(&json!({ "models": [] })))
        });
        let provider = TypeSafeProvider::new(&transports(http));
        let cx = ProviderCx {
            endpoint: Some("https://typesafe.example/custom/".into()),
            ..Default::default()
        };
        assert!(provider.models(&cx).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn models_preserves_upstream_and_malformed_response_errors() {
        let upstream = TypeSafeProvider::new(&transports(FakeHttpTransport::status(503, "down")));
        assert!(matches!(
            upstream.models(&ProviderCx::default()).await,
            Err(ModelError::Upstream { status: 503, message }) if message == "down"
        ));

        let malformed = TypeSafeProvider::new(&transports(FakeHttpTransport::json(json!({
            "models": { "name": "jev-latest" }
        }))));
        assert!(malformed
            .models(&ProviderCx::default())
            .await
            .unwrap()
            .is_empty());

        let invalid_json =
            TypeSafeProvider::new(&transports(FakeHttpTransport::status(200, "nope")));
        assert!(matches!(
            invalid_json.models(&ProviderCx::default()).await,
            Err(ModelError::Mapping(_))
        ));
    }
}
