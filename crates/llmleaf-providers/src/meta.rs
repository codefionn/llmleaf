//! Meta Model API (`api.meta.ai`).
//!
//! Meta retired the hosted Llama API at `api.llama.com`; its replacement serves Muse models over
//! OpenAI-compatible Responses and Chat Completions endpoints. Keep the public `MetaProvider` type as
//! a thin edge wrapper, while the shared compatibility layer owns the wire mapping.

use async_trait::async_trait;
use llmleaf_model::{ChatRequest, ModelError, ModelInfo, ResponseStream};
use llmleaf_provider::{Provider, ProviderCx};

use crate::compat::{Brand, OpenAiCompatProvider};
use crate::transport::Transports;

pub struct MetaProvider {
    compat: OpenAiCompatProvider,
}

impl MetaProvider {
    pub fn new(transports: &Transports) -> Self {
        let brand = Brand::for_kind("meta").expect("Meta brand is registered");
        Self {
            compat: OpenAiCompatProvider::new(brand, transports),
        }
    }
}

#[async_trait]
impl Provider for MetaProvider {
    fn name(&self) -> &str {
        "meta"
    }

    async fn chat(&self, req: ChatRequest, cx: &ProviderCx) -> Result<ResponseStream, ModelError> {
        self.compat.chat(req, cx).await
    }

    async fn models(&self, cx: &ProviderCx) -> Result<Vec<ModelInfo>, ModelError> {
        self.compat.models(cx).await
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use llmleaf_model::{collect, ChatRequest, Message, Role};
    use serde_json::{json, Map};

    use super::*;
    use crate::fake::{FakeHttpTransport, FakeRealtimeTransport, FakeResponse};
    use crate::transport::{HttpBody, Method};

    fn fake_transports(http: FakeHttpTransport) -> Transports {
        Transports {
            http: Arc::new(http),
            realtime: Arc::new(FakeRealtimeTransport::scripted(Vec::new())),
        }
    }

    fn request() -> ChatRequest {
        ChatRequest {
            model: "muse-spark-1.2".into(),
            messages: vec![Message::text(Role::User, "hello")],
            max_tokens: Some(128),
            temperature: None,
            top_p: None,
            stop: vec![],
            stream: false,
            tools: vec![],
            tool_choice: None,
            thinking: None,
            extra: Map::new(),
        }
    }

    fn collected_cx() -> ProviderCx {
        ProviderCx {
            credential: Some("test-meta-key".into()),
            settings: serde_json::from_value(json!({ "upstream_streaming": "never" })).unwrap(),
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn provider_uses_meta_model_responses_api_with_bearer_auth() {
        let http = FakeHttpTransport::new(|outbound| {
            assert_eq!(outbound.method, Method::Post);
            assert_eq!(outbound.url, "https://api.meta.ai/v1/responses");
            assert!(outbound.headers.iter().any(|(name, value)| {
                name == "Authorization" && value == "Bearer test-meta-key"
            }));
            let HttpBody::Json(body) = &outbound.body else {
                panic!("expected JSON request body")
            };
            assert_eq!(body["model"], "muse-spark-1.2");
            assert_eq!(body["max_output_tokens"], 128);
            assert_eq!(body["stream"], false);
            assert_eq!(body["store"], false);
            assert_eq!(body["input"][0]["role"], "user");

            Ok(FakeResponse::ok_json(&json!({
                "id": "resp_1",
                "object": "response",
                "model": "muse-spark-1.2",
                "status": "completed",
                "output": [{
                    "type": "message",
                    "content": [{ "type": "output_text", "text": "hello" }]
                }],
                "usage": {
                    "input_tokens": 4,
                    "output_tokens": 1,
                    "total_tokens": 5
                }
            })))
        });
        let provider = MetaProvider::new(&fake_transports(http));

        let response = collect(provider.chat(request(), &collected_cx()).await.unwrap())
            .await
            .unwrap();
        assert_eq!(response.id, "resp_1");
        assert_eq!(response.model, "muse-spark-1.2");
        assert_eq!(response.choices[0].text, "hello");
        assert_eq!(response.usage.total_tokens, 5);
    }

    #[tokio::test]
    async fn provider_lists_muse_models_and_preserves_optional_metadata() {
        let http = FakeHttpTransport::new(|outbound| {
            assert_eq!(outbound.method, Method::Get);
            assert_eq!(outbound.url, "https://api.meta.ai/v1/models");
            assert!(outbound.headers.iter().any(|(name, value)| {
                name == "Authorization" && value == "Bearer test-meta-key"
            }));
            Ok(FakeResponse::ok_json(&json!({
                "object": "list",
                "data": [{
                    "id": "muse-spark-1.2",
                    "object": "model",
                    "created": 0,
                    "owned_by": "meta",
                    "metadata": {
                        "context_window": 1048576,
                        "client": "example"
                    }
                }]
            })))
        });
        let provider = MetaProvider::new(&fake_transports(http));

        let models = provider.models(&collected_cx()).await.unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "muse-spark-1.2");
        assert_eq!(models[0].extra["owned_by"], "meta");
        assert_eq!(models[0].extra["created"], 0);
        assert_eq!(models[0].extra["metadata"]["context_window"], 1048576);
    }

    #[test]
    fn factory_registers_meta_model_api_kinds() {
        for kind in ["meta", "meta-ai", "meta-model-api", "muse"] {
            let provider = crate::build(kind, &Transports::fake()).expect("known Meta kind");
            assert_eq!(provider.name(), "meta");
            assert!(crate::known_kinds().contains(&kind));
        }
        assert!(crate::build("llama-api", &Transports::fake()).is_none());
        assert!(crate::build("meta-llama", &Transports::fake()).is_none());
    }
}
