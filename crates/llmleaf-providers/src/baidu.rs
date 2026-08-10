//! Baidu AI Cloud Qianfan — its OpenAI-compatible `/v2` inference wire plus the provider-specific
//! catalog and capability boundaries that keep the generic compatibility layer honest.
//!
//! Chat Completions, embeddings, rerank, model listing, and batch jobs reuse
//! [`OpenAiCompatProvider`]. The wrapper corrects two facts the shared family cannot infer: Qianfan's
//! catalog prices are CNY per 1,000 tokens (not USD per token), and `max_tokens` is its final-answer
//! cap. Raw vendor fields remain in [`ModelInfo::extra`], while the typed USD rates stay unset rather
//! than publishing a fabricated conversion. Qianfan speech APIs use unrelated native dialects, so
//! this provider deliberately leaves speech/transcription on the trait's `Unsupported` defaults.

use async_trait::async_trait;
use llmleaf_model::{
    BatchHandle, BatchResultStream, BatchSpec, ChatRequest, EmbeddingRequest, EmbeddingResponse,
    ModelError, ModelInfo, RerankRequest, RerankResponse, ResponseStream, Thinking,
};
use llmleaf_provider::{Provider, ProviderCx};
use serde_json::Value;

use crate::compat::OpenAiCompatProvider;
use crate::transport::Transports;

/// The first-class Qianfan edge around the shared OpenAI-wire transport.
pub struct BaiduProvider {
    inner: OpenAiCompatProvider,
}

impl BaiduProvider {
    /// Construct from any supported Baidu/Qianfan config kind.
    pub fn for_kind(kind: &str, transports: &Transports) -> Option<Self> {
        if !matches!(
            kind,
            "baidu" | "qianfan" | "baidu-qianfan" | "baidu-ai-cloud"
        ) {
            return None;
        }
        Some(Self {
            inner: OpenAiCompatProvider::for_kind(kind, transports)?,
        })
    }
}

fn catalog_u32(value: Option<&Value>) -> Option<u32> {
    value
        .and_then(|v| {
            v.as_u64()
                .or_else(|| v.as_str().and_then(|s| s.parse::<u64>().ok()))
        })
        .map(|n| n.min(u32::MAX as u64) as u32)
}

/// Normalize Qianfan catalog facts after the tolerant OpenAI-family parse.
fn normalize_catalog(info: &mut ModelInfo) {
    // Qianfan reports the final-answer cap as `max_tokens`; its separate, intentionally preserved
    // `max_completions_tokens` (plural) covers reasoning + answer.
    info.max_output = catalog_u32(info.extra.get("max_tokens")).or(info.max_output);

    // The shared parser interpreted `pricing.prompt`/`completion` as USD per token. Qianfan defines
    // those same keys as CNY per 1,000 tokens, so neither the currency nor scale matches the typed
    // USD-per-Mtok fields. Keep the raw `pricing` object in `extra` and publish no false USD rate.
    info.input_per_mtok = None;
    info.output_per_mtok = None;
}

/// Qianfan documents two reasoning-effort levels. Its compatibility aliases accept lower OpenAI
/// levels but treat them as `high`; llmleaf's two upper rungs map to Qianfan's real `max` level.
fn normalize_reasoning(req: &mut ChatRequest) {
    let Some(thinking) = req.thinking.take() else {
        return;
    };
    let effort = match thinking {
        Thinking::Low | Thinking::Med | Thinking::High => "high",
        Thinking::Highx | Thinking::Max => "max",
    };
    req.extra
        .insert("reasoning_effort".into(), Value::from(effort));
}

#[async_trait]
impl Provider for BaiduProvider {
    fn name(&self) -> &str {
        self.inner.name()
    }

    async fn chat(
        &self,
        mut req: ChatRequest,
        cx: &ProviderCx,
    ) -> Result<ResponseStream, ModelError> {
        normalize_reasoning(&mut req);
        self.inner.chat(req, cx).await
    }

    async fn embed(
        &self,
        req: EmbeddingRequest,
        cx: &ProviderCx,
    ) -> Result<EmbeddingResponse, ModelError> {
        self.inner.embed(req, cx).await
    }

    async fn rerank(
        &self,
        req: RerankRequest,
        cx: &ProviderCx,
    ) -> Result<RerankResponse, ModelError> {
        self.inner.rerank(req, cx).await
    }

    async fn models(&self, cx: &ProviderCx) -> Result<Vec<ModelInfo>, ModelError> {
        let mut models = self.inner.models(cx).await?;
        models.iter_mut().for_each(normalize_catalog);
        Ok(models)
    }

    async fn batch_create(
        &self,
        mut req: BatchSpec,
        cx: &ProviderCx,
    ) -> Result<BatchHandle, ModelError> {
        if req.items.len() > 5_000 {
            return Err(ModelError::Mapping(
                "Baidu Qianfan batches accept at most 5000 requests".into(),
            ));
        }
        if let Some(model) = req.items.first().map(|item| item.request.model.as_str()) {
            if req
                .items
                .iter()
                .any(|item| item.request.model.as_str() != model)
            {
                return Err(ModelError::Mapping(
                    "Baidu Qianfan requires every request in a batch to use the same model".into(),
                ));
            }
        }
        for item in &mut req.items {
            normalize_reasoning(&mut item.request);
        }
        self.inner.batch_create(req, cx).await
    }

    async fn batch_retrieve(
        &self,
        upstream_id: &str,
        cx: &ProviderCx,
    ) -> Result<BatchHandle, ModelError> {
        self.inner.batch_retrieve(upstream_id, cx).await
    }

    async fn batch_results(
        &self,
        upstream_id: &str,
        cx: &ProviderCx,
    ) -> Result<BatchResultStream, ModelError> {
        self.inner.batch_results(upstream_id, cx).await
    }

    async fn batch_cancel(
        &self,
        upstream_id: &str,
        cx: &ProviderCx,
    ) -> Result<BatchHandle, ModelError> {
        self.inner.batch_cancel(upstream_id, cx).await
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use llmleaf_model::{BatchItem, Message, Role, SpeechRequest, TranscriptionRequest};
    use serde_json::json;

    use super::*;
    use crate::fake::{FakeHttpTransport, FakeRealtimeTransport, FakeResponse};
    use crate::transport::HttpBody;

    fn transports(http: FakeHttpTransport) -> Transports {
        Transports {
            http: Arc::new(http),
            realtime: Arc::new(FakeRealtimeTransport::scripted(Vec::new())),
        }
    }

    fn chat_request(model: &str) -> ChatRequest {
        ChatRequest {
            model: model.into(),
            messages: vec![Message::text(Role::User, "hello")],
            max_tokens: Some(123),
            temperature: None,
            top_p: None,
            stop: Vec::new(),
            stream: false,
            tools: Vec::new(),
            tool_choice: None,
            thinking: None,
            extra: Default::default(),
        }
    }

    #[tokio::test]
    async fn chat_uses_bearer_v2_and_final_answer_max_tokens() {
        let http = FakeHttpTransport::new(|req| {
            assert_eq!(req.url, "https://qianfan.baidubce.com/v2/chat/completions");
            assert!(req
                .headers
                .iter()
                .any(|(name, value)| name == "Authorization" && value == "Bearer qf-key"));
            let HttpBody::Json(body) = &req.body else {
                panic!("expected JSON body")
            };
            assert_eq!(body["max_tokens"], 123);
            assert!(body.get("max_completion_tokens").is_none());
            assert_eq!(body["reasoning_effort"], "max");
            Ok(FakeResponse::ok_json(&json!({
                "id": "as-1",
                "model": "ernie-4.5-turbo-32k",
                "choices": [{
                    "index": 0,
                    "message": { "role": "assistant", "content": "hello" },
                    "finish_reason": "stop"
                }],
                "usage": { "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2 }
            })))
        });
        let provider = BaiduProvider::for_kind("qianfan", &transports(http)).unwrap();
        let cx = ProviderCx {
            credential: Some("qf-key".into()),
            settings: serde_json::from_value(json!({ "upstream_streaming": "never" })).unwrap(),
            ..Default::default()
        };
        let mut request = chat_request("ernie-4.5-turbo-32k");
        request.thinking = Some(Thinking::Max);
        let response = llmleaf_model::collect(provider.chat(request, &cx).await.unwrap())
            .await
            .unwrap();
        assert_eq!(response.choices[0].text, "hello");
    }

    #[tokio::test]
    async fn catalog_preserves_cny_pricing_without_claiming_usd_rates() {
        let http = FakeHttpTransport::new(|req| {
            assert_eq!(req.url, "https://qianfan.baidubce.com/v2/models");
            Ok(FakeResponse::ok_json(&json!({
                "object": "list",
                "data": [{
                    "id": "deepseek-v3.1-think-250821",
                    "object": "model",
                    "type": "chat",
                    "context_length": 163840,
                    "max_completions_tokens": 65536,
                    "max_tokens": 32768,
                    "pricing": { "prompt": "0.004", "completion": "0.012" }
                }]
            })))
        });
        let provider = BaiduProvider::for_kind("baidu", &transports(http)).unwrap();
        let models = provider.models(&ProviderCx::default()).await.unwrap();
        assert_eq!(models[0].max_context, Some(163840));
        assert_eq!(models[0].max_output, Some(32768));
        assert_eq!(models[0].input_per_mtok, None);
        assert_eq!(models[0].output_per_mtok, None);
        assert_eq!(
            models[0].extra["pricing"],
            json!({ "prompt": "0.004", "completion": "0.012" })
        );
        assert_eq!(models[0].extra["max_completions_tokens"], 65536);
    }

    #[tokio::test]
    async fn non_openai_audio_dialects_are_unsupported() {
        let provider = BaiduProvider::for_kind("qianfan", &Transports::fake()).unwrap();
        let speech: SpeechRequest = serde_json::from_value(json!({
            "model": "tts", "input": "hi", "voice": "default"
        }))
        .unwrap();
        assert!(matches!(
            provider.speech(speech, &ProviderCx::default()).await,
            Err(ModelError::Unsupported(_))
        ));

        let transcription: TranscriptionRequest = serde_json::from_value(json!({
            "model": "asr", "audio": [], "filename": "clip.wav"
        }))
        .unwrap();
        assert!(matches!(
            provider
                .transcribe(transcription, &ProviderCx::default())
                .await,
            Err(ModelError::Unsupported(_))
        ));
    }

    #[tokio::test]
    async fn mixed_model_batch_is_rejected_before_transport() {
        let provider = BaiduProvider::for_kind("qianfan", &Transports::fake()).unwrap();
        let req = BatchSpec {
            items: vec![
                BatchItem {
                    custom_id: "a".into(),
                    request: chat_request("ernie-a"),
                },
                BatchItem {
                    custom_id: "b".into(),
                    request: chat_request("ernie-b"),
                },
            ],
        };
        assert!(matches!(
            provider.batch_create(req, &ProviderCx::default()).await,
            Err(ModelError::Mapping(message)) if message.contains("same model")
        ));
    }
}
