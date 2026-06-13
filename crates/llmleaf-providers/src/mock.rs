//! A zero-dependency echo provider. Useful as a route fallback target, for local development, and
//! for exercising the whole hot path offline. It speaks the canonical model directly.

use async_trait::async_trait;
use bytes::Bytes;
use futures::stream;
use llmleaf_model::{
    AudioChunk, AudioStream, ChatRequest, Embedding, EmbeddingRequest, EmbeddingResponse,
    FinishReason, Modality, ModelError, ModelInfo, ResponseStream, Role, SpeechRequest,
    StreamChunk, TranscriptionRequest, TranscriptionResponse, Usage, VoiceInfo,
};
use llmleaf_provider::{Provider, ProviderCx};

#[derive(Default)]
pub struct EchoProvider;

#[async_trait]
impl Provider for EchoProvider {
    fn name(&self) -> &str {
        "echo"
    }

    async fn chat(&self, req: ChatRequest, cx: &ProviderCx) -> Result<ResponseStream, ModelError> {
        let last_user = req
            .messages
            .iter()
            .rev()
            .find(|m| m.role == Role::User)
            .map(|m| m.text_content())
            .unwrap_or_default();
        let reply = format!("echo: {last_user}");

        // The core never counts tokens; the provider reports them. Here, a crude word count stands in.
        let prompt_tokens: u64 = req
            .messages
            .iter()
            .map(|m| count_words(&m.text_content()))
            .sum();
        let completion_tokens = count_words(&reply);
        let usage = Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
            cost_usd: None,
        };

        let id = if cx.request_id.is_empty() {
            "echo".to_string()
        } else {
            format!("echo-{}", cx.request_id)
        };

        // A real, ordered canonical stream. Non-streaming consumers get this collected (principle 4).
        let chunks: Vec<Result<StreamChunk, ModelError>> = vec![
            Ok(StreamChunk::Start {
                id,
                model: req.model.clone(),
            }),
            Ok(StreamChunk::Content {
                index: 0,
                delta: reply,
            }),
            Ok(StreamChunk::Finish {
                index: 0,
                reason: FinishReason::Stop,
            }),
            Ok(StreamChunk::Usage(usage)),
        ];
        Ok(Box::pin(stream::iter(chunks)))
    }

    async fn embed(
        &self,
        req: EmbeddingRequest,
        _cx: &ProviderCx,
    ) -> Result<EmbeddingResponse, ModelError> {
        // One synthetic vector per input, deterministic from the text: [byte length, word count].
        // Enough to exercise the embeddings hot path (ordering, indices, usage) without a network.
        let embeddings: Vec<Embedding> = req
            .input
            .iter()
            .enumerate()
            .map(|(i, text)| Embedding {
                index: i as u32,
                vector: vec![text.len() as f32, count_words(text) as f32],
            })
            .collect();

        let prompt_tokens: u64 = req.input.iter().map(|t| count_words(t)).sum();
        let usage = Usage {
            prompt_tokens,
            completion_tokens: 0,
            total_tokens: prompt_tokens,
            cost_usd: None,
        };

        Ok(EmbeddingResponse {
            model: req.model,
            embeddings,
            usage,
        })
    }

    async fn speech(
        &self,
        req: SpeechRequest,
        _cx: &ProviderCx,
    ) -> Result<AudioStream, ModelError> {
        // Synthetic TTS: a canonical Start → Data → Finish stream whose "audio" is the input text
        // bytes verbatim. Lets operators drive the whole speech hot path offline.
        let content_type = match req.response_format.as_deref() {
            Some("opus") => "audio/opus",
            Some("aac") => "audio/aac",
            Some("flac") => "audio/flac",
            Some("wav") => "audio/wav",
            Some("pcm") => "audio/pcm",
            // "mp3" and anything unset/unknown.
            _ => "audio/mpeg",
        }
        .to_string();

        let chunks: Vec<Result<AudioChunk, ModelError>> = vec![
            Ok(AudioChunk::Start { content_type }),
            Ok(AudioChunk::Data(Bytes::from(req.input.into_bytes()))),
            Ok(AudioChunk::Finish),
        ];
        Ok(Box::pin(stream::iter(chunks)))
    }

    async fn voices(&self, _model: &str, _cx: &ProviderCx) -> Result<Vec<VoiceInfo>, ModelError> {
        // A small synthetic catalog so the voices hot path can be driven offline, mirroring the
        // synthetic speech above.
        Ok(vec![
            VoiceInfo {
                id: "echo".to_string(),
                name: Some("Echo".to_string()),
                languages: vec!["en".to_string()],
                extra: Default::default(),
            },
            VoiceInfo {
                id: "reverb".to_string(),
                name: Some("Reverb".to_string()),
                languages: vec!["en".to_string()],
                extra: Default::default(),
            },
        ])
    }

    async fn models(&self, _cx: &ProviderCx) -> Result<Vec<ModelInfo>, ModelError> {
        // A small synthetic catalog so the model-listing surface can be driven offline (e.g. give the
        // echo provider a prefix to see passthrough without a real upstream). Mirrors the synthetic
        // voices catalog above.
        let mut chat = ModelInfo::new("echo-chat");
        chat.name = Some("Echo Chat".to_string());
        chat.modality = Some(Modality::Llm);
        chat.max_context = Some(8192);
        let mut embed = ModelInfo::new("echo-embed");
        embed.modality = Some(Modality::Embedding);
        embed.max_context = Some(2048);
        Ok(vec![chat, embed])
    }

    async fn transcribe(
        &self,
        req: TranscriptionRequest,
        _cx: &ProviderCx,
    ) -> Result<TranscriptionResponse, ModelError> {
        // Synthetic STT: no real decoding, just a deterministic stand-in transcript that echoes the
        // upload's size and name. Usage borrows the byte length as a crude "token" count.
        Ok(TranscriptionResponse {
            text: format!(
                "echo transcript of {} bytes from {}",
                req.audio.len(),
                req.filename
            ),
            language: None,
            duration: None,
            usage: Usage {
                total_tokens: req.audio.len() as u64,
                ..Default::default()
            },
        })
    }
}

fn count_words(s: &str) -> u64 {
    s.split_whitespace().count() as u64
}
