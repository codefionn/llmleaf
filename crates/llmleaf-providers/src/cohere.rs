//! Cohere Chat API v2 provider.
//!
//! Close to OpenAI in spirit but distinct in detail: endpoint `/v2/chat`, nucleus sampling is `p`
//! (not `top_p`), the assistant reply is `message.content` (a list of typed blocks), finish reasons
//! use Cohere's vocabulary, and usage is nested under the top-level `meta.tokens` object (there is no
//! `usage` object in v2). Mapped here.

use async_trait::async_trait;
use futures::stream;
use llmleaf_model::{
    ChatRequest, Embedding, EmbeddingRequest, EmbeddingResponse, FinishReason, Message, Modality,
    ModelError, ModelInfo, ResponseStream, Role, StreamChunk, ToolCallDelta, ToolChoice, Usage,
};
use llmleaf_provider::{Provider, ProviderCx};
use serde_json::{json, Map, Value};

use std::sync::Arc;

use crate::http::post_json;
use crate::transport::{HttpRequest, HttpTransport, Transports};

const DEFAULT_ENDPOINT: &str = "https://api.cohere.com";

pub struct CohereProvider {
    http: Arc<dyn HttpTransport>,
}

impl CohereProvider {
    pub fn new(transports: &Transports) -> Self {
        CohereProvider {
            http: transports.http.clone(),
        }
    }
}

#[async_trait]
impl Provider for CohereProvider {
    fn name(&self) -> &str {
        "cohere"
    }

    /// Enumerate Cohere's catalog via `GET /v1/models` (Bearer auth; this path serves both API
    /// versions even though chat/embed use `/v2`). Paginated via `next_page_token`. Cohere's id field is
    /// literally `name`; modality comes from `endpoints[]`; `context_length` maps; there is no pricing.
    async fn models(&self, cx: &ProviderCx) -> Result<Vec<ModelInfo>, ModelError> {
        let endpoint = cx
            .endpoint
            .as_deref()
            .unwrap_or(DEFAULT_ENDPOINT)
            .trim_end_matches('/');
        let mut out: Vec<ModelInfo> = Vec::new();
        let mut page_token: Option<String> = None;
        for _ in 0..100 {
            let mut url = format!("{endpoint}/v1/models?page_size=1000");
            if let Some(tok) = &page_token {
                url.push_str(&format!("&page_token={tok}"));
            }
            let mut req = HttpRequest::get(&url).header("Accept", "application/json");
            if let Some(cred) = &cx.credential {
                req = req.bearer(cred);
            }
            let value = post_json(&*self.http, req).await?;
            if let Some(items) = value.get("models").and_then(Value::as_array) {
                out.extend(items.iter().filter_map(cohere_model_to_info));
            }
            match value.get("next_page_token").and_then(Value::as_str) {
                Some(tok) if !tok.is_empty() => page_token = Some(tok.to_string()),
                _ => break,
            }
        }
        Ok(out)
    }

    async fn chat(&self, req: ChatRequest, cx: &ProviderCx) -> Result<ResponseStream, ModelError> {
        let endpoint = cx
            .endpoint
            .as_deref()
            .unwrap_or(DEFAULT_ENDPOINT)
            .trim_end_matches('/');
        let url = format!("{endpoint}/v2/chat");
        let body = request_to_cohere(&req);

        let mut http_req = HttpRequest::post(&url)
            .header("Accept", "application/json")
            .json(body);
        if let Some(cred) = &cx.credential {
            http_req = http_req.bearer(cred);
        }

        let value = post_json(&*self.http, http_req).await?;
        let chunks = cohere_to_chunks(value, &req.model);
        Ok(Box::pin(stream::iter(chunks.into_iter().map(Ok))))
    }

    async fn embed(
        &self,
        req: EmbeddingRequest,
        cx: &ProviderCx,
    ) -> Result<EmbeddingResponse, ModelError> {
        let endpoint = cx
            .endpoint
            .as_deref()
            .unwrap_or(DEFAULT_ENDPOINT)
            .trim_end_matches('/');
        let url = format!("{endpoint}/v2/embed");
        // `input_type` is mandatory for Cohere v3 embed models — hence the default; an operator can
        // override it per instance (`search_query` for queries, `classification`, `clustering`, …).
        let input_type = cx.setting_str("input_type").unwrap_or("search_document");
        let body = embedding_request_to_cohere(&req, input_type);

        let mut http_req = HttpRequest::post(&url)
            .header("Accept", "application/json")
            .json(body);
        if let Some(cred) = &cx.credential {
            http_req = http_req.bearer(cred);
        }

        let value = post_json(&*self.http, http_req).await?;
        Ok(cohere_to_embeddings(value, &req.model))
    }
}

fn embedding_request_to_cohere(req: &EmbeddingRequest, input_type: &str) -> Value {
    let mut obj = Map::new();
    obj.insert("model".into(), json!(req.model));
    obj.insert("texts".into(), json!(req.input));
    obj.insert("input_type".into(), json!(input_type));
    // The canonical vector is `f32`; ask for the float wire encoding explicitly (v2 can return several).
    obj.insert("embedding_types".into(), json!(["float"]));
    for (k, v) in &req.extra {
        obj.entry(k.clone()).or_insert_with(|| v.clone());
    }
    Value::Object(obj)
}

/// Map one `GET /v1/models` entry to canonical [`ModelInfo`]. `None` (skipped) when it has no `name`
/// (Cohere's id field IS `name`). No separate display name exists, so [`ModelInfo::name`] stays `None`.
fn cohere_model_to_info(m: &Value) -> Option<ModelInfo> {
    let id = m.get("name").and_then(Value::as_str)?;
    let mut info = ModelInfo::new(id);
    // Modality from `endpoints[]`: chat/generate → Llm (wins), embed → Embedding; rerank/classify/
    // summarize/rate have no canonical modality → None.
    let endpoints: Vec<&str> = m
        .get("endpoints")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    info.modality = if endpoints.contains(&"chat") || endpoints.contains(&"generate") {
        Some(Modality::Llm)
    } else if endpoints.contains(&"embed") {
        Some(Modality::Embedding)
    } else {
        None
    };
    info.max_context = m
        .get("context_length")
        .and_then(Value::as_u64)
        .map(|n| n.min(u32::MAX as u64) as u32);
    // No output cap (do NOT use sampling_defaults.max_tokens_per_doc — a RAG doc-truncation default,
    // not a generation cap), no thinking budget, no pricing — all None, enhanced downstream.
    for key in [
        "endpoints",
        "default_endpoints",
        "finetuned",
        "is_deprecated",
        "tokenizer_url",
        "features",
        "sampling_defaults",
    ] {
        if let Some(v) = m.get(key) {
            if !v.is_null() {
                info.extra.insert(key.to_string(), v.clone());
            }
        }
    }
    Some(info)
}

fn cohere_to_embeddings(value: Value, model: &str) -> EmbeddingResponse {
    // v2 nests the requested encodings under `embeddings`; we asked for `float`, so read that list.
    let embeddings = value
        .get("embeddings")
        .and_then(|e| e.get("float"))
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .enumerate()
                .map(|(i, item)| {
                    let vector = item
                        .as_array()
                        .map(|vs| {
                            vs.iter()
                                .filter_map(Value::as_f64)
                                .map(|f| f as f32)
                                .collect()
                        })
                        .unwrap_or_default();
                    Embedding {
                        index: i as u32,
                        vector,
                    }
                })
                .collect()
        })
        .unwrap_or_default();

    // Cohere reports billed input tokens under `meta.billed_units`; there are no completion tokens
    // for embeddings, so total mirrors the prompt count (the core relays, never computes — principle 5).
    let prompt_tokens = value
        .get("meta")
        .and_then(|m| m.get("billed_units"))
        .and_then(|b| b.get("input_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let usage = Usage {
        prompt_tokens,
        completion_tokens: 0,
        total_tokens: prompt_tokens,
        cost_usd: None,
    };

    EmbeddingResponse {
        model: model.to_string(),
        embeddings,
        usage,
    }
}

fn request_to_cohere(req: &ChatRequest) -> Value {
    let mut obj = Map::new();
    obj.insert("model".into(), json!(req.model));
    obj.insert(
        "messages".into(),
        Value::Array(req.messages.iter().map(message_to_cohere).collect()),
    );
    if let Some(v) = req.max_tokens {
        obj.insert("max_tokens".into(), json!(v));
    }
    if let Some(v) = req.temperature {
        obj.insert("temperature".into(), json!(v));
    }
    if let Some(v) = req.top_p {
        obj.insert("p".into(), json!(v)); // Cohere names nucleus sampling `p`.
    }
    if !req.stop.is_empty() {
        obj.insert("stop_sequences".into(), json!(req.stop));
    }
    if !req.tools.is_empty() {
        let tools: Vec<Value> = req
            .tools
            .iter()
            .map(|t| {
                json!({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    }
                })
            })
            .collect();
        obj.insert("tools".into(), json!(tools));
    }
    if let Some(choice) = req.tool_choice.as_ref().and_then(tool_choice_to_cohere) {
        obj.insert("tool_choice".into(), choice);
    }
    if let Some(t) = req.thinking {
        obj.insert(
            "thinking".into(),
            json!({ "type": "enabled", "token_budget": crate::thinking::budget_tokens(t) }),
        );
    }
    for (k, v) in &req.extra {
        obj.entry(k.clone()).or_insert_with(|| v.clone());
    }
    Value::Object(obj)
}

fn message_to_cohere(msg: &Message) -> Value {
    let role = match msg.role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    };
    let mut obj = Map::new();
    obj.insert("role".into(), json!(role));

    if msg.role == Role::Tool {
        obj.insert(
            "tool_call_id".into(),
            json!(msg.tool_call_id.clone().unwrap_or_default()),
        );
        // Cohere v2 tool results are a list of `document` content blocks, not a bare string.
        obj.insert(
            "content".into(),
            json!([{ "type": "document", "document": { "data": msg.text_content() } }]),
        );
        return Value::Object(obj);
    }

    // Cohere accepts a plain string for content; use it for the common single-text case.
    let text = msg.text_content();
    if !text.is_empty() {
        obj.insert("content".into(), json!(text));
    }

    if !msg.tool_calls.is_empty() {
        let calls: Vec<Value> = msg
            .tool_calls
            .iter()
            .map(|c| {
                json!({
                    "id": c.id,
                    "type": "function",
                    "function": { "name": c.name, "arguments": c.arguments },
                })
            })
            .collect();
        obj.insert("tool_calls".into(), json!(calls));
    }
    Value::Object(obj)
}

/// Cohere v2 supports `tool_choice: REQUIRED | NONE`. Auto is the default (omit); a forced named tool
/// is not expressible, so it falls back to REQUIRED.
fn tool_choice_to_cohere(tc: &ToolChoice) -> Option<Value> {
    match tc {
        ToolChoice::Required | ToolChoice::Named(_) => Some(json!("REQUIRED")),
        ToolChoice::None => Some(json!("NONE")),
        ToolChoice::Auto => None,
    }
}

fn cohere_to_chunks(value: Value, fallback_model: &str) -> Vec<StreamChunk> {
    let id = value
        .get("id")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let mut chunks = vec![StreamChunk::Start {
        id,
        model: fallback_model.to_string(),
    }];

    let message = value.get("message");

    if let Some(blocks) = message
        .and_then(|m| m.get("content"))
        .and_then(Value::as_array)
    {
        for block in blocks {
            if let Some(text) = block.get("text").and_then(Value::as_str) {
                if !text.is_empty() {
                    chunks.push(StreamChunk::Content {
                        index: 0,
                        delta: text.to_string(),
                    });
                }
            }
        }
    }

    if let Some(tool_calls) = message
        .and_then(|m| m.get("tool_calls"))
        .and_then(Value::as_array)
    {
        for (i, tc) in tool_calls.iter().enumerate() {
            let func = tc.get("function");
            chunks.push(StreamChunk::ToolCall {
                index: 0,
                call: ToolCallDelta {
                    index: i as u32,
                    id: tc.get("id").and_then(Value::as_str).map(str::to_owned),
                    name: func
                        .and_then(|f| f.get("name"))
                        .and_then(Value::as_str)
                        .map(str::to_owned),
                    arguments: func
                        .and_then(|f| f.get("arguments"))
                        .and_then(Value::as_str)
                        .map(str::to_owned),
                },
            });
        }
    }

    let reason = value
        .get("finish_reason")
        .and_then(Value::as_str)
        .map(map_finish_reason)
        .unwrap_or(FinishReason::Stop);
    chunks.push(StreamChunk::Finish { index: 0, reason });

    // Cohere v2 reports usage under the top-level `meta.tokens` object (true counts), not `usage`.
    if let Some(tokens) = value.get("meta").and_then(|m| m.get("tokens")) {
        let prompt = tokens
            .get("input_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let completion = tokens
            .get("output_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        chunks.push(StreamChunk::Usage(Usage {
            prompt_tokens: prompt,
            completion_tokens: completion,
            total_tokens: prompt + completion,
            cost_usd: None,
        }));
    }

    chunks
}

fn map_finish_reason(reason: &str) -> FinishReason {
    // Cohere v2 vocabulary: COMPLETE | STOP_SEQUENCE | MAX_TOKENS | TOOL_CALL | ERROR | TIMEOUT.
    // There is no dedicated content-filter reason.
    match reason {
        "MAX_TOKENS" => FinishReason::Length,
        "TOOL_CALL" => FinishReason::ToolCalls,
        "ERROR" | "TIMEOUT" => FinishReason::Error,
        // "COMPLETE", "STOP_SEQUENCE".
        _ => FinishReason::Stop,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_top_p_to_p() {
        let req = ChatRequest {
            model: "command-r-plus".into(),
            messages: vec![Message::text(Role::User, "hi")],
            max_tokens: None,
            temperature: None,
            top_p: Some(0.9),
            stop: vec![],
            stream: false,
            tools: vec![],
            tool_choice: None,
            thinking: None,
            extra: Default::default(),
        };
        let wire = request_to_cohere(&req);
        // top_p maps to `p`; compare with tolerance (f32 → JSON f64 isn't bit-exact).
        assert!((wire["p"].as_f64().unwrap() - 0.9).abs() < 1e-6);
        assert_eq!(wire["messages"][0]["content"], "hi");
        assert!(wire.get("thinking").is_none());
    }

    #[test]
    fn thinking_sets_token_budget() {
        let req = ChatRequest {
            model: "command-a-reasoning".into(),
            messages: vec![Message::text(Role::User, "hi")],
            max_tokens: None,
            temperature: None,
            top_p: None,
            stop: vec![],
            stream: false,
            tools: vec![],
            tool_choice: None,
            thinking: Some(llmleaf_model::Thinking::Low),
            extra: Default::default(),
        };
        let wire = request_to_cohere(&req);
        assert_eq!(wire["thinking"]["type"], "enabled");
        assert_eq!(wire["thinking"]["token_budget"], 1024);
    }

    #[test]
    fn parses_content_blocks_and_usage() {
        let resp = json!({
            "id": "c1",
            "message": { "role": "assistant", "content": [{ "type": "text", "text": "hello" }] },
            "finish_reason": "COMPLETE",
            "meta": { "tokens": { "input_tokens": 7, "output_tokens": 2 } }
        });
        let chunks = cohere_to_chunks(resp, "command-r-plus");
        assert!(matches!(&chunks[1], StreamChunk::Content { delta, .. } if delta == "hello"));
        assert!(matches!(
            chunks[2],
            StreamChunk::Finish {
                reason: FinishReason::Stop,
                ..
            }
        ));
        assert!(matches!(chunks[3], StreamChunk::Usage(u) if u.total_tokens == 9));
    }

    #[test]
    fn builds_embed_request_with_default_input_type() {
        let req = EmbeddingRequest {
            model: "embed-v4.0".into(),
            input: vec!["hello".into(), "world".into()],
            dimensions: None,
            encoding_format: None,
            extra: Default::default(),
        };
        let wire = embedding_request_to_cohere(&req, "search_document");
        assert_eq!(wire["model"], "embed-v4.0");
        assert_eq!(wire["texts"][0], "hello");
        assert_eq!(wire["texts"][1], "world");
        assert_eq!(wire["input_type"], "search_document");
        assert_eq!(wire["embedding_types"][0], "float");
    }

    #[test]
    fn parses_float_embeddings_and_billed_tokens() {
        let resp = json!({
            "embeddings": { "float": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]] },
            "meta": { "billed_units": { "input_tokens": 11 } }
        });
        let out = cohere_to_embeddings(resp, "embed-v4.0");
        assert_eq!(out.embeddings[0].index, 0);
        assert_eq!(out.embeddings[1].index, 1);
        assert_eq!(out.embeddings[0].vector.len(), 3);
        assert!((out.embeddings[1].vector[0] - 0.4).abs() < 1e-6);
        // Embeddings have no completion tokens; total mirrors the billed input count.
        assert_eq!(out.usage.prompt_tokens, 11);
        assert_eq!(out.usage.completion_tokens, 0);
        assert_eq!(out.usage.total_tokens, 11);
    }
}
