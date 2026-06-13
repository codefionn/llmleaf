//! Google Gemini provider (Generative Language API `generateContent`).
//!
//! Distinct dialect: turns are `contents` with `parts`, roles are `user`/`model` (no system role —
//! system text goes in `systemInstruction`), sampling lives under `generationConfig`, and the key is
//! sent in the `x-goog-api-key` header. All mapped here.

use async_trait::async_trait;
use futures::stream;
use llmleaf_model::{
    collect_chunks, BatchCounts, BatchHandle, BatchOutcome, BatchResult, BatchResultStream,
    BatchSpec, BatchStatus, ChatRequest, ContentPart, Embedding, EmbeddingRequest,
    EmbeddingResponse, FinishReason, Message, Modality, ModelError, ModelInfo, ResponseStream,
    Role, StreamChunk, ToolCallDelta, ToolChoice, Usage,
};
use llmleaf_provider::{Provider, ProviderCx};
use serde_json::{json, Map, Value};

use std::sync::Arc;

use crate::http::{post_json, send_checked};
use crate::transport::{HttpRequest, HttpTransport, Transports};

const DEFAULT_ENDPOINT: &str = "https://generativelanguage.googleapis.com/v1beta";

pub struct GeminiProvider {
    http: Arc<dyn HttpTransport>,
}

impl GeminiProvider {
    pub fn new(transports: &Transports) -> Self {
        GeminiProvider {
            http: transports.http.clone(),
        }
    }

    /// This instance's base endpoint (config override or the default), trailing slash trimmed.
    fn endpoint(&self, cx: &ProviderCx) -> String {
        cx.endpoint
            .as_deref()
            .unwrap_or(DEFAULT_ENDPOINT)
            .trim_end_matches('/')
            .to_string()
    }

    /// Apply Gemini's header auth (`x-goog-api-key`). Shared by chat, embeddings, and batch.
    fn auth(&self, req: HttpRequest, cx: &ProviderCx) -> HttpRequest {
        match &cx.credential {
            Some(cred) => req.header("x-goog-api-key", cred),
            None => req,
        }
    }
}

#[async_trait]
impl Provider for GeminiProvider {
    fn name(&self) -> &str {
        "gemini"
    }

    /// Enumerate Gemini's catalog via `GET /v1beta/models` (header `x-goog-api-key`). The response is
    /// paginated, so loop on `nextPageToken`. Token limits map across (`inputTokenLimit`/
    /// `outputTokenLimit`); modality comes from `supportedGenerationMethods`; there is no pricing.
    async fn models(&self, cx: &ProviderCx) -> Result<Vec<ModelInfo>, ModelError> {
        let base = self.endpoint(cx);
        let mut out: Vec<ModelInfo> = Vec::new();
        let mut page_token: Option<String> = None;
        // Bound the loop so a misbehaving/repeating token cannot spin forever.
        for _ in 0..100 {
            let mut url = format!("{base}/models?pageSize=1000");
            if let Some(tok) = &page_token {
                url.push_str(&format!("&pageToken={tok}"));
            }
            let req = self.auth(HttpRequest::get(&url), cx);
            let value = post_json(&*self.http, req).await?;
            if let Some(items) = value.get("models").and_then(Value::as_array) {
                out.extend(items.iter().filter_map(gemini_model_to_info));
            }
            match value.get("nextPageToken").and_then(Value::as_str) {
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
        let url = format!("{endpoint}/models/{}:generateContent", req.model);
        let body = request_to_gemini(&req);

        let mut http_req = HttpRequest::post(&url).json(body);
        if let Some(cred) = &cx.credential {
            // Header auth keeps the key out of URLs/logs (query `?key=` is the documented alternative).
            http_req = http_req.header("x-goog-api-key", cred);
        }

        let value = post_json(&*self.http, http_req).await?;
        let chunks = gemini_to_chunks(value, &req.model);
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
        // batchEmbedContents embeds every input in a single round-trip; the per-request `model` field
        // is mandatory and must carry the `models/` prefix even though it repeats the path's model.
        let url = format!("{endpoint}/models/{}:batchEmbedContents", req.model);
        // `taskType` (RETRIEVAL_DOCUMENT, RETRIEVAL_QUERY, SEMANTIC_SIMILARITY, …) is optional but
        // materially changes embedding quality for Gemini embedding models. Operator-configurable, like
        // Cohere's `input_type`; omitted when unset so Gemini applies its own default.
        let task_type = cx.setting_str("task_type");
        let body = embedding_request_to_gemini(&req, task_type);

        let mut http_req = HttpRequest::post(&url).json(body);
        if let Some(cred) = &cx.credential {
            http_req = http_req.header("x-goog-api-key", cred);
        }

        let value = post_json(&*self.http, http_req).await?;
        Ok(gemini_to_embeddings(value, &req.model))
    }

    /// Submit a batch via Gemini's *inline* batch mode (`:batchGenerateContent`). Each canonical item
    /// maps through the very same [`request_to_gemini`] the live `chat` path uses, tagged with its
    /// `custom_id` under `metadata.key`. Gemini's batch is per-model (the model is in the URL), so all
    /// items use the first item's model — the core already guarantees they routed to this one provider.
    /// The create returns a long-running-operation name (`batches/{id}`); we keep the id past that prefix.
    async fn batch_create(
        &self,
        req: BatchSpec,
        cx: &ProviderCx,
    ) -> Result<BatchHandle, ModelError> {
        let model = req
            .items
            .first()
            .map(|i| i.request.model.clone())
            .unwrap_or_default();
        let url = format!("{}/models/{model}:batchGenerateContent", self.endpoint(cx));
        let requests: Vec<Value> = req
            .items
            .iter()
            .map(|item| {
                json!({
                    "request": request_to_gemini(&item.request),
                    "metadata": { "key": item.custom_id },
                })
            })
            .collect();
        let body = json!({
            "batch": { "input_config": { "requests": { "requests": requests } } }
        });
        let req = self.auth(HttpRequest::post(&url).json(body), cx);
        let value = post_json(&*self.http, req).await?;
        Ok(gemini_batch_to_handle(&value))
    }

    async fn batch_retrieve(
        &self,
        upstream_id: &str,
        cx: &ProviderCx,
    ) -> Result<BatchHandle, ModelError> {
        let url = format!("{}/batches/{upstream_id}", self.endpoint(cx));
        let req = self.auth(HttpRequest::get(&url), cx);
        let value = post_json(&*self.http, req).await?;
        Ok(gemini_batch_to_handle(&value))
    }

    async fn batch_cancel(
        &self,
        upstream_id: &str,
        cx: &ProviderCx,
    ) -> Result<BatchHandle, ModelError> {
        // Gemini's cancel returns an empty body, so report the transitional state and let the consumer
        // poll for the terminal one.
        let url = format!("{}/batches/{upstream_id}:cancel", self.endpoint(cx));
        let req = self.auth(HttpRequest::post(&url), cx);
        send_checked(&*self.http, req).await?;
        Ok(BatchHandle {
            id: upstream_id.to_string(),
            status: BatchStatus::Canceling,
            counts: BatchCounts::default(),
            created_at: None,
            expires_at: None,
            ended_at: None,
            endpoint: Some("/v1beta/models:generateContent".to_string()),
            extra: Map::new(),
        })
    }

    /// Stream a batch's results. For an inline batch the results live on the batch object itself
    /// (`response.inlinedResponses`), so this fetches it and maps each entry — keyed by `metadata.key`
    /// (the `custom_id`) — through the same chunk path the chat surface uses.
    async fn batch_results(
        &self,
        upstream_id: &str,
        cx: &ProviderCx,
    ) -> Result<BatchResultStream, ModelError> {
        let url = format!("{}/batches/{upstream_id}", self.endpoint(cx));
        let req = self.auth(HttpRequest::get(&url), cx);
        let value = post_json(&*self.http, req).await?;
        let entries = value
            .get("response")
            .and_then(|r| r.get("inlinedResponses"))
            // The array is nested under a second `inlinedResponses` key; tolerate a flat array too.
            .map(|ir| ir.get("inlinedResponses").unwrap_or(ir))
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let results: Vec<Result<BatchResult, ModelError>> = entries
            .iter()
            .filter_map(gemini_inline_result)
            .map(Ok)
            .collect();
        Ok(Box::pin(stream::iter(results)))
    }
}

fn embedding_request_to_gemini(req: &EmbeddingRequest, task_type: Option<&str>) -> Value {
    let requests: Vec<Value> = req
        .input
        .iter()
        .map(|text| {
            let mut obj = Map::new();
            // Gemini requires the `models/` prefix on the per-request model id.
            obj.insert("model".into(), json!(format!("models/{}", req.model)));
            obj.insert("content".into(), json!({ "parts": [{ "text": text }] }));
            // Matryoshka truncation, when the consumer asked for a specific output width.
            if let Some(dims) = req.dimensions {
                obj.insert("outputDimensionality".into(), json!(dims));
            }
            if let Some(task) = task_type {
                obj.insert("taskType".into(), json!(task));
            }
            Value::Object(obj)
        })
        .collect();
    json!({ "requests": requests })
}

fn gemini_to_embeddings(value: Value, model: &str) -> EmbeddingResponse {
    // Response is `{ "embeddings": [ { "values": [..] }, .. ] }` in input order; batchEmbedContents
    // reports no usage, so accounting stays at the canonical default (the core never invents counts).
    let embeddings = value
        .get("embeddings")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .enumerate()
                .map(|(i, item)| {
                    let vector = item
                        .get("values")
                        .and_then(Value::as_array)
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

    EmbeddingResponse {
        model: model.to_string(),
        embeddings,
        usage: Usage::default(),
    }
}

/// Map a canonical [`ChatRequest`] onto Gemini's `generateContent` body. `pub(crate)` because the
/// Vertex provider's `generateContent` body is byte-for-byte the same schema (only the URL host/path
/// and auth differ), so it reuses this verbatim rather than duplicating the mapping.
pub(crate) fn request_to_gemini(req: &ChatRequest) -> Value {
    let mut obj = Map::new();

    // System text → systemInstruction (Gemini has no system role).
    let system: String = req
        .messages
        .iter()
        .filter(|m| m.role == Role::System)
        .map(|m| m.text_content())
        .collect::<Vec<_>>()
        .join("\n\n");
    if !system.is_empty() {
        obj.insert(
            "systemInstruction".into(),
            json!({ "parts": [{ "text": system }] }),
        );
    }

    let contents: Vec<Value> = req
        .messages
        .iter()
        .filter(|m| m.role != Role::System)
        .map(message_to_gemini)
        .collect();
    obj.insert("contents".into(), json!(contents));

    let mut gen = Map::new();
    if let Some(v) = req.max_tokens {
        gen.insert("maxOutputTokens".into(), json!(v));
    }
    if let Some(v) = req.temperature {
        gen.insert("temperature".into(), json!(v));
    }
    if let Some(v) = req.top_p {
        gen.insert("topP".into(), json!(v));
    }
    if !req.stop.is_empty() {
        gen.insert("stopSequences".into(), json!(req.stop));
    }
    // Gemini carries the thinking budget inside generationConfig (clamped to the model's range upstream).
    if let Some(t) = req.thinking {
        gen.insert(
            "thinkingConfig".into(),
            json!({ "thinkingBudget": crate::thinking::budget_tokens(t) }),
        );
    }
    if !gen.is_empty() {
        obj.insert("generationConfig".into(), Value::Object(gen));
    }

    if !req.tools.is_empty() {
        let decls: Vec<Value> = req
            .tools
            .iter()
            .map(|t| json!({ "name": t.name, "description": t.description, "parameters": t.parameters }))
            .collect();
        obj.insert("tools".into(), json!([{ "functionDeclarations": decls }]));
    }
    if let Some(tc) = &req.tool_choice {
        // Gemini expresses tool choice via toolConfig.functionCallingConfig.mode (AUTO|ANY|NONE).
        let cfg = match tc {
            ToolChoice::Auto => json!({ "mode": "AUTO" }),
            ToolChoice::None => json!({ "mode": "NONE" }),
            ToolChoice::Required => json!({ "mode": "ANY" }),
            ToolChoice::Named(name) => json!({ "mode": "ANY", "allowedFunctionNames": [name] }),
        };
        obj.insert("toolConfig".into(), json!({ "functionCallingConfig": cfg }));
    }
    for (k, v) in &req.extra {
        obj.entry(k.clone()).or_insert_with(|| v.clone());
    }
    Value::Object(obj)
}

fn message_to_gemini(msg: &Message) -> Value {
    if msg.role == Role::Tool {
        return json!({
            "role": "user",
            "parts": [{
                "functionResponse": {
                    "name": msg.name.clone().unwrap_or_default(),
                    "response": { "content": msg.text_content() },
                }
            }],
        });
    }

    let role = if msg.role == Role::Assistant {
        "model"
    } else {
        "user"
    };
    let mut parts: Vec<Value> = msg
        .content
        .iter()
        .map(|p| match p {
            ContentPart::Text { text } => json!({ "text": text }),
            ContentPart::ImageUrl { url, .. } => json!({ "fileData": { "fileUri": url } }),
        })
        .collect();

    for call in &msg.tool_calls {
        let args: Value = serde_json::from_str(&call.arguments).unwrap_or_else(|_| json!({}));
        parts.push(json!({ "functionCall": { "name": call.name, "args": args } }));
    }

    json!({ "role": role, "parts": parts })
}

/// Map a Gemini `generateContent` *response* onto canonical [`StreamChunk`]s. `pub(crate)` because
/// Vertex returns the identical response schema (additive Vertex-only fields aside), so its chat path
/// reuses this verbatim.
pub(crate) fn gemini_to_chunks(value: Value, fallback_model: &str) -> Vec<StreamChunk> {
    let id = value
        .get("responseId")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let model = value
        .get("modelVersion")
        .and_then(Value::as_str)
        .unwrap_or(fallback_model)
        .to_string();

    let mut chunks = vec![StreamChunk::Start { id, model }];
    let mut emitted_finish = false;

    if let Some(candidates) = value.get("candidates").and_then(Value::as_array) {
        for candidate in candidates {
            let index = candidate.get("index").and_then(Value::as_u64).unwrap_or(0) as u32;
            let mut saw_tool = false;
            let mut tool_index = 0u32;

            if let Some(parts) = candidate
                .get("content")
                .and_then(|c| c.get("parts"))
                .and_then(Value::as_array)
            {
                for part in parts {
                    if let Some(text) = part.get("text").and_then(Value::as_str) {
                        if !text.is_empty() {
                            chunks.push(StreamChunk::Content {
                                index,
                                delta: text.to_string(),
                            });
                        }
                    } else if let Some(fc) = part.get("functionCall") {
                        saw_tool = true;
                        let arguments = fc.get("args").map(|v| v.to_string());
                        chunks.push(StreamChunk::ToolCall {
                            index,
                            call: ToolCallDelta {
                                index: tool_index,
                                // Gemini 3 emits a per-call id to echo in the functionResponse;
                                // older models omit it (pair by name in that case).
                                id: fc.get("id").and_then(Value::as_str).map(str::to_owned),
                                name: fc.get("name").and_then(Value::as_str).map(str::to_owned),
                                arguments,
                            },
                        });
                        tool_index += 1;
                    }
                }
            }

            let reason = if saw_tool {
                FinishReason::ToolCalls
            } else {
                candidate
                    .get("finishReason")
                    .and_then(Value::as_str)
                    .map(map_finish_reason)
                    .unwrap_or(FinishReason::Stop)
            };
            chunks.push(StreamChunk::Finish { index, reason });
            emitted_finish = true;
        }
    }

    // A prompt blocked before generation has no candidates; the block surfaces under
    // `promptFeedback.blockReason`. Always emit a terminal Finish so the stream never ends without
    // one (content_filter when blocked, otherwise a benign stop for an empty/candidate-less reply).
    if !emitted_finish {
        let blocked = value
            .get("promptFeedback")
            .and_then(|f| f.get("blockReason"))
            .and_then(Value::as_str)
            .is_some();
        let reason = if blocked {
            FinishReason::ContentFilter
        } else {
            FinishReason::Stop
        };
        chunks.push(StreamChunk::Finish { index: 0, reason });
    }

    if let Some(usage) = value.get("usageMetadata") {
        let prompt = usage
            .get("promptTokenCount")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let completion = usage
            .get("candidatesTokenCount")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let total = usage
            .get("totalTokenCount")
            .and_then(Value::as_u64)
            .unwrap_or(prompt + completion);
        chunks.push(StreamChunk::Usage(Usage {
            prompt_tokens: prompt,
            completion_tokens: completion,
            total_tokens: total,
            cost_usd: None,
        }));
    }

    chunks
}

fn map_finish_reason(reason: &str) -> FinishReason {
    match reason {
        "MAX_TOKENS" => FinishReason::Length,
        "SAFETY" | "RECITATION" | "BLOCKLIST" | "PROHIBITED_CONTENT" | "SPII" | "IMAGE_SAFETY" => {
            FinishReason::ContentFilter
        }
        // Abnormal terminations — distinct from a clean stop (spec groups these as error cases).
        "MALFORMED_FUNCTION_CALL"
        | "UNEXPECTED_TOOL_CALL"
        | "TOO_MANY_TOOL_CALLS"
        | "OTHER"
        | "LANGUAGE" => FinishReason::Error,
        // STOP, FINISH_REASON_UNSPECIFIED, and unknown values are a normal stop.
        _ => FinishReason::Stop,
    }
}

/// A Gemini batch long-running-operation → canonical [`BatchHandle`]. State and counts live under
/// `metadata`; the resource `name` is `batches/{id}` — keep the id past that prefix so retrieve/cancel
/// address it. Instants are RFC-3339 strings (not unix seconds), so the numeric fields stay `None`.
/// Map one `GET /v1beta/models` entry to canonical [`ModelInfo`]. `None` (skipped) when it has no
/// `name`. The id is `name` with the `models/` prefix stripped (what the generate API takes).
fn gemini_model_to_info(m: &Value) -> Option<ModelInfo> {
    let raw = m.get("name").and_then(Value::as_str)?;
    let id = raw.strip_prefix("models/").unwrap_or(raw);
    let mut info = ModelInfo::new(id);
    info.name = m
        .get("displayName")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .map(str::to_string);
    let methods: Vec<&str> = m
        .get("supportedGenerationMethods")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    info.modality = gemini_modality(&methods);
    // Token limits may arrive as JSON numbers or numeric strings (Google occasionally stringifies int64).
    let token = |key: &str| {
        m.get(key)
            .and_then(|v| {
                v.as_u64()
                    .or_else(|| v.as_str().and_then(|s| s.parse::<u64>().ok()))
            })
            .map(|n| n.min(u32::MAX as u64) as u32)
    };
    info.max_context = token("inputTokenLimit");
    info.max_output = token("outputTokenLimit");
    // No pricing, no published thinking budget — left None, enhanced from the bundled dataset. Stash
    // the descriptive/capability fields verbatim (principle 7).
    for key in [
        "version",
        "description",
        "supportedGenerationMethods",
        "thinking",
        "temperature",
        "maxTemperature",
        "topP",
        "topK",
        "baseModelId",
    ] {
        if let Some(v) = m.get(key) {
            if !v.is_null() {
                info.extra.insert(key.to_string(), v.clone());
            }
        }
    }
    Some(info)
}

/// Infer modality from `supportedGenerationMethods` — Llm wins over Embedding when both are present;
/// predict-only (Imagen/Veo) and method-less entries are `None`. No id-substring guessing.
fn gemini_modality(methods: &[&str]) -> Option<Modality> {
    let has = |m: &str| methods.contains(&m);
    if has("generateContent") || has("countTokens") || has("batchGenerateContent") {
        Some(Modality::Llm)
    } else if has("embedContent") || has("batchEmbedContents") {
        Some(Modality::Embedding)
    } else {
        None
    }
}

fn gemini_batch_to_handle(value: &Value) -> BatchHandle {
    let id = value
        .get("name")
        .and_then(Value::as_str)
        .map(|n| n.strip_prefix("batches/").unwrap_or(n).to_string())
        .unwrap_or_default();
    let metadata = value.get("metadata");
    let status = metadata
        .and_then(|m| m.get("state"))
        .and_then(Value::as_str)
        .map(map_gemini_batch_state)
        .unwrap_or(BatchStatus::InProgress);
    let counts = metadata
        .and_then(|m| m.get("batchStats"))
        .map(gemini_batch_counts)
        .unwrap_or_default();
    BatchHandle {
        id,
        status,
        counts,
        created_at: None,
        expires_at: None,
        ended_at: None,
        endpoint: Some("/v1beta/models:generateContent".to_string()),
        extra: Map::new(),
    }
}

/// Map a Gemini job-state string onto the canonical ladder. The wire uses `JOB_STATE_*` (cookbook/curl)
/// while the schema lists `BATCH_STATE_*`; match on the suffix so either prefix works.
fn map_gemini_batch_state(s: &str) -> BatchStatus {
    match s.rsplit('_').next().unwrap_or(s) {
        "PENDING" | "RUNNING" => BatchStatus::InProgress,
        "SUCCEEDED" => BatchStatus::Completed,
        "FAILED" => BatchStatus::Failed,
        "CANCELLED" => BatchStatus::Canceled,
        "EXPIRED" => BatchStatus::Expired,
        _ => BatchStatus::InProgress,
    }
}

/// `batchStats` → canonical counts. Google often serializes int64 as JSON *strings*, so read either.
fn gemini_batch_counts(stats: &Value) -> BatchCounts {
    let n = |k: &str| {
        stats
            .get(k)
            .and_then(|v| {
                v.as_u64()
                    .or_else(|| v.as_str().and_then(|s| s.parse().ok()))
            })
            .unwrap_or(0)
    };
    BatchCounts {
        total: n("requestCount"),
        processing: n("pendingRequestCount"),
        succeeded: n("successfulRequestCount"),
        errored: n("failedRequestCount"),
        canceled: 0,
        expired: 0,
    }
}

/// One inline batch response entry → canonical [`BatchResult`]. Keyed by `metadata.key` (the
/// `custom_id`); a `response` maps through [`gemini_to_chunks`] like a live turn, an `error` becomes an
/// errored outcome. Entries without a key are skipped.
fn gemini_inline_result(entry: &Value) -> Option<BatchResult> {
    let custom_id = entry
        .get("metadata")
        .and_then(|m| m.get("key"))
        .and_then(Value::as_str)?
        .to_string();
    if let Some(err) = entry.get("error").filter(|e| !e.is_null()) {
        let message = err
            .get("message")
            .and_then(Value::as_str)
            .map(str::to_string)
            .unwrap_or_else(|| err.to_string());
        return Some(BatchResult {
            custom_id,
            outcome: BatchOutcome::Errored { status: 0, message },
        });
    }
    let response = entry.get("response")?;
    Some(BatchResult {
        custom_id,
        outcome: BatchOutcome::Succeeded(collect_chunks(gemini_to_chunks(response.clone(), ""))),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_contents_and_generation_config() {
        let req = ChatRequest {
            model: "gemini-2.0-flash".into(),
            messages: vec![
                Message::text(Role::System, "be brief"),
                Message::text(Role::User, "hi"),
                Message::text(Role::Assistant, "hello"),
            ],
            max_tokens: Some(256),
            temperature: Some(0.5),
            top_p: None,
            stop: vec![],
            stream: false,
            tools: vec![],
            tool_choice: None,
            thinking: None,
            extra: Default::default(),
        };
        let wire = request_to_gemini(&req);
        assert_eq!(wire["systemInstruction"]["parts"][0]["text"], "be brief");
        assert_eq!(wire["contents"][0]["role"], "user");
        assert_eq!(wire["contents"][1]["role"], "model"); // assistant -> model
        assert_eq!(wire["generationConfig"]["maxOutputTokens"], 256);
        // No thinking requested -> no thinkingConfig.
        assert!(wire["generationConfig"].get("thinkingConfig").is_none());
    }

    #[test]
    fn thinking_sets_thinking_budget() {
        let req = ChatRequest {
            model: "gemini-2.5-flash".into(),
            messages: vec![Message::text(Role::User, "hi")],
            max_tokens: None,
            temperature: None,
            top_p: None,
            stop: vec![],
            stream: false,
            tools: vec![],
            tool_choice: None,
            thinking: Some(llmleaf_model::Thinking::Max),
            extra: Default::default(),
        };
        let wire = request_to_gemini(&req);
        assert_eq!(
            wire["generationConfig"]["thinkingConfig"]["thinkingBudget"],
            32768
        );
    }

    #[test]
    fn parses_candidate_text_and_usage() {
        let resp = json!({
            "candidates": [{
                "index": 0,
                "content": { "role": "model", "parts": [{ "text": "hello" }] },
                "finishReason": "STOP"
            }],
            "usageMetadata": { "promptTokenCount": 4, "candidatesTokenCount": 1, "totalTokenCount": 5 }
        });
        let chunks = gemini_to_chunks(resp, "gemini-2.0-flash");
        assert!(matches!(&chunks[1], StreamChunk::Content { delta, .. } if delta == "hello"));
        assert!(matches!(
            chunks[2],
            StreamChunk::Finish {
                reason: FinishReason::Stop,
                ..
            }
        ));
        assert!(matches!(chunks[3], StreamChunk::Usage(u) if u.total_tokens == 5));
    }

    #[test]
    fn function_call_sets_tool_finish() {
        let resp = json!({
            "candidates": [{
                "index": 0,
                "content": { "role": "model", "parts": [{ "functionCall": { "name": "get_weather", "args": { "city": "Paris" } } }] },
                "finishReason": "STOP"
            }]
        });
        let chunks = gemini_to_chunks(resp, "gemini-2.0-flash");
        match &chunks[1] {
            StreamChunk::ToolCall { call, .. } => {
                assert_eq!(call.name.as_deref(), Some("get_weather"))
            }
            other => panic!("expected ToolCall, got {other:?}"),
        }
        assert!(matches!(
            chunks[2],
            StreamChunk::Finish {
                reason: FinishReason::ToolCalls,
                ..
            }
        ));
    }

    #[test]
    fn builds_batch_embed_requests() {
        let req = EmbeddingRequest {
            model: "text-embedding-004".into(),
            input: vec!["hello".into(), "world".into()],
            dimensions: Some(256),
            encoding_format: None,
            extra: Default::default(),
        };
        let wire = embedding_request_to_gemini(&req, Some("RETRIEVAL_DOCUMENT"));
        // One request per input, in order, each carrying the `models/` prefix and the part text.
        assert_eq!(wire["requests"][0]["model"], "models/text-embedding-004");
        assert_eq!(wire["requests"][0]["content"]["parts"][0]["text"], "hello");
        assert_eq!(wire["requests"][1]["content"]["parts"][0]["text"], "world");
        assert_eq!(wire["requests"][0]["outputDimensionality"], 256);
        assert_eq!(wire["requests"][0]["taskType"], "RETRIEVAL_DOCUMENT");
        // taskType is omitted entirely when the operator did not configure one.
        let bare = embedding_request_to_gemini(&req, None);
        assert!(bare["requests"][0].get("taskType").is_none());
    }

    #[test]
    fn parses_batch_embeddings_in_order() {
        let resp = json!({
            "embeddings": [
                { "values": [0.1, 0.2, 0.3] },
                { "values": [0.4, 0.5] }
            ]
        });
        let out = gemini_to_embeddings(resp, "text-embedding-004");
        assert_eq!(out.model, "text-embedding-004");
        assert_eq!(out.embeddings[0].index, 0);
        assert_eq!(out.embeddings[1].index, 1);
        assert_eq!(out.embeddings[0].vector.len(), 3);
        assert!((out.embeddings[1].vector[0] - 0.4).abs() < 1e-6);
        // batchEmbedContents reports no usage.
        assert_eq!(out.usage, Usage::default());
    }

    #[test]
    fn batch_handle_maps_lro_state_and_stats() {
        // State strings use JOB_STATE_*; counts may be JSON strings; the name keeps the id past
        // "batches/".
        let lro = json!({
            "name": "batches/abc123",
            "metadata": {
                "state": "JOB_STATE_RUNNING",
                "batchStats": {
                    "requestCount": "10",
                    "successfulRequestCount": 4,
                    "failedRequestCount": "1",
                    "pendingRequestCount": 5
                }
            }
        });
        let h = gemini_batch_to_handle(&lro);
        assert_eq!(h.id, "abc123");
        assert_eq!(h.status, BatchStatus::InProgress);
        assert_eq!(h.counts.total, 10);
        assert_eq!(h.counts.succeeded, 4);
        assert_eq!(h.counts.errored, 1);
        assert_eq!(h.counts.processing, 5);
        // The schema's BATCH_STATE_* prefix is tolerated too.
        assert_eq!(
            map_gemini_batch_state("BATCH_STATE_SUCCEEDED"),
            BatchStatus::Completed
        );
        assert_eq!(
            map_gemini_batch_state("JOB_STATE_CANCELLED"),
            BatchStatus::Canceled
        );
    }

    #[test]
    fn inline_result_maps_response_and_error() {
        let ok = json!({
            "metadata": { "key": "a" },
            "response": { "candidates": [{ "content": { "parts": [{ "text": "hi" }] } }] }
        });
        let r = gemini_inline_result(&ok).unwrap();
        assert_eq!(r.custom_id, "a");
        match r.outcome {
            BatchOutcome::Succeeded(resp) => assert_eq!(resp.choices[0].text, "hi"),
            other => panic!("expected success, got {other:?}"),
        }

        let err = json!({ "metadata": { "key": "b" }, "error": { "message": "boom" } });
        let r = gemini_inline_result(&err).unwrap();
        assert!(matches!(r.outcome, BatchOutcome::Errored { .. }));

        // No key → skipped.
        assert!(gemini_inline_result(&json!({ "response": {} })).is_none());
    }
}
