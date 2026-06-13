//! Anthropic Messages API provider.
//!
//! Its wire format is its own (decision filter: a provider quirk lives in the provider): `system` is
//! a top-level field (not a message), content is a list of typed blocks, auth is the `x-api-key`
//! header plus a required `anthropic-version`, and `max_tokens` is mandatory. All of that is mapped
//! here so the core stays dialect-free.
//!
//! Extended thinking is one more quirk handled here. When the canonical request carries a `thinking`
//! effort, it maps to a `thinking: { type: "enabled", budget_tokens }` block. Anthropic counts thinking
//! tokens toward `max_tokens` and *requires* `budget_tokens < max_tokens`, so the wire `max_tokens` is
//! raised to the caller's requested response room plus the budget (translating two canonical intents —
//! response cap and thinking budget — onto Anthropic's single combined cap). Anthropic also rejects
//! `temperature`/`top_p` alongside thinking, so both are omitted from the wire when thinking is on.
//! These adjustments apply only when the caller opts in via `thinking`; a request without it is mapped
//! verbatim (no silent mutation — principle 7).
//!
//! Prompt caching is another such quirk and stays here too. It is opt-in per provider instance via
//! `settings = { prompt_cache = ... }` (config is the base — principle 6); when unset the wire request
//! is left untouched (no silent mutation — principle 7). When enabled, `cache_control` breakpoints are
//! stamped onto the stable prefix (the `system` block — which also caches the `tools` rendered before
//! it — or the tool list when there is no system) and onto the running conversation, so repeat
//! requests read the prefix from Anthropic's cache instead of re-billing it. The cache lifetime
//! defaults to Anthropic's own 5-minute window and can be extended to 1 hour ([`CacheTtl`]).

use async_trait::async_trait;
use futures::stream;
use llmleaf_model::{
    collect_chunks, BatchCounts, BatchHandle, BatchOutcome, BatchResult, BatchResultStream,
    BatchSpec, BatchStatus, ChatRequest, ContentPart, FinishReason, Message, Modality, ModelError,
    ModelInfo, ResponseStream, Role, StreamChunk, ToolCallDelta, ToolChoice, Usage,
};
use llmleaf_provider::{Provider, ProviderCx};
use serde_json::{json, Map, Value};

use std::sync::Arc;

use crate::batch::jsonl_result_stream;
use crate::http::{post_json, send_checked};
use crate::transport::{HttpRequest, HttpTransport, Transports};

const DEFAULT_ENDPOINT: &str = "https://api.anthropic.com";
const DEFAULT_VERSION: &str = "2023-06-01";
/// Anthropic requires `max_tokens`; used when the canonical request leaves it unset.
const DEFAULT_MAX_TOKENS: u32 = 4096;

pub struct AnthropicProvider {
    http: Arc<dyn HttpTransport>,
}

impl AnthropicProvider {
    pub fn new(transports: &Transports) -> Self {
        AnthropicProvider {
            http: transports.http.clone(),
        }
    }

    /// The base endpoint for this instance (config override or Anthropic's default), trailing slash
    /// trimmed.
    fn endpoint(&self, cx: &ProviderCx) -> String {
        cx.endpoint
            .as_deref()
            .unwrap_or(DEFAULT_ENDPOINT)
            .trim_end_matches('/')
            .to_string()
    }

    /// Apply Anthropic's auth: the required `anthropic-version` header plus the `x-api-key` credential.
    /// Shared by chat and batch so they authenticate identically.
    fn auth(&self, req: HttpRequest, cx: &ProviderCx) -> HttpRequest {
        let version = cx
            .setting_str("anthropic_version")
            .unwrap_or(DEFAULT_VERSION);
        let mut req = req.header("anthropic-version", version);
        if let Some(cred) = &cx.credential {
            req = req.header("x-api-key", cred);
        }
        req
    }
}

#[async_trait]
impl Provider for AnthropicProvider {
    fn name(&self) -> &str {
        "anthropic"
    }

    /// Enumerate Anthropic's catalog via `GET /v1/models` (the same `x-api-key` + `anthropic-version`
    /// auth as chat). The catalog is LLM-only and carries no pricing, so only id/display_name and the
    /// optional `max_input_tokens`/`max_tokens` map across; everything else is enhanced downstream.
    async fn models(&self, cx: &ProviderCx) -> Result<Vec<ModelInfo>, ModelError> {
        // `?limit=1000` returns the whole catalog in one shot (well under the cursor-paginated cap).
        let url = format!("{}/v1/models?limit=1000", self.endpoint(cx));
        let req = self.auth(HttpRequest::get(&url), cx);
        let value = post_json(&*self.http, req).await?;
        let models = value
            .get("data")
            .and_then(Value::as_array)
            .map(|items| items.iter().filter_map(anthropic_model_to_info).collect())
            .unwrap_or_default();
        Ok(models)
    }

    async fn chat(&self, req: ChatRequest, cx: &ProviderCx) -> Result<ResponseStream, ModelError> {
        let endpoint = cx
            .endpoint
            .as_deref()
            .unwrap_or(DEFAULT_ENDPOINT)
            .trim_end_matches('/');
        let url = format!("{endpoint}/v1/messages");
        let version = cx
            .setting_str("anthropic_version")
            .unwrap_or(DEFAULT_VERSION);
        let body = request_to_anthropic(&req, prompt_cache(cx));

        let mut http_req = HttpRequest::post(&url)
            .header("anthropic-version", version)
            .json(body);
        if let Some(cred) = &cx.credential {
            http_req = http_req.header("x-api-key", cred);
        }

        let value = post_json(&*self.http, http_req).await?;
        let chunks = anthropic_to_chunks(value, &req.model);
        Ok(Box::pin(stream::iter(chunks.into_iter().map(Ok))))
    }

    /// Submit a Message Batch. Anthropic takes the requests *inline* (no file step), so each canonical
    /// item maps through the very same [`request_to_anthropic`] the live `chat` path uses, wrapped as
    /// `{ custom_id, params }`. The whole submission completes in one call — nothing is held after.
    async fn batch_create(
        &self,
        req: BatchSpec,
        cx: &ProviderCx,
    ) -> Result<BatchHandle, ModelError> {
        let url = format!("{}/v1/messages/batches", self.endpoint(cx));
        let cache = prompt_cache(cx);
        let requests: Vec<Value> = req
            .items
            .iter()
            .map(|item| {
                json!({
                    "custom_id": item.custom_id,
                    "params": request_to_anthropic(&item.request, cache),
                })
            })
            .collect();
        let req = self.auth(
            HttpRequest::post(&url).json(json!({ "requests": requests })),
            cx,
        );
        let value = post_json(&*self.http, req).await?;
        Ok(anthropic_batch_to_handle(&value))
    }

    async fn batch_retrieve(
        &self,
        upstream_id: &str,
        cx: &ProviderCx,
    ) -> Result<BatchHandle, ModelError> {
        let url = format!("{}/v1/messages/batches/{upstream_id}", self.endpoint(cx));
        let req = self.auth(HttpRequest::get(&url), cx);
        let value = post_json(&*self.http, req).await?;
        Ok(anthropic_batch_to_handle(&value))
    }

    async fn batch_cancel(
        &self,
        upstream_id: &str,
        cx: &ProviderCx,
    ) -> Result<BatchHandle, ModelError> {
        let url = format!(
            "{}/v1/messages/batches/{upstream_id}/cancel",
            self.endpoint(cx)
        );
        let req = self.auth(HttpRequest::post(&url), cx);
        let value = post_json(&*self.http, req).await?;
        Ok(anthropic_batch_to_handle(&value))
    }

    /// Stream a batch's results. Anthropic exposes them as JSONL at `…/results`; each line maps to a
    /// canonical [`BatchResult`] by [`anthropic_result_line`].
    async fn batch_results(
        &self,
        upstream_id: &str,
        cx: &ProviderCx,
    ) -> Result<BatchResultStream, ModelError> {
        let url = format!(
            "{}/v1/messages/batches/{upstream_id}/results",
            self.endpoint(cx)
        );
        let req = self.auth(HttpRequest::get(&url), cx);
        let resp = send_checked(&*self.http, req).await?;
        Ok(jsonl_result_stream(resp.body, anthropic_result_line))
    }
}

/// Canonical request → Anthropic Messages JSON. `cache`, when set, threads the configured prompt-cache
/// TTL through so the stable prefix and the running conversation get `cache_control` breakpoints.
fn request_to_anthropic(req: &ChatRequest, cache: Option<CacheTtl>) -> Value {
    let mut obj = Map::new();
    obj.insert("model".into(), json!(req.model));

    // Extended thinking maps the canonical ladder to a token budget. Anthropic counts thinking tokens
    // toward `max_tokens` and *requires* `budget_tokens < max_tokens`, so raise the cap to fit the
    // budget on top of the response room the caller asked for. With thinking on, Anthropic also rejects
    // `temperature`/`top_p` — both are omitted below. Opt-in via `thinking`; a documented dialect
    // mapping, not silent magic (principle 7).
    let thinking_budget = req.thinking.map(crate::thinking::budget_tokens);
    let effective_max = req.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS);
    let max_tokens = match thinking_budget {
        Some(budget) => effective_max.saturating_add(budget),
        None => effective_max,
    };
    obj.insert("max_tokens".into(), json!(max_tokens));

    // System messages are hoisted to the top-level `system` field, concatenated.
    let system: String = req
        .messages
        .iter()
        .filter(|m| m.role == Role::System)
        .map(|m| m.text_content())
        .collect::<Vec<_>>()
        .join("\n\n");
    let has_system = !system.is_empty();
    if has_system {
        // Cache the stable prefix. A breakpoint on the `system` block also caches `tools` (which
        // Anthropic renders before `system`), so this one marker covers both. Attaching `cache_control`
        // requires emitting `system` as a one-element block list rather than a bare string.
        let system_value = match cache {
            Some(ttl) => {
                json!([{ "type": "text", "text": system, "cache_control": ttl.control() }])
            }
            None => json!(system),
        };
        obj.insert("system".into(), system_value);
    }

    let mut messages: Vec<Value> = req
        .messages
        .iter()
        .filter(|m| m.role != Role::System)
        .map(message_to_anthropic)
        .collect();
    // Cache the running conversation: with the breakpoint on the final content block, the whole message
    // prefix is a cache read on the next turn (the standard multi-turn caching pattern).
    if let Some(ttl) = cache {
        mark_last_message(&mut messages, ttl);
    }
    obj.insert("messages".into(), json!(messages));

    // Anthropic rejects sampling params when extended thinking is enabled; omit them in that case.
    if thinking_budget.is_none() {
        if let Some(v) = req.temperature {
            obj.insert("temperature".into(), json!(v));
        }
        if let Some(v) = req.top_p {
            obj.insert("top_p".into(), json!(v));
        }
    }
    if !req.stop.is_empty() {
        obj.insert("stop_sequences".into(), json!(req.stop));
    }
    if !req.tools.is_empty() {
        let mut tools: Vec<Value> = req
            .tools
            .iter()
            .map(|t| {
                json!({
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.parameters,
                })
            })
            .collect();
        // With no system prompt to carry the prefix breakpoint, cache the (otherwise stable) tool list
        // directly by marking its last entry; with a system prompt the breakpoint there already covers
        // the tools, so they need no separate marker.
        if let Some(ttl) = cache {
            if !has_system {
                if let Some(last) = tools.last_mut().and_then(Value::as_object_mut) {
                    last.insert("cache_control".into(), ttl.control());
                }
            }
        }
        obj.insert("tools".into(), json!(tools));
    }
    if let Some(tc) = &req.tool_choice {
        obj.insert(
            "tool_choice".into(),
            match tc {
                ToolChoice::Auto => json!({ "type": "auto" }),
                ToolChoice::None => json!({ "type": "none" }),
                ToolChoice::Required => json!({ "type": "any" }),
                ToolChoice::Named(name) => json!({ "type": "tool", "name": name }),
            },
        );
    }
    if let Some(budget) = thinking_budget {
        obj.insert(
            "thinking".into(),
            json!({ "type": "enabled", "budget_tokens": budget }),
        );
    }
    for (k, v) in &req.extra {
        obj.entry(k.clone()).or_insert_with(|| v.clone());
    }
    Value::Object(obj)
}

fn message_to_anthropic(msg: &Message) -> Value {
    // Tool results map to a `tool_result` content block carried by a user-role message.
    if msg.role == Role::Tool {
        return json!({
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": msg.tool_call_id.clone().unwrap_or_default(),
                "content": msg.text_content(),
            }],
        });
    }

    let role = if msg.role == Role::Assistant {
        "assistant"
    } else {
        "user"
    };
    let mut blocks: Vec<Value> = msg
        .content
        .iter()
        .map(|p| match p {
            ContentPart::Text { text } => json!({ "type": "text", "text": text }),
            ContentPart::ImageUrl { url, .. } => {
                json!({ "type": "image", "source": { "type": "url", "url": url } })
            }
        })
        .collect();

    // Assistant tool calls become `tool_use` blocks; arguments parse from the canonical JSON string.
    for call in &msg.tool_calls {
        let input: Value = serde_json::from_str(&call.arguments).unwrap_or_else(|_| json!({}));
        blocks
            .push(json!({ "type": "tool_use", "id": call.id, "name": call.name, "input": input }));
    }

    json!({ "role": role, "content": blocks })
}

/// Anthropic ephemeral prompt-cache TTL. The API offers exactly two cache lifetimes: the default
/// 5-minute window and an extended 1-hour window. Enabling the cache is opt-in per provider instance
/// (`settings = { prompt_cache = ... }`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CacheTtl {
    /// Anthropic's default 5-minute window — expressed on the wire by omitting `ttl`.
    FiveMinutes,
    /// The extended 1-hour window (`ttl: "1h"`).
    OneHour,
}

impl CacheTtl {
    /// Parse a configured TTL. Anthropic recognizes only `"5m"` and `"1h"`; anything else falls back to
    /// the 5-minute default rather than failing the request — caching is best-effort, never a hard
    /// precondition.
    fn parse(raw: &str) -> Self {
        match raw.trim() {
            "1h" => CacheTtl::OneHour,
            _ => CacheTtl::FiveMinutes,
        }
    }

    /// The `cache_control` block to stamp onto a cached content block. 5 minutes is the API default, so
    /// it is expressed by omitting `ttl`; 1 hour names it explicitly.
    fn control(self) -> Value {
        match self {
            CacheTtl::FiveMinutes => json!({ "type": "ephemeral" }),
            CacheTtl::OneHour => json!({ "type": "ephemeral", "ttl": "1h" }),
        }
    }
}

/// Resolve the prompt-cache setting for this instance. `prompt_cache` is opt-in — absent (or `false`)
/// leaves the wire request untouched — and accepts three shapes: `true` (enable, default 5-minute
/// TTL), `"5m"` / `"1h"` (enable with that TTL), or `{ ttl = "1h" }` (enable with that TTL).
fn prompt_cache(cx: &ProviderCx) -> Option<CacheTtl> {
    match cx.settings.get("prompt_cache") {
        Some(Value::Bool(true)) => Some(CacheTtl::FiveMinutes),
        Some(Value::String(s)) => Some(CacheTtl::parse(s)),
        Some(Value::Object(o)) => Some(
            o.get("ttl")
                .and_then(Value::as_str)
                .map(CacheTtl::parse)
                .unwrap_or(CacheTtl::FiveMinutes),
        ),
        _ => None,
    }
}

/// Stamp `cache_control` onto the final content block of the conversation, so the whole message prefix
/// becomes a cache read on the next turn. A no-op when the last message carries no content blocks.
fn mark_last_message(messages: &mut [Value], ttl: CacheTtl) {
    let Some(block) = messages
        .last_mut()
        .and_then(|m| m.get_mut("content"))
        .and_then(Value::as_array_mut)
        .and_then(|blocks| blocks.last_mut())
        .and_then(Value::as_object_mut)
    else {
        return;
    };
    block.insert("cache_control".into(), ttl.control());
}

/// Anthropic response → canonical chunks.
fn anthropic_to_chunks(value: Value, fallback_model: &str) -> Vec<StreamChunk> {
    let id = value
        .get("id")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let model = value
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or(fallback_model)
        .to_string();

    let mut chunks = vec![StreamChunk::Start { id, model }];

    if let Some(blocks) = value.get("content").and_then(Value::as_array) {
        let mut tool_index = 0u32;
        for block in blocks {
            match block.get("type").and_then(Value::as_str) {
                Some("text") => {
                    if let Some(text) = block.get("text").and_then(Value::as_str) {
                        if !text.is_empty() {
                            chunks.push(StreamChunk::Content {
                                index: 0,
                                delta: text.to_string(),
                            });
                        }
                    }
                }
                Some("tool_use") => {
                    let arguments = block.get("input").map(|v| v.to_string());
                    chunks.push(StreamChunk::ToolCall {
                        index: 0,
                        call: ToolCallDelta {
                            index: tool_index,
                            id: block.get("id").and_then(Value::as_str).map(str::to_owned),
                            name: block.get("name").and_then(Value::as_str).map(str::to_owned),
                            arguments,
                        },
                    });
                    tool_index += 1;
                }
                _ => {}
            }
        }
    }

    let reason = value
        .get("stop_reason")
        .and_then(Value::as_str)
        .map(map_stop_reason)
        .unwrap_or(FinishReason::Stop);
    chunks.push(StreamChunk::Finish { index: 0, reason });

    if let Some(usage) = value.get("usage") {
        let prompt = usage
            .get("input_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let completion = usage
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

/// Anthropic Message Batch object → canonical [`BatchHandle`]. Anthropic reports instants as ISO-8601
/// strings (not unix seconds), so rather than pull in a date dependency just to reformat them, the
/// numeric instant fields stay `None` and the raw strings ride through verbatim in `extra` (principle 7
/// — never silently dropped). Status maps Anthropic's `processing_status` to the canonical ladder;
/// `ended` means processing is done and results are available, i.e. `Completed`.
/// Map one `GET /v1/models` entry to canonical [`ModelInfo`]. `None` (skipped) when it has no id.
fn anthropic_model_to_info(value: &Value) -> Option<ModelInfo> {
    let id = value.get("id").and_then(Value::as_str)?;
    let mut info = ModelInfo::new(id);
    info.name = value
        .get("display_name")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .map(str::to_string);
    // The catalog is LLM-only; `type` is always the literal "model" (an object discriminator, NOT a
    // modality), so there is no per-entry modality signal to read — Llm is a safe constant, not a guess.
    info.modality = Some(Modality::Llm);
    // `max_input_tokens`/`max_tokens` are optional and the docs show literal `0` placeholders — treat
    // both `0` and absent as unknown (`None`), never report a 0-token window (no silent magic).
    let positive = |key: &str| {
        value
            .get(key)
            .and_then(Value::as_u64)
            .filter(|&n| n > 0)
            .map(|n| n as u32)
    };
    info.max_context = positive("max_input_tokens");
    info.max_output = positive("max_tokens");
    // max_thinking / pricing have no field in this API — left None, enhanced from the bundled dataset.
    for key in ["created_at", "capabilities", "type"] {
        if let Some(v) = value.get(key) {
            if !v.is_null() {
                info.extra.insert(key.to_string(), v.clone());
            }
        }
    }
    Some(info)
}

fn anthropic_batch_to_handle(value: &Value) -> BatchHandle {
    let id = value
        .get("id")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let status = match value.get("processing_status").and_then(Value::as_str) {
        Some("canceling") => BatchStatus::Canceling,
        Some("ended") => BatchStatus::Completed,
        // "in_progress" and anything unrecognized read as still-processing.
        _ => BatchStatus::InProgress,
    };
    let rc = value.get("request_counts");
    let count = |k: &str| {
        rc.and_then(|c| c.get(k))
            .and_then(Value::as_u64)
            .unwrap_or(0)
    };
    let processing = count("processing");
    let succeeded = count("succeeded");
    let errored = count("errored");
    let canceled = count("canceled");
    let expired = count("expired");
    let counts = BatchCounts {
        total: processing + succeeded + errored + canceled + expired,
        processing,
        succeeded,
        errored,
        canceled,
        expired,
    };

    let mut extra = Map::new();
    for k in ["created_at", "expires_at", "ended_at"] {
        if let Some(v) = value.get(k) {
            if !v.is_null() {
                extra.insert(k.to_string(), v.clone());
            }
        }
    }

    BatchHandle {
        id,
        status,
        counts,
        created_at: None,
        expires_at: None,
        ended_at: None,
        endpoint: Some("/v1/messages".to_string()),
        extra,
    }
}

/// One line of an Anthropic batch results file → canonical [`BatchResult`]. A `succeeded` result
/// carries a full Messages response, mapped to a [`crate::ChatResponse`] via the same chunk mapping the
/// chat path uses, then folded. A line we cannot read at all is skipped (returns `None`).
fn anthropic_result_line(value: Value) -> Option<BatchResult> {
    let custom_id = value.get("custom_id")?.as_str()?.to_string();
    let result = value.get("result")?;
    let outcome = match result.get("type").and_then(Value::as_str)? {
        "succeeded" => {
            let message = result.get("message")?;
            BatchOutcome::Succeeded(collect_chunks(anthropic_to_chunks(message.clone(), "")))
        }
        "errored" => {
            let message = result
                .get("error")
                .and_then(|e| e.get("error"))
                .and_then(|e| e.get("message"))
                .and_then(Value::as_str)
                .map(str::to_string)
                .or_else(|| result.get("error").map(Value::to_string))
                .unwrap_or_else(|| "errored".to_string());
            BatchOutcome::Errored { status: 0, message }
        }
        "canceled" => BatchOutcome::Canceled,
        "expired" => BatchOutcome::Expired,
        _ => return None,
    };
    Some(BatchResult { custom_id, outcome })
}

fn map_stop_reason(reason: &str) -> FinishReason {
    match reason {
        "max_tokens" => FinishReason::Length,
        "tool_use" => FinishReason::ToolCalls,
        "refusal" => FinishReason::ContentFilter,
        // "end_turn", "stop_sequence", and anything else are a normal stop.
        _ => FinishReason::Stop,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llmleaf_model::ToolDef;

    fn chat_req(messages: Vec<Message>, tools: Vec<ToolDef>) -> ChatRequest {
        ChatRequest {
            model: "claude-sonnet-4".into(),
            messages,
            max_tokens: None,
            temperature: None,
            top_p: None,
            stop: vec![],
            stream: false,
            tools,
            tool_choice: None,
            thinking: None,
            extra: Default::default(),
        }
    }

    fn cache_setting(value: Value) -> ProviderCx {
        let mut settings = Map::new();
        settings.insert("prompt_cache".into(), value);
        ProviderCx {
            settings,
            ..Default::default()
        }
    }

    #[test]
    fn hoists_system_and_requires_max_tokens() {
        let req = ChatRequest {
            model: "claude-sonnet-4".into(),
            messages: vec![
                Message::text(Role::System, "be terse"),
                Message::text(Role::User, "hi"),
            ],
            max_tokens: None,
            temperature: None,
            top_p: None,
            stop: vec![],
            stream: false,
            tools: vec![],
            tool_choice: None,
            thinking: None,
            extra: Default::default(),
        };
        let wire = request_to_anthropic(&req, None);
        assert_eq!(wire["system"], "be terse");
        assert_eq!(wire["max_tokens"], DEFAULT_MAX_TOKENS);
        assert_eq!(wire["messages"].as_array().unwrap().len(), 1); // system not in messages
        assert_eq!(wire["messages"][0]["role"], "user");
    }

    #[test]
    fn thinking_enables_budget_and_drops_sampling() {
        let mut req = chat_req(vec![Message::text(Role::User, "hi")], vec![]);
        req.max_tokens = Some(1000);
        req.temperature = Some(0.7);
        req.top_p = Some(0.9);
        req.thinking = Some(llmleaf_model::Thinking::High);
        let wire = request_to_anthropic(&req, None);

        assert_eq!(wire["thinking"]["type"], "enabled");
        assert_eq!(wire["thinking"]["budget_tokens"], 8192);
        // Anthropic counts thinking toward max_tokens and demands budget < max_tokens, so the cap is
        // raised to the requested response room plus the budget.
        assert_eq!(wire["max_tokens"], 1000 + 8192);
        // ...and sampling params are forbidden alongside thinking, so they are omitted.
        assert!(wire.get("temperature").is_none());
        assert!(wire.get("top_p").is_none());
    }

    #[test]
    fn no_thinking_keeps_sampling_and_cap() {
        let mut req = chat_req(vec![Message::text(Role::User, "hi")], vec![]);
        req.max_tokens = Some(1000);
        req.temperature = Some(0.7);
        let wire = request_to_anthropic(&req, None);
        assert!(wire.get("thinking").is_none());
        assert_eq!(wire["max_tokens"], 1000);
        // f32 → JSON f64 isn't bit-exact, so compare with tolerance.
        assert!((wire["temperature"].as_f64().unwrap() - 0.7).abs() < 1e-6);
    }

    #[test]
    fn parses_text_and_usage() {
        let resp = json!({
            "id": "msg_1",
            "model": "claude-sonnet-4",
            "content": [{ "type": "text", "text": "hello" }],
            "stop_reason": "end_turn",
            "usage": { "input_tokens": 10, "output_tokens": 5 }
        });
        let chunks = anthropic_to_chunks(resp, "claude-sonnet-4");
        assert!(matches!(&chunks[1], StreamChunk::Content { delta, .. } if delta == "hello"));
        assert!(matches!(
            chunks[2],
            StreamChunk::Finish {
                reason: FinishReason::Stop,
                ..
            }
        ));
        assert!(matches!(chunks[3], StreamChunk::Usage(u) if u.total_tokens == 15));
    }

    #[test]
    fn parses_tool_use() {
        let resp = json!({
            "id": "msg_2",
            "model": "claude-opus-4",
            "content": [{ "type": "tool_use", "id": "tu_1", "name": "get_weather", "input": { "city": "Paris" } }],
            "stop_reason": "tool_use",
            "usage": { "input_tokens": 8, "output_tokens": 12 }
        });
        let chunks = anthropic_to_chunks(resp, "claude-opus-4");
        match &chunks[1] {
            StreamChunk::ToolCall { call, .. } => {
                assert_eq!(call.name.as_deref(), Some("get_weather"));
                assert!(call.arguments.as_deref().unwrap().contains("Paris"));
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
    fn prompt_cache_setting_shapes() {
        assert_eq!(
            prompt_cache(&cache_setting(json!(true))),
            Some(CacheTtl::FiveMinutes)
        );
        assert_eq!(
            prompt_cache(&cache_setting(json!("5m"))),
            Some(CacheTtl::FiveMinutes)
        );
        assert_eq!(
            prompt_cache(&cache_setting(json!("1h"))),
            Some(CacheTtl::OneHour)
        );
        assert_eq!(
            prompt_cache(&cache_setting(json!({ "ttl": "1h" }))),
            Some(CacheTtl::OneHour)
        );
        assert_eq!(prompt_cache(&cache_setting(json!(false))), None);
        // An absent setting disables caching (the wire request is left untouched).
        assert_eq!(prompt_cache(&ProviderCx::default()), None);
    }

    #[test]
    fn caches_system_prefix_and_conversation() {
        let req = chat_req(
            vec![
                Message::text(Role::System, "be terse"),
                Message::text(Role::User, "hi"),
            ],
            vec![],
        );
        let wire = request_to_anthropic(&req, Some(CacheTtl::FiveMinutes));
        // System becomes a one-block list carrying a 5-minute (ttl-less) breakpoint.
        let sys = &wire["system"][0];
        assert_eq!(sys["text"], "be terse");
        assert_eq!(sys["cache_control"], json!({ "type": "ephemeral" }));
        // The last (here only) message also carries a breakpoint on its final content block.
        let last = wire["messages"].as_array().unwrap().last().unwrap();
        let block = last["content"].as_array().unwrap().last().unwrap();
        assert_eq!(block["cache_control"], json!({ "type": "ephemeral" }));
    }

    #[test]
    fn cache_ttl_one_hour_names_ttl() {
        let req = chat_req(
            vec![
                Message::text(Role::System, "sys"),
                Message::text(Role::User, "hi"),
            ],
            vec![],
        );
        let wire = request_to_anthropic(&req, Some(CacheTtl::OneHour));
        assert_eq!(
            wire["system"][0]["cache_control"],
            json!({ "type": "ephemeral", "ttl": "1h" })
        );
    }

    #[test]
    fn caches_tool_list_when_no_system() {
        let tool = ToolDef {
            name: "get_weather".into(),
            description: Some("weather".into()),
            parameters: json!({ "type": "object" }),
        };
        let req = chat_req(vec![Message::text(Role::User, "hi")], vec![tool]);
        let wire = request_to_anthropic(&req, Some(CacheTtl::FiveMinutes));
        // With no system prompt, the prefix breakpoint lands on the last tool instead.
        let tools = wire["tools"].as_array().unwrap();
        assert_eq!(
            tools.last().unwrap()["cache_control"],
            json!({ "type": "ephemeral" })
        );
        assert!(wire.get("system").is_none());
    }

    #[test]
    fn no_cache_control_when_disabled() {
        let req = chat_req(
            vec![
                Message::text(Role::System, "sys"),
                Message::text(Role::User, "hi"),
            ],
            vec![],
        );
        let wire = request_to_anthropic(&req, None);
        // System stays a bare string; nothing anywhere is stamped with cache_control.
        assert_eq!(wire["system"], "sys");
        assert!(!wire.to_string().contains("cache_control"));
    }
}
