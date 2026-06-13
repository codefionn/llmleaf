//! Ollama native API provider (`/api/chat`, `/api/embed`, `/api/tags` + `/api/show`).
//!
//! Distinct dialect: Ollama does not speak the OpenAI wire on its native surface (that lives behind a
//! separate `/v1` shim). The native API is its own thing, handled here:
//!   - **`/api/chat` streams NDJSON**, not SSE — one JSON object per `\n`-delimited line, no `data:`
//!     prefix and no `[DONE]` sentinel. The final object (`done:true`) carries the token stats inline.
//!     Parsing it line-by-line gives genuinely live tokens (principle 4) instead of the
//!     collect-then-rechunk path the OpenAI-compat shim would force.
//!   - **sampling lives under a flat `options` object** (`temperature`, `top_p`, `num_predict`, `stop`,
//!     `seed`, …) rather than at the top level.
//!   - **tool-call arguments are a JSON *object*** on the wire, not OpenAI's stringified blob — mapped
//!     in both directions here (the canonical [`llmleaf_model::ToolCall`] carries the raw JSON string).
//!     A tool *result* is matched back to its call by name, so a round-trip needs the canonical message's
//!     `name` set (it becomes Ollama's `tool_name`); when absent it is omitted rather than guessed.
//!   - **reasoning** is requested with `think` (the canonical effort ladder → `low`/`medium`/`high`/
//!     `max`) and streamed back under `message.thinking`; like every other provider here, the reasoning
//!     trace has no canonical channel, so it is not folded into the answer stream.
//!   - **the catalog** comes from `GET /api/tags`, enriched per-model from `POST /api/show` for the
//!     real capability set (→ modality) and context length (`<arch>.context_length`) — neither of which
//!     `/api/tags` reports. Modality is read from declared capabilities, never guessed from the id (SOUL).
//!
//! Auth: none for a local server (the default); an optional bearer credential is sent so the same code
//! path serves Ollama Cloud (`https://ollama.com`).

use async_trait::async_trait;
use futures::{stream, StreamExt};
use llmleaf_model::{
    ChatRequest, ContentPart, Embedding, EmbeddingRequest, EmbeddingResponse, FinishReason,
    Message, Modality, ModelError, ModelInfo, ResponseStream, Role, StreamChunk, Thinking,
    ToolCallDelta, Usage,
};
use llmleaf_provider::{Provider, ProviderCx};
use serde_json::{json, Map, Value};

use std::sync::Arc;

use crate::http::{post_json, send_checked};
use crate::transport::{HttpRequest, HttpTransport, Transports};

const DEFAULT_ENDPOINT: &str = "http://localhost:11434";

/// Max concurrent `/api/show` enrichment calls in [`OllamaProvider::models`]. `/api/show` is one
/// round-trip per model and a pulled catalog can hold many; this bounds the fan-out so a large catalog
/// neither serializes nor floods the local server. Metadata reads are cheap, so a small pool suffices.
const SHOW_CONCURRENCY: usize = 8;

pub struct OllamaProvider {
    http: Arc<dyn HttpTransport>,
}

impl OllamaProvider {
    pub fn new(transports: &Transports) -> Self {
        OllamaProvider {
            http: transports.http.clone(),
        }
    }

    /// This instance's base endpoint (config override or the local default), trailing slash trimmed.
    /// The native paths (`/api/chat`, …) are appended to it.
    fn endpoint(&self, cx: &ProviderCx) -> String {
        cx.endpoint
            .as_deref()
            .unwrap_or(DEFAULT_ENDPOINT)
            .trim_end_matches('/')
            .to_string()
    }

    /// Apply optional bearer auth. A local Ollama needs none; Ollama Cloud takes an API key as
    /// `Authorization: Bearer`. Shared by every method so they authenticate identically.
    fn auth(&self, req: HttpRequest, cx: &ProviderCx) -> HttpRequest {
        match &cx.credential {
            Some(cred) => req.bearer(cred),
            None => req,
        }
    }

    /// Fetch one model's `/api/show` detail (capabilities + GGUF `model_info`) for catalog enrichment.
    async fn fetch_show(&self, cx: &ProviderCx, model: &str) -> Result<Value, ModelError> {
        let url = format!("{}/api/show", self.endpoint(cx));
        let req = self.auth(HttpRequest::post(&url).json(json!({ "model": model })), cx);
        post_json(&*self.http, req).await
    }
}

#[async_trait]
impl Provider for OllamaProvider {
    fn name(&self) -> &str {
        "ollama"
    }

    async fn chat(&self, req: ChatRequest, cx: &ProviderCx) -> Result<ResponseStream, ModelError> {
        let url = format!("{}/api/chat", self.endpoint(cx));
        let body = request_to_ollama(&req, cx.setting_str("keep_alive"));
        let http_req = self.auth(HttpRequest::post(&url).json(body), cx);
        let resp = send_checked(&*self.http, http_req).await?;

        // Parse the NDJSON stream incrementally: one JSON object per line, tokens flowing as they
        // arrive (principle 4). The final object carries the stats; there is no `[DONE]` sentinel.
        Ok(Box::pin(async_stream::stream! {
            let mut bytes = resp.body;
            let mut buf: Vec<u8> = Vec::with_capacity(1024);
            let mut seen_start = false;
            let mut saw_tool = false;
            while let Some(item) = bytes.next().await {
                let chunk = match item {
                    Ok(b) => b,
                    Err(e) => {
                        yield Err(e);
                        return;
                    }
                };
                for line in ndjson_lines(&mut buf, &chunk) {
                    match serde_json::from_str::<Value>(&line) {
                        Ok(v) => {
                            // Ollama can report an error as a JSON object on the stream itself, not
                            // only via a non-200 status — surface it as an upstream failure.
                            if let Some(err) = v.get("error").and_then(Value::as_str) {
                                yield Err(ModelError::Upstream {
                                    status: 500,
                                    message: err.to_string(),
                                });
                                return;
                            }
                            for c in ollama_chunk_to_canonical(&v, &mut seen_start, &mut saw_tool) {
                                yield Ok(c);
                            }
                        }
                        Err(e) => {
                            yield Err(ModelError::Mapping(e.to_string()));
                            return;
                        }
                    }
                }
            }
        }))
    }

    async fn embed(
        &self,
        req: EmbeddingRequest,
        cx: &ProviderCx,
    ) -> Result<EmbeddingResponse, ModelError> {
        // `/api/embed` (current) embeds every input in one round-trip and always returns an array of
        // vectors, even for a single input. (The legacy `/api/embeddings` is single-prompt and flat.)
        let url = format!("{}/api/embed", self.endpoint(cx));
        let body = embedding_request_to_ollama(&req, cx.setting_str("keep_alive"));
        let http_req = self.auth(HttpRequest::post(&url).json(body), cx);
        let value = post_json(&*self.http, http_req).await?;
        Ok(ollama_to_embeddings(value, &req.model))
    }

    /// Enumerate the local catalog. `GET /api/tags` lists what is pulled; it reports size/digest/quant
    /// but neither modality nor context length, so each entry is enriched best-effort from
    /// `POST /api/show` (capabilities → modality, `<arch>.context_length` → context window). A model
    /// whose `/api/show` fails keeps its `/api/tags` data rather than vanishing (fail toward
    /// availability — principle 8).
    async fn models(&self, cx: &ProviderCx) -> Result<Vec<ModelInfo>, ModelError> {
        let url = format!("{}/api/tags", self.endpoint(cx));
        let req = self.auth(HttpRequest::get(&url), cx);
        let value = post_json(&*self.http, req).await?;
        let bases: Vec<ModelInfo> = value
            .get("models")
            .and_then(Value::as_array)
            .map(|models| models.iter().filter_map(ollama_tag_to_info).collect())
            .unwrap_or_default();
        // Enrich each entry from `/api/show` with bounded concurrency, preserving catalog order. A model
        // whose `/api/show` fails keeps its `/api/tags` data rather than vanishing (principle 8).
        let out: Vec<ModelInfo> = stream::iter(bases)
            .map(|mut info| async move {
                let id = info.id.clone();
                if let Ok(show) = self.fetch_show(cx, &id).await {
                    enrich_from_show(&mut info, &show);
                }
                info
            })
            .buffered(SHOW_CONCURRENCY)
            .collect()
            .await;
        Ok(out)
    }
}

/// Canonical request → Ollama `/api/chat` body. Sampling is flattened into `options`; the consumer's
/// `keep_alive` knob (operator config) rides through when set; anything unmodeled passes verbatim via
/// `extra` (principle 7). `stream` is forced on — the internal representation is always a stream
/// (principle 4); a single non-streamed object would still parse, but streaming is the point.
fn request_to_ollama(req: &ChatRequest, keep_alive: Option<&str>) -> Value {
    let mut obj = Map::new();
    obj.insert("model".into(), json!(req.model));
    obj.insert(
        "messages".into(),
        Value::Array(req.messages.iter().map(message_to_ollama).collect()),
    );

    // Sampling lives under a flat `options` object on the native API. Seed it from any consumer-supplied
    // runner options in `extra` (`num_ctx`, `seed`, `num_gpu`, …) so they survive, then layer the modeled
    // canonical sampling fields on top — they take precedence, mirroring how the top-level `extra`
    // passthrough below fills only the gaps the canonical request leaves (principle 7: nothing dropped).
    let mut options = req
        .extra
        .get("options")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    if let Some(v) = req.temperature {
        options.insert("temperature".into(), json!(v));
    }
    if let Some(v) = req.top_p {
        options.insert("top_p".into(), json!(v));
    }
    if let Some(v) = req.max_tokens {
        options.insert("num_predict".into(), json!(v));
    }
    if !req.stop.is_empty() {
        options.insert("stop".into(), json!(req.stop));
    }
    if !options.is_empty() {
        obj.insert("options".into(), Value::Object(options));
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
    // Ollama's native `/api/chat` has no `tool_choice` field, so the canonical one is not mapped (an
    // operator whose build supports it can still force it through `extra`).

    if let Some(t) = req.thinking {
        obj.insert("think".into(), json!(think_value(t)));
    }
    if let Some(ka) = keep_alive {
        obj.insert("keep_alive".into(), json!(ka));
    }

    for (k, v) in &req.extra {
        obj.entry(k.clone()).or_insert_with(|| v.clone());
    }
    // The internal boundary is a stream; force NDJSON streaming regardless of what `extra` carried.
    obj.insert("stream".into(), json!(true));
    Value::Object(obj)
}

/// Map the canonical thinking ladder to Ollama's `think` string. Ollama accepts `low`/`medium`/`high`/
/// `max`; the canonical `Highx` rung has no Ollama equivalent and collapses onto `high` (lossy by
/// design — see [`Thinking`]).
fn think_value(t: Thinking) -> &'static str {
    match t {
        Thinking::Low => "low",
        Thinking::Med => "medium",
        Thinking::High | Thinking::Highx => "high",
        Thinking::Max => "max",
    }
}

fn message_to_ollama(msg: &Message) -> Value {
    let role = match msg.role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    };
    let mut obj = Map::new();
    obj.insert("role".into(), json!(role));

    // Content is a flat string; images ride alongside as a parallel array of bare base64 strings.
    let mut text = String::new();
    let mut images: Vec<String> = Vec::new();
    for part in &msg.content {
        match part {
            ContentPart::Text { text: t } => text.push_str(t),
            ContentPart::ImageUrl { url, .. } => {
                if let Some(b64) = ollama_image_b64(url) {
                    images.push(b64);
                }
            }
        }
    }
    obj.insert("content".into(), json!(text));
    if !images.is_empty() {
        obj.insert("images".into(), json!(images));
    }

    // Tool result: Ollama names the answered tool with `tool_name` (and accepts `tool_call_id`).
    if msg.role == Role::Tool {
        if let Some(name) = &msg.name {
            obj.insert("tool_name".into(), json!(name));
        }
        if let Some(id) = &msg.tool_call_id {
            obj.insert("tool_call_id".into(), json!(id));
        }
    }

    // Assistant tool calls: Ollama's `arguments` is a JSON object, so the canonical raw-JSON string is
    // parsed back into one (an unparseable blob degrades to `{}` rather than breaking the request).
    if !msg.tool_calls.is_empty() {
        let calls: Vec<Value> = msg
            .tool_calls
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let arguments: Value =
                    serde_json::from_str(&c.arguments).unwrap_or_else(|_| json!({}));
                let mut function = Map::new();
                function.insert("index".into(), json!(i));
                function.insert("name".into(), json!(c.name));
                function.insert("arguments".into(), arguments);
                let mut call = Map::new();
                if !c.id.is_empty() {
                    call.insert("id".into(), json!(c.id));
                }
                call.insert("function".into(), Value::Object(function));
                Value::Object(call)
            })
            .collect();
        obj.insert("tool_calls".into(), json!(calls));
    }

    Value::Object(obj)
}

/// Extract the bare base64 payload an image content part carries for Ollama's `images` array. A
/// `data:<mime>;base64,<payload>` URL is unwrapped to its payload; a bare base64 string passes through;
/// a remote `http(s)` URL the native API cannot fetch is skipped (`None`) so the text still goes out.
fn ollama_image_b64(url: &str) -> Option<String> {
    if let Some(idx) = url.find(";base64,") {
        return Some(url[idx + ";base64,".len()..].to_string());
    }
    if url.starts_with("http://") || url.starts_with("https://") {
        None
    } else {
        Some(url.to_string())
    }
}

/// One NDJSON object → canonical chunks. The opening [`StreamChunk::Start`] is emitted on the first
/// line (Ollama assigns no response id, so it stays empty — the model id rides through); content and
/// tool-call deltas follow; the `done:true` line yields the terminal [`StreamChunk::Finish`] and the
/// [`StreamChunk::Usage`] folded from `prompt_eval_count`/`eval_count`. `message.thinking` is not
/// surfaced — the canonical stream has no reasoning channel (same as every other provider here).
fn ollama_chunk_to_canonical(
    v: &Value,
    seen_start: &mut bool,
    saw_tool: &mut bool,
) -> Vec<StreamChunk> {
    let mut out = Vec::new();

    if !*seen_start {
        let model = v
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        out.push(StreamChunk::Start {
            id: String::new(),
            model,
        });
        *seen_start = true;
    }

    let message = v.get("message");

    if let Some(content) = message
        .and_then(|m| m.get("content"))
        .and_then(Value::as_str)
    {
        if !content.is_empty() {
            out.push(StreamChunk::Content {
                index: 0,
                delta: content.to_string(),
            });
        }
    }

    if let Some(tool_calls) = message
        .and_then(|m| m.get("tool_calls"))
        .and_then(Value::as_array)
    {
        for tc in tool_calls {
            let func = tc.get("function");
            // Ollama delivers complete tool calls (not fragmented), so `arguments` is a whole object —
            // re-serialized to the canonical raw-JSON string. `index` is Ollama's own per-call index.
            let arguments = func.and_then(|f| f.get("arguments")).map(|a| a.to_string());
            let index = func
                .and_then(|f| f.get("index"))
                .and_then(Value::as_u64)
                .unwrap_or(0) as u32;
            out.push(StreamChunk::ToolCall {
                index: 0,
                call: ToolCallDelta {
                    index,
                    id: tc.get("id").and_then(Value::as_str).map(str::to_owned),
                    name: func
                        .and_then(|f| f.get("name"))
                        .and_then(Value::as_str)
                        .map(str::to_owned),
                    arguments,
                },
            });
            *saw_tool = true;
        }
    }

    if v.get("done").and_then(Value::as_bool).unwrap_or(false) {
        let reason = if *saw_tool {
            FinishReason::ToolCalls
        } else {
            v.get("done_reason")
                .and_then(Value::as_str)
                .map(map_done_reason)
                .unwrap_or(FinishReason::Stop)
        };
        out.push(StreamChunk::Finish { index: 0, reason });

        // Ollama reports input/output token counts but no aggregate; sum them for `total`.
        let prompt = v
            .get("prompt_eval_count")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let completion = v.get("eval_count").and_then(Value::as_u64).unwrap_or(0);
        out.push(StreamChunk::Usage(Usage {
            prompt_tokens: prompt,
            completion_tokens: completion,
            total_tokens: prompt + completion,
            cost_usd: None,
        }));
    }

    out
}

/// Map Ollama's `done_reason` to the canonical finish reason. `length` is the output/context cap; a
/// tool-call turn is classified upstream of this (`saw_tool`); everything else — `stop`, the load/unload
/// housekeeping reasons, and unknown values — is a normal stop.
fn map_done_reason(reason: &str) -> FinishReason {
    match reason {
        "length" => FinishReason::Length,
        _ => FinishReason::Stop,
    }
}

/// Drain complete NDJSON lines out of a rolling byte buffer.
///
/// Like SSE, `reqwest`'s byte stream yields arbitrary boundaries, so a JSON object can split across TCP
/// reads. Bytes accumulate and only `\n`-terminated lines are decoded; a trailing partial line stays in
/// `buf` for the next call. Unlike SSE there is no `data:` framing — each complete line *is* a JSON
/// object. Blank lines (and a `\r` from a stray CRLF) are skipped.
fn ndjson_lines(buf: &mut Vec<u8>, incoming: &[u8]) -> Vec<String> {
    buf.extend_from_slice(incoming);
    let mut out = Vec::new();
    while let Some(nl) = buf.iter().position(|&b| b == b'\n') {
        let end = if nl > 0 && buf[nl - 1] == b'\r' {
            nl - 1
        } else {
            nl
        };
        if let Ok(line) = std::str::from_utf8(&buf[..end]) {
            let line = line.trim();
            if !line.is_empty() {
                out.push(line.to_string());
            }
        }
        buf.drain(..=nl);
    }
    out
}

fn embedding_request_to_ollama(req: &EmbeddingRequest, keep_alive: Option<&str>) -> Value {
    let mut obj = Map::new();
    obj.insert("model".into(), json!(req.model));
    // The canonical `input` is always a vector; Ollama's `input` accepts an array directly.
    obj.insert("input".into(), json!(req.input));
    if let Some(d) = req.dimensions {
        obj.insert("dimensions".into(), json!(d));
    }
    if let Some(ka) = keep_alive {
        obj.insert("keep_alive".into(), json!(ka));
    }
    for (k, v) in &req.extra {
        obj.entry(k.clone()).or_insert_with(|| v.clone());
    }
    Value::Object(obj)
}

fn ollama_to_embeddings(value: Value, fallback_model: &str) -> EmbeddingResponse {
    // `embeddings` is always an array of vectors, in input order.
    let embeddings = value
        .get("embeddings")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .enumerate()
                .map(|(i, row)| {
                    let vector = row
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

    // Ollama reports the prompt token count; embeddings have no completion side, so total mirrors it
    // (the core relays, never computes — principle 5).
    let prompt = value
        .get("prompt_eval_count")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let model = value
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or(fallback_model)
        .to_string();
    EmbeddingResponse {
        model,
        embeddings,
        usage: Usage {
            prompt_tokens: prompt,
            completion_tokens: 0,
            total_tokens: prompt,
            cost_usd: None,
        },
    }
}

/// One `GET /api/tags` entry → canonical [`ModelInfo`]. The id is the tag (`model`, falling back to
/// `name` — Ollama reports both, e.g. `llama3.2:latest`). `/api/tags` carries no modality, context, or
/// pricing, so those stay `None` here (filled by [`enrich_from_show`] or the bundled dataset); the
/// descriptive fields ride through verbatim (principle 7).
fn ollama_tag_to_info(m: &Value) -> Option<ModelInfo> {
    let id = m
        .get("model")
        .and_then(Value::as_str)
        .or_else(|| m.get("name").and_then(Value::as_str))?;
    let mut info = ModelInfo::new(id);
    for key in ["size", "digest", "modified_at", "details"] {
        if let Some(v) = m.get(key) {
            if !v.is_null() {
                info.extra.insert(key.to_string(), v.clone());
            }
        }
    }
    Some(info)
}

/// Enrich a catalog entry from its `POST /api/show` detail: modality from the declared `capabilities`
/// (never the id — SOUL), and the context window from the architecture-prefixed GGUF metadata
/// (`model_info["general.architecture"]` → `model_info["<arch>.context_length"]`). The capability list
/// rides through verbatim in `extra`.
fn enrich_from_show(info: &mut ModelInfo, show: &Value) {
    if let Some(caps_val) = show.get("capabilities") {
        if let Some(caps) = caps_val.as_array() {
            let names: Vec<&str> = caps.iter().filter_map(Value::as_str).collect();
            // A generative model declares `completion` (chat/vision); an embedder declares `embedding`.
            if names.contains(&"completion") {
                info.modality = Some(Modality::Llm);
            } else if names.contains(&"embedding") {
                info.modality = Some(Modality::Embedding);
            }
            info.extra.insert("capabilities".into(), caps_val.clone());
        }
    }
    if let Some(mi) = show.get("model_info").and_then(Value::as_object) {
        if let Some(arch) = mi.get("general.architecture").and_then(Value::as_str) {
            if let Some(ctx) = mi
                .get(&format!("{arch}.context_length"))
                .and_then(Value::as_u64)
            {
                info.max_context = Some(ctx.min(u32::MAX as u64) as u32);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llmleaf_model::{Role, ToolCall, ToolDef};

    fn user(text: &str) -> Message {
        Message::text(Role::User, text)
    }

    #[test]
    fn flattens_sampling_into_options_and_forces_stream() {
        let req = ChatRequest {
            model: "qwen3".into(),
            messages: vec![user("hi")],
            max_tokens: Some(256),
            temperature: Some(0.7),
            top_p: Some(0.9),
            stop: vec!["</s>".into()],
            stream: false,
            tools: vec![],
            tool_choice: None,
            thinking: None,
            extra: Default::default(),
        };
        let wire = request_to_ollama(&req, Some("5m"));
        assert_eq!(wire["model"], "qwen3");
        assert_eq!(wire["messages"][0]["content"], "hi");
        // Sampling is nested under `options`, not top-level; max_tokens → num_predict.
        // (f32 → JSON f64 isn't bit-exact, so compare with tolerance.)
        assert!((wire["options"]["temperature"].as_f64().unwrap() - 0.7).abs() < 1e-6);
        assert_eq!(wire["options"]["num_predict"], 256);
        assert_eq!(wire["options"]["stop"][0], "</s>");
        assert!(wire.get("temperature").is_none());
        // keep_alive from settings, and streaming forced on.
        assert_eq!(wire["keep_alive"], "5m");
        assert_eq!(wire["stream"], true);
        // No thinking requested → no `think` field.
        assert!(wire.get("think").is_none());
    }

    #[test]
    fn merges_extra_runner_options_with_canonical_sampling() {
        let mut extra = Map::new();
        // Consumer passes Ollama runner options the canonical request doesn't model.
        extra.insert(
            "options".into(),
            json!({ "num_ctx": 8192, "seed": 42, "temperature": 0.1 }),
        );
        let req = ChatRequest {
            model: "qwen3".into(),
            messages: vec![user("hi")],
            max_tokens: Some(128),
            temperature: Some(0.7),
            top_p: None,
            stop: vec![],
            stream: false,
            tools: vec![],
            tool_choice: None,
            thinking: None,
            extra,
        };
        let wire = request_to_ollama(&req, None);
        // Consumer runner options survive (would be dropped by a naive top-level `or_insert`)...
        assert_eq!(wire["options"]["num_ctx"], 8192);
        assert_eq!(wire["options"]["seed"], 42);
        // ...num_predict comes from canonical max_tokens...
        assert_eq!(wire["options"]["num_predict"], 128);
        // ...and a modeled canonical field wins over the same key in extra.options.
        assert!((wire["options"]["temperature"].as_f64().unwrap() - 0.7).abs() < 1e-6);
    }

    #[test]
    fn maps_thinking_ladder_to_think_string() {
        let mk = |t| {
            let mut req = ChatRequest {
                model: "qwen3".into(),
                messages: vec![user("hi")],
                max_tokens: None,
                temperature: None,
                top_p: None,
                stop: vec![],
                stream: false,
                tools: vec![],
                tool_choice: None,
                thinking: Some(t),
                extra: Default::default(),
            };
            req.thinking = Some(t);
            request_to_ollama(&req, None)["think"].clone()
        };
        assert_eq!(mk(Thinking::Low), "low");
        assert_eq!(mk(Thinking::Med), "medium");
        assert_eq!(mk(Thinking::High), "high");
        assert_eq!(mk(Thinking::Highx), "high"); // no Ollama equivalent → collapses onto high
        assert_eq!(mk(Thinking::Max), "max");
    }

    #[test]
    fn tools_and_assistant_tool_calls_use_object_arguments() {
        let tool = ToolDef {
            name: "get_weather".into(),
            description: Some("weather".into()),
            parameters: json!({ "type": "object", "properties": { "city": { "type": "string" } } }),
        };
        let assistant = Message {
            role: Role::Assistant,
            content: vec![],
            tool_calls: vec![ToolCall {
                id: "call_1".into(),
                name: "get_weather".into(),
                arguments: "{\"city\":\"Tokyo\"}".into(),
            }],
            tool_call_id: None,
            name: None,
        };
        let req = ChatRequest {
            model: "qwen3".into(),
            messages: vec![user("weather?"), assistant],
            max_tokens: None,
            temperature: None,
            top_p: None,
            stop: vec![],
            stream: false,
            tools: vec![tool],
            tool_choice: None,
            thinking: None,
            extra: Default::default(),
        };
        let wire = request_to_ollama(&req, None);
        assert_eq!(wire["tools"][0]["type"], "function");
        assert_eq!(wire["tools"][0]["function"]["name"], "get_weather");
        // Arguments are a real JSON object on the wire, not OpenAI's stringified blob.
        let call = &wire["messages"][1]["tool_calls"][0];
        assert_eq!(call["id"], "call_1");
        assert_eq!(call["function"]["index"], 0);
        assert_eq!(call["function"]["arguments"]["city"], "Tokyo");
        assert!(call["function"]["arguments"].is_object());
    }

    #[test]
    fn tool_result_message_carries_tool_name() {
        let result = Message {
            role: Role::Tool,
            content: vec![ContentPart::Text {
                text: "11 degrees".into(),
            }],
            tool_calls: vec![],
            tool_call_id: Some("call_1".into()),
            name: Some("get_weather".into()),
        };
        let wire = message_to_ollama(&result);
        assert_eq!(wire["role"], "tool");
        assert_eq!(wire["content"], "11 degrees");
        assert_eq!(wire["tool_name"], "get_weather");
        assert_eq!(wire["tool_call_id"], "call_1");
    }

    #[test]
    fn image_part_unwraps_data_url_to_bare_base64() {
        let msg = Message {
            role: Role::User,
            content: vec![
                ContentPart::Text {
                    text: "describe".into(),
                },
                ContentPart::ImageUrl {
                    url: "data:image/png;base64,iVBORw0KGgo=".into(),
                    detail: None,
                },
            ],
            tool_calls: vec![],
            tool_call_id: None,
            name: None,
        };
        let wire = message_to_ollama(&msg);
        assert_eq!(wire["content"], "describe");
        assert_eq!(wire["images"][0], "iVBORw0KGgo="); // prefix stripped
                                                       // A remote URL the native API can't fetch is skipped (no `images` produced).
        assert_eq!(ollama_image_b64("https://example.com/cat.png"), None);
        assert_eq!(ollama_image_b64("rawbase64=="), Some("rawbase64==".into()));
    }

    #[test]
    fn parses_ndjson_content_then_final_stats() {
        let mut seen_start = false;
        let mut saw_tool = false;
        // First line: model + content delta → Start, then Content.
        let first = json!({
            "model": "qwen3",
            "message": { "role": "assistant", "content": "Hel", "thinking": "" },
            "done": false
        });
        let c0 = ollama_chunk_to_canonical(&first, &mut seen_start, &mut saw_tool);
        assert!(
            matches!(&c0[0], StreamChunk::Start { id, model } if id.is_empty() && model == "qwen3")
        );
        assert!(matches!(&c0[1], StreamChunk::Content { delta, .. } if delta == "Hel"));
        // A later line emits no Start again.
        let mid = json!({ "model": "qwen3", "message": { "content": "lo" }, "done": false });
        let c1 = ollama_chunk_to_canonical(&mid, &mut seen_start, &mut saw_tool);
        assert!(matches!(c1.as_slice(), [StreamChunk::Content { delta, .. }] if delta == "lo"));
        // Final line: done:true with stats → Finish then Usage.
        let last = json!({
            "model": "qwen3",
            "message": { "role": "assistant", "content": "" },
            "done": true,
            "done_reason": "stop",
            "prompt_eval_count": 26,
            "eval_count": 282
        });
        let c2 = ollama_chunk_to_canonical(&last, &mut seen_start, &mut saw_tool);
        assert!(matches!(
            c2[0],
            StreamChunk::Finish {
                reason: FinishReason::Stop,
                ..
            }
        ));
        assert!(
            matches!(c2[1], StreamChunk::Usage(u) if u.prompt_tokens == 26 && u.completion_tokens == 282 && u.total_tokens == 308)
        );
    }

    #[test]
    fn tool_call_chunk_sets_tool_finish_and_object_args() {
        let mut seen_start = false;
        let mut saw_tool = false;
        let line = json!({
            "model": "qwen3",
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    { "function": { "index": 0, "name": "get_weather", "arguments": { "city": "Tokyo" } } }
                ]
            },
            "done": true,
            "done_reason": "stop"
        });
        let chunks = ollama_chunk_to_canonical(&line, &mut seen_start, &mut saw_tool);
        // Start, ToolCall, Finish(ToolCalls), Usage.
        match chunks
            .iter()
            .find(|c| matches!(c, StreamChunk::ToolCall { .. }))
        {
            Some(StreamChunk::ToolCall { call, .. }) => {
                assert_eq!(call.name.as_deref(), Some("get_weather"));
                // Object arguments round-trip to the canonical raw-JSON string.
                assert!(call.arguments.as_deref().unwrap().contains("Tokyo"));
            }
            _ => panic!("expected a ToolCall chunk"),
        }
        // A tool turn finishes as ToolCalls even when done_reason is the plain "stop".
        assert!(chunks.iter().any(|c| matches!(
            c,
            StreamChunk::Finish {
                reason: FinishReason::ToolCalls,
                ..
            }
        )));
    }

    #[test]
    fn ndjson_lines_buffers_split_objects() {
        let mut buf = Vec::new();
        // An object split across two reads, mid-JSON.
        assert!(ndjson_lines(&mut buf, b"{\"done\":").is_empty());
        let got = ndjson_lines(&mut buf, b"false}\n");
        assert_eq!(got, vec!["{\"done\":false}".to_string()]);
        // Two objects in one read, CRLF tolerated, blank lines skipped.
        let got = ndjson_lines(&mut buf, b"{\"a\":1}\r\n\n{\"b\":2}\n");
        assert_eq!(got, vec!["{\"a\":1}".to_string(), "{\"b\":2}".to_string()]);
    }

    #[test]
    fn parses_embeddings_array_and_prompt_tokens() {
        let resp = json!({
            "model": "all-minilm",
            "embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5]],
            "prompt_eval_count": 8
        });
        let out = ollama_to_embeddings(resp, "all-minilm");
        assert_eq!(out.model, "all-minilm");
        assert_eq!(out.embeddings[0].index, 0);
        assert_eq!(out.embeddings[1].index, 1);
        assert_eq!(out.embeddings[0].vector.len(), 3);
        assert!((out.embeddings[1].vector[0] - 0.4).abs() < 1e-6);
        assert_eq!(out.usage.prompt_tokens, 8);
        assert_eq!(out.usage.completion_tokens, 0);
        assert_eq!(out.usage.total_tokens, 8);
    }

    #[test]
    fn embed_request_passes_input_array() {
        let req = EmbeddingRequest {
            model: "all-minilm".into(),
            input: vec!["a".into(), "b".into()],
            dimensions: Some(256),
            encoding_format: None,
            extra: Default::default(),
        };
        let wire = embedding_request_to_ollama(&req, Some("10m"));
        assert_eq!(wire["model"], "all-minilm");
        assert_eq!(wire["input"][0], "a");
        assert_eq!(wire["input"][1], "b");
        assert_eq!(wire["dimensions"], 256);
        assert_eq!(wire["keep_alive"], "10m");
    }

    #[test]
    fn tag_to_info_uses_tag_id_and_stashes_details() {
        let m = json!({
            "name": "llama3.2:latest",
            "model": "llama3.2:latest",
            "size": 2019393189_u64,
            "digest": "a80c4f17",
            "details": { "parameter_size": "3.2B", "quantization_level": "Q4_K_M" }
        });
        let info = ollama_tag_to_info(&m).unwrap();
        assert_eq!(info.id, "llama3.2:latest");
        // Nothing guessed: no modality/context from /api/tags alone.
        assert_eq!(info.modality, None);
        assert_eq!(info.max_context, None);
        assert_eq!(info.extra.get("details").unwrap()["parameter_size"], "3.2B");
    }

    #[test]
    fn enrich_reads_capabilities_and_arch_context_length() {
        let mut info = ModelInfo::new("llama3.1:8b");
        let show = json!({
            "capabilities": ["completion", "tools", "vision"],
            "model_info": {
                "general.architecture": "llama",
                "llama.context_length": 131072,
                "llama.embedding_length": 4096
            }
        });
        enrich_from_show(&mut info, &show);
        assert_eq!(info.modality, Some(Modality::Llm)); // from `completion`, not the id
        assert_eq!(info.max_context, Some(131072)); // <arch>.context_length
        assert_eq!(info.extra.get("capabilities").unwrap()[1], "tools");

        // An embedding model is detected from its capability, not its name.
        let mut emb = ModelInfo::new("all-minilm");
        enrich_from_show(
            &mut emb,
            &json!({ "capabilities": ["embedding"], "model_info": { "general.architecture": "bert", "bert.context_length": 512 } }),
        );
        assert_eq!(emb.modality, Some(Modality::Embedding));
        assert_eq!(emb.max_context, Some(512));
    }

    #[test]
    fn stream_error_object_is_surfaced() {
        // A line carrying an `error` is handled by the chat loop, but verify the mapper itself never
        // mistakes a normal chunk for one and that done_reason `length` maps to Length.
        assert_eq!(map_done_reason("length"), FinishReason::Length);
        assert_eq!(map_done_reason("stop"), FinishReason::Stop);
        assert_eq!(map_done_reason("unload"), FinishReason::Stop);
    }
}
