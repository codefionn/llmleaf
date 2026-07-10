//! The OpenAI Chat Completions *wire* mapping, shared by every OpenAI-compatible provider.
//!
//! This is the canonical ⇄ OpenAI-JSON translation, factored out so each compatible brand
//! (OpenRouter, Groq, DeepSeek, xAI, Mistral, Together, Fireworks, Perplexity, Azure …)
//! reuses one correct implementation and contributes only its endpoint/auth quirks (decision filter:
//! "A provider quirk goes in that provider's extension, never in the core").

use base64::Engine as _;
use futures::StreamExt;
use llmleaf_model::{
    ChatRequest, ContentPart, Embedding, EmbeddingRequest, EmbeddingResponse, FinishReason,
    Message, Modality, ModelError, ModelInfo, RerankRequest, RerankResponse, RerankResult,
    ResponseStream, Role, SpeechRequest, StreamChunk, Thinking, ToolCallDelta, ToolChoice,
    TranscriptionRequest, TranscriptionResponse, Usage, VoiceInfo,
};
use serde_json::{json, Map, Value};

/// Canonical request → OpenAI wire JSON. Dialect-specific `extra` fields ride back out verbatim
/// (principle 7).
///
/// `max_tokens_field` selects the output-cap field name per brand: OpenAI/Azure/xAI use
/// `max_completion_tokens` (o-series/reasoning models reject the deprecated `max_tokens`), while
/// most compatible vendors use `max_tokens`.
///
/// `stream` selects the upstream transport: `false` collects the whole response in one shot (the
/// shape used when no incremental output is needed); `true` asks upstream to stream SSE and adds
/// `stream_options.include_usage` so a terminal usage frame still arrives. The `false` branch is
/// byte-identical to the original collect-only mapping.
pub fn request_to_openai(req: &ChatRequest, max_tokens_field: &str, stream: bool) -> Value {
    let mut obj = Map::new();
    obj.insert("model".into(), json!(req.model));
    obj.insert(
        "messages".into(),
        Value::Array(req.messages.iter().map(message_to_openai).collect()),
    );
    if let Some(v) = req.max_tokens {
        obj.insert(max_tokens_field.to_string(), json!(v));
    }
    if let Some(v) = req.temperature {
        obj.insert("temperature".into(), json!(v));
    }
    if let Some(v) = req.top_p {
        obj.insert("top_p".into(), json!(v));
    }
    if !req.stop.is_empty() {
        obj.insert("stop".into(), json!(req.stop));
    }
    if !req.tools.is_empty() {
        let tools: Vec<Value> = req
            .tools
            .iter()
            .map(|t| {
                let mut function = Map::new();
                function.insert("name".into(), json!(t.name));
                if let Some(desc) = &t.description {
                    function.insert("description".into(), json!(desc));
                }
                function.insert("parameters".into(), t.parameters.clone());
                json!({ "type": "function", "function": Value::Object(function) })
            })
            .collect();
        obj.insert("tools".into(), json!(tools));
    }
    if let Some(tc) = &req.tool_choice {
        obj.insert("tool_choice".into(), tool_choice_to_openai(tc));
    }
    if let Some(t) = req.thinking {
        obj.insert("reasoning_effort".into(), json!(thinking_effort(t)));
    }
    // Pass through anything the consumer sent that we don't model (principle 7).
    for (k, v) in &req.extra {
        obj.entry(k.clone()).or_insert_with(|| v.clone());
    }
    obj.insert("stream".into(), json!(stream));
    if stream {
        // Ask upstream to report usage in a terminal SSE frame so the canonical stream still carries
        // token accounting (principle 5: the core relays provider usage, it never counts).
        obj.insert("stream_options".into(), json!({ "include_usage": true }));
    }
    Value::Object(obj)
}

/// Map the canonical thinking ladder onto OpenAI's `reasoning_effort`. OpenAI exposes only
/// low/medium/high, so the two highest rungs collapse onto `high` (lossy by design — see [`Thinking`]).
/// `pub(crate)` so the Responses wire ([`crate::openai_responses_wire`]) reuses the same lossy mapping
/// for its `reasoning.effort` field rather than duplicating the ladder.
pub(crate) fn thinking_effort(t: Thinking) -> &'static str {
    match t {
        Thinking::Low => "low",
        Thinking::Med => "medium",
        Thinking::High | Thinking::Highx | Thinking::Max => "high",
    }
}

fn tool_choice_to_openai(tc: &ToolChoice) -> Value {
    match tc {
        ToolChoice::Auto => json!("auto"),
        ToolChoice::None => json!("none"),
        ToolChoice::Required => json!("required"),
        ToolChoice::Named(name) => json!({ "type": "function", "function": { "name": name } }),
    }
}

fn message_to_openai(msg: &Message) -> Value {
    let role = match msg.role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    };
    let mut obj = Map::new();
    obj.insert("role".into(), json!(role));

    // A single text part collapses to a string (the common, cheapest case); otherwise emit parts.
    if msg.content.len() == 1 {
        if let ContentPart::Text { text } = &msg.content[0] {
            obj.insert("content".into(), json!(text));
        } else {
            obj.insert("content".into(), content_parts(msg));
        }
    } else if msg.content.is_empty() {
        obj.insert("content".into(), Value::Null);
    } else {
        obj.insert("content".into(), content_parts(msg));
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
    if let Some(id) = &msg.tool_call_id {
        obj.insert("tool_call_id".into(), json!(id));
    }
    if let Some(name) = &msg.name {
        obj.insert("name".into(), json!(name));
    }
    Value::Object(obj)
}

fn content_parts(msg: &Message) -> Value {
    let parts: Vec<Value> = msg
        .content
        .iter()
        .filter_map(|p| match p {
            ContentPart::Text { text } => Some(json!({ "type": "text", "text": text })),
            ContentPart::ImageUrl { url, detail } => {
                let mut image_url = Map::new();
                image_url.insert("url".into(), json!(url));
                if let Some(detail) = detail {
                    image_url.insert("detail".into(), json!(detail));
                }
                Some(json!({ "type": "image_url", "image_url": Value::Object(image_url) }))
            }
            // OpenAI chat completions has no field for replayed reasoning (it is server-side on o-series
            // models); reasoning does not port across providers, so drop these blocks at this edge.
            ContentPart::Thinking { .. } | ContentPart::RedactedThinking { .. } => None,
        })
        .collect();
    Value::Array(parts)
}

/// OpenAI response object → canonical chunks (Start, per-choice content/tool calls, Usage, Finish).
pub fn openai_to_chunks(value: Value, fallback_model: &str) -> Vec<StreamChunk> {
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

    if let Some(choices) = value.get("choices").and_then(Value::as_array) {
        for choice in choices {
            let index = choice.get("index").and_then(Value::as_u64).unwrap_or(0) as u32;
            let message = choice.get("message");

            // Open reasoning text, ordered ahead of the visible answer it justifies. OpenRouter
            // emits `reasoning`, DeepSeek `reasoning_content`; stock OpenAI keeps o-series
            // reasoning server-side and emits neither.
            if let Some(reasoning) = message
                .and_then(|m| m.get("reasoning").or_else(|| m.get("reasoning_content")))
                .and_then(Value::as_str)
            {
                if !reasoning.is_empty() {
                    chunks.push(StreamChunk::Thinking {
                        index,
                        delta: reasoning.to_string(),
                    });
                }
            }

            if let Some(content) = message
                .and_then(|m| m.get("content"))
                .and_then(Value::as_str)
            {
                if !content.is_empty() {
                    chunks.push(StreamChunk::Content {
                        index,
                        delta: content.to_string(),
                    });
                }
            }

            if let Some(tool_calls) = message
                .and_then(|m| m.get("tool_calls"))
                .and_then(Value::as_array)
            {
                for (i, tc) in tool_calls.iter().enumerate() {
                    let func = tc.get("function");
                    chunks.push(StreamChunk::ToolCall {
                        index,
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

            let reason = choice
                .get("finish_reason")
                .and_then(Value::as_str)
                .map(map_finish)
                .unwrap_or(FinishReason::Stop);
            chunks.push(StreamChunk::Finish { index, reason });
        }
    }

    if let Some(usage) = value.get("usage") {
        chunks.push(StreamChunk::Usage(Usage {
            prompt_tokens: usage
                .get("prompt_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0),
            completion_tokens: usage
                .get("completion_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0),
            total_tokens: usage
                .get("total_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0),
            cost_usd: None,
            // OpenAI/OpenRouter report cache hits under `prompt_tokens_details.cached_tokens`
            // (automatic prompt caching); DeepSeek uses `prompt_cache_hit_tokens`. Either maps to the
            // canonical cache-read counter; there is no client-driven cache write, so creation is 0.
            cache_read_tokens: openai_cache_read_tokens(usage),
            cache_creation_tokens: 0,
        }));
    }

    chunks
}

/// Cache-read (hit) input tokens from an OpenAI-style `usage` object. OpenAI and OpenRouter (a
/// first-class OpenAI-compatible provider) report this as `prompt_tokens_details.cached_tokens` under
/// automatic prompt caching; DeepSeek reports `prompt_cache_hit_tokens`. Either maps to the canonical
/// cache-read counter; absent (most compatible vendors) → 0.
fn openai_cache_read_tokens(usage: &Value) -> u64 {
    usage
        .get("prompt_tokens_details")
        .and_then(|d| d.get("cached_tokens"))
        .and_then(Value::as_u64)
        .or_else(|| usage.get("prompt_cache_hit_tokens").and_then(Value::as_u64))
        .unwrap_or(0)
}

fn map_finish(reason: &str) -> FinishReason {
    match reason {
        "length" => FinishReason::Length,
        "tool_calls" | "function_call" => FinishReason::ToolCalls,
        "content_filter" => FinishReason::ContentFilter,
        _ => FinishReason::Stop,
    }
}

// ---------------------------------------------------------------------------------------------
// Incremental streaming (SSE): one `chat.completion.chunk` → canonical chunks
// ---------------------------------------------------------------------------------------------

/// One OpenAI `chat.completion.chunk` (the JSON after `data: `, never `[DONE]`) → canonical chunks.
///
/// The streaming mirror of [`openai_to_chunks`]: it reads `choices[].delta` (not `.message`) and the
/// final `include_usage` frame's top-level `usage`. `seen_start` guards the single opening
/// [`StreamChunk::Start`]: the first frame carries the response id/model and emits it, later frames do
/// not. Streaming tool-call deltas carry their own `index`, so it is read from the frame rather than
/// synthesized by enumeration.
pub fn openai_chunk_to_canonical(
    value: &Value,
    seen_start: &mut bool,
    fallback_model: &str,
) -> Vec<StreamChunk> {
    let mut out = Vec::new();

    if !*seen_start {
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
        out.push(StreamChunk::Start { id, model });
        *seen_start = true;
    }

    if let Some(choices) = value.get("choices").and_then(Value::as_array) {
        for choice in choices {
            let index = choice.get("index").and_then(Value::as_u64).unwrap_or(0) as u32;
            let delta = choice.get("delta");

            // Open reasoning deltas — OpenRouter's `delta.reasoning` / DeepSeek's
            // `delta.reasoning_content` — become canonical Thinking chunks, same as the collected
            // mirror above.
            if let Some(reasoning) = delta
                .and_then(|d| d.get("reasoning").or_else(|| d.get("reasoning_content")))
                .and_then(Value::as_str)
            {
                if !reasoning.is_empty() {
                    out.push(StreamChunk::Thinking {
                        index,
                        delta: reasoning.to_string(),
                    });
                }
            }

            if let Some(content) = delta.and_then(|d| d.get("content")).and_then(Value::as_str) {
                if !content.is_empty() {
                    out.push(StreamChunk::Content {
                        index,
                        delta: content.to_string(),
                    });
                }
            }

            if let Some(tool_calls) = delta
                .and_then(|d| d.get("tool_calls"))
                .and_then(Value::as_array)
            {
                for tc in tool_calls {
                    let func = tc.get("function");
                    out.push(StreamChunk::ToolCall {
                        index,
                        call: ToolCallDelta {
                            index: tc.get("index").and_then(Value::as_u64).unwrap_or(0) as u32,
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

            if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str) {
                out.push(StreamChunk::Finish {
                    index,
                    reason: map_finish(reason),
                });
            }
        }
    }

    if let Some(usage) = value.get("usage").filter(|u| !u.is_null()) {
        out.push(StreamChunk::Usage(Usage {
            prompt_tokens: usage
                .get("prompt_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0),
            completion_tokens: usage
                .get("completion_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0),
            total_tokens: usage
                .get("total_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0),
            cost_usd: None,
            // OpenAI/OpenRouter report cache hits under `prompt_tokens_details.cached_tokens`
            // (automatic prompt caching); DeepSeek uses `prompt_cache_hit_tokens`. Either maps to the
            // canonical cache-read counter; there is no client-driven cache write, so creation is 0.
            cache_read_tokens: openai_cache_read_tokens(usage),
            cache_creation_tokens: 0,
        }));
    }

    out
}

/// Drain complete SSE `data:` payloads out of a rolling byte buffer.
///
/// `reqwest`'s byte stream yields arbitrary boundaries — a single SSE frame (or even one multi-byte
/// UTF-8 character) can split across TCP reads — so bytes are accumulated and only *complete* lines
/// (terminated by `\n`) are decoded. Returns the payload of each `data:` line (the text after
/// `data: `), including the `[DONE]` sentinel for the caller to detect; comment lines (`:`...), blank
/// lines, and other SSE fields (`event:`/`id:`/`retry:`) are skipped. A trailing partial line stays in
/// `buf` for the next call.
pub fn sse_payloads(buf: &mut Vec<u8>, incoming: &[u8]) -> Vec<String> {
    buf.extend_from_slice(incoming);
    let mut out = Vec::new();
    while let Some(nl) = buf.iter().position(|&b| b == b'\n') {
        // The line is buf[..nl]; tolerate a CRLF terminator.
        let end = if nl > 0 && buf[nl - 1] == b'\r' {
            nl - 1
        } else {
            nl
        };
        if let Ok(line) = std::str::from_utf8(&buf[..end]) {
            let line = line.trim();
            if !line.is_empty() && !line.starts_with(':') {
                if let Some(rest) = line.strip_prefix("data:") {
                    out.push(rest.strip_prefix(' ').unwrap_or(rest).to_string());
                }
            }
        }
        buf.drain(..=nl);
    }
    out
}

/// Stream an OpenAI-wire SSE chat response into canonical chunks.
///
/// The shared streaming core for every OpenAI-wire provider that asks upstream for `stream:true` — the
/// compat family ([`crate::compat`]) and LM Studio's `/api/v0` ([`crate::lmstudio`]). It owns the rolling
/// SSE buffer, rejoins each `data:` frame via [`sse_payloads`] (a frame split across TCP reads is
/// stitched back together), maps every `chat.completion.chunk` through [`openai_chunk_to_canonical`], and
/// stops at the `[DONE]` sentinel — tokens flow as they arrive (principle 4), never collected then
/// replayed. `model` is the fallback id stamped on the opening [`StreamChunk::Start`] when a frame omits
/// one. A transport error becomes [`ModelError::Unavailable`] (fallback-eligible); an undecodable frame
/// becomes [`ModelError::Mapping`].
pub fn openai_sse_to_stream(body: crate::transport::BytesStream, model: String) -> ResponseStream {
    openai_sse_to_stream_checked(body, model, |_| None)
}

/// [`openai_sse_to_stream`] with a per-frame envelope check, for a brand that can smuggle an error
/// into a 2xx SSE frame (MiniMax's `base_resp` — see [`crate::compat`]). `check` runs on every decoded
/// frame *before* canonical mapping; a `Some(err)` ends the stream with that error instead of mapping
/// the frame. The brand-specific classification lives at the provider edge (the caller passes it in);
/// this shared core only provides the hook, and the plain [`openai_sse_to_stream`] passes a no-op.
pub fn openai_sse_to_stream_checked(
    body: crate::transport::BytesStream,
    model: String,
    check: fn(&Value) -> Option<ModelError>,
) -> ResponseStream {
    Box::pin(async_stream::stream! {
        let mut bytes = body;
        let mut buf: Vec<u8> = Vec::with_capacity(1024);
        let mut seen_start = false;
        let mut done = false;
        while let Some(item) = bytes.next().await {
            let chunk = match item {
                Ok(b) => b,
                Err(e) => {
                    // The transport already mapped this to the canonical taxonomy.
                    yield Err(e);
                    return;
                }
            };
            for payload in sse_payloads(&mut buf, &chunk) {
                if payload == "[DONE]" {
                    done = true;
                    break;
                }
                match serde_json::from_str::<Value>(&payload) {
                    Ok(v) => {
                        if let Some(e) = check(&v) {
                            yield Err(e);
                            return;
                        }
                        for c in openai_chunk_to_canonical(&v, &mut seen_start, &model) {
                            yield Ok(c);
                        }
                    }
                    Err(e) => {
                        yield Err(ModelError::Mapping(e.to_string()));
                        return;
                    }
                }
            }
            if done {
                break;
            }
        }
    })
}

// ---------------------------------------------------------------------------------------------
// Embeddings (`/embeddings`)
// ---------------------------------------------------------------------------------------------

/// Canonical embeddings request → OpenAI wire JSON. We always ask upstream for `float` vectors so the
/// canonical representation stays `Vec<f32>`; the *consumer* surface re-encodes to base64 if asked
/// (a documented edge mapping, not a silent change to what the provider returns).
pub fn embedding_request_to_openai(req: &EmbeddingRequest) -> Value {
    let mut obj = Map::new();
    obj.insert("model".into(), json!(req.model));
    obj.insert("input".into(), json!(req.input));
    obj.insert("encoding_format".into(), json!("float"));
    if let Some(d) = req.dimensions {
        obj.insert("dimensions".into(), json!(d));
    }
    for (k, v) in &req.extra {
        obj.entry(k.clone()).or_insert_with(|| v.clone());
    }
    Value::Object(obj)
}

/// OpenAI embeddings response → canonical [`EmbeddingResponse`]. Vectors are `float` (we requested so);
/// usage carries `prompt_tokens`/`total_tokens` (there is no completion side for embeddings).
pub fn openai_to_embeddings(value: Value, fallback_model: &str) -> EmbeddingResponse {
    let model = value
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or(fallback_model)
        .to_string();

    let mut embeddings = Vec::new();
    if let Some(data) = value.get("data").and_then(Value::as_array) {
        for (i, item) in data.iter().enumerate() {
            let index = item
                .get("index")
                .and_then(Value::as_u64)
                .map(|n| n as u32)
                .unwrap_or(i as u32);
            let vector = item
                .get("embedding")
                .and_then(Value::as_array)
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_f64().map(|f| f as f32))
                        .collect()
                })
                .unwrap_or_default();
            embeddings.push(Embedding { index, vector });
        }
    }

    let usage = value
        .get("usage")
        .map(|u| Usage {
            prompt_tokens: u.get("prompt_tokens").and_then(Value::as_u64).unwrap_or(0),
            completion_tokens: 0,
            total_tokens: u.get("total_tokens").and_then(Value::as_u64).unwrap_or(0),
            cost_usd: None,
            cache_read_tokens: 0,
            cache_creation_tokens: 0,
        })
        .unwrap_or_default();

    EmbeddingResponse {
        model,
        embeddings,
        usage,
    }
}

// ---------------------------------------------------------------------------------------------
// Rerank (`/rerank`)
// ---------------------------------------------------------------------------------------------

/// Canonical rerank request → the Jina/Cohere-compatible `/v1/rerank` JSON that Together, OpenRouter,
/// Jina, Voyage, SiliconFlow and self-hosted vLLM/Infinity/TEI all accept: `{ model, query, documents,
/// top_n?, return_documents? }`. Each document rides out verbatim — a plain string or the structured
/// object the consumer sent (multimodal `{ text?, image? }`) — and dialect-specific `extra` fields
/// (`max_tokens_per_doc`, `truncation`, …) merge back in (principle 7).
pub fn rerank_request_to_openai(req: &RerankRequest) -> Value {
    let mut obj = Map::new();
    obj.insert("model".into(), json!(req.model));
    obj.insert("query".into(), json!(req.query));
    obj.insert("documents".into(), json!(req.documents));
    if let Some(n) = req.top_n {
        obj.insert("top_n".into(), json!(n));
    }
    if let Some(rd) = req.return_documents {
        obj.insert("return_documents".into(), json!(rd));
    }
    for (k, v) in &req.extra {
        obj.entry(k.clone()).or_insert_with(|| v.clone());
    }
    Value::Object(obj)
}

/// Jina/Cohere-compatible `/v1/rerank` response → canonical [`RerankResponse`]. Reads `results[]`
/// (`index`, `relevance_score`, optional echoed `document`) and relays the billed usage: token-billed
/// upstreams (Jina/Voyage/OpenRouter) report `usage.total_tokens`, a Cohere-compatible one (Together)
/// reports `meta.billed_units.search_units` — carried into `total_tokens` as its billed count (the core
/// relays, never computes — principle 5).
pub fn openai_to_rerank(value: Value, fallback_model: &str) -> RerankResponse {
    let model = value
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or(fallback_model)
        .to_string();

    let mut results = Vec::new();
    if let Some(items) = value.get("results").and_then(Value::as_array) {
        for (i, item) in items.iter().enumerate() {
            let index = item
                .get("index")
                .and_then(Value::as_u64)
                .map(|n| n as u32)
                .unwrap_or(i as u32);
            let relevance_score = item
                .get("relevance_score")
                .and_then(Value::as_f64)
                .map(|f| f as f32)
                .unwrap_or(0.0);
            // Echoed back only when the consumer asked (`return_documents`); carried verbatim so a
            // structured/multimodal document round-trips unchanged.
            let document = item
                .get("document")
                .filter(|v| !v.is_null())
                .cloned();
            results.push(RerankResult {
                index,
                relevance_score,
                document,
            });
        }
    }

    RerankResponse {
        model,
        results,
        usage: rerank_usage(&value),
    }
}

/// Relay a rerank response's billed usage. Prefers a token-billed `usage.total_tokens`; falls back to a
/// Cohere-compatible `meta.billed_units.{search_units,total_tokens,input_tokens}`. Zero when neither is
/// present (never fabricated).
fn rerank_usage(value: &Value) -> Usage {
    let total = value
        .get("usage")
        .and_then(|u| u.get("total_tokens"))
        .and_then(Value::as_u64)
        .or_else(|| {
            let billed = value.get("meta").and_then(|m| m.get("billed_units"))?;
            billed
                .get("search_units")
                .and_then(Value::as_u64)
                .or_else(|| billed.get("total_tokens").and_then(Value::as_u64))
                .or_else(|| billed.get("input_tokens").and_then(Value::as_u64))
        })
        .unwrap_or(0);
    Usage {
        prompt_tokens: total,
        completion_tokens: 0,
        total_tokens: total,
        cost_usd: None,
        cache_read_tokens: 0,
        cache_creation_tokens: 0,
    }
}

// ---------------------------------------------------------------------------------------------
// Speech (`/audio/speech`) and transcription (`/audio/transcriptions`)
// ---------------------------------------------------------------------------------------------

/// Canonical speech request → OpenAI wire JSON. OpenAI's own response is the raw audio file (the
/// provider streams it as bytes), but some OpenAI-wire brands wrap it in JSON — see
/// [`decode_speech_envelope`] for that path.
pub fn speech_request_to_openai(req: &SpeechRequest) -> Value {
    let mut obj = Map::new();
    obj.insert("model".into(), json!(req.model));
    obj.insert("input".into(), json!(req.input));
    obj.insert("voice".into(), json!(req.voice));
    if let Some(fmt) = &req.response_format {
        obj.insert("response_format".into(), json!(fmt));
    }
    if let Some(speed) = req.speed {
        obj.insert("speed".into(), json!(speed));
    }
    for (k, v) in &req.extra {
        obj.entry(k.clone()).or_insert_with(|| v.clone());
    }
    Value::Object(obj)
}

/// Lift the audio bytes out of a JSON-enveloped TTS response. Where OpenAI returns the audio file as
/// the raw `POST /v1/audio/speech` body, Mistral instead returns `{ "audio_data": "<base64>" }` — a
/// JSON object with the audio base64-encoded. This decodes that payload back to raw bytes. Returns
/// `None` when there is no decodable `audio_data` string (the caller surfaces that as a mapping error);
/// it is only consulted when the upstream marked the body `application/json`, so a missing field is a
/// genuine schema mismatch, not the ordinary raw-bytes path.
pub fn decode_speech_envelope(value: &Value) -> Option<Vec<u8>> {
    let b64 = value.get("audio_data")?.as_str()?;
    base64::engine::general_purpose::STANDARD.decode(b64).ok()
}

/// OpenAI's documented TTS voice catalog. OpenAI has no list-voices API — the set is published, not
/// fetchable — so it is declared statically here (SOUL: static provider data is a const in the provider,
/// not a bundled dataset). The catalog is per-model: `tts-1`/`tts-1-hd` ship the original six voices;
/// the `gpt-4o`-family speech models add five expressive voices. Other OpenAI-wire brands have no
/// published voice set, so they report the modality unsupported rather than guess.
pub fn openai_voices(model: &str) -> Vec<VoiceInfo> {
    const CLASSIC: &[&str] = &["alloy", "echo", "fable", "onyx", "nova", "shimmer"];
    const EXPRESSIVE: &[&str] = &["ash", "ballad", "coral", "sage", "verse"];
    let mut ids: Vec<&str> = CLASSIC.to_vec();
    if model.contains("gpt-4o") {
        ids.extend_from_slice(EXPRESSIVE);
    }
    ids.into_iter()
        .map(|id| VoiceInfo {
            id: id.to_string(),
            name: None,
            languages: Vec::new(),
            extra: Map::new(),
        })
        .collect()
}

/// The OpenAI speech model id `brand` would serve for `model` when an OpenAI voice catalog applies, or
/// `None` for a brand with no OpenAI voices. A brand either *is* OpenAI directly, or proxies it under a
/// `vendor/model` namespace (OpenRouter's `openai/<model>`), in which case the namespace is stripped so
/// [`openai_voices`] scopes on the real OpenAI model id (e.g. the `gpt-4o` family's expressive voices).
/// Keeping this here puts the namespace quirk in the wire layer every compatible brand already shares
/// (decision filter: a provider quirk lives at the edge, never in the core).
pub fn openai_speech_model<'a>(brand: &str, model: &'a str) -> Option<&'a str> {
    match brand {
        // Azure OpenAI deployments serve OpenAI's own speech models, so the same voice catalog applies
        // (the model here is the deployment name, which only affects the gpt-4o expressive-voice check).
        "openai" | "azure-openai" => Some(model),
        // OpenRouter addresses upstreams as `vendor/model`; its `openai/*` ids reach OpenAI's own
        // speech endpoint, so OpenAI's voice catalog applies to them.
        "openrouter" => model.strip_prefix("openai/"),
        _ => None,
    }
}

/// Mistral's `GET /v1/audio/voices` reply → canonical voices. The body wraps the list in `items`
/// (paginated: `{ "items": [ { id, name, … } ], "total", … }`); a bare array is tolerated too. `id` is
/// required (entries without one are skipped); `name` and a string-array `languages` are lifted into the
/// canonical fields, and every other field (e.g. `created_at`, `user_id`) rides through verbatim in
/// `extra` (principle 7) — so the mapping does not depend on Mistral's exact, evolving schema.
pub fn mistral_voices_to_canonical(value: Value) -> Vec<VoiceInfo> {
    let items = match &value {
        Value::Object(obj) => obj.get("items").and_then(Value::as_array).cloned(),
        Value::Array(arr) => Some(arr.clone()),
        _ => None,
    };
    let Some(items) = items else {
        return Vec::new();
    };
    items
        .into_iter()
        .filter_map(|item| {
            let Value::Object(mut obj) = item else {
                return None;
            };
            let id = obj.remove("id").and_then(|v| match v {
                Value::String(s) => Some(s),
                _ => None,
            })?;
            let name = obj.remove("name").and_then(|v| match v {
                Value::String(s) => Some(s),
                _ => None,
            });
            let languages = obj
                .remove("languages")
                .and_then(|v| v.as_array().cloned())
                .map(|a| {
                    a.iter()
                        .filter_map(Value::as_str)
                        .map(str::to_string)
                        .collect()
                })
                .unwrap_or_default();
            // Everything left over is preserved verbatim (created_at, user_id, vendor extras, …).
            Some(VoiceInfo {
                id,
                name,
                languages,
                extra: obj,
            })
        })
        .collect()
}

/// Parse an OpenAI-wire `GET /models` listing into canonical [`ModelInfo`]s. ONE tolerant parser for
/// the whole OpenAI-compatible family: OpenAI returns id-only; OpenRouter adds `context_length`,
/// `pricing` (per-TOKEN strings) and `architecture`; Together is a bare top-level array with `pricing`
/// (per-Mtok numbers) + `context_length`; Groq adds `context_window` + `max_completion_tokens` (plus an
/// `active` flag the provider layer filters on, preserved here verbatim); Cerebras's public catalog nests
/// the caps under `limits` (`max_context_length`/`max_completion_tokens`) and prices per-TOKEN under
/// `pricing.prompt`/`completion`; some surfaces wrap the list under a `models[]` array and key items by
/// `key`. This reads the *union* of those shapes and leaves every field a given response omits as
/// `None` — it never guesses (SOUL). Gaps are enhanced downstream from the bundled dataset.
pub fn openai_wire_models_to_canonical(value: Value) -> Vec<ModelInfo> {
    // Wrapper: a bare array (Together), else `data` (OpenAI family), else a `models` array (tolerated).
    let items = match value {
        Value::Array(arr) => arr,
        Value::Object(mut obj) => obj
            .remove("data")
            .or_else(|| obj.remove("models"))
            .and_then(|v| match v {
                Value::Array(a) => Some(a),
                _ => None,
            })
            .unwrap_or_default(),
        _ => Vec::new(),
    };
    items.into_iter().filter_map(model_item_to_info).collect()
}

/// Map one OpenAI-wire model object to [`ModelInfo`]. `None` (skipped) when it has no string id.
fn model_item_to_info(item: Value) -> Option<ModelInfo> {
    let Value::Object(mut obj) = item else {
        return None;
    };
    // id: `id` (OpenAI family) else `key` (tolerated alternative). Required — never fabricated.
    let id = wire_take_str(&mut obj, "id").or_else(|| wire_take_str(&mut obj, "key"))?;
    let mut info = ModelInfo::new(id);
    info.name = wire_take_str(&mut obj, "name").or_else(|| wire_take_str(&mut obj, "display_name"));
    info.modality = wire_modality(&obj);
    // The nested `limits` source is Cerebras's public-catalog model-cap object; only its
    // `max_context_length`/`max_completion_tokens` keys are token caps (sibling rate limits like
    // `requests_per_minute` are NOT caps and are left in `extra`). It is probed last, so a brand that
    // reports a cap at top level (or under OpenRouter's `top_provider`) still wins.
    info.max_context = wire_first_u32(
        &obj,
        &["context_window", "context_length", "max_context_length"],
    )
    .or_else(|| wire_nested_u32(&obj, "top_provider", "context_length"))
    .or_else(|| wire_nested_u32(&obj, "limits", "max_context_length"));
    info.max_output = wire_first_u32(&obj, &["max_completion_tokens", "max_output_tokens"])
        .or_else(|| wire_nested_u32(&obj, "top_provider", "max_completion_tokens"))
        .or_else(|| wire_nested_u32(&obj, "limits", "max_completion_tokens"));
    // Pricing differs by brand AND unit: OpenRouter `pricing.prompt`/`completion` are per-TOKEN decimal
    // STRINGS (×1e6 → per-Mtok); Together `pricing.input`/`output` are per-Mtok NUMBERS (verbatim). A
    // `<0` sentinel ("-1") means "not priced" → None; `0` is a real free rate → Some(0.0).
    if let Some(p) = obj.get("pricing").and_then(Value::as_object) {
        info.input_per_mtok = wire_price(p, "prompt", "input");
        info.output_per_mtok = wire_price(p, "completion", "output");
    }
    // max_thinking stays None — no compat brand's /models publishes a reasoning-budget number.
    // OpenRouter-style param metadata, captured verbatim when reported (other brands omit these):
    // `supported_parameters` is a positive string list — authoritative, so it passes straight through;
    // `default_parameters` is the recommended-values object. Lifted out of `extra` into typed fields so
    // the catalog surface renders them first-class. Their absence leaves the typed fields empty/None,
    // and the catalog computes a list from modality instead.
    if let Some(Value::Array(a)) = obj.remove("supported_parameters") {
        info.supported_parameters = Some(
            a.iter()
                .filter_map(|x| x.as_str().map(str::to_string))
                .collect(),
        );
    }
    if let Some(Value::Object(m)) = obj.remove("default_parameters") {
        info.default_parameters = m;
    }
    obj.remove("object"); // structural discriminator, not data
    info.extra = obj;
    Some(info)
}

/// Remove and return a non-empty string field.
fn wire_take_str(obj: &mut Map<String, Value>, key: &str) -> Option<String> {
    match obj.remove(key) {
        Some(Value::String(s)) if !s.is_empty() => Some(s),
        _ => None,
    }
}

/// A token count as u32, accepting a JSON number or a numeric string (some upstreams stringify int64).
fn wire_u32(v: &Value) -> Option<u32> {
    v.as_u64()
        .or_else(|| v.as_str().and_then(|s| s.parse::<u64>().ok()))
        .map(|n| n.min(u32::MAX as u64) as u32)
}

fn wire_first_u32(obj: &Map<String, Value>, keys: &[&str]) -> Option<u32> {
    keys.iter().find_map(|k| obj.get(*k).and_then(wire_u32))
}

fn wire_nested_u32(obj: &Map<String, Value>, parent: &str, key: &str) -> Option<u32> {
    obj.get(parent)
        .and_then(Value::as_object)
        .and_then(|o| o.get(key))
        .and_then(wire_u32)
}

/// Resolve a price to USD-per-Mtok. `per_token_key` carries an OpenRouter-style per-token value
/// (string or number; scaled ×1e6); `per_mtok_key` carries a Together-style per-Mtok value (verbatim).
/// A negative value is the "unpriced" sentinel → `None`; `0` is a real free rate → `Some(0.0)`.
fn wire_price(p: &Map<String, Value>, per_token_key: &str, per_mtok_key: &str) -> Option<f64> {
    if let Some(x) = p.get(per_token_key).and_then(wire_f64) {
        return (x >= 0.0).then_some(x * 1_000_000.0);
    }
    if let Some(x) = p.get(per_mtok_key).and_then(wire_f64) {
        return (x >= 0.0).then_some(x);
    }
    None
}

fn wire_f64(v: &Value) -> Option<f64> {
    v.as_f64()
        .or_else(|| v.as_str().and_then(|s| s.parse::<f64>().ok()))
}

/// Infer modality from explicit catalog signals only — NEVER from an id substring (SOUL: no guessing).
/// `type` (Together and similar), then OpenRouter `architecture.{input,output}_modalities`, then Azure
/// `capabilities`. `None` when no signal is decisive.
fn wire_modality(obj: &Map<String, Value>) -> Option<Modality> {
    if let Some(t) = obj.get("type").and_then(Value::as_str) {
        match t.to_ascii_lowercase().as_str() {
            "chat" | "language" | "code" | "llm" | "vlm" => return Some(Modality::Llm),
            "embedding" | "embeddings" | "embed" => return Some(Modality::Embedding),
            "rerank" | "reranker" | "rank" => return Some(Modality::Rerank),
            _ => {} // image / moderation / … → no canonical modality
        }
    }
    // OpenRouter (and look-alikes) describe a model by the modalities it accepts and emits. Decide from
    // BOTH sides: audio OUT is text→speech (TTS); audio IN with text OUT is speech→text (STT). Without
    // the input side an STT model (audio in, text out — e.g. Voxtral/Whisper) collapses to `llm` and is
    // unreachable via `?type=stt`. Only decide when the output side is reported (we never guess output).
    if let Some(arch) = obj.get("architecture").and_then(Value::as_object) {
        let list = |key: &str| -> Vec<&str> {
            arch.get(key)
                .and_then(Value::as_array)
                .map(|a| a.iter().filter_map(Value::as_str).collect())
                .unwrap_or_default()
        };
        let has_audio = |m: &[&str]| m.iter().any(|x| *x == "audio" || *x == "speech");
        let has_vision = |m: &[&str]| m.iter().any(|x| *x == "image" || *x == "video");
        let outs = list("output_modalities");
        let ins = list("input_modalities");
        if has_audio(&outs) {
            return Some(Modality::Tts);
        }
        // OpenRouter labels a speech-to-text model's output `transcription` (not `text`) — e.g. Voxtral
        // Mini Transcribe / gpt-4o-transcribe report `modality: "audio->transcription"`. That is an
        // explicit catalog signal (audio in → transcribed text out), so it is STT, not a guess. Without
        // this they fall through to `None` and are unreachable via `?type=stt`.
        if outs.contains(&"transcription") {
            return Some(Modality::Stt);
        }
        if outs.contains(&"text") {
            // Text out with audio in is speech-to-text ONLY when audio is the model's *distinguishing*
            // rich input. A natively-multimodal chat LLM — Gemini (`text+image+video+file+audio->text`),
            // GPT-4o — also accepts audio, but its image/video inputs betray a general language model;
            // classifying it STT would hide the whole Gemini catalog from `?type=llm` (its normal home)
            // and mis-file it under `?type=stt`. A true STT model (Voxtral/Whisper: audio[+text] in, no
            // vision) has no such vision input, so it still resolves to STT.
            return Some(if has_audio(&ins) && !has_vision(&ins) {
                Modality::Stt
            } else {
                Modality::Llm
            });
        }
    }
    if let Some(caps) = obj.get("capabilities").and_then(Value::as_object) {
        let on = |k: &str| caps.get(k).and_then(Value::as_bool).unwrap_or(false);
        if on("embeddings") {
            return Some(Modality::Embedding);
        }
        if on("chat_completion") || on("completion") {
            return Some(Modality::Llm);
        }
    }
    None
}

/// The audio MIME type for a `response_format`, used when the upstream response omits a `Content-Type`.
pub fn audio_content_type(format: Option<&str>) -> &'static str {
    match format.unwrap_or("mp3") {
        "mp3" => "audio/mpeg",
        // OpenAI returns opus framed in an Ogg container, not a raw opus stream.
        "opus" => "audio/ogg",
        "aac" => "audio/aac",
        "flac" => "audio/flac",
        "wav" => "audio/wav",
        "pcm" => "audio/pcm",
        _ => "application/octet-stream",
    }
}

/// OpenAI transcription response (`json`/`verbose_json`) → canonical [`TranscriptionResponse`]. We
/// always drive the upstream toward a JSON shape, so `text` is present; `language`/`duration`/`usage`
/// ride along when the model reports them.
pub fn openai_to_transcription(value: Value) -> TranscriptionResponse {
    let text = value
        .get("text")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let language = value
        .get("language")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let duration = value
        .get("duration")
        .and_then(Value::as_f64)
        .map(|f| f as f32);
    let usage = value
        .get("usage")
        .map(|u| {
            let prompt = u.get("input_tokens").and_then(Value::as_u64).unwrap_or(0);
            let completion = u.get("output_tokens").and_then(Value::as_u64).unwrap_or(0);
            Usage {
                prompt_tokens: prompt,
                completion_tokens: completion,
                total_tokens: u
                    .get("total_tokens")
                    .and_then(Value::as_u64)
                    .unwrap_or(prompt + completion),
                cost_usd: None,
                cache_read_tokens: 0,
                cache_creation_tokens: 0,
            }
        })
        .unwrap_or_default();
    TranscriptionResponse {
        text,
        language,
        duration,
        usage,
    }
}

/// Canonical transcription request → OpenRouter's JSON `/audio/transcriptions` body. OpenRouter is the
/// one compatible brand that does *not* accept the OpenAI multipart upload here: it wants a JSON body
/// with the audio base64-encoded under `input_audio.data`, tagged with its container `format` (the
/// OpenRouter format token derived from the upload filename). Honoring that exactly is the decision
/// filter "dialect mapping fidelity beats convenience" in practice. Dialect-/provider-specific `extra`
/// fields (e.g. `provider`) ride along verbatim (principle 7); OpenRouter documents no `prompt` or
/// `response_format` on this endpoint, so neither is sent.
pub fn transcription_request_to_openrouter(req: &TranscriptionRequest, format: &str) -> Value {
    let data = base64::engine::general_purpose::STANDARD.encode(&req.audio);
    let mut obj = Map::new();
    obj.insert("model".into(), json!(req.model));
    obj.insert(
        "input_audio".into(),
        json!({ "data": data, "format": format }),
    );
    if let Some(language) = &req.language {
        obj.insert("language".into(), json!(language));
    }
    if let Some(temp) = req.temperature {
        obj.insert("temperature".into(), json!(temp));
    }
    for (k, v) in &req.extra {
        obj.entry(k.clone()).or_insert_with(|| v.clone());
    }
    Value::Object(obj)
}

/// OpenRouter transcription response → canonical [`TranscriptionResponse`]. OpenRouter returns the
/// transcript under `text` and an accounting block whose `seconds` is the audio duration and whose
/// `cost` is the upstream-reported USD price — relayed verbatim, never computed (principle 5).
pub fn openrouter_to_transcription(value: Value) -> TranscriptionResponse {
    let text = value
        .get("text")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let language = value
        .get("language")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let usage_obj = value.get("usage");
    let duration = usage_obj
        .and_then(|u| u.get("seconds"))
        .and_then(Value::as_f64)
        .map(|f| f as f32);
    let usage = usage_obj
        .map(|u| {
            let prompt = u.get("input_tokens").and_then(Value::as_u64).unwrap_or(0);
            let completion = u.get("output_tokens").and_then(Value::as_u64).unwrap_or(0);
            Usage {
                prompt_tokens: prompt,
                completion_tokens: completion,
                total_tokens: u
                    .get("total_tokens")
                    .and_then(Value::as_u64)
                    .unwrap_or(prompt + completion),
                cost_usd: u.get("cost").and_then(Value::as_f64),
                cache_read_tokens: 0,
                cache_creation_tokens: 0,
            }
        })
        .unwrap_or_default();
    TranscriptionResponse {
        text,
        language,
        duration,
        usage,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn models_openai_shape_is_id_only_with_extras() {
        // OpenAI's `/models`: `{object:list, data:[{id,object,created,owned_by}]}` → id-only + extras.
        let v = json!({ "object": "list", "data": [
            { "id": "gpt-4o", "object": "model", "created": 1715000000, "owned_by": "openai" }
        ]});
        let out = openai_wire_models_to_canonical(v);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].id, "gpt-4o");
        assert_eq!(out[0].modality, None); // never guessed from the id
        assert_eq!(out[0].max_context, None);
        assert_eq!(out[0].input_per_mtok, None);
        assert_eq!(
            out[0].extra.get("owned_by").and_then(Value::as_str),
            Some("openai")
        );
        assert!(out[0].extra.get("object").is_none()); // structural key dropped
    }

    #[test]
    fn models_openrouter_pricing_is_per_token_string_scaled() {
        // OpenRouter prices are per-TOKEN decimal strings → ×1e6 for per-Mtok; context + modality from
        // its rich fields.
        let v = json!({ "data": [{
            "id": "anthropic/claude-x",
            "context_length": 200000,
            "pricing": { "prompt": "0.000003", "completion": "0.000015" },
            "architecture": { "output_modalities": ["text"] },
            "top_provider": { "max_completion_tokens": 64000 }
        }]});
        let m = &openai_wire_models_to_canonical(v)[0];
        assert_eq!(m.max_context, Some(200000));
        assert_eq!(m.max_output, Some(64000));
        assert_eq!(m.modality, Some(Modality::Llm));
        assert_eq!(m.input_per_mtok, Some(3.0)); // 0.000003 * 1e6
        assert_eq!(m.output_per_mtok, Some(15.0));
    }

    #[test]
    fn models_openrouter_audio_output_is_tts() {
        let v = json!({ "data": [{ "id": "x/tts", "architecture": { "output_modalities": ["audio"] } }]});
        assert_eq!(
            openai_wire_models_to_canonical(v)[0].modality,
            Some(Modality::Tts)
        );
    }

    #[test]
    fn models_openrouter_audio_input_text_output_is_stt() {
        // Voxtral/Whisper-shaped: audio in, text out. Must classify as STT (not LLM) so it is reachable
        // via `?type=stt`; a text-only model with no audio input stays LLM.
        let v = json!({ "data": [
            { "id": "x/voxtral", "architecture": {
                "input_modalities": ["text", "audio"], "output_modalities": ["text"] } },
            { "id": "x/plain", "architecture": {
                "input_modalities": ["text"], "output_modalities": ["text"] } },
        ]});
        let out = openai_wire_models_to_canonical(v);
        assert_eq!(out[0].modality, Some(Modality::Stt));
        assert_eq!(out[1].modality, Some(Modality::Llm));
    }

    #[test]
    fn models_openrouter_multimodal_llm_with_audio_input_is_llm() {
        // Gemini-on-OpenRouter shape: a native-multimodal chat model reports audio AMONG its inputs
        // (`text+image+video+file+audio -> text`). Audio-in must NOT flip it to STT — its image/video
        // input marks it a language model, so it stays LLM and remains listed under `?type=llm` (the
        // regression that hid the entire Gemini catalog from the chat model picker).
        let v = json!({ "data": [
            { "id": "google/gemini-2.5-pro", "architecture": {
                "input_modalities": ["text", "image", "file", "audio", "video"],
                "output_modalities": ["text"] } },
            { "id": "google/gemini-3.5-flash", "architecture": {
                "input_modalities": ["file", "image", "text", "audio", "video"],
                "output_modalities": ["text"] } },
        ]});
        let out = openai_wire_models_to_canonical(v);
        assert_eq!(out[0].modality, Some(Modality::Llm));
        assert_eq!(out[1].modality, Some(Modality::Llm));
    }

    #[test]
    fn models_openrouter_transcription_output_is_stt() {
        // Dedicated transcribe models (Voxtral Mini Transcribe, gpt-4o-transcribe) report
        // `modality: "audio->transcription"` with `output_modalities: ["transcription"]` — not "text".
        // That explicit signal must classify STT so they appear under `?type=stt`.
        let v = json!({ "data": [{ "id": "mistralai/voxtral-mini-transcribe", "architecture": {
            "modality": "audio->transcription",
            "input_modalities": ["audio"], "output_modalities": ["transcription"] } }]});
        assert_eq!(
            openai_wire_models_to_canonical(v)[0].modality,
            Some(Modality::Stt)
        );
    }

    #[test]
    fn models_together_bare_array_per_mtok_number_verbatim() {
        // Together returns a BARE top-level array; its `pricing.input/output` are per-Mtok NUMBERS (no
        // scaling); `type` drives modality; `context_length` is top-level.
        let v = json!([
            { "id": "together/llama", "type": "chat", "context_length": 128000,
              "pricing": { "input": 0.2, "output": 0.6 } },
            { "id": "together/embed", "type": "embedding", "context_length": 8192 }
        ]);
        let out = openai_wire_models_to_canonical(v);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].modality, Some(Modality::Llm));
        assert_eq!(out[0].max_context, Some(128000));
        assert_eq!(out[0].input_per_mtok, Some(0.2)); // verbatim, NOT *1e6
        assert_eq!(out[0].output_per_mtok, Some(0.6));
        assert_eq!(out[1].modality, Some(Modality::Embedding));
    }

    #[test]
    fn models_negative_price_sentinel_is_none_zero_is_some() {
        let v = json!({ "data": [{ "id": "x", "pricing": { "input": -1, "output": 0 } }]});
        let m = &openai_wire_models_to_canonical(v)[0];
        assert_eq!(m.input_per_mtok, None); // "-1" sentinel = not priced
        assert_eq!(m.output_per_mtok, Some(0.0)); // a real free rate
    }

    #[test]
    fn models_groq_context_window_no_id_modality_guess() {
        // Groq exposes `context_window` (read by the tolerant parser) but no modality field — and we do
        // NOT infer modality from the id substring.
        let v = json!({ "data": [{ "id": "whisper-large-v3", "context_window": 448 }]});
        let m = &openai_wire_models_to_canonical(v)[0];
        assert_eq!(m.max_context, Some(448));
        assert_eq!(m.modality, None);
    }

    #[test]
    fn models_groq_full_shape_lifts_caps_and_preserves_active() {
        // Groq's `GET /openai/v1/models` (verbatim from the docs): the standard OpenAI envelope plus the
        // Groq-specific `active`, `context_window`, `public_apps`, `max_completion_tokens`. The tolerant
        // parser lifts the two caps; everything else rides through verbatim in `extra` (principle 7) —
        // including `active`, which the *provider layer* (not this pure parser) reads to drop inactive
        // models. Modality is NOT inferred from the id.
        let v = json!({ "object": "list", "data": [{
            "id": "llama3-8b-8192",
            "object": "model",
            "created": 1693721698,
            "owned_by": "Meta",
            "active": true,
            "context_window": 8192,
            "public_apps": null,
            "max_completion_tokens": 8192
        }]});
        let out = openai_wire_models_to_canonical(v);
        assert_eq!(out.len(), 1);
        let m = &out[0];
        assert_eq!(m.id, "llama3-8b-8192");
        assert_eq!(m.max_context, Some(8192)); // context_window
        assert_eq!(m.max_output, Some(8192)); // max_completion_tokens
        assert_eq!(m.modality, None); // never guessed from the id
        assert_eq!(m.name, None); // Groq reports no display name
                                  // Groq-specific fields survive verbatim (the provider layer reads `active` to filter).
        assert_eq!(m.extra.get("active").and_then(Value::as_bool), Some(true));
        assert_eq!(
            m.extra.get("owned_by").and_then(Value::as_str),
            Some("Meta")
        );
        assert!(m.extra.contains_key("public_apps")); // a JSON null still rides through
        assert!(m.extra.contains_key("created"));
        assert!(m.extra.get("object").is_none()); // structural key dropped
    }

    #[test]
    fn models_cerebras_public_shape_reads_nested_limits_and_per_token_pricing() {
        // Cerebras's public catalog nests the caps under `limits` and prices per-TOKEN under
        // `pricing.prompt`/`completion`; everything else (capabilities, architecture, quantization,
        // owned_by) is preserved verbatim in `extra`. Modality is NOT guessed from `architecture.modality`.
        let v = json!({ "object": "list", "data": [{
            "id": "gpt-oss-120b",
            "object": "model",
            "created": 0,
            "owned_by": "Cerebras",
            "name": "GPT-OSS 120B",
            "pricing": { "prompt": "0.00000025", "completion": "0.00000069" },
            "limits": {
                "max_context_length": 131072,
                "max_completion_tokens": 32768,
                "requests_per_minute": null,
                "tokens_per_minute": null
            },
            "capabilities": { "streaming": true, "function_calling": true, "reasoning": true },
            "architecture": { "modality": "text", "tokenizer": "GPT", "instruct_type": "chatml" },
            "quantization": "fp16",
            "deprecated": false,
            "preview": false
        }]});
        let out = openai_wire_models_to_canonical(v);
        assert_eq!(out.len(), 1);
        let m = &out[0];
        assert_eq!(m.id, "gpt-oss-120b");
        assert_eq!(m.name.as_deref(), Some("GPT-OSS 120B"));
        assert_eq!(m.max_context, Some(131072)); // limits.max_context_length
        assert_eq!(m.max_output, Some(32768)); // limits.max_completion_tokens
        assert!((m.input_per_mtok.unwrap() - 0.25).abs() < 1e-9); // per-token string ×1e6
        assert!((m.output_per_mtok.unwrap() - 0.69).abs() < 1e-9);
        assert_eq!(m.modality, None); // "text" alone is ambiguous — left for downstream enrichment
                                      // The rich fields the authed `/v1/models` lacks survive verbatim (principle 7).
        assert_eq!(
            m.extra.get("owned_by").and_then(Value::as_str),
            Some("Cerebras")
        );
        assert_eq!(
            m.extra.get("quantization").and_then(Value::as_str),
            Some("fp16")
        );
        assert!(m.extra.contains_key("capabilities"));
        assert!(m.extra.contains_key("limits"));
        assert!(m.extra.get("object").is_none()); // structural key dropped
    }

    #[test]
    fn models_openrouter_supported_and_default_parameters_captured() {
        // OpenRouter reports a positive `supported_parameters` list and a `default_parameters` object;
        // both are lifted into typed fields (authoritative passthrough) and must NOT linger in `extra`.
        let v = json!({ "data": [{
            "id": "openai/gpt-5",
            "architecture": { "output_modalities": ["text"] },
            "supported_parameters": ["max_tokens", "tools", "tool_choice", "reasoning", "reasoning_effort"],
            "default_parameters": { "temperature": 1, "top_p": 1 }
        }]});
        let m = &openai_wire_models_to_canonical(v)[0];
        assert_eq!(
            m.supported_parameters,
            Some(vec![
                "max_tokens".to_string(),
                "tools".to_string(),
                "tool_choice".to_string(),
                "reasoning".to_string(),
                "reasoning_effort".to_string(),
            ])
        );
        assert_eq!(m.default_parameters.get("temperature"), Some(&json!(1)));
        // Lifted out of `extra`, not duplicated there.
        assert!(m.extra.get("supported_parameters").is_none());
        assert!(m.extra.get("default_parameters").is_none());
        // OpenRouter reports the positive list only — we never invent a negative one.
        assert!(m.unsupported_parameters.is_empty());
    }

    #[test]
    fn models_wrapper_and_key_id_fallback() {
        // Tolerance: a surface that wraps the list under `models[]` and keys items by `key` still parses.
        let v = json!({ "models": [{ "key": "qwen3", "max_context_length": 32768 }]});
        let m = &openai_wire_models_to_canonical(v)[0];
        assert_eq!(m.id, "qwen3");
        assert_eq!(m.max_context, Some(32768));
    }

    #[test]
    fn models_skips_items_without_id_and_tolerates_empty() {
        assert!(openai_wire_models_to_canonical(json!({})).is_empty());
        assert!(
            openai_wire_models_to_canonical(json!({ "data": [{ "object": "model" }] })).is_empty()
        );
    }

    #[test]
    fn openai_speech_model_recognizes_direct_and_openrouter_namespace() {
        // Direct OpenAI: the model id rides through unchanged.
        assert_eq!(openai_speech_model("openai", "tts-1"), Some("tts-1"));
        // Azure OpenAI deployments serve the same OpenAI voices.
        assert_eq!(
            openai_speech_model("azure-openai", "tts-deploy"),
            Some("tts-deploy")
        );
        // OpenRouter proxying OpenAI: the `openai/` namespace is stripped to the real model id, so the
        // per-model catalog (gpt-4o expressive voices) still resolves.
        assert_eq!(
            openai_speech_model("openrouter", "openai/gpt-4o-mini-tts"),
            Some("gpt-4o-mini-tts")
        );
        // OpenRouter routing to a non-OpenAI vendor has no OpenAI voices.
        assert_eq!(openai_speech_model("openrouter", "google/gemini-tts"), None);
        // A brand with no catalog at all.
        assert_eq!(openai_speech_model("groq", "whatever"), None);
    }

    #[test]
    fn speech_envelope_decodes_mistral_base64_audio() {
        // Mistral wraps the audio in `{ "audio_data": "<base64>" }`. The decoder must hand back the
        // raw bytes — here the three-byte `ID3` MP3 marker, base64 "SUQz".
        let body = json!({ "audio_data": "SUQz" });
        assert_eq!(decode_speech_envelope(&body), Some(b"ID3".to_vec()));
        // No `audio_data`, a non-string value, or invalid base64 → None (caller maps to an error).
        assert_eq!(decode_speech_envelope(&json!({ "other": "x" })), None);
        assert_eq!(decode_speech_envelope(&json!({ "audio_data": 7 })), None);
        assert_eq!(
            decode_speech_envelope(&json!({ "audio_data": "not base64!!" })),
            None
        );
    }

    #[test]
    fn mistral_voices_map_lifts_known_fields_and_passes_the_rest() {
        let body = json!({
            "items": [
                { "id": "aurora", "name": "Aurora", "languages": ["en", "fr"],
                  "created_at": "2025-10-07T20:56:01.974Z", "user_id": null },
                { "id": "basalt", "name": "Basalt" },
                { "name": "no-id-skipped" }
            ],
            "total": 2, "page": 1, "page_size": 1000
        });
        let voices = mistral_voices_to_canonical(body);
        assert_eq!(voices.len(), 2); // the id-less entry is dropped

        assert_eq!(voices[0].id, "aurora");
        assert_eq!(voices[0].name.as_deref(), Some("Aurora"));
        assert_eq!(voices[0].languages, vec!["en", "fr"]);
        // Unknown fields ride through verbatim (principle 7), not lost or hardcoded.
        assert_eq!(
            voices[0].extra.get("created_at").and_then(Value::as_str),
            Some("2025-10-07T20:56:01.974Z")
        );
        assert!(voices[0].extra.contains_key("user_id"));
        // id/name/languages are lifted out, not duplicated into extra.
        assert!(!voices[0].extra.contains_key("id"));
        assert!(!voices[0].extra.contains_key("name"));

        assert_eq!(voices[1].id, "basalt");
        assert!(voices[1].languages.is_empty());
    }

    #[test]
    fn mistral_voices_map_tolerates_a_bare_array_and_junk() {
        let bare = json!([{ "id": "solo", "name": "Solo" }]);
        assert_eq!(mistral_voices_to_canonical(bare).len(), 1);
        // Neither object-with-items nor array → empty, never a panic.
        assert!(mistral_voices_to_canonical(json!("nonsense")).is_empty());
        assert!(mistral_voices_to_canonical(json!({ "no_items": true })).is_empty());
    }

    #[test]
    fn openai_voices_scope_is_per_model() {
        // tts-1/tts-1-hd ship the classic six; the gpt-4o family adds five expressive voices.
        assert_eq!(openai_voices("tts-1").len(), 6);
        assert_eq!(openai_voices("gpt-4o-mini-tts").len(), 11);
        // The namespace must be stripped before scoping, else `openai/tts-1` would falsely match gpt-4o.
        let stripped = openai_speech_model("openrouter", "openai/tts-1").unwrap();
        assert_eq!(openai_voices(stripped).len(), 6);
    }

    #[test]
    fn maps_response_to_chunks() {
        let resp = json!({
            "id": "chatcmpl-x",
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": "hello" },
                "finish_reason": "stop"
            }],
            "usage": { "prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6 }
        });
        let chunks = openai_to_chunks(resp, "gpt-4o");
        assert!(matches!(chunks[0], StreamChunk::Start { .. }));
        assert!(matches!(&chunks[1], StreamChunk::Content { delta, .. } if delta == "hello"));
        assert!(matches!(
            chunks[2],
            StreamChunk::Finish {
                reason: FinishReason::Stop,
                ..
            }
        ));
        assert!(matches!(chunks[3], StreamChunk::Usage(u) if u.total_tokens == 6));
    }

    #[test]
    fn maps_upstream_reasoning_to_thinking_chunks() {
        // Collected: OpenRouter's `message.reasoning` leads the visible answer.
        let resp = json!({
            "id": "chatcmpl-x",
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "reasoning": "let me think", "content": "hello" },
                "finish_reason": "stop"
            }]
        });
        let chunks = openai_to_chunks(resp, "gpt-4o");
        assert!(
            matches!(&chunks[1], StreamChunk::Thinking { delta, .. } if delta == "let me think")
        );
        assert!(matches!(&chunks[2], StreamChunk::Content { delta, .. } if delta == "hello"));

        // Streaming: DeepSeek's `delta.reasoning_content` maps the same way.
        let mut seen_start = true;
        let chunks = openai_chunk_to_canonical(
            &json!({ "choices": [{ "index": 0, "delta": { "reasoning_content": "hmm" } }] }),
            &mut seen_start,
            "gpt-4o",
        );
        assert!(
            matches!(chunks.as_slice(), [StreamChunk::Thinking { index: 0, delta }] if delta == "hmm")
        );
    }

    #[test]
    fn request_collapses_single_text_to_string() {
        let req = ChatRequest {
            model: "gpt-4o".into(),
            messages: vec![Message::text(Role::User, "hi")],
            max_tokens: Some(16),
            temperature: None,
            top_p: None,
            stop: vec![],
            stream: false,
            tools: vec![],
            tool_choice: Some(ToolChoice::Required),
            thinking: None,
            extra: Default::default(),
        };
        // Default brand field name.
        let wire = request_to_openai(&req, "max_tokens", false);
        assert_eq!(wire["messages"][0]["content"], "hi");
        assert_eq!(wire["max_tokens"], 16);
        assert_eq!(wire["stream"], false);
        assert!(wire.get("stream_options").is_none());
        assert_eq!(wire["tool_choice"], "required");
        assert!(wire.get("max_completion_tokens").is_none());
    }

    #[test]
    fn reasoning_brands_use_max_completion_tokens() {
        let req = ChatRequest {
            model: "o3-mini".into(),
            messages: vec![Message::text(Role::User, "hi")],
            max_tokens: Some(32),
            temperature: None,
            top_p: None,
            stop: vec![],
            stream: false,
            tools: vec![],
            tool_choice: None,
            thinking: None,
            extra: Default::default(),
        };
        let wire = request_to_openai(&req, "max_completion_tokens", false);
        assert_eq!(wire["max_completion_tokens"], 32);
        assert!(wire.get("max_tokens").is_none());
        assert!(wire.get("tool_choice").is_none());
    }

    #[test]
    fn thinking_maps_to_reasoning_effort() {
        let effort = |t| {
            let mut req = sample_req();
            req.thinking = Some(t);
            request_to_openai(&req, "max_tokens", false)["reasoning_effort"].clone()
        };
        assert_eq!(effort(Thinking::Low), "low");
        assert_eq!(effort(Thinking::Med), "medium");
        assert_eq!(effort(Thinking::High), "high");
        // OpenAI exposes only three tiers, so the upper rungs collapse onto `high`.
        assert_eq!(effort(Thinking::Highx), "high");
        assert_eq!(effort(Thinking::Max), "high");
        // Absent unless the caller asks for it — extra stays the escape hatch.
        assert!(request_to_openai(&sample_req(), "max_tokens", false)
            .get("reasoning_effort")
            .is_none());
    }

    fn sample_req() -> ChatRequest {
        ChatRequest {
            model: "gpt-4o".into(),
            messages: vec![Message::text(Role::User, "hi")],
            max_tokens: Some(16),
            temperature: None,
            top_p: None,
            stop: vec![],
            stream: false,
            tools: vec![],
            tool_choice: None,
            thinking: None,
            extra: Default::default(),
        }
    }

    #[test]
    fn streaming_request_sets_stream_and_usage() {
        let wire = request_to_openai(&sample_req(), "max_tokens", true);
        assert_eq!(wire["stream"], true);
        assert_eq!(wire["stream_options"]["include_usage"], true);
    }

    #[test]
    fn parse_delta_content_and_finish() {
        let mut seen = false;
        // First frame: role only — no content, but emits Start.
        let c0 = openai_chunk_to_canonical(
            &json!({ "id": "chatcmpl-x", "model": "gpt-4o", "choices": [{ "index": 0, "delta": { "role": "assistant" } }] }),
            &mut seen,
            "gpt-4o",
        );
        assert!(
            matches!(c0.as_slice(), [StreamChunk::Start { id, model }] if id == "chatcmpl-x" && model == "gpt-4o")
        );
        // Subsequent content frame: no Start again.
        let c1 = openai_chunk_to_canonical(
            &json!({ "choices": [{ "index": 0, "delta": { "content": "Hel" } }] }),
            &mut seen,
            "gpt-4o",
        );
        assert!(
            matches!(c1.as_slice(), [StreamChunk::Content { index: 0, delta }] if delta == "Hel")
        );
        // Finish frame.
        let c2 = openai_chunk_to_canonical(
            &json!({ "choices": [{ "index": 0, "delta": {}, "finish_reason": "stop" }] }),
            &mut seen,
            "gpt-4o",
        );
        assert!(matches!(
            c2.as_slice(),
            [StreamChunk::Finish {
                index: 0,
                reason: FinishReason::Stop
            }]
        ));
    }

    #[test]
    fn parse_delta_tool_call_and_final_usage() {
        let mut seen = true; // Start already emitted earlier in the stream.
        let tc = openai_chunk_to_canonical(
            &json!({ "choices": [{ "index": 0, "delta": { "tool_calls": [
                { "index": 0, "id": "call_1", "function": { "name": "get_weather", "arguments": "{\"c" } }
            ] } }] }),
            &mut seen,
            "gpt-4o",
        );
        match tc.as_slice() {
            [StreamChunk::ToolCall { index: 0, call }] => {
                assert_eq!(call.index, 0);
                assert_eq!(call.id.as_deref(), Some("call_1"));
                assert_eq!(call.name.as_deref(), Some("get_weather"));
                assert_eq!(call.arguments.as_deref(), Some("{\"c"));
            }
            other => panic!("expected ToolCall, got {other:?}"),
        }
        // The terminal include_usage frame: empty choices, top-level usage.
        let u = openai_chunk_to_canonical(
            &json!({ "choices": [], "usage": { "prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12 } }),
            &mut seen,
            "gpt-4o",
        );
        assert!(
            matches!(u.as_slice(), [StreamChunk::Usage(usage)] if usage.total_tokens == 12 && usage.cost_usd.is_none())
        );
    }

    #[test]
    fn sse_payloads_buffers_split_frames() {
        let mut buf = Vec::new();
        // A frame split across two reads, mid-JSON.
        assert!(sse_payloads(&mut buf, b"data: {\"a\":").is_empty());
        let got = sse_payloads(&mut buf, b"1}\n\n");
        assert_eq!(got, vec!["{\"a\":1}".to_string()]);
        // Comment lines and CRLF terminators are skipped/handled; [DONE] is surfaced.
        let got = sse_payloads(&mut buf, b": keep-alive\r\ndata: [DONE]\n\n");
        assert_eq!(got, vec!["[DONE]".to_string()]);
    }

    #[test]
    fn sse_payloads_handles_split_done_sentinel() {
        let mut buf = Vec::new();
        assert!(sse_payloads(&mut buf, b"data: [DO").is_empty());
        let got = sse_payloads(&mut buf, b"NE]\n");
        assert_eq!(got, vec!["[DONE]".to_string()]);
    }

    #[test]
    fn openrouter_transcription_request_carries_base64_audio() {
        let req = TranscriptionRequest {
            model: "mistralai/voxtral-mini-transcribe".into(),
            audio: bytes::Bytes::from_static(b"hello"),
            filename: "clip.mp3".into(),
            language: Some("en".into()),
            prompt: Some("ignored upstream".into()),
            response_format: Some("json".into()),
            temperature: Some(0.0),
            extra: [("provider".to_string(), json!({ "only": ["mistral"] }))]
                .into_iter()
                .collect(),
        };
        let body = transcription_request_to_openrouter(&req, "mp3");

        assert_eq!(body["model"], json!("mistralai/voxtral-mini-transcribe"));
        // Audio rides base64-encoded in a JSON object, never as a multipart part.
        assert_eq!(body["input_audio"]["data"], json!("aGVsbG8=")); // base64("hello")
        assert_eq!(body["input_audio"]["format"], json!("mp3"));
        assert_eq!(body["language"], json!("en"));
        assert_eq!(body["temperature"], json!(0.0));
        // Verbatim passthrough (principle 7); `prompt`/`response_format` are not OpenRouter STT fields.
        assert_eq!(body["provider"], json!({ "only": ["mistral"] }));
        assert!(body.get("prompt").is_none());
        assert!(body.get("response_format").is_none());
    }

    #[test]
    fn openrouter_transcription_response_maps_text_duration_and_cost() {
        let value = json!({
            "text": "Hello, this is a test of speech-to-text transcription.",
            "usage": { "seconds": 9.2, "total_tokens": 113, "input_tokens": 83, "output_tokens": 30, "cost": 0.000508 }
        });
        let out = openrouter_to_transcription(value);
        assert_eq!(
            out.text,
            "Hello, this is a test of speech-to-text transcription."
        );
        assert_eq!(out.duration, Some(9.2)); // OpenRouter reports duration as `usage.seconds`
        assert_eq!(out.usage.prompt_tokens, 83);
        assert_eq!(out.usage.completion_tokens, 30);
        assert_eq!(out.usage.total_tokens, 113);
        assert_eq!(out.usage.cost_usd, Some(0.000508)); // relayed verbatim, never computed
    }
}
