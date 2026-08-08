//! Meta's hosted Llama API (`api.llama.com`).
//!
//! The request looks deliberately familiar to OpenAI Chat Completions, but the response is a
//! different dialect: collected replies live under `completion_message`, while SSE frames carry an
//! `event` with a typed `delta`. Keep that difference at this provider edge instead of adding Meta
//! conditionals to the shared OpenAI wire mapper.

use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use futures::stream;
use llmleaf_model::{
    ChatRequest, ContentPart, FinishReason, Message, ModelError, ModelInfo, ResponseStream, Role,
    StreamChunk, ToolCallDelta, ToolChoice, Usage,
};
use llmleaf_provider::{Provider, ProviderCx};
use serde_json::{json, Map, Value};

use crate::http::{json_sse_to_stream, post_json, send_checked};
use crate::openai_wire::openai_wire_models_to_canonical;
use crate::transport::{HttpRequest, HttpTransport, Transports};

const DEFAULT_ENDPOINT: &str = "https://api.llama.com/v1";

pub struct MetaProvider {
    http: Arc<dyn HttpTransport>,
}

impl MetaProvider {
    pub fn new(transports: &Transports) -> Self {
        Self {
            http: transports.http.clone(),
        }
    }

    fn endpoint<'a>(&self, cx: &'a ProviderCx) -> &'a str {
        cx.endpoint
            .as_deref()
            .unwrap_or(DEFAULT_ENDPOINT)
            .trim_end_matches('/')
    }

    fn auth(&self, req: HttpRequest, cx: &ProviderCx) -> HttpRequest {
        match &cx.credential {
            Some(credential) => req.bearer(credential),
            None => req,
        }
    }
}

#[async_trait]
impl Provider for MetaProvider {
    fn name(&self) -> &str {
        "meta"
    }

    async fn models(&self, cx: &ProviderCx) -> Result<Vec<ModelInfo>, ModelError> {
        let url = format!("{}/models", self.endpoint(cx));
        let req = self.auth(
            HttpRequest::get(&url).header("Accept", "application/json"),
            cx,
        );
        let value = post_json(&*self.http, req).await?;
        // Meta returns the conventional `{ "data": [{ "id", "created", "owned_by", ... }] }`
        // catalog envelope. The tolerant shared catalog parser preserves every vendor field.
        Ok(openai_wire_models_to_canonical(value))
    }

    async fn chat(&self, req: ChatRequest, cx: &ProviderCx) -> Result<ResponseStream, ModelError> {
        validate_request(&req)?;

        let use_stream = cx.use_upstream_streaming(req.stream, true);
        let mut body = request_to_meta(&req)?;
        body["stream"] = json!(use_stream);
        let accept = if use_stream {
            "text/event-stream"
        } else {
            "application/json"
        };
        let url = format!("{}/chat/completions", self.endpoint(cx));
        let http_req = self.auth(
            HttpRequest::post(&url).header("Accept", accept).json(body),
            cx,
        );

        if use_stream {
            let response = send_checked(&*self.http, http_req).await?;
            Ok(meta_sse_to_stream(response.body, req.model))
        } else {
            let value = post_json(&*self.http, http_req).await?;
            Ok(Box::pin(stream::iter(
                meta_to_chunks(value, &req.model).into_iter().map(Ok),
            )))
        }
    }
}

/// Reject canonical features the hosted API cannot represent. Returning `Unsupported` lets routing
/// continue to the next target without treating a healthy Meta endpoint as degraded.
fn validate_request(req: &ChatRequest) -> Result<(), ModelError> {
    if req.has_input_audio() {
        return Err(ModelError::Unsupported(
            "provider 'meta' does not support audio input in chat".into(),
        ));
    }
    if !req.stop.is_empty() {
        return Err(ModelError::Unsupported(
            "provider 'meta' does not support stop sequences".into(),
        ));
    }
    if req.thinking.is_some() {
        return Err(ModelError::Unsupported(
            "provider 'meta' does not expose a reasoning-effort control".into(),
        ));
    }
    for message in &req.messages {
        for part in &message.content {
            if matches!(
                part,
                ContentPart::Thinking { .. } | ContentPart::RedactedThinking { .. }
            ) {
                return Err(ModelError::Unsupported(
                    "provider 'meta' cannot replay reasoning blocks".into(),
                ));
            }
            if matches!(part, ContentPart::ImageUrl { .. }) && message.role != Role::User {
                return Err(ModelError::Unsupported(
                    "provider 'meta' accepts image content only in user messages".into(),
                ));
            }
        }
    }
    Ok(())
}

/// Canonical chat request to Meta's documented Llama API request.
fn request_to_meta(req: &ChatRequest) -> Result<Value, ModelError> {
    let mut obj = Map::new();
    obj.insert("model".into(), json!(req.model));
    obj.insert(
        "messages".into(),
        Value::Array(
            req.messages
                .iter()
                .map(message_to_meta)
                .collect::<Result<Vec<_>, _>>()?,
        ),
    );
    if let Some(value) = req.max_tokens {
        obj.insert("max_completion_tokens".into(), json!(value));
    }
    if let Some(value) = req.temperature {
        obj.insert("temperature".into(), json!(value));
    }
    if let Some(value) = req.top_p {
        obj.insert("top_p".into(), json!(value));
    }
    if !req.tools.is_empty() {
        let tools = req
            .tools
            .iter()
            .map(|tool| {
                let mut function = Map::new();
                function.insert("name".into(), json!(tool.name));
                if let Some(description) = &tool.description {
                    function.insert("description".into(), json!(description));
                }
                function.insert("parameters".into(), tool.parameters.clone());
                json!({ "type": "function", "function": Value::Object(function) })
            })
            .collect::<Vec<_>>();
        obj.insert("tools".into(), Value::Array(tools));
    }
    if let Some(choice) = &req.tool_choice {
        obj.insert("tool_choice".into(), tool_choice_to_meta(choice));
    }
    // Native additions such as `top_k`, `repetition_penalty`, `response_format`, and `user` arrive
    // through `extra`; modeled fields above win if a consumer supplied both representations.
    for (key, value) in &req.extra {
        obj.entry(key.clone()).or_insert_with(|| value.clone());
    }
    Ok(Value::Object(obj))
}

fn message_to_meta(message: &Message) -> Result<Value, ModelError> {
    let role = match message.role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    };
    let mut obj = Map::new();
    obj.insert("role".into(), json!(role));

    let has_images = message
        .content
        .iter()
        .any(|part| matches!(part, ContentPart::ImageUrl { .. }));
    if has_images {
        let parts = message
            .content
            .iter()
            .map(|part| match part {
                ContentPart::Text { text } => Ok(json!({ "type": "text", "text": text })),
                ContentPart::ImageUrl { url, .. } => {
                    Ok(json!({ "type": "image_url", "image_url": { "url": url } }))
                }
                _ => Err(ModelError::Unsupported(
                    "provider 'meta' cannot represent this message content".into(),
                )),
            })
            .collect::<Result<Vec<_>, _>>()?;
        obj.insert("content".into(), Value::Array(parts));
    } else {
        // All non-image message kinds accept a string. Concatenating typed text parts preserves their
        // canonical semantics while avoiding array shapes the assistant response type does not accept.
        obj.insert("content".into(), json!(message.text_content()));
    }

    if !message.tool_calls.is_empty() {
        obj.insert(
            "tool_calls".into(),
            Value::Array(
                message
                    .tool_calls
                    .iter()
                    .map(|call| {
                        json!({
                            "id": call.id,
                            "function": { "name": call.name, "arguments": call.arguments },
                        })
                    })
                    .collect(),
            ),
        );
    }
    if let Some(id) = &message.tool_call_id {
        obj.insert("tool_call_id".into(), json!(id));
    }
    Ok(Value::Object(obj))
}

fn tool_choice_to_meta(choice: &ToolChoice) -> Value {
    match choice {
        ToolChoice::Auto => json!("auto"),
        ToolChoice::None => json!("none"),
        ToolChoice::Required => json!("required"),
        ToolChoice::Named(name) => {
            json!({ "type": "function", "function": { "name": name } })
        }
    }
}

fn meta_to_chunks(value: Value, fallback_model: &str) -> Vec<StreamChunk> {
    let id = value
        .get("id")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let mut chunks = vec![StreamChunk::Start {
        id,
        model: fallback_model.to_string(),
    }];

    let message = value.get("completion_message").unwrap_or(&Value::Null);
    if let Some(text) = message_content_text(message.get("content")) {
        if !text.is_empty() {
            chunks.push(StreamChunk::Content {
                index: 0,
                delta: text,
            });
        }
    }
    if let Some(calls) = message.get("tool_calls").and_then(Value::as_array) {
        for (index, call) in calls.iter().enumerate() {
            let function = call.get("function").unwrap_or(&Value::Null);
            chunks.push(StreamChunk::ToolCall {
                index: 0,
                call: ToolCallDelta {
                    index: index as u32,
                    id: call.get("id").and_then(Value::as_str).map(str::to_owned),
                    name: function
                        .get("name")
                        .and_then(Value::as_str)
                        .map(str::to_owned),
                    arguments: function
                        .get("arguments")
                        .and_then(Value::as_str)
                        .map(str::to_owned),
                },
            });
        }
    }
    let reason = message
        .get("stop_reason")
        .and_then(Value::as_str)
        .map(meta_finish_reason)
        .unwrap_or(FinishReason::Stop);
    chunks.push(StreamChunk::Finish { index: 0, reason });
    if let Some(usage) = metrics_to_usage(value.get("metrics")) {
        chunks.push(StreamChunk::Usage(usage));
    }
    chunks
}

fn message_content_text(content: Option<&Value>) -> Option<String> {
    match content? {
        Value::String(text) => Some(text.clone()),
        Value::Object(item) => item.get("text").and_then(Value::as_str).map(str::to_owned),
        // Be tolerant of a future additive list form; concatenate only documented text items.
        Value::Array(items) => Some(
            items
                .iter()
                .filter_map(|item| item.get("text").and_then(Value::as_str))
                .collect(),
        ),
        _ => None,
    }
}

#[derive(Default)]
struct MetaStreamState {
    fallback_model: String,
    seen_start: bool,
    finished: bool,
    next_tool_index: u32,
    tool_indexes: BTreeMap<String, u32>,
    active_tool_index: Option<u32>,
}

fn meta_sse_to_stream(
    body: crate::transport::BytesStream,
    fallback_model: String,
) -> ResponseStream {
    json_sse_to_stream(
        body,
        MetaStreamState {
            fallback_model,
            ..Default::default()
        },
        meta_event_to_chunks,
    )
}

fn meta_event_to_chunks(
    value: &Value,
    state: &mut MetaStreamState,
) -> Result<Vec<StreamChunk>, ModelError> {
    if let Some(error) = value.get("error") {
        let status = error
            .get("status")
            .or_else(|| error.get("code"))
            .and_then(Value::as_u64)
            .unwrap_or(500) as u16;
        let message = error
            .get("message")
            .and_then(Value::as_str)
            .unwrap_or("Meta Llama API stream error")
            .to_string();
        return Err(ModelError::Upstream { status, message });
    }

    let mut chunks = Vec::new();
    if !state.seen_start {
        chunks.push(StreamChunk::Start {
            id: value
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string(),
            model: state.fallback_model.clone(),
        });
        state.seen_start = true;
    }

    let event = value.get("event").unwrap_or(&Value::Null);
    let delta = event.get("delta").unwrap_or(&Value::Null);
    match delta.get("type").and_then(Value::as_str) {
        Some("text") => {
            if let Some(text) = delta
                .get("text")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
            {
                chunks.push(StreamChunk::Content {
                    index: 0,
                    delta: text.to_string(),
                });
            }
        }
        Some("tool_call") => {
            let id = delta.get("id").and_then(Value::as_str).map(str::to_owned);
            let tool_index = if let Some(id) = &id {
                if let Some(index) = state.tool_indexes.get(id) {
                    *index
                } else {
                    let index = state.next_tool_index;
                    state.next_tool_index += 1;
                    state.tool_indexes.insert(id.clone(), index);
                    index
                }
            } else if let Some(index) = state.active_tool_index {
                index
            } else {
                let index = state.next_tool_index;
                state.next_tool_index += 1;
                index
            };
            state.active_tool_index = Some(tool_index);
            let function = delta.get("function").unwrap_or(&Value::Null);
            chunks.push(StreamChunk::ToolCall {
                index: 0,
                call: ToolCallDelta {
                    index: tool_index,
                    id,
                    name: function
                        .get("name")
                        .and_then(Value::as_str)
                        .map(str::to_owned),
                    arguments: function
                        .get("arguments")
                        .and_then(Value::as_str)
                        .filter(|s| !s.is_empty())
                        .map(str::to_owned),
                },
            });
        }
        _ => {}
    }

    if let Some(usage) = metrics_to_usage(event.get("metrics")) {
        chunks.push(StreamChunk::Usage(usage));
    }
    let event_type = event.get("event_type").and_then(Value::as_str);
    if !state.finished && (event_type == Some("complete") || event.get("stop_reason").is_some()) {
        let reason = event
            .get("stop_reason")
            .and_then(Value::as_str)
            .map(meta_finish_reason)
            .unwrap_or(FinishReason::Stop);
        chunks.push(StreamChunk::Finish { index: 0, reason });
        state.finished = true;
    }
    Ok(chunks)
}

/// Meta reports generation measurements as a list of `{metric, value, unit}` objects. Token metric
/// names have changed during the API preview, so accept both prompt/completion and input/output names.
fn metrics_to_usage(metrics: Option<&Value>) -> Option<Usage> {
    let mut prompt = None;
    let mut completion = None;
    let mut total = None;
    for metric in metrics.and_then(Value::as_array)? {
        let name = metric.get("metric").and_then(Value::as_str)?;
        let value = metric
            .get("value")
            .and_then(Value::as_f64)
            .filter(|value| value.is_finite() && *value >= 0.0)
            .map(|value| value as u64);
        match name {
            "prompt_tokens" | "input_tokens" | "num_prompt_tokens" | "num_input_tokens" => {
                prompt = value
            }
            "completion_tokens"
            | "output_tokens"
            | "num_completion_tokens"
            | "num_output_tokens" => completion = value,
            "total_tokens" | "num_total_tokens" => total = value,
            _ => {}
        }
    }
    if prompt.is_none() && completion.is_none() && total.is_none() {
        return None;
    }
    let prompt_tokens = prompt.unwrap_or(0);
    let completion_tokens = completion.unwrap_or(0);
    Some(Usage {
        prompt_tokens,
        completion_tokens,
        total_tokens: total.unwrap_or(prompt_tokens + completion_tokens),
        cost_usd: None,
        cache_read_tokens: 0,
        cache_creation_tokens: 0,
    })
}

fn meta_finish_reason(reason: &str) -> FinishReason {
    match reason {
        "length" => FinishReason::Length,
        "tool_calls" => FinishReason::ToolCalls,
        _ => FinishReason::Stop,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llmleaf_model::{collect, collect_chunks, ToolCall, ToolDef};

    use crate::fake::{FakeHttpTransport, FakeRealtimeTransport, FakeResponse};
    use crate::transport::{HttpBody, Transports};

    fn fake_transports(http: FakeHttpTransport) -> Transports {
        Transports {
            http: Arc::new(http),
            realtime: Arc::new(FakeRealtimeTransport::scripted(Vec::new())),
        }
    }

    fn request() -> ChatRequest {
        ChatRequest {
            model: "Llama-4-Maverick-17B-128E-Instruct-FP8".into(),
            messages: vec![Message::text(Role::User, "hello")],
            max_tokens: Some(128),
            temperature: Some(0.4),
            top_p: Some(0.9),
            stop: vec![],
            stream: false,
            tools: vec![],
            tool_choice: None,
            thinking: None,
            extra: Map::new(),
        }
    }

    #[test]
    fn maps_request_fields_and_native_extras() {
        let mut req = request();
        req.extra.insert("top_k".into(), json!(40));
        req.tools.push(ToolDef {
            name: "weather".into(),
            description: Some("Get weather".into()),
            parameters: json!({ "type": "object" }),
        });
        req.tool_choice = Some(ToolChoice::Named("weather".into()));

        let wire = request_to_meta(&req).unwrap();
        assert_eq!(wire["max_completion_tokens"], 128);
        assert_eq!(wire["messages"][0]["content"], "hello");
        assert_eq!(wire["tools"][0]["function"]["name"], "weather");
        assert_eq!(wire["tool_choice"]["function"]["name"], "weather");
        assert_eq!(wire["top_k"], 40);
        assert!(wire.get("stream_options").is_none());
    }

    #[test]
    fn maps_multimodal_user_and_tool_history() {
        let mut req = request();
        req.messages = vec![
            Message {
                role: Role::User,
                content: vec![
                    ContentPart::Text {
                        text: "look".into(),
                    },
                    ContentPart::ImageUrl {
                        url: "https://example.test/cat.png".into(),
                        detail: Some("high".into()),
                    },
                ],
                tool_calls: vec![],
                tool_call_id: None,
                name: None,
            },
            Message {
                role: Role::Assistant,
                content: vec![],
                tool_calls: vec![ToolCall {
                    id: "call_1".into(),
                    name: "weather".into(),
                    arguments: "{}".into(),
                }],
                tool_call_id: None,
                name: None,
            },
            Message {
                role: Role::Tool,
                content: vec![ContentPart::Text {
                    text: "sunny".into(),
                }],
                tool_calls: vec![],
                tool_call_id: Some("call_1".into()),
                name: None,
            },
        ];

        let wire = request_to_meta(&req).unwrap();
        assert_eq!(wire["messages"][0]["content"][1]["type"], "image_url");
        assert_eq!(wire["messages"][1]["tool_calls"][0]["id"], "call_1");
        assert_eq!(wire["messages"][2]["tool_call_id"], "call_1");
    }

    #[test]
    fn parses_collected_content_tools_and_metrics() {
        let chunks = meta_to_chunks(
            json!({
                "id": "req_1",
                "completion_message": {
                    "role": "assistant",
                    "content": { "type": "text", "text": "hello" },
                    "tool_calls": [{
                        "id": "call_1",
                        "function": { "name": "weather", "arguments": "{\"city\":\"Paris\"}" }
                    }],
                    "stop_reason": "tool_calls"
                },
                "metrics": [
                    { "metric": "input_tokens", "value": 9, "unit": "tokens" },
                    { "metric": "output_tokens", "value": 4, "unit": "tokens" }
                ]
            }),
            "Llama-4-Maverick",
        );
        let response = collect_chunks(chunks);
        assert_eq!(response.id, "req_1");
        assert_eq!(response.model, "Llama-4-Maverick");
        assert_eq!(response.choices[0].text, "hello");
        assert_eq!(response.choices[0].tool_calls[0].name, "weather");
        assert_eq!(
            response.choices[0].finish_reason,
            Some(FinishReason::ToolCalls)
        );
        assert_eq!(response.usage.total_tokens, 13);
    }

    #[test]
    fn parses_streamed_text_and_incremental_tool_call() {
        let mut state = MetaStreamState {
            fallback_model: "Llama-4-Maverick".into(),
            ..Default::default()
        };
        let events = [
            json!({
                "id": "req_1",
                "event": { "event_type": "start", "delta": { "type": "text", "text": "" } }
            }),
            json!({
                "id": "req_1",
                "event": { "event_type": "progress", "delta": { "type": "text", "text": "hello" } }
            }),
            json!({
                "id": "req_1",
                "event": { "event_type": "progress", "delta": {
                    "type": "tool_call", "id": "call_1",
                    "function": { "name": "weather", "arguments": "{\"city\":" }
                } }
            }),
            json!({
                "id": "req_1",
                "event": { "event_type": "progress", "delta": {
                    "type": "tool_call", "function": { "arguments": "\"Paris\"}" }
                } }
            }),
            json!({
                "id": "req_1",
                "event": {
                    "event_type": "complete",
                    "delta": { "type": "text", "text": "" },
                    "stop_reason": "tool_calls",
                    "metrics": [
                        { "metric": "prompt_tokens", "value": 8 },
                        { "metric": "completion_tokens", "value": 3 }
                    ]
                }
            }),
        ];
        let chunks = events
            .iter()
            .flat_map(|event| meta_event_to_chunks(event, &mut state).unwrap())
            .collect::<Vec<_>>();
        let response = collect_chunks(chunks);
        assert_eq!(response.id, "req_1");
        assert_eq!(response.choices[0].text, "hello");
        assert_eq!(response.choices[0].tool_calls[0].id, "call_1");
        assert_eq!(
            response.choices[0].tool_calls[0].arguments,
            "{\"city\":\"Paris\"}"
        );
        assert_eq!(response.usage.total_tokens, 11);
    }

    #[test]
    fn rejects_unrepresentable_canonical_features() {
        let mut req = request();
        req.stop.push("END".into());
        assert!(matches!(
            validate_request(&req),
            Err(ModelError::Unsupported(_))
        ));

        let mut req = request();
        req.messages[0].content = vec![ContentPart::InputAudio {
            data: "AAAA".into(),
            format: "wav".into(),
        }];
        assert!(matches!(
            validate_request(&req),
            Err(ModelError::Unsupported(_))
        ));
    }

    #[tokio::test]
    async fn provider_posts_native_streaming_wire_with_bearer_auth() {
        let sse = concat!(
            "data: {\"id\":\"req_1\",\"event\":{\"event_type\":\"start\",\"delta\":{\"type\":\"text\",\"text\":\"\"}}}\n\n",
            "data: {\"id\":\"req_1\",\"event\":{\"event_type\":\"progress\",\"delta\":{\"type\":\"text\",\"text\":\"Hello\"}}}\n\n",
            "data: {\"id\":\"req_1\",\"event\":{\"event_type\":\"complete\",\"delta\":{\"type\":\"text\",\"text\":\"\"},\"stop_reason\":\"stop\"}}\n\n",
        );
        let http = FakeHttpTransport::new(move |outbound| {
            assert_eq!(outbound.url, "https://api.llama.com/v1/chat/completions");
            assert!(outbound.headers.iter().any(|(name, value)| {
                name == "Authorization" && value == "Bearer test-meta-key"
            }));
            let HttpBody::Json(body) = &outbound.body else {
                panic!("expected JSON request body")
            };
            assert_eq!(body["model"], "Llama-4-Maverick-17B-128E-Instruct-FP8");
            // Streaming upstream is the provider default even for a collected consumer.
            assert_eq!(body["stream"], true);
            assert!(body.get("stream_options").is_none());
            Ok(FakeResponse::ok_bytes("text/event-stream", sse))
        });
        let provider = MetaProvider::new(&fake_transports(http));
        let cx = ProviderCx {
            credential: Some("test-meta-key".into()),
            ..Default::default()
        };

        let response = collect(provider.chat(request(), &cx).await.unwrap())
            .await
            .unwrap();
        assert_eq!(response.id, "req_1");
        assert_eq!(response.choices[0].text, "Hello");
        assert_eq!(response.choices[0].finish_reason, Some(FinishReason::Stop));
    }

    #[tokio::test]
    async fn provider_lists_models_and_preserves_meta_fields() {
        let http = FakeHttpTransport::new(|outbound| {
            assert_eq!(outbound.url, "https://api.llama.com/v1/models");
            Ok(FakeResponse::ok_json(&json!({
                "data": [{
                    "id": "Llama-4-Maverick-17B-128E-Instruct-FP8",
                    "object": "model",
                    "created": 1743811200,
                    "owned_by": "meta"
                }]
            })))
        });
        let provider = MetaProvider::new(&fake_transports(http));

        let models = provider.models(&ProviderCx::default()).await.unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "Llama-4-Maverick-17B-128E-Instruct-FP8");
        assert_eq!(models[0].extra["owned_by"], "meta");
        assert_eq!(models[0].extra["created"], 1743811200);
    }

    #[test]
    fn factory_registers_meta_kinds() {
        for kind in ["meta", "llama-api", "meta-llama"] {
            let provider = crate::build(kind, &Transports::fake()).expect("known Meta kind");
            assert_eq!(provider.name(), "meta");
            assert!(crate::known_kinds().contains(&kind));
        }
    }
}
