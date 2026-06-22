//! Wire-fidelity tests: prove the serde types serialise to / parse from the exact
//! OpenAI/OpenRouter JSON shapes SPEC.md mandates. SPEC.md: "The SDK calls must produce
//! byte-identical request bodies."

use llmleaf_client::*;
use serde_json::json;

#[test]
fn chat_request_minimal_body() {
    let req = ChatRequest::new("gpt-4o-mini", vec![ChatMessage::user("hi")]);
    let v = serde_json::to_value(&req).unwrap();
    assert_eq!(
        v,
        json!({
            "model": "gpt-4o-mini",
            "messages": [{ "role": "user", "content": "hi" }]
        })
    );
}

#[test]
fn content_is_string_for_text_and_array_for_parts() {
    let text = ChatMessage::user("plain");
    assert_eq!(
        serde_json::to_value(&text).unwrap()["content"],
        json!("plain")
    );

    let mut multi = ChatMessage::user("");
    multi.content = Some(Content::Parts(vec![
        ContentPart::text("look:"),
        ContentPart::image_url("https://x/y.png"),
    ]));
    assert_eq!(
        serde_json::to_value(&multi).unwrap()["content"],
        json!([
            { "type": "text", "text": "look:" },
            { "type": "image_url", "image_url": { "url": "https://x/y.png" } }
        ])
    );
}

#[test]
fn stop_collapses_to_string_for_single_element() {
    let one = Stop::from_vec(vec!["END".into()]).unwrap();
    assert_eq!(serde_json::to_value(&one).unwrap(), json!("END"));

    let many = Stop::from_vec(vec!["A".into(), "B".into()]).unwrap();
    assert_eq!(serde_json::to_value(&many).unwrap(), json!(["A", "B"]));

    assert!(Stop::from_vec(vec![]).is_none());
}

#[test]
fn extra_merges_at_top_level() {
    let mut req = ChatRequest::new("m", vec![ChatMessage::user("x")]);
    let mut extra = serde_json::Map::new();
    extra.insert("provider".into(), json!({ "order": ["openai"] }));
    extra.insert("transforms".into(), json!(["middle-out"]));
    req.extra = Some(extra);

    let v = serde_json::to_value(&req).unwrap();
    // Spliced verbatim at the top level — not nested under "extra", not stringified.
    assert_eq!(v["provider"], json!({ "order": ["openai"] }));
    assert_eq!(v["transforms"], json!(["middle-out"]));
    assert!(v.get("extra").is_none());
}

#[test]
fn free_form_json_schema_spliced_not_stringified() {
    let mut req = ChatRequest::new("m", vec![ChatMessage::user("x")]);
    req.response_format = Some(ResponseFormat {
        kind: "json_schema".into(),
        json_schema: Some(json!({ "name": "foo", "schema": { "type": "object" } })),
    });
    let v = serde_json::to_value(&req).unwrap();
    assert_eq!(v["response_format"]["type"], json!("json_schema"));
    // The schema is a real object on the wire, not an escaped string.
    assert!(v["response_format"]["json_schema"].is_object());
    assert_eq!(
        v["response_format"]["json_schema"]["schema"]["type"],
        json!("object")
    );
}

#[test]
fn tool_choice_string_vs_named_object() {
    assert_eq!(
        serde_json::to_value(ToolChoice::mode("auto")).unwrap(),
        json!("auto")
    );
    assert_eq!(
        serde_json::to_value(ToolChoice::named("get_weather")).unwrap(),
        json!({ "type": "function", "function": { "name": "get_weather" } })
    );
}

#[test]
fn tool_def_parameters_are_raw_json() {
    let tool = ToolDef::function(FunctionDef {
        name: "get_weather".into(),
        description: Some("Get weather".into()),
        parameters: Some(json!({ "type": "object", "properties": {} })),
    });
    let v = serde_json::to_value(&tool).unwrap();
    assert_eq!(v["type"], json!("function"));
    assert!(v["function"]["parameters"].is_object());
}

#[test]
fn enum_wire_tokens_are_lowercased() {
    assert_eq!(
        serde_json::to_value(Role::Assistant).unwrap(),
        json!("assistant")
    );
    assert_eq!(
        serde_json::to_value(FinishReason::ToolCalls).unwrap(),
        json!("tool_calls")
    );
    assert_eq!(
        serde_json::to_value(BatchStatus::InProgress).unwrap(),
        json!("in_progress")
    );
}

#[test]
fn chat_response_parses_and_extracts_text() {
    let body = json!({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1700000000_i64,
        "model": "gpt-4o-mini",
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": "Hello!" },
            "finish_reason": "stop"
        }],
        "usage": { "prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5, "cost_usd": 0.0001 }
    });
    let resp: ChatResponse = serde_json::from_value(body).unwrap();
    assert_eq!(resp.first_text(), Some("Hello!"));
    assert_eq!(resp.choices[0].finish_reason, Some(FinishReason::Stop));
    assert_eq!(resp.usage.unwrap().cost_usd, Some(0.0001));
}

#[test]
fn embedding_request_input_string_or_array() {
    let one = EmbeddingRequest::new("emb", "hello");
    assert_eq!(serde_json::to_value(&one).unwrap()["input"], json!("hello"));

    let many = EmbeddingRequest::new("emb", vec!["a".to_string(), "b".to_string()]);
    assert_eq!(
        serde_json::to_value(&many).unwrap()["input"],
        json!(["a", "b"])
    );
}

#[test]
fn batch_create_body_shape() {
    let req = BatchCreateRequest {
        requests: vec![BatchRequestItem {
            custom_id: "req-1".into(),
            body: ChatRequest::new("m", vec![ChatMessage::user("hi")]),
        }],
    };
    let v = serde_json::to_value(&req).unwrap();
    assert_eq!(v["requests"][0]["custom_id"], json!("req-1"));
    assert_eq!(v["requests"][0]["body"]["model"], json!("m"));
}

#[test]
fn pb_module_is_usable() {
    // Proves the prost-generated codegen is genuinely compiled into the crate.
    let u = llmleaf_client::pb::Usage {
        prompt_tokens: 1,
        completion_tokens: 2,
        total_tokens: 3,
        cost_usd: None,
        prompt_tokens_details: None,
        cache_creation_tokens: None,
    };
    assert_eq!(u.total_tokens, 3);
    // Enum casing on the proto side is i32; wire casing lives in the serde types.
    assert_eq!(llmleaf_client::pb::Role::Assistant as i32, 3);
}
