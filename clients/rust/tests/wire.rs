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

// ---------------------------------------------------------------------------
// Responses (POST /v1/responses)
// ---------------------------------------------------------------------------

#[test]
fn responses_input_is_bare_string_for_one_message() {
    let req = ResponsesRequest::new("gpt-4o-mini", "Say hi.");
    let v = serde_json::to_value(&req).unwrap();
    assert_eq!(
        v,
        json!({ "model": "gpt-4o-mini", "input": "Say hi." })
    );
}

#[test]
fn responses_request_item_array_flat_tools_and_reasoning_replay() {
    // A multi-turn replay: a user message, a reasoning item (summary + content), the
    // model's function_call, and the caller's function_call_output — plus a flat tool
    // and a flat named tool_choice. This exercises every dialect quirk in one body.
    let mut req = ResponsesRequest::new(
        "gpt-4o-mini",
        vec![
            ResponseItem::Message(ResponseMessageItem::user(vec![
                ResponseContentPart::input_text("What's the weather?"),
                ResponseContentPart::input_image("https://x/y.png"),
            ])),
            ResponseItem::Reasoning(ResponseReasoningItem {
                id: Some("rs_1".into()),
                summary: vec![ResponseReasoningText::new("Think about weather.")],
                content: vec![ResponseReasoningText::new("The user wants the weather.")],
                encrypted_content: Some("opaque-blob".into()),
            }),
            ResponseItem::function_call("call_1", "get_weather", "{\"city\":\"Paris\"}"),
            ResponseItem::function_call_output("call_1", "{\"temp_c\":21}"),
        ],
    );
    req.tools = vec![ResponsesToolDef {
        kind: "function".into(),
        name: "get_weather".into(),
        description: Some("Look up the weather".into()),
        parameters: Some(json!({ "type": "object", "properties": {} })),
        strict: Some(false),
    }];
    req.tool_choice = Some(ResponsesToolChoice::named("get_weather"));

    let v = serde_json::to_value(&req).unwrap();

    // Message item: role-keyed, NO "type"; input_image.image_url is a bare string.
    assert_eq!(
        v["input"][0],
        json!({
            "role": "user",
            "content": [
                { "type": "input_text", "text": "What's the weather?" },
                { "type": "input_image", "image_url": "https://x/y.png" }
            ]
        })
    );
    assert!(v["input"][0].get("type").is_none());

    // Reasoning item: summary entries -> "summary_text", content -> "reasoning_text".
    assert_eq!(
        v["input"][1],
        json!({
            "type": "reasoning",
            "id": "rs_1",
            "summary": [{ "type": "summary_text", "text": "Think about weather." }],
            "content": [{ "type": "reasoning_text", "text": "The user wants the weather." }],
            "encrypted_content": "opaque-blob"
        })
    );

    // Typed items carry their "type".
    assert_eq!(
        v["input"][2],
        json!({
            "type": "function_call",
            "call_id": "call_1",
            "name": "get_weather",
            "arguments": "{\"city\":\"Paris\"}"
        })
    );
    assert_eq!(
        v["input"][3],
        json!({
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "{\"temp_c\":21}"
        })
    );

    // Tools are FLAT (type/name/parameters at the top level, no nested "function"), and
    // "parameters" is a real object, not a stringified schema.
    assert_eq!(v["tools"][0]["type"], json!("function"));
    assert_eq!(v["tools"][0]["name"], json!("get_weather"));
    assert!(v["tools"][0]["parameters"].is_object());
    assert!(v["tools"][0].get("function").is_none());

    // Flat named tool_choice: {"type":"function","name":"..."} — no nested "function".
    assert_eq!(
        v["tool_choice"],
        json!({ "type": "function", "name": "get_weather" })
    );
}

#[test]
fn responses_output_text_part_emits_empty_annotations() {
    let part = ResponseContentPart::output_text("hello");
    assert_eq!(
        serde_json::to_value(&part).unwrap(),
        json!({ "type": "output_text", "text": "hello", "annotations": [] })
    );
}

#[test]
fn responses_tool_choice_mode_is_bare_string() {
    assert_eq!(
        serde_json::to_value(ResponsesToolChoice::mode("auto")).unwrap(),
        json!("auto")
    );
}

#[test]
fn responses_extra_merges_at_top_level() {
    let mut req = ResponsesRequest::new("m", "hi");
    let mut extra = serde_json::Map::new();
    extra.insert("metadata".into(), json!({ "trace": "abc" }));
    req.extra = Some(extra);
    let v = serde_json::to_value(&req).unwrap();
    assert_eq!(v["metadata"], json!({ "trace": "abc" }));
    assert!(v.get("extra").is_none());
}

#[test]
fn responses_response_decodes_output_usage_and_store() {
    let body = json!({
        "id": "resp_1",
        "object": "response",
        "created_at": 1_700_000_000_i64,
        "status": "completed",
        "model": "gpt-4o-mini",
        "store": false,
        "output": [
            {
                "type": "reasoning",
                "id": "rs_1",
                "summary": [{ "type": "summary_text", "text": "thinking" }],
                "content": []
            },
            {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "content": [
                    { "type": "output_text", "text": "Hello!", "annotations": [] }
                ]
            }
        ],
        "usage": {
            "input_tokens": 20,
            "input_tokens_details": { "cached_tokens": 12 },
            "output_tokens": 5,
            "output_tokens_details": { "reasoning_tokens": 3 },
            "total_tokens": 25
        }
    });
    let resp: ResponsesResponse = serde_json::from_value(body).unwrap();
    assert_eq!(resp.status, "completed");
    assert_eq!(resp.store, Some(false));
    assert_eq!(resp.output_text(), "Hello!");
    let usage = resp.usage.as_ref().unwrap();
    assert_eq!(usage.input_tokens, 20);
    assert_eq!(usage.cached_tokens(), 12);
    assert_eq!(usage.reasoning_tokens(), 3);

    // The reasoning item decoded into its typed variant.
    assert!(matches!(resp.output[0], ResponseItem::Reasoning(_)));
}

#[test]
fn responses_item_decodes_role_keyed_message_without_type() {
    // A bare role-keyed object (no "type") is a message item.
    let item: ResponseItem =
        serde_json::from_value(json!({ "role": "user", "content": "hi" })).unwrap();
    match item {
        ResponseItem::Message(m) => {
            assert_eq!(m.role, "user");
            assert_eq!(m.content, Some(ResponseContent::Text("hi".into())));
        }
        other => panic!("expected Message, got {other:?}"),
    }
}

#[test]
fn responses_item_keeps_unknown_type_verbatim() {
    // A future item type this SDK doesn't model round-trips through `Other`.
    let raw = json!({ "type": "web_search_call", "id": "ws_1", "status": "completed" });
    let item: ResponseItem = serde_json::from_value(raw.clone()).unwrap();
    assert!(matches!(item, ResponseItem::Other(_)));
    assert_eq!(serde_json::to_value(&item).unwrap(), raw);
}

#[test]
fn responses_failed_response_carries_error_body() {
    let body = json!({
        "id": "resp_err",
        "object": "response",
        "status": "failed",
        "model": "m",
        "error": { "message": "boom", "code": "server_error" }
    });
    let resp: ResponsesResponse = serde_json::from_value(body).unwrap();
    assert_eq!(resp.status, "failed");
    assert_eq!(resp.error.as_ref().unwrap().message, "boom");
    assert_eq!(resp.error.as_ref().unwrap().code.as_deref(), Some("server_error"));
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
