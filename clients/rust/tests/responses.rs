//! End-to-end tests for the Responses dialect (`POST /v1/responses`) through the real
//! [`Client`], against a tiny in-process HTTP mock. These prove the client sends the
//! byte-exact request the wire spec mandates and decodes the responses it gets back —
//! including the typed SSE stream (no `[DONE]`) and the error envelope.
//!
//! The mock is a one-shot blocking server on its own OS thread (no extra dependency): the
//! `reqwest` call runs on the tokio runtime while the mock serves a single canned response
//! and hands back what it received.

use futures::StreamExt;
use llmleaf_client::{
    Client, Error, ResponseItem, ResponsesRequest, ResponsesToolChoice, ResponsesToolDef,
};
use serde_json::json;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc::{self, Receiver};
use std::thread;
use std::time::Duration;

/// What the mock captured from the single request it served.
struct Captured {
    method: String,
    path: String,
    body: String,
}

/// Spawn a one-shot mock that serves `response` (a fully assembled HTTP response) to the
/// first connection, then reports what it received. Returns the base URL to point a
/// [`Client`] at and a channel carrying the captured request.
fn spawn_mock(response: Vec<u8>) -> (String, Receiver<Captured>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        if let Ok((mut stream, _)) = listener.accept() {
            let captured = read_request(&mut stream);
            let _ = stream.write_all(&response);
            let _ = stream.flush();
            let _ = tx.send(captured);
        }
    });
    (format!("http://{addr}"), rx)
}

/// Read one HTTP/1.1 request: the request line, then the body per `Content-Length`.
fn read_request(stream: &mut TcpStream) -> Captured {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 1024];
    let header_end = loop {
        if let Some(pos) = find_subslice(&buf, b"\r\n\r\n") {
            break pos;
        }
        let n = stream.read(&mut tmp).unwrap();
        if n == 0 {
            break buf.len();
        }
        buf.extend_from_slice(&tmp[..n]);
    };

    let head = String::from_utf8_lossy(&buf[..header_end]).into_owned();
    let mut lines = head.lines();
    let request_line = lines.next().unwrap_or_default();
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or_default().to_string();
    let path = parts.next().unwrap_or_default().to_string();

    let content_length = lines
        .filter_map(|l| l.split_once(':'))
        .find(|(k, _)| k.trim().eq_ignore_ascii_case("content-length"))
        .and_then(|(_, v)| v.trim().parse::<usize>().ok())
        .unwrap_or(0);

    let body_start = header_end + 4; // past the "\r\n\r\n" terminator
    let mut body = buf.get(body_start..).unwrap_or_default().to_vec();
    while body.len() < content_length {
        let n = stream.read(&mut tmp).unwrap();
        if n == 0 {
            break;
        }
        body.extend_from_slice(&tmp[..n]);
    }

    Captured {
        method,
        path,
        body: String::from_utf8_lossy(&body).into_owned(),
    }
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

/// Assemble a canned HTTP response with an explicit `Content-Length` and `Connection: close`.
fn http_response(status: u16, reason: &str, content_type: &str, body: &str) -> Vec<u8> {
    format!(
        "HTTP/1.1 {status} {reason}\r\n\
         Content-Type: {content_type}\r\n\
         Content-Length: {len}\r\n\
         Connection: close\r\n\
         \r\n\
         {body}",
        len = body.len(),
    )
    .into_bytes()
}

fn recv(rx: &Receiver<Captured>) -> Captured {
    rx.recv_timeout(Duration::from_secs(5))
        .expect("mock never received a request")
}

#[tokio::test]
async fn responses_non_streaming_round_trip() {
    // A canned completed response with cached-token accounting and store:false.
    let response_body = json!({
        "id": "resp_1",
        "object": "response",
        "created_at": 1_700_000_000_i64,
        "status": "completed",
        "model": "gpt-4o-mini",
        "store": false,
        "output": [
            {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "content": [{ "type": "output_text", "text": "It's 21C.", "annotations": [] }]
            }
        ],
        "usage": {
            "input_tokens": 42,
            "input_tokens_details": { "cached_tokens": 16 },
            "output_tokens": 6,
            "total_tokens": 48
        }
    })
    .to_string();
    let (base, rx) = spawn_mock(http_response(200, "OK", "application/json", &response_body));

    let client = Client::new(base, "sk-test").unwrap();
    let mut request = ResponsesRequest::new(
        "gpt-4o-mini",
        vec![
            ResponseItem::user("What's the weather in Paris?"),
            ResponseItem::function_call("call_1", "get_weather", "{\"city\":\"Paris\"}"),
            ResponseItem::function_call_output("call_1", "{\"temp_c\":21}"),
        ],
    );
    request.tools = vec![ResponsesToolDef::function("get_weather")];
    request.tool_choice = Some(ResponsesToolChoice::mode("auto"));

    let resp = client.responses(request).await.unwrap();
    assert_eq!(resp.status, "completed");
    assert_eq!(resp.store, Some(false));
    assert_eq!(resp.output_text(), "It's 21C.");
    let usage = resp.usage.as_ref().unwrap();
    assert_eq!(usage.input_tokens, 42);
    assert_eq!(usage.cached_tokens(), 16);

    // Assert the exact request the client put on the wire.
    let captured = recv(&rx);
    assert_eq!(captured.method, "POST");
    assert_eq!(captured.path, "/v1/responses");
    let sent: serde_json::Value = serde_json::from_str(&captured.body).unwrap();
    // Non-streaming call must not force a `stream` field.
    assert!(sent.get("stream").is_none());
    assert_eq!(sent["model"], json!("gpt-4o-mini"));
    // Item array: message role-keyed with no "type"; typed items carry "type".
    assert_eq!(
        sent["input"],
        json!([
            { "role": "user", "content": "What's the weather in Paris?" },
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "get_weather",
                "arguments": "{\"city\":\"Paris\"}"
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "{\"temp_c\":21}"
            }
        ])
    );
    // Flat tool + bare-string tool_choice.
    assert_eq!(sent["tools"], json!([{ "type": "function", "name": "get_weather" }]));
    assert_eq!(sent["tool_choice"], json!("auto"));
}

#[tokio::test]
async fn responses_streaming_round_trip() {
    let sse = concat!(
        "event: response.created\n",
        "data: {\"type\":\"response.created\",\"sequence_number\":0,\"response\":{\"id\":\"resp_1\",\"object\":\"response\",\"status\":\"in_progress\",\"model\":\"m\"}}\n\n",
        "event: response.output_text.delta\n",
        "data: {\"type\":\"response.output_text.delta\",\"sequence_number\":1,\"delta\":\"Hi\"}\n\n",
        "event: response.output_text.delta\n",
        "data: {\"type\":\"response.output_text.delta\",\"sequence_number\":2,\"delta\":\"!\"}\n\n",
        "event: response.completed\n",
        "data: {\"type\":\"response.completed\",\"sequence_number\":3,\"response\":{\"id\":\"resp_1\",\"object\":\"response\",\"status\":\"completed\",\"model\":\"m\",\"usage\":{\"input_tokens\":5,\"output_tokens\":2,\"total_tokens\":7}}}\n\n",
    );
    let (base, rx) = spawn_mock(http_response(200, "OK", "text/event-stream", sse));

    let client = Client::new(base, "sk-test").unwrap();
    let mut stream = client
        .responses_stream(ResponsesRequest::new("m", "Say hi."))
        .await
        .unwrap();

    let mut kinds = Vec::new();
    let mut text = String::new();
    let mut terminal_input_tokens = None;
    while let Some(event) = stream.next().await {
        let event = event.unwrap();
        if let Some(delta) = event.output_text_delta() {
            text.push_str(delta);
        }
        if let Some(snapshot) = event.terminal_response() {
            terminal_input_tokens = snapshot.usage.as_ref().map(|u| u.input_tokens);
        }
        kinds.push(event.kind);
    }

    assert_eq!(
        kinds,
        vec![
            "response.created",
            "response.output_text.delta",
            "response.output_text.delta",
            "response.completed",
        ]
    );
    assert_eq!(text, "Hi!");
    assert_eq!(terminal_input_tokens, Some(5));

    // The streaming call forces stream:true.
    let captured = recv(&rx);
    let sent: serde_json::Value = serde_json::from_str(&captured.body).unwrap();
    assert_eq!(sent["stream"], json!(true));
    assert_eq!(sent["input"], json!("Say hi."));
}

#[tokio::test]
async fn responses_error_envelope_maps_to_api_error() {
    let (base, _rx) = spawn_mock(http_response(
        403,
        "Forbidden",
        "application/json",
        r#"{"error":{"message":"model not allowed"}}"#,
    ));

    let client = Client::new(base, "sk-test").unwrap();
    let err = client
        .responses(ResponsesRequest::new("m", "hi"))
        .await
        .unwrap_err();
    match err {
        Error::Api { status, message } => {
            assert_eq!(status, 403);
            assert_eq!(message, "model not allowed");
        }
        other => panic!("expected Error::Api, got {other:?}"),
    }
}
