//! Decisions transport tests against a small one-request HTTP mock.

use llmleaf_client::{Client, DecisionsRequest, Error};
use serde_json::json;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc::{self, Receiver};
use std::thread;
use std::time::Duration;

struct Captured {
    method: String,
    path: String,
    authorization: Option<String>,
    body: String,
}

fn spawn_mock(response: Vec<u8>) -> (String, Receiver<Captured>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        if let Ok((mut stream, _)) = listener.accept() {
            let captured = read_request(&mut stream);
            stream.write_all(&response).unwrap();
            stream.flush().unwrap();
            tx.send(captured).unwrap();
        }
    });
    (format!("http://{addr}"), rx)
}

fn read_request(stream: &mut TcpStream) -> Captured {
    let mut buf = Vec::new();
    let mut chunk = [0_u8; 1024];
    let header_end = loop {
        if let Some(pos) = buf.windows(4).position(|part| part == b"\r\n\r\n") {
            break pos;
        }
        let n = stream.read(&mut chunk).unwrap();
        if n == 0 {
            break buf.len();
        }
        buf.extend_from_slice(&chunk[..n]);
    };
    let head = String::from_utf8_lossy(&buf[..header_end]).into_owned();
    let mut lines = head.lines();
    let mut request = lines.next().unwrap_or_default().split_whitespace();
    let method = request.next().unwrap_or_default().to_string();
    let path = request.next().unwrap_or_default().to_string();
    let headers: Vec<_> = lines.filter_map(|line| line.split_once(':')).collect();
    let authorization = headers
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case("authorization"))
        .map(|(_, value)| value.trim().to_string());
    let length = headers
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case("content-length"))
        .and_then(|(_, value)| value.trim().parse::<usize>().ok())
        .unwrap_or(0);
    let mut body = buf[header_end + 4..].to_vec();
    while body.len() < length {
        let n = stream.read(&mut chunk).unwrap();
        if n == 0 {
            break;
        }
        body.extend_from_slice(&chunk[..n]);
    }
    Captured {
        method,
        path,
        authorization,
        body: String::from_utf8(body).unwrap(),
    }
}

fn response(status: u16, reason: &str, body: &str) -> Vec<u8> {
    format!("HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).into_bytes()
}

#[test]
fn decisions_request_keeps_json_and_protects_canonical_fields() {
    let mut request = DecisionsRequest::new(
        "jev",
        json!({"turn": 2, "context": ["a", "b"]}),
        serde_json::Map::from_iter([(
            "route".to_string(),
            json!({"type": "choice", "instructions": "Which route?", "criteria": {"left": null, "right": null}}),
        )]),
    );
    request
        .extra
        .insert("metadata".to_string(), json!({"trace": true}));
    request.extra.insert("model".to_string(), json!("wrong"));
    request.extra.insert("state".to_string(), json!("wrong"));
    let wire = serde_json::to_value(request).unwrap();
    assert_eq!(wire["model"], json!("jev"));
    assert_eq!(wire["state"]["context"], json!(["a", "b"]));
    assert_eq!(
        wire["questions"]["route"]["criteria"],
        json!({"left": null, "right": null})
    );
    assert_eq!(wire["metadata"], json!({"trace": true}));

    let decoded: DecisionsRequest = serde_json::from_value(json!({
        "model": "jev",
        "state": {"turn": 3},
        "questions": {"route": {"type": "noul", "instructions": "Proceed?"}},
        "metadata": {"trace": "abc"}
    }))
    .unwrap();
    assert_eq!(decoded.extra["metadata"], json!({"trace": "abc"}));
}

#[tokio::test]
async fn decisions_posts_auth_and_decodes_structured_response() {
    let body = json!({
        "id": "dec_1",
        "model": "jev",
        "provider": "openrouter",
        "answers": {"route": {"type": "choice", "choice": "left", "probabilities": {"left": 0.7, "right": 0.3}}},
        "usage": {"input_tokens": 12, "output_tokens": 4, "cost": 0.0, "cached": {"tokens": 8}},
        "trace_id": "tr_1"
    }).to_string();
    let (base, received) = spawn_mock(response(200, "OK", &body));
    let client = Client::new(base, "sk-test").unwrap();
    let result = client
        .decisions(DecisionsRequest::new(
            "jev",
            json!({"a": 1}),
            serde_json::Map::from_iter([(
                "route".to_string(),
                json!({"type": "noul", "instructions": "Should we proceed?"}),
            )]),
        ))
        .await
        .unwrap();
    assert_eq!(result.id.as_deref(), Some("dec_1"));
    assert_eq!(result.provider.as_deref(), Some("openrouter"));
    assert_eq!(result.answers["route"]["probabilities"]["left"], json!(0.7));
    assert_eq!(result.usage.cost, Some(0.0));
    assert_eq!(result.usage.extra["cached"], json!({"tokens": 8}));
    assert_eq!(result.extra["trace_id"], json!("tr_1"));
    let sent = received.recv_timeout(Duration::from_secs(5)).unwrap();
    assert_eq!(sent.method, "POST");
    assert_eq!(sent.path, "/v1/decisions");
    assert_eq!(sent.authorization.as_deref(), Some("Bearer sk-test"));
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&sent.body).unwrap()["state"],
        json!({"a": 1})
    );
}

#[tokio::test]
async fn decisions_surfaces_error_envelope() {
    let (base, _) = spawn_mock(response(
        403,
        "Forbidden",
        r#"{"error":{"message":"model not allowed"}}"#,
    ));
    let client = Client::new(base, "sk-test").unwrap();
    let error = client
        .decisions(DecisionsRequest::new(
            "m",
            json!({}),
            serde_json::Map::new(),
        ))
        .await
        .unwrap_err();
    assert!(
        matches!(error, Error::Api { status: 403, ref message } if message == "model not allowed")
    );
}
