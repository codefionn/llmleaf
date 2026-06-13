//! Example: a streaming CLI chat client for a stood-up llmleaf server.
//!
//! A *consumer* of the proxy — it talks the OpenAI-compatible surface (`POST /v1/chat/completions`)
//! over HTTP, like any other client. It reads its connection settings from `llmleaf.toml` via the
//! shared [`common`] helper (where to connect, what bearer to present, which models exist), so there
//! is nothing extra to configure for a local demo.
//!
//! Streaming is the default (SOUL.md principle 4): the request sets `stream: true` and the reply is
//! printed token-by-token as the SSE `chat.completion.chunk` frames arrive.
//!
//! Run it (with the server up via `cargo run -p llmleaf`):
//!
//! ```text
//! cargo run -p llmleaf --example chat            # default model = first route in llmleaf.toml
//! cargo run -p llmleaf --example chat -- smart   # pick a model by name
//! ```
//!
//! See [`common`] for the auth/base-url/config-path environment overrides.

#[path = "common/mod.rs"]
mod common;

use common::BoxError;
use futures::StreamExt;
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, Stdout};

/// Session state. The bearer is fixed for the session; only the model changes (via `/model`).
struct Session {
    base_url: String,
    bearer: String,
    model: String,
}

#[tokio::main]
async fn main() -> Result<(), BoxError> {
    let conn = common::connect()?;
    let model = common::pick_model(&conn.config, std::env::args().nth(1))
        .ok_or("no model given and no [[routes]] in the config to default to")?;
    let session = &mut Session {
        base_url: conn.base_url,
        bearer: conn.bearer,
        model,
    };

    let mut out = tokio::io::stdout();
    banner(&mut out, session, &conn.id_label, conn.config.routes.len()).await?;

    // The REPL keeps the running conversation so the model sees prior turns (the proxy is stateless —
    // context lives with the client, here in `messages`).
    let client = reqwest::Client::new();
    let mut messages: Vec<Value> = Vec::new();
    let mut lines = BufReader::new(tokio::io::stdin()).lines();

    loop {
        out.write_all(b"\nyou> ").await?;
        out.flush().await?;

        let Some(line) = lines.next_line().await? else {
            out.write_all(b"\n").await?; // Ctrl-D / EOF
            break;
        };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        // Slash commands are handled locally and never sent upstream.
        if let Some(cmd) = line.strip_prefix('/') {
            match handle_command(cmd, session, &mut messages, &conn.config, &mut out).await? {
                Flow::Continue => continue,
                Flow::Quit => break,
            }
        }

        messages.push(json!({ "role": "user", "content": line }));
        match send(&client, session, &messages, &mut out).await {
            Ok(reply) if !reply.is_empty() => {
                messages.push(json!({ "role": "assistant", "content": reply }));
            }
            // The reply carried no text (e.g. a tool-call-only response we don't render). Keep the
            // user turn but don't append an empty assistant message that some providers reject.
            Ok(_) => {
                out.write_all(b"  (no text content returned)\n").await?;
                out.flush().await?;
            }
            Err(e) => {
                // The turn failed; drop the user message so history stays consistent for a retry.
                messages.pop();
                out.write_all(format!("\n  ! request failed: {e}\n").as_bytes())
                    .await?;
                out.flush().await?;
            }
        }
    }

    Ok(())
}

enum Flow {
    Continue,
    Quit,
}

/// Handle a `/command`. Returns whether the REPL should continue or quit.
async fn handle_command(
    cmd: &str,
    session: &mut Session,
    messages: &mut Vec<Value>,
    config: &llmleaf_core::Config,
    out: &mut Stdout,
) -> Result<Flow, BoxError> {
    let mut parts = cmd.splitn(2, char::is_whitespace);
    let name = parts.next().unwrap_or("");
    let arg = parts.next().map(str::trim).filter(|s| !s.is_empty());

    let reply = match name {
        "quit" | "exit" | "q" => return Ok(Flow::Quit),
        "help" | "h" | "?" => HELP.to_string(),
        "reset" | "clear" => {
            messages.clear();
            "  (conversation history cleared)".to_string()
        }
        "model" => match arg {
            Some(m) => {
                session.model = m.to_string();
                format!("  model is now `{}`", session.model)
            }
            None => format!("  current model: `{}`", session.model),
        },
        "models" => {
            let mut s = String::from("  routable models (from the config):");
            for r in &config.routes {
                s.push_str("\n    - ");
                s.push_str(&r.model);
            }
            if config.routes.is_empty() {
                s.push_str("\n    (none)");
            }
            s
        }
        "system" => match arg {
            Some(text) => {
                set_system(messages, text);
                "  (system prompt set)".to_string()
            }
            None => "  usage: /system <prompt text>".to_string(),
        },
        other => format!("  unknown command `/{other}` — try /help"),
    };

    out.write_all(reply.as_bytes()).await?;
    out.write_all(b"\n").await?;
    out.flush().await?;
    Ok(Flow::Continue)
}

const HELP: &str = "  commands:\n\
    \x20   /model [name]   show or switch the model\n\
    \x20   /models         list routable models from the config\n\
    \x20   /system <text>  set a system prompt for the conversation\n\
    \x20   /reset          clear the conversation history\n\
    \x20   /help           show this help\n\
    \x20   /quit           exit (also Ctrl-D)";

/// Insert or replace the leading system message.
fn set_system(messages: &mut Vec<Value>, text: &str) {
    let sys = json!({ "role": "system", "content": text });
    if messages.first().and_then(|m| m.get("role")) == Some(&json!("system")) {
        messages[0] = sys;
    } else {
        messages.insert(0, sys);
    }
}

/// What one decoded SSE line contributes to the turn. Pure/sync so the same decoder runs both inside
/// the receive loop and on any unterminated remainder after the stream ends.
#[derive(Default)]
struct Decoded {
    content: String,
    usage: Option<Value>,
    error: Option<String>,
    done: bool,
}

/// Decode one SSE line (`data: <json>`). Anything that isn't a recognisable data frame yields an
/// empty `Decoded` (skipped): event boundaries, other SSE fields, or a frame we can't parse.
fn decode_sse_line(line: &str) -> Decoded {
    let mut decoded = Decoded::default();
    // axum writes the field as `data: ` (with a trailing space); the `trim()` below absorbs it, so
    // stripping the no-space `data:` prefix is load-bearing together with that trim.
    let Some(data) = line.trim_end().strip_prefix("data:") else {
        return decoded;
    };
    let data = data.trim();
    if data.is_empty() {
        return decoded;
    }
    if data == "[DONE]" {
        decoded.done = true;
        return decoded;
    }
    let Ok(frame) = serde_json::from_str::<Value>(data) else {
        return decoded;
    };
    // A mid-stream error frame (e.g. the upstream failed after headers were sent).
    if let Some(err) = frame
        .get("error")
        .and_then(|e| e.get("message"))
        .and_then(Value::as_str)
    {
        decoded.error = Some(err.to_string());
        return decoded;
    }
    if let Some(choices) = frame.get("choices").and_then(Value::as_array) {
        for ch in choices {
            if let Some(delta) = ch
                .get("delta")
                .and_then(|d| d.get("content"))
                .and_then(Value::as_str)
            {
                decoded.content.push_str(delta);
            }
        }
    }
    if let Some(u) = frame.get("usage") {
        if !u.is_null() {
            decoded.usage = Some(u.clone());
        }
    }
    decoded
}

/// Send one turn and stream the assistant's reply to stdout. Returns the full reply text so the
/// caller can append it to the conversation history.
async fn send(
    client: &reqwest::Client,
    session: &Session,
    messages: &[Value],
    out: &mut Stdout,
) -> Result<String, BoxError> {
    let body = json!({
        "model": session.model,
        "messages": messages,
        "stream": true,
    });

    let resp = client
        .post(format!("{}/v1/chat/completions", session.base_url))
        .bearer_auth(&session.bearer)
        .json(&body)
        .send()
        .await?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return Err(format!("{status}: {}", common::error_detail(&text)).into());
    }

    out.write_all(b"asst> ").await?;
    out.flush().await?;

    let mut stream = resp.bytes_stream();
    // Raw byte buffer: we only decode *complete* lines, so a multibyte char split across two network
    // chunks is never decoded mid-sequence. Each SSE frame is one `data:` line (the server emits
    // compact, newline-free JSON), terminated by `\n`.
    let mut buf: Vec<u8> = Vec::new();
    let mut reply = String::new();
    let mut usage: Option<Value> = None;
    let mut ended = false;

    'recv: while !ended {
        match stream.next().await {
            Some(chunk) => buf.extend_from_slice(&chunk?),
            // Stream closed. Append a terminator so an unterminated trailing frame (abrupt close /
            // non-compliant peer) is decoded by the same path below, then exit after this pass.
            None => {
                buf.push(b'\n');
                ended = true;
            }
        }

        while let Some(nl) = buf.iter().position(|&b| b == b'\n') {
            let raw: Vec<u8> = buf.drain(..=nl).collect();
            let decoded = decode_sse_line(&String::from_utf8_lossy(&raw));

            if let Some(err) = decoded.error {
                out.write_all(b"\n").await?;
                out.flush().await?;
                return Err(format!("stream error: {err}").into());
            }
            if !decoded.content.is_empty() {
                out.write_all(decoded.content.as_bytes()).await?;
                out.flush().await?;
                reply.push_str(&decoded.content);
            }
            if decoded.usage.is_some() {
                usage = decoded.usage;
            }
            if decoded.done {
                break 'recv;
            }
        }
    }

    out.write_all(b"\n").await?;
    if let Some(footer) = usage.as_ref().and_then(usage_footer) {
        out.write_all(footer.as_bytes()).await?;
        out.write_all(b"\n").await?;
    }
    out.flush().await?;
    Ok(reply)
}

/// Format the usage object the server appends to the stream into a one-line footer. Each field is
/// optional (`total` falls back to prompt+completion); cost is an llmleaf extension present only when
/// the pricing dataset covers the model (the `echo` provider reports no cost).
fn usage_footer(usage: &Value) -> Option<String> {
    let n = |k: &str| usage.get(k).and_then(Value::as_u64);
    let prompt = n("prompt_tokens");
    let completion = n("completion_tokens");
    let total = n("total_tokens");
    let cost = usage.get("cost_usd").and_then(Value::as_f64);
    if prompt.is_none() && completion.is_none() && total.is_none() && cost.is_none() {
        return None;
    }
    let (p, c) = (prompt.unwrap_or(0), completion.unwrap_or(0));
    let t = total.unwrap_or(p + c);
    let mut s = format!("  [{p} prompt + {c} completion = {t} tokens");
    if let Some(cost) = cost {
        s.push_str(&format!(", ${cost:.6}"));
    }
    s.push(']');
    Some(s)
}

async fn banner(
    out: &mut Stdout,
    session: &Session,
    id_label: &str,
    route_count: usize,
) -> Result<(), BoxError> {
    let text = format!(
        "llmleaf chat — connected to {}\n  \
         model: {}    key: {}    ({route_count} route(s) configured)\n  \
         type a message, or /help for commands. Ctrl-D to exit.\n",
        session.base_url, session.model, id_label,
    );
    out.write_all(text.as_bytes()).await?;
    out.flush().await?;
    Ok(())
}
