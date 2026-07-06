//! Example: a streaming CLI chat client for a stood-up llmleaf server.
//!
//! A *consumer* of the proxy — it talks the OpenAI-compatible surface over HTTP, like any other
//! client: `POST /v1/chat/completions` by default, or the Responses dialect (`POST /v1/responses`)
//! with `--responses` (or `/api responses` at the prompt). Both drive the same conversation through
//! the same gateway — only the edge dialect differs (SOUL.md principle 3). It reads its connection
//! settings from `llmleaf.toml` via the shared [`common`] helper (where to connect, what bearer to
//! present, which models exist), so there is nothing extra to configure for a local demo.
//!
//! Streaming is the default (SOUL.md principle 4): the request sets `stream: true` and the reply is
//! printed token-by-token as the SSE frames arrive — `chat.completion.chunk` frames terminated by
//! `[DONE]` on the chat surface; typed `response.*` events ending at `response.completed` on the
//! Responses surface.
//!
//! Run it (with the server up via `cargo run -p llmleaf`):
//!
//! ```text
//! cargo run -p llmleaf --example chat                         # default model = first route in llmleaf.toml
//! cargo run -p llmleaf --example chat -- smart                # pick a model by name
//! cargo run -p llmleaf --example chat -- smart --responses    # same conversation over POST /v1/responses
//! ```
//!
//! See [`common`] for the auth/base-url/config-path environment overrides.

#[path = "common/mod.rs"]
mod common;

use common::BoxError;
use futures::StreamExt;
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, Stdout};

/// Which consumer dialect a turn is sent over. Both reach the same models through the same gateway;
/// the conversation history is dialect-agnostic (role-keyed messages parse as chat `messages` and as
/// Responses `input` items alike), so switching mid-conversation keeps the context.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Api {
    /// `POST /v1/chat/completions` — `chat.completion.chunk` SSE frames, `[DONE]`-terminated.
    Chat,
    /// `POST /v1/responses` — typed `response.*` SSE events, ending at the terminal event.
    Responses,
}

impl Api {
    fn label(self) -> &'static str {
        match self {
            Api::Chat => "chat completions (/v1/chat/completions)",
            Api::Responses => "responses (/v1/responses)",
        }
    }
}

/// Session state. The bearer is fixed for the session; the model (`/model`) and the consumer
/// dialect (`/api`) can change between turns.
struct Session {
    base_url: String,
    bearer: String,
    model: String,
    api: Api,
}

#[tokio::main]
async fn main() -> Result<(), BoxError> {
    let conn = common::connect()?;
    // `--responses` picks the Responses dialect; the first non-flag argument is the model.
    let mut api = Api::Chat;
    let mut model_arg: Option<String> = None;
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "--responses" => api = Api::Responses,
            "--chat" => api = Api::Chat,
            other if !other.starts_with("--") && model_arg.is_none() => {
                model_arg = Some(other.to_string());
            }
            other => return Err(format!("unknown argument `{other}` (try --responses)").into()),
        }
    }
    let model = common::pick_model(&conn.config, model_arg)
        .ok_or("no model given and no [[routes]] in the config to default to")?;
    let session = &mut Session {
        base_url: conn.base_url,
        bearer: conn.bearer,
        model,
        api,
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
        // Switch the consumer dialect mid-conversation. The history survives the switch: the same
        // role-keyed messages are chat `messages` and Responses `input` items alike.
        "api" => match arg {
            Some("chat") | Some("chat_completions") | Some("completions") => {
                session.api = Api::Chat;
                format!("  api is now {}", session.api.label())
            }
            Some("responses") => {
                session.api = Api::Responses;
                format!("  api is now {}", session.api.label())
            }
            Some(other) => format!("  unknown api `{other}` — chat | responses"),
            None => format!("  current api: {}", session.api.label()),
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
    \x20   /api [name]     show or switch the consumer dialect: chat | responses\n\
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
fn decode_sse_line(api: Api, line: &str) -> Decoded {
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
        decoded.done = true; // chat-surface sentinel; the Responses stream never sends one
        return decoded;
    }
    let Ok(frame) = serde_json::from_str::<Value>(data) else {
        return decoded;
    };
    match api {
        Api::Chat => decode_chat_frame(&frame, decoded),
        Api::Responses => decode_responses_event(&frame, decoded),
    }
}

/// One OpenAI `chat.completion.chunk` (or the mid-stream error envelope) → its contribution.
fn decode_chat_frame(frame: &Value, mut decoded: Decoded) -> Decoded {
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

/// One Responses typed event (self-describing via `type`) → its contribution. Only the events this
/// REPL renders are read — text deltas, the terminal snapshot (for usage), and errors; everything
/// else (item/part lifecycle, reasoning deltas) is skipped, as the dialect prescribes for
/// unrecognised types. There is no `[DONE]`: the terminal `response.*` event ends the turn.
fn decode_responses_event(event: &Value, mut decoded: Decoded) -> Decoded {
    match event.get("type").and_then(Value::as_str).unwrap_or("") {
        "response.output_text.delta" => {
            if let Some(delta) = event.get("delta").and_then(Value::as_str) {
                decoded.content.push_str(delta);
            }
        }
        "response.completed" | "response.incomplete" => {
            decoded.done = true;
            if let Some(u) = event.get("response").and_then(|r| r.get("usage")) {
                if !u.is_null() {
                    decoded.usage = Some(u.clone());
                }
            }
        }
        "response.failed" => {
            decoded.done = true;
            let message = event
                .get("response")
                .and_then(|r| r.get("error"))
                .and_then(|e| e.get("message"))
                .and_then(Value::as_str)
                .unwrap_or("response failed");
            decoded.error = Some(message.to_string());
        }
        "error" => {
            let message = event
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("stream error");
            decoded.error = Some(message.to_string());
        }
        _ => {}
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
    // The same role-keyed history serves both dialects: chat takes it as `messages`, the Responses
    // surface as `input` items (a role-keyed object with string content is a message item there).
    let (path, body) = match session.api {
        Api::Chat => (
            "/v1/chat/completions",
            json!({ "model": session.model, "messages": messages, "stream": true }),
        ),
        Api::Responses => (
            "/v1/responses",
            json!({ "model": session.model, "input": messages, "stream": true }),
        ),
    };

    let resp = client
        .post(format!("{}{path}", session.base_url))
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
            let decoded = decode_sse_line(session.api, &String::from_utf8_lossy(&raw));

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
/// optional (`total` falls back to prompt+completion). Cost is an llmleaf extension present only when
/// the pricing dataset covers the model (the `echo` provider reports no cost). Prompt-cache hits ride
/// OpenAI-style in `prompt_tokens_details.cached_tokens` (a cache *read*, shown as "cached"); cache
/// *writes* ride in the llmleaf `cache_creation_tokens` extension (shown as "cache-write"). Both are
/// present only when the upstream reported caching, so a non-caching turn's footer is unchanged.
fn usage_footer(usage: &Value) -> Option<String> {
    let n = |k: &str| usage.get(k).and_then(Value::as_u64);
    // The two dialects name the same counters differently: chat says `prompt_tokens` /
    // `completion_tokens` with cache hits under `prompt_tokens_details`; the Responses surface says
    // `input_tokens` / `output_tokens` with hits under `input_tokens_details`. Read either.
    let prompt = n("prompt_tokens").or_else(|| n("input_tokens"));
    let completion = n("completion_tokens").or_else(|| n("output_tokens"));
    let total = n("total_tokens");
    let cost = usage.get("cost_usd").and_then(Value::as_f64);
    let cached = usage
        .get("prompt_tokens_details")
        .or_else(|| usage.get("input_tokens_details"))
        .and_then(|d| d.get("cached_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let cache_write = n("cache_creation_tokens").unwrap_or(0);
    if prompt.is_none() && completion.is_none() && total.is_none() && cost.is_none() {
        return None;
    }
    let (p, c) = (prompt.unwrap_or(0), completion.unwrap_or(0));
    let t = total.unwrap_or(p + c);
    let mut s = format!("  [{p} prompt + {c} completion = {t} tokens");
    if cached > 0 {
        s.push_str(&format!(", {cached} cached"));
    }
    if cache_write > 0 {
        s.push_str(&format!(", {cache_write} cache-write"));
    }
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
         api: {}\n  \
         type a message, or /help for commands. Ctrl-D to exit.\n",
        session.base_url,
        session.model,
        id_label,
        session.api.label(),
    );
    out.write_all(text.as_bytes()).await?;
    out.flush().await?;
    Ok(())
}
