//! Example: a capability *probe* — exercise every modality a route can serve and report a matrix.
//!
//! A single *consumer* of the proxy that runs one battery per logical model: **chat**, **tool calls**,
//! **text-to-speech**, and **speech-to-text**, each against the OpenAI-compatible surface the other
//! examples speak one-at-a-time (`/v1/chat/completions`, `/v1/audio/speech`,
//! `/v1/audio/transcriptions`). It is the "does this provider actually do X?" smoke test.
//!
//! Each capability lands in one of three states, so a clean run distinguishes a real fault from a
//! provider that simply lacks the modality:
//!   - **PASS** — the surface answered and the answer was well-formed.
//!   - **n/a**  — every target in the route returned `Unsupported` (HTTP 502, "does not support …");
//!     the route is healthy, this modality just isn't wired for it. Not a failure.
//!   - **FAIL** — anything else (transport error, bad upstream, empty/wrong answer). Detail is shown.
//!
//! Against the bundled `llmleaf.toml` the `demo` route (the built-in `echo` provider) runs fully
//! offline: chat/tts/stt PASS, and **tools FAIL** — `echo` mirrors text and never emits a tool call,
//! which is exactly the gap a probe should surface. Point it at a real route for the real matrix.
//!
//! Run it (with the server up via `cargo run -p llmleaf`); `--help` lists every flag:
//!
//! ```text
//! cargo run -p llmleaf --example probe                  # default model (first route)
//! cargo run -p llmleaf --example probe -- smart         # probe one model by name
//! cargo run -p llmleaf --example probe -- smart gpt-4o  # probe several
//! cargo run -p llmleaf --example probe -- --all         # probe every route in the config
//! cargo run -p llmleaf --example probe -- --only chat,tools smart   # a subset of capabilities
//! cargo run -p llmleaf --example probe -- --help        # full usage
//! ```
//!
//! TTS and STT are usually served by *different* models than chat (e.g. `tts-1`,
//! `gpt-4o-transcribe`), so each capability takes its own model override; whatever is unset falls back
//! to the base (positional) model, and tools default to the chat model. This probes a whole stack in
//! one run — with no base model at all when every probed capability is overridden:
//!
//! ```text
//! cargo run -p llmleaf --example probe -- gpt-4o --tts-model tts-1 --stt-model gpt-4o-transcribe
//! cargo run -p llmleaf --example probe -- --only tts,stt --tts-model tts-1 --stt-model gpt-4o-transcribe
//! ```
//!
//! Exit status is non-zero if any capability FAILED (n/a does not count), so it drops into CI as-is.
//! Auth/base-url/config follow the same `LLMLEAF_*` overrides documented in [`common`].

#[path = "common/mod.rs"]
mod common;

use std::fmt::Write as _;
use std::io::IsTerminal;
use std::time::Instant;

use clap::Parser;
use common::BoxError;
use reqwest::multipart::{Form, Part};
use reqwest::Client;
use serde_json::{json, Value};

/// Capability probe for a stood-up llmleaf server: chat, tool calls, TTS, STT.
#[derive(Parser)]
#[command(name = "probe")]
struct Args {
    /// Models (routes) to probe. Omit to probe the first configured route (or $LLMLEAF_MODEL).
    models: Vec<String>,

    /// Probe every route declared in the config instead of naming models.
    #[arg(short, long)]
    all: bool,

    /// Restrict to a comma-separated subset of capabilities: chat,tools,tts,stt [default: all].
    #[arg(short, long, value_delimiter = ',')]
    only: Vec<Capability>,

    /// Model for the chat probe (and tools, unless --tools-model). Default: the base model.
    #[arg(long, env = "LLMLEAF_CHAT_MODEL")]
    chat_model: Option<String>,

    /// Model for the tool-call probe. Default: the chat model (then the base model).
    #[arg(long, env = "LLMLEAF_TOOLS_MODEL")]
    tools_model: Option<String>,

    /// Model for the text-to-speech probe. Default: the base model.
    #[arg(long, env = "LLMLEAF_TTS_MODEL")]
    tts_model: Option<String>,

    /// Model for the speech-to-text probe. Default: the base model.
    #[arg(long, env = "LLMLEAF_STT_MODEL")]
    stt_model: Option<String>,

    /// Voice for the text-to-speech probe.
    #[arg(short, long, env = "LLMLEAF_VOICE", default_value = "alloy")]
    voice: String,

    /// Audio `response_format` for the text-to-speech probe. Defaults to OpenAI's `mp3`; some
    /// targets serve only one format (Groq's Orpheus is `wav`-only), so it must be selectable.
    #[arg(long, env = "LLMLEAF_TTS_FORMAT", default_value = "mp3")]
    tts_format: String,

    /// Per-request timeout, in seconds.
    #[arg(short, long, default_value_t = 60)]
    timeout: u64,
}

impl Args {
    /// The per-capability model override, if the user set one for this capability. Tools inherit the
    /// chat model when no tools-specific override is given (a tool call is a chat turn).
    fn override_for(&self, cap: Capability) -> Option<&str> {
        match cap {
            Capability::Chat => self.chat_model.as_deref(),
            Capability::Tools => self.tools_model.as_deref().or(self.chat_model.as_deref()),
            Capability::Tts => self.tts_model.as_deref(),
            Capability::Stt => self.stt_model.as_deref(),
        }
    }

    /// Whether any per-capability override was given at all (affects whether a base model is required).
    fn any_override(&self) -> bool {
        self.chat_model.is_some()
            || self.tools_model.is_some()
            || self.tts_model.is_some()
            || self.stt_model.is_some()
    }
}

/// The four probed capabilities, in display order.
#[derive(Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum Capability {
    Chat,
    Tools,
    Tts,
    Stt,
}

impl Capability {
    const ALL: [Capability; 4] = [
        Capability::Chat,
        Capability::Tools,
        Capability::Tts,
        Capability::Stt,
    ];

    /// Short fixed-width label used in both the per-model lines and the summary header.
    fn label(self) -> &'static str {
        match self {
            Capability::Chat => "chat",
            Capability::Tools => "tools",
            Capability::Tts => "tts",
            Capability::Stt => "stt",
        }
    }
}

/// The verdict for one capability against one model.
enum Status {
    Pass,
    /// The route exists but no target serves this modality (provider returned `Unsupported`).
    Unsupported,
    Fail,
}

struct Outcome {
    cap: Capability,
    status: Status,
    /// A one-line human note (the reply snippet, byte count, or failure reason).
    detail: String,
    elapsed_ms: u128,
}

#[tokio::main]
async fn main() -> Result<(), BoxError> {
    let args = Args::parse();
    let conn = common::connect()?;

    // Which capabilities to run, in canonical order, honouring `--only` (deduped, order-stable).
    let caps: Vec<Capability> = if args.only.is_empty() {
        Capability::ALL.to_vec()
    } else {
        Capability::ALL
            .into_iter()
            .filter(|c| args.only.contains(c))
            .collect()
    };

    // The base models, one row each: every route with `--all`, the named ones, else the single
    // default. `None` is a base-less row — valid only when every probed capability is overridden, so a
    // pure TTS/STT stack (different models per modality) needs no chat route to anchor it.
    let bases: Vec<Option<String>> = if args.all {
        if conn.config.routes.is_empty() {
            return Err("--all: the config declares no [[routes]] to probe".into());
        }
        conn.config
            .routes
            .iter()
            .map(|r| Some(r.model.clone()))
            .collect()
    } else if !args.models.is_empty() {
        args.models.iter().cloned().map(Some).collect()
    } else {
        // A base model is only needed if some *selected* capability has no override of its own; when
        // every probed capability is overridden, the run is base-less (header "(per-capability)") and
        // the default route is never pulled in as dead weight.
        let need_base = caps.iter().any(|&c| args.override_for(c).is_none());
        match common::pick_model(&conn.config, None) {
            Some(m) if need_base => vec![Some(m)],
            _ if !need_base => vec![None],
            // Some selected capability needs a base but there is none; if anything else is overridden
            // we still run (those caps are skipped with a note), otherwise there is nothing to do.
            _ if args.any_override() => vec![None],
            _ => return Err(
                "no model: name one (`probe smart`), pass --all, set LLMLEAF_MODEL, give per-capability \
                 --tts-model/--stt-model/…, or add a [[routes]] entry"
                    .into(),
            ),
        }
    };

    let client = Client::builder()
        .timeout(std::time::Duration::from_secs(args.timeout))
        .build()?;
    let painter = Painter::detect();

    println!(
        "probe → {}   key: {}   ({} model(s) × {} capabilit{})\n",
        conn.base_url,
        conn.id_label,
        bases.len(),
        caps.len(),
        if caps.len() == 1 { "y" } else { "ies" },
    );

    let ctx = Probe {
        client: &client,
        base_url: &conn.base_url,
        bearer: &conn.bearer,
        voice: &args.voice,
        tts_format: &args.tts_format,
    };

    // row label -> its per-capability outcomes, kept for the closing summary matrix.
    let mut matrix: Vec<(String, Vec<Outcome>)> = Vec::new();
    let mut any_fail = false;

    for base in &bases {
        let base = base.as_deref();
        println!("model `{}`", base.unwrap_or("(per-capability)"));

        let mut outcomes = Vec::with_capacity(caps.len());
        // TTS hands its synthesized bytes to STT for a real round-trip when both run.
        let mut tts_audio: Option<Vec<u8>> = None;

        for &cap in &caps {
            // Resolve the model for this capability: its override, else the row's base model. A
            // capability with neither is skipped (it cannot be probed) with a note, not a failure.
            let Some(model) = args.override_for(cap).or(base) else {
                println!(
                    "  {} {:<6} skipped — no model (give a base model or --{}-model)",
                    painter.paint("·", DIM),
                    cap.label(),
                    cap.label(),
                );
                continue;
            };

            let started = Instant::now();
            let (status, detail) = match cap {
                Capability::Chat => ctx.probe_chat(model).await,
                Capability::Tools => ctx.probe_tools(model).await,
                Capability::Tts => {
                    let (status, detail, audio) = ctx.probe_tts(model).await;
                    tts_audio = audio;
                    (status, detail)
                }
                Capability::Stt => ctx.probe_stt(model, tts_audio.as_deref()).await,
            };
            // Tag the line with the model when it differs from the row base (or there is no base), so a
            // mixed-model stack is legible at a glance.
            let detail = if Some(model) == base {
                detail
            } else {
                format!("[{model}] {detail}")
            };
            let outcome = Outcome {
                cap,
                status,
                detail,
                elapsed_ms: started.elapsed().as_millis(),
            };
            if matches!(outcome.status, Status::Fail) {
                any_fail = true;
            }
            println!("  {}", painter.line(&outcome));
            outcomes.push(outcome);
        }

        println!();
        matrix.push((base.unwrap_or("(per-capability)").to_string(), outcomes));
    }

    // A compact matrix is only worth printing when there is more than one row to line up.
    if matrix.len() > 1 {
        print!("{}", painter.summary(&caps, &matrix));
    }

    // Non-zero exit on a genuine failure (not on n/a) so the probe is CI-droppable.
    std::process::exit(if any_fail { 1 } else { 0 });
}

/// The shared connection one capability probe needs; the per-capability model is passed per call.
struct Probe<'a> {
    client: &'a Client,
    base_url: &'a str,
    bearer: &'a str,
    /// Voice requested by the text-to-speech probe.
    voice: &'a str,
    /// Audio `response_format` requested by the text-to-speech probe (and the container the STT
    /// round-trip then re-uploads as).
    tts_format: &'a str,
}

impl Probe<'_> {
    /// Chat: a plain turn must come back with non-empty assistant text.
    async fn probe_chat(&self, model: &str) -> (Status, String) {
        let body = json!({
            "model": model,
            "messages": [{ "role": "user", "content": "Reply with a short greeting." }],
            "stream": false,
        });
        let reply = match self.post_json("/v1/chat/completions", &body).await {
            Ok(r) => r,
            Err(verdict) => return verdict,
        };
        match content_text(&reply) {
            Some(text) if !text.trim().is_empty() => (Status::Pass, snippet(text)),
            _ => (
                Status::Fail,
                "200 OK but no assistant text in the reply".into(),
            ),
        }
    }

    /// Tool calls: offer one function and require the model to actually call it. On a call we close
    /// the loop with a tool result and confirm a follow-up answer, but the call itself is the verdict.
    async fn probe_tools(&self, model: &str) -> (Status, String) {
        let tools = json!([{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": { "location": { "type": "string", "description": "City name" } },
                    "required": ["location"],
                },
            },
        }]);
        let first_user = json!({
            "role": "user",
            "content": "What is the weather in Paris right now? Use the get_weather tool.",
        });
        let body = json!({
            "model": model,
            "messages": [first_user],
            "tools": tools,
            "tool_choice": "auto",
            "stream": false,
        });
        let reply = match self.post_json("/v1/chat/completions", &body).await {
            Ok(r) => r,
            Err(verdict) => return verdict,
        };

        let message = reply
            .pointer("/choices/0/message")
            .cloned()
            .unwrap_or(Value::Null);
        let calls = message.get("tool_calls").and_then(Value::as_array);
        let Some(call) = calls.and_then(|c| c.first()) else {
            return (
                Status::Fail,
                "no tool_call emitted (model answered with text instead)".into(),
            );
        };
        let func = call.get("function");
        let name = func
            .and_then(|f| f.get("name"))
            .and_then(Value::as_str)
            .unwrap_or("");
        let arguments = func
            .and_then(|f| f.get("arguments"))
            .and_then(Value::as_str)
            .unwrap_or("");
        if name != "get_weather" {
            return (
                Status::Fail,
                format!("called an unexpected function `{name}`"),
            );
        }
        let call_id = call.get("id").and_then(Value::as_str).unwrap_or("call_0");

        // Close the loop: feed a tool result back and confirm the model can continue from it. A hiccup
        // here doesn't void the capability — the model proved it can request a tool — so we still PASS,
        // just noting the follow-up outcome.
        let follow = json!({
            "model": model,
            "messages": [
                first_user,
                message,
                { "role": "tool", "tool_call_id": call_id, "content": "{\"temperature_c\": 15, \"sky\": \"sunny\"}" },
            ],
            "stream": false,
        });
        let follow_note = match self.post_json("/v1/chat/completions", &follow).await {
            Ok(r) if content_text(&r).is_some_and(|t| !t.trim().is_empty()) => "+ answered",
            Ok(_) => "+ follow-up empty",
            Err(_) => "+ follow-up failed",
        };

        (
            Status::Pass,
            format!("called get_weather({}) {follow_note}", snippet(arguments)),
        )
    }

    /// TTS: a synthesis must return non-empty audio bytes. Returns the bytes so STT can round-trip them.
    async fn probe_tts(&self, model: &str) -> (Status, String, Option<Vec<u8>>) {
        let body = json!({
            "model": model,
            "input": "The quick brown fox jumps over the lazy dog.",
            "voice": self.voice,
            "response_format": self.tts_format,
        });
        let (bytes, content_type) = match self.post_bytes("/v1/audio/speech", &body).await {
            Ok(ok) => ok,
            Err((status, detail)) => return (status, detail, None),
        };
        if bytes.is_empty() {
            return (
                Status::Fail,
                "200 OK but zero audio bytes returned".into(),
                None,
            );
        }
        let detail = format!(
            "{} bytes, {content_type}, voice={}",
            bytes.len(),
            self.voice
        );
        (Status::Pass, detail, Some(bytes))
    }

    /// STT: transcribe audio and require some text back. Round-trips the TTS bytes when available,
    /// otherwise uploads a tiny synthetic clip so the path is still exercised.
    async fn probe_stt(&self, model: &str, tts_audio: Option<&[u8]>) -> (Status, String) {
        // Label the upload to match the bytes: round-tripped TTS audio carries whatever container the
        // TTS probe requested (Groq's Orpheus only emits `wav`), so a hardcoded `.mp3`/`audio/mpeg`
        // would mislabel it and risk the transcriber rejecting or mis-sniffing the clip.
        let (audio, filename, mime): (Vec<u8>, String, &str) = match tts_audio {
            Some(bytes) => {
                let (ext, mime) = audio_ext_mime(self.tts_format);
                (bytes.to_vec(), format!("probe.{ext}"), mime)
            }
            None => (
                b"llmleaf probe synthetic audio".to_vec(),
                "probe.bin".to_string(),
                "application/octet-stream",
            ),
        };
        let part = match Part::bytes(audio).file_name(filename).mime_str(mime) {
            Ok(p) => p,
            Err(e) => return (Status::Fail, format!("building upload: {e}")),
        };
        let form = Form::new()
            .part("file", part)
            .text("model", model.to_string())
            .text("response_format", "json");

        let started = self
            .client
            .post(format!("{}/v1/audio/transcriptions", self.base_url))
            .bearer_auth(self.bearer)
            .multipart(form)
            .send()
            .await;
        let resp = match started {
            Ok(r) => r,
            Err(e) => return (Status::Fail, format!("request error: {e}")),
        };
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        if !status.is_success() {
            return classify_error(status.as_u16(), &body);
        }
        // `json` returns `{ "text": "…" }`; a bare-text format would arrive as the body verbatim.
        let text = serde_json::from_str::<Value>(&body)
            .ok()
            .and_then(|v| v.get("text").and_then(Value::as_str).map(str::to_string))
            .unwrap_or_else(|| body.trim().to_string());
        if text.trim().is_empty() {
            (Status::Fail, "200 OK but empty transcript".into())
        } else {
            (Status::Pass, snippet(&text))
        }
    }

    /// POST a JSON body and decode a JSON reply. A non-2xx is turned into a ready verdict (the
    /// "unsupported modality" case rendered as n/a, everything else FAIL) so callers just `return` it.
    async fn post_json(&self, path: &str, body: &Value) -> Result<Value, (Status, String)> {
        let resp = self
            .client
            .post(format!("{}{path}", self.base_url))
            .bearer_auth(self.bearer)
            .json(body)
            .send()
            .await
            .map_err(|e| (Status::Fail, format!("request error: {e}")))?;
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        if !status.is_success() {
            return Err(classify_error(status.as_u16(), &text));
        }
        serde_json::from_str::<Value>(&text)
            .map_err(|e| (Status::Fail, format!("200 OK but invalid JSON: {e}")))
    }

    /// POST a JSON body and keep the raw response bytes plus the resolved content type (the audio
    /// surfaces return bytes, not JSON). Non-2xx is classified into a ready verdict like `post_json`.
    async fn post_bytes(
        &self,
        path: &str,
        body: &Value,
    ) -> Result<(Vec<u8>, String), (Status, String)> {
        let resp = self
            .client
            .post(format!("{}{path}", self.base_url))
            .bearer_auth(self.bearer)
            .json(body)
            .send()
            .await
            .map_err(|e| (Status::Fail, format!("request error: {e}")))?;
        let status = resp.status();
        let content_type = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("application/octet-stream")
            .to_string();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(classify_error(status.as_u16(), &text));
        }
        let bytes = resp
            .bytes()
            .await
            .map_err(|e| (Status::Fail, format!("reading body: {e}")))?
            .to_vec();
        Ok((bytes, content_type))
    }
}

/// File extension and MIME type for a TTS `response_format`, so the STT round-trip labels the
/// re-uploaded audio with the container it actually is. Unknown formats fall back to a generic
/// binary upload (the transcriber sniffs the bytes).
fn audio_ext_mime(format: &str) -> (&'static str, &'static str) {
    match format {
        "mp3" => ("mp3", "audio/mpeg"),
        "wav" => ("wav", "audio/wav"),
        "flac" => ("flac", "audio/flac"),
        // OpenAI frames opus in an Ogg container.
        "opus" | "ogg" => ("ogg", "audio/ogg"),
        "aac" => ("aac", "audio/aac"),
        "pcm" => ("pcm", "audio/pcm"),
        _ => ("bin", "application/octet-stream"),
    }
}

/// Pull the assistant text out of a non-streaming chat completion (`choices[0].message.content`).
fn content_text(body: &Value) -> Option<&str> {
    body.pointer("/choices/0/message/content")
        .and_then(Value::as_str)
}

/// Classify a non-2xx response: the proxy maps a fully-unsupported route to **502 + "does not
/// support …"** (the canonical phrasing from the provider trait), which we render as n/a rather than a
/// failure. Everything else is a real FAIL with the parsed message.
fn classify_error(status: u16, body: &str) -> (Status, String) {
    let message = common::error_detail(body);
    let lower = message.to_ascii_lowercase();
    if status == 502 && (lower.contains("does not support") || lower.contains("unsupported")) {
        (Status::Unsupported, message)
    } else {
        (Status::Fail, format!("{status}: {message}"))
    }
}

/// Truncate a detail string to one tidy line (single-spaced, capped length).
fn snippet(text: &str) -> String {
    let one_line: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
    const MAX: usize = 64;
    if one_line.chars().count() > MAX {
        let mut s: String = one_line.chars().take(MAX - 1).collect();
        s.push('…');
        s
    } else {
        one_line
    }
}

/// Renders outcomes, optionally with ANSI colour when stdout is a real terminal.
struct Painter {
    color: bool,
}

impl Painter {
    fn detect() -> Self {
        Painter {
            color: std::io::stdout().is_terminal(),
        }
    }

    /// `<sym> <CAP>  (<ms>)  <detail>` for one capability, left-padded to align the labels.
    fn line(&self, o: &Outcome) -> String {
        let (sym, word) = self.badge(&o.status);
        format!(
            "{sym} {:<6} {word:<11} ({:>5}ms)  {}",
            o.cap.label(),
            o.elapsed_ms,
            o.detail,
        )
    }

    /// The closing model × capability grid: one row per model, a cell symbol per capability.
    fn summary(&self, caps: &[Capability], matrix: &[(String, Vec<Outcome>)]) -> String {
        let name_w = matrix
            .iter()
            .map(|(m, _)| m.len())
            .max()
            .unwrap_or(5)
            .max("model".len());
        let mut out = String::new();
        let _ = write!(out, "summary  {:<name_w$}", "model");
        for cap in caps {
            let _ = write!(out, "  {:<6}", cap.label());
        }
        out.push('\n');
        for (model, outcomes) in matrix {
            let _ = write!(out, "         {model:<name_w$}");
            for cap in caps {
                let cell = outcomes
                    .iter()
                    .find(|o| o.cap == *cap)
                    .map(|o| self.cell(&o.status))
                    .unwrap_or_else(|| "  ".into());
                let _ = write!(out, "  {cell:<6}");
            }
            out.push('\n');
        }
        out
    }

    /// (symbol, word) for the per-line view.
    fn badge(&self, status: &Status) -> (String, &'static str) {
        match status {
            Status::Pass => (self.paint("✓", GREEN), "PASS"),
            Status::Unsupported => (self.paint("–", DIM), "n/a"),
            Status::Fail => (self.paint("✗", RED), "FAIL"),
        }
    }

    /// Single-glyph cell for the matrix.
    fn cell(&self, status: &Status) -> String {
        match status {
            Status::Pass => self.paint("✓", GREEN),
            Status::Unsupported => self.paint("–", DIM),
            Status::Fail => self.paint("✗", RED),
        }
    }

    fn paint(&self, s: &str, code: &str) -> String {
        if self.color {
            format!("\x1b[{code}m{s}\x1b[0m")
        } else {
            s.to_string()
        }
    }
}

const GREEN: &str = "32";
const RED: &str = "31";
const DIM: &str = "2";
