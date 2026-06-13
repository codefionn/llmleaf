//! Example: speech-to-text against a stood-up llmleaf server (`POST /v1/audio/transcriptions`).
//!
//! A *consumer* of the proxy's OpenAI-compatible transcription surface. Like OpenAI, it uploads the
//! audio as `multipart/form-data` (`file`, `model`, and optional `response_format` / `language`) and
//! prints the transcript.
//!
//! Against the default model (the first configured route — `demo`, served by the built-in `echo`
//! provider, in the bundled `llmleaf.toml`) it works offline: `echo` returns a synthetic transcript
//! describing the upload, so it round-trips with the `tts` example — transcribe the `speech.mp3` that
//! `tts` wrote.
//!
//! Run it (with the server up via `cargo run -p llmleaf`); `--help` lists every flag:
//!
//! ```text
//! cargo run -p llmleaf --example tts -- "round trip"   # writes speech.mp3
//! cargo run -p llmleaf --example stt -- speech.mp3     # transcribes it
//! cargo run -p llmleaf --example stt -- --model gpt-4o-transcribe speech.mp3   # pick the model
//! cargo run -p llmleaf --example stt -- --help         # full usage
//! ```
//!
//! Every flag falls back to its `LLMLEAF_*` env var (`--model`/`LLMLEAF_MODEL`,
//! `--format`/`LLMLEAF_FORMAT`, `--language`/`LLMLEAF_LANGUAGE`); the model ultimately defaults to the
//! first configured route. See also the auth/base-url/config overrides documented in [`common`].

#[path = "common/mod.rs"]
mod common;

use clap::Parser;
use common::BoxError;
use reqwest::multipart::{Form, Part};
use serde_json::Value;
use tokio::io::AsyncWriteExt;

/// Speech-to-text client for a stood-up llmleaf server (POST /v1/audio/transcriptions).
#[derive(Parser)]
#[command(name = "stt")]
struct Args {
    /// Audio file to transcribe (e.g. the speech.mp3 written by the tts example).
    file: String,

    /// Logical model (route) to request [default: the first configured route].
    #[arg(short, long, env = "LLMLEAF_MODEL")]
    model: Option<String>,

    /// Response format: json | text | verbose_json | srt | vtt.
    #[arg(short, long, env = "LLMLEAF_FORMAT", default_value = "json")]
    format: String,

    /// Optional ISO-639-1 language hint.
    #[arg(short, long, env = "LLMLEAF_LANGUAGE")]
    language: Option<String>,
}

#[tokio::main]
async fn main() -> Result<(), BoxError> {
    let args = Args::parse();
    let conn = common::connect()?;
    let model = common::pick_model(&conn.config, args.model).ok_or(
        "no model: pass --model NAME, set LLMLEAF_MODEL, or add a [[routes]] entry to the config",
    )?;
    let format = args.format;

    let path = args.file;
    let audio = tokio::fs::read(&path)
        .await
        .map_err(|e| format!("reading audio file {path}: {e}"))?;
    let filename = std::path::Path::new(&path)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("audio")
        .to_string();

    let mut out = tokio::io::stdout();
    out.write_all(
        format!(
            "stt → {}  (model: {}, file: {}, {} bytes, format: {})\n",
            conn.base_url,
            model,
            filename,
            audio.len(),
            format
        )
        .as_bytes(),
    )
    .await?;
    out.flush().await?;

    // Build the multipart body. The MIME type is a best-effort hint from the extension; the proxy
    // forwards the bytes to the provider regardless.
    let file_part = Part::bytes(audio)
        .file_name(filename.clone())
        .mime_str(mime_for(&filename))?;
    let mut form = Form::new()
        .part("file", file_part)
        .text("model", model)
        .text("response_format", format.clone());
    if let Some(language) = args.language.filter(|l| !l.is_empty()) {
        form = form.text("language", language);
    }

    let resp = reqwest::Client::new()
        .post(format!("{}/v1/audio/transcriptions", conn.base_url))
        .bearer_auth(&conn.bearer)
        .multipart(form)
        .send()
        .await?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(format!("{status}: {}", common::error_detail(&body)).into());
    }

    // `text` / `srt` / `vtt` come back as a plain body; `json` / `verbose_json` as an object whose
    // `text` field holds the transcript (verbose_json adds language/duration).
    let body = resp.text().await?;
    let rendered = match serde_json::from_str::<Value>(&body) {
        Ok(v) => render_json(&v),
        Err(_) => body.trim().to_string(),
    };
    out.write_all(format!("\n{rendered}\n").as_bytes()).await?;
    out.flush().await?;
    Ok(())
}

/// Pretty-print a transcription JSON response: the transcript, plus language/duration when present.
fn render_json(v: &Value) -> String {
    let text = v
        .get("text")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let mut extras = Vec::new();
    if let Some(lang) = v.get("language").and_then(Value::as_str) {
        extras.push(format!("language: {lang}"));
    }
    if let Some(dur) = v.get("duration").and_then(Value::as_f64) {
        extras.push(format!("duration: {dur}s"));
    }
    if extras.is_empty() {
        text
    } else {
        format!("{text}\n  ({})", extras.join(", "))
    }
}

/// Best-effort audio MIME type from a file extension. Defaults to `application/octet-stream`.
fn mime_for(filename: &str) -> &'static str {
    match filename
        .rsplit_once('.')
        .map(|(_, ext)| ext.to_ascii_lowercase())
        .as_deref()
    {
        Some("mp3") => "audio/mpeg",
        Some("wav") => "audio/wav",
        Some("flac") => "audio/flac",
        Some("aac") => "audio/aac",
        Some("ogg" | "opus") => "audio/ogg",
        Some("m4a") => "audio/mp4",
        Some("webm") => "audio/webm",
        Some("pcm") => "audio/pcm",
        _ => "application/octet-stream",
    }
}
