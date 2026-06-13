//! Example: text-to-speech against a stood-up llmleaf server (`POST /v1/audio/speech`).
//!
//! A *consumer* of the proxy's OpenAI-compatible speech surface. It sends `{model, input, voice,
//! response_format}` and receives **raw audio bytes** (no JSON envelope — the server sets
//! `Content-Type` from the provider's audio stream), which it writes to a file.
//!
//! Against the default model (the first configured route — `demo`, served by the built-in `echo`
//! provider, in the bundled `llmleaf.toml`) it works offline: the synthetic "audio" is just the input
//! text bytes, so you can round-trip it through the `stt` example.
//!
//! Run it (with the server up via `cargo run -p llmleaf`); `--help` lists every flag:
//!
//! ```text
//! cargo run -p llmleaf --example tts -- "Hello from llmleaf"        # -> speech.mp3
//! cargo run -p llmleaf --example tts -- "Hello" out.wav             # explicit output path
//! cargo run -p llmleaf --example tts -- --model tts-1 "Hello"       # pick the model (route)
//! echo "piped text" | cargo run -p llmleaf --example tts           # read input from stdin
//! cargo run -p llmleaf --example tts -- --help                     # full usage
//! ```
//!
//! Every flag falls back to its `LLMLEAF_*` env var (`--model`/`LLMLEAF_MODEL`, etc.); the model
//! ultimately defaults to the first configured route. See also the auth/base-url/config overrides
//! documented in [`common`].

#[path = "common/mod.rs"]
mod common;

use clap::Parser;
use common::BoxError;
use serde_json::json;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// Text-to-speech client for a stood-up llmleaf server (POST /v1/audio/speech).
#[derive(Parser)]
#[command(name = "tts")]
struct Args {
    /// Text to synthesize. `-` (or omit, on a pipe) reads it from stdin.
    input: Option<String>,

    /// Output file [default: speech.<ext>, the extension derived from --format].
    output: Option<String>,

    /// Logical model (route) to request [default: the first configured route].
    #[arg(short, long, env = "LLMLEAF_MODEL")]
    model: Option<String>,

    /// Voice name.
    #[arg(short, long, env = "LLMLEAF_VOICE", default_value = "alloy")]
    voice: String,

    /// Audio format: mp3 | opus | aac | flac | wav | pcm.
    #[arg(short, long, env = "LLMLEAF_FORMAT", default_value = "mp3")]
    format: String,
}

#[tokio::main]
async fn main() -> Result<(), BoxError> {
    let args = Args::parse();
    let conn = common::connect()?;
    let model = common::pick_model(&conn.config, args.model).ok_or(
        "no model: pass --model NAME, set LLMLEAF_MODEL, or add a [[routes]] entry to the config",
    )?;
    let voice = args.voice;
    let format = args.format;

    // Input text: the positional arg, an explicit `-` to read stdin, or piped stdin. If it is simply
    // forgotten on an interactive terminal, say so instead of blocking on a TTY read forever.
    let input = match args.input.as_deref() {
        Some("-") => read_stdin().await?, // explicit `-`
        Some(text) => text.to_string(),
        None if std::io::IsTerminal::is_terminal(&std::io::stdin()) => {
            return Err(
                "no input text. Pass it as an argument — `tts \"your text\"` — \
                        or pipe it on stdin."
                    .into(),
            );
        }
        None => read_stdin().await?,
    };
    let input = input.trim();
    if input.is_empty() {
        return Err("no input text (pass it as an argument or pipe it on stdin)".into());
    }

    // Output path: the second positional arg, or `speech.<ext>` derived from the format.
    let out_path = args
        .output
        .unwrap_or_else(|| format!("speech.{}", ext_for(&format)));

    let mut out = tokio::io::stdout();
    out.write_all(
        format!(
            "tts → {}  (model: {}, voice: {}, format: {})\n",
            conn.base_url, model, voice, format
        )
        .as_bytes(),
    )
    .await?;
    out.flush().await?;

    let resp = reqwest::Client::new()
        .post(format!("{}/v1/audio/speech", conn.base_url))
        .bearer_auth(&conn.bearer)
        .json(&json!({
            "model": model,
            "input": input,
            "voice": voice,
            "response_format": format,
        }))
        .send()
        .await?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(format!("{status}: {}", common::error_detail(&body)).into());
    }

    // The audio MIME type the server resolved from the provider's stream (informational).
    let content_type = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("application/octet-stream")
        .to_string();

    let audio = resp.bytes().await?;
    tokio::fs::write(&out_path, &audio).await?;

    out.write_all(
        format!(
            "wrote {} bytes of {} to {}\n",
            audio.len(),
            content_type,
            out_path
        )
        .as_bytes(),
    )
    .await?;
    out.flush().await?;
    Ok(())
}

/// Read all of stdin to a string (used when no input argument is given).
async fn read_stdin() -> Result<String, BoxError> {
    let mut s = String::new();
    tokio::io::stdin().read_to_string(&mut s).await?;
    Ok(s)
}

/// File extension for an OpenAI speech `response_format`. `opus` is delivered Ogg-framed.
fn ext_for(format: &str) -> &str {
    match format {
        "opus" => "ogg",
        "aac" => "aac",
        "flac" => "flac",
        "wav" => "wav",
        "pcm" => "pcm",
        _ => "mp3",
    }
}
