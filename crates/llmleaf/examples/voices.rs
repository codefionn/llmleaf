//! Example: list the voices a speech model can use (`GET /v1/audio/voices`).
//!
//! A *consumer* of the proxy's voice-catalog surface. It sends `?model=<route>` and receives
//! `{ "model": …, "voices": [ { "id", "name"?, "languages"?, … } ] }` — the `id` of each voice is
//! exactly what the `tts` example puts in its `--voice` flag. Voices are provider-declared: a provider
//! with a documented catalog (e.g. OpenAI) answers from a static list, so this never leaves the node;
//! a model whose provider does not enumerate voices comes back as a friendly "not listed" note rather
//! than an error (the same `Unsupported` fall-through the `probe` example renders as n/a).
//!
//! Against the default model (the first configured route — `demo`, served by the built-in `echo`
//! provider, in the bundled `llmleaf.toml`) it works offline, returning the echo provider's synthetic
//! catalog.
//!
//! Run it (with the server up via `cargo run -p llmleaf`); `--help` lists every flag:
//!
//! ```text
//! cargo run -p llmleaf --example voices                    # the default model (first route)
//! cargo run -p llmleaf --example voices -- --model tts-1   # pick the model (route)
//! cargo run -p llmleaf --example voices -- --json          # raw JSON, for scripting
//! cargo run -p llmleaf --example voices -- --help          # full usage
//! ```
//!
//! Every flag falls back to its `LLMLEAF_*` env var (`--model`/`LLMLEAF_MODEL`); the model ultimately
//! defaults to the first configured route. See also the auth/base-url/config overrides documented in
//! [`common`].

#[path = "common/mod.rs"]
mod common;

use clap::Parser;
use common::BoxError;
use serde_json::Value;

/// List a speech model's voices from a stood-up llmleaf server (GET /v1/audio/voices).
#[derive(Parser)]
#[command(name = "voices")]
struct Args {
    /// Logical model (route) whose voices to list [default: the first configured route].
    #[arg(short, long, env = "LLMLEAF_MODEL")]
    model: Option<String>,

    /// Print the raw JSON response instead of the formatted table.
    #[arg(short, long)]
    json: bool,
}

#[tokio::main]
async fn main() -> Result<(), BoxError> {
    let args = Args::parse();
    let conn = common::connect()?;
    let model = common::pick_model(&conn.config, args.model).ok_or(
        "no model: pass --model NAME, set LLMLEAF_MODEL, or add a [[routes]] entry to the config",
    )?;

    // `model` is a config route name (a plain identifier), so it needs no percent-encoding to ride in
    // the query string — keep the example free of a urlencoding dependency.
    let resp = reqwest::Client::new()
        .get(format!("{}/v1/audio/voices?model={model}", conn.base_url))
        .bearer_auth(&conn.bearer)
        .send()
        .await?;

    let status = resp.status();
    let body = resp.text().await.unwrap_or_default();

    if !status.is_success() {
        let detail = common::error_detail(&body);
        // A provider that does not enumerate voices surfaces as 502 + "does not support …" (the
        // canonical `Unsupported` phrasing) — report that as information, not a hard error.
        let lower = detail.to_ascii_lowercase();
        if status.as_u16() == 502
            && (lower.contains("does not support") || lower.contains("does not list"))
        {
            println!("model `{model}` → no voice catalog ({detail})");
            return Ok(());
        }
        return Err(format!("{status}: {detail}").into());
    }

    if args.json {
        println!("{body}");
        return Ok(());
    }

    let parsed: Value =
        serde_json::from_str(&body).map_err(|e| format!("200 OK but invalid JSON: {e}"))?;
    let voices = parsed
        .get("voices")
        .and_then(Value::as_array)
        .ok_or("response had no `voices` array")?;

    println!(
        "voices → {}   model: {}   ({} voice{})\n",
        conn.base_url,
        model,
        voices.len(),
        if voices.len() == 1 { "" } else { "s" },
    );
    if voices.is_empty() {
        println!("(no voices reported for this model)");
        return Ok(());
    }

    // Left-pad the id column so the optional name/languages line up.
    let id_w = voices
        .iter()
        .filter_map(|v| v.get("id").and_then(Value::as_str))
        .map(str::len)
        .max()
        .unwrap_or(0);

    for v in voices {
        let id = v.get("id").and_then(Value::as_str).unwrap_or("?");
        let name = v.get("name").and_then(Value::as_str);
        let langs = v
            .get("languages")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(Value::as_str)
                    .collect::<Vec<_>>()
                    .join(", ")
            })
            .unwrap_or_default();

        let mut line = format!("  {id:<id_w$}");
        if let Some(name) = name {
            line.push_str(&format!("  {name}"));
        }
        if !langs.is_empty() {
            line.push_str(&format!("  [{langs}]"));
        }
        println!("{line}");
    }

    Ok(())
}
