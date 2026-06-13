//! Example: embed two inputs and compare them, against a stood-up llmleaf server
//! (`POST /v1/embeddings`).
//!
//! A *consumer* of the proxy's OpenAI-compatible embeddings surface. It sends BOTH inputs in a single
//! request (`input: [a, b]`) — batch embeddings, with the response indices lined up to the inputs in
//! order — then computes the cosine similarity between the two returned vectors and prints it.
//!
//! Against the default model (the first configured route — `demo`, served by the built-in `echo`
//! provider, in the bundled `llmleaf.toml`) it works offline: `echo` returns a synthetic 2-D vector
//! per input (`[byte-length, word-count]`), enough to exercise the embeddings path and show a real
//! similarity number without a network or credentials. For a meaningful semantic comparison, point
//! `--model` at a real embeddings route (e.g. `text-embedding-3-small` from `llmleaf.example.toml`).
//!
//! Run it (with the server up via `cargo run -p llmleaf`); `--help` lists every flag:
//!
//! ```text
//! cargo run -p llmleaf --example embed -- "cat" "dog"                       # default model (route)
//! cargo run -p llmleaf --example embed -- --model text-embedding-3-small "a king" "a queen"
//! cargo run -p llmleaf --example embed -- --dimensions 256 "short" "shorter"  # Matryoshka truncation
//! cargo run -p llmleaf --example embed -- --help                            # full usage
//! ```
//!
//! Every flag falls back to its `LLMLEAF_*` env var (`--model`/`LLMLEAF_MODEL`,
//! `--dimensions`/`LLMLEAF_DIMENSIONS`); the model ultimately defaults to the first configured route.
//! See also the auth/base-url/config overrides documented in [`common`].

#[path = "common/mod.rs"]
mod common;

use clap::Parser;
use common::BoxError;
use serde_json::{json, Value};
use tokio::io::AsyncWriteExt;

/// Embed two inputs against a stood-up llmleaf server and report their cosine similarity.
#[derive(Parser)]
#[command(name = "embed")]
struct Args {
    /// First text to embed.
    a: String,

    /// Second text to embed (compared against the first).
    b: String,

    /// Logical model (route) to request [default: the first configured route].
    #[arg(short, long, env = "LLMLEAF_MODEL")]
    model: Option<String>,

    /// Optional output dimensionality (Matryoshka truncation, e.g. text-embedding-3-*).
    #[arg(short, long, env = "LLMLEAF_DIMENSIONS")]
    dimensions: Option<u32>,
}

#[tokio::main]
async fn main() -> Result<(), BoxError> {
    let args = Args::parse();
    let conn = common::connect()?;
    let model = common::pick_model(&conn.config, args.model).ok_or(
        "no model: pass --model NAME, set LLMLEAF_MODEL, or add a [[routes]] entry to the config",
    )?;

    let mut out = tokio::io::stdout();
    out.write_all(format!("embed → {}  (model: {})\n", conn.base_url, model).as_bytes())
        .await?;
    out.flush().await?;

    // Both inputs ride in one request, so the response indices line up with `[a, b]` in order.
    let mut body = json!({
        "model": model,
        "input": [args.a, args.b],
        "encoding_format": "float",
    });
    if let Some(d) = args.dimensions {
        body["dimensions"] = json!(d);
    }

    let resp = reqwest::Client::new()
        .post(format!("{}/v1/embeddings", conn.base_url))
        .bearer_auth(&conn.bearer)
        .json(&body)
        .send()
        .await?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return Err(format!("{status}: {}", common::error_detail(&text)).into());
    }

    let payload: Value = resp.json().await?;
    let (va, vb) = two_vectors(&payload)?;

    let similarity = cosine_similarity(&va, &vb)
        .ok_or("cannot compare: the two embeddings have different lengths or a zero vector")?;

    let report = format!(
        "\n  a: {:?}\n  b: {:?}\n  dimensions: {}\n  cosine similarity: {:.6}  ({})\n",
        args.a,
        args.b,
        va.len(),
        similarity,
        interpret(similarity),
    );
    out.write_all(report.as_bytes()).await?;
    if let Some(footer) = payload.get("usage").and_then(usage_footer) {
        out.write_all(footer.as_bytes()).await?;
        out.write_all(b"\n").await?;
    }
    out.flush().await?;
    Ok(())
}

/// Pull exactly the two embedding vectors out of an OpenAI `{ object: "list", data: [...] }` response,
/// ordered by their `index` so they line up with the `[a, b]` inputs regardless of wire order.
fn two_vectors(payload: &Value) -> Result<(Vec<f64>, Vec<f64>), BoxError> {
    let data = payload
        .get("data")
        .and_then(Value::as_array)
        .ok_or("malformed response: missing `data` array")?;
    if data.len() != 2 {
        return Err(format!("expected 2 embeddings, got {}", data.len()).into());
    }

    let mut indexed: Vec<(u64, Vec<f64>)> = data
        .iter()
        .map(|item| {
            let index = item.get("index").and_then(Value::as_u64).unwrap_or(0);
            let vector = item
                .get("embedding")
                .and_then(Value::as_array)
                .ok_or("malformed response: an entry is missing its `embedding` array")?
                .iter()
                .map(|n| n.as_f64().ok_or("embedding component was not a number"))
                .collect::<Result<Vec<f64>, _>>()?;
            Ok::<_, BoxError>((index, vector))
        })
        .collect::<Result<_, _>>()?;
    indexed.sort_by_key(|(i, _)| *i);

    let mut it = indexed.into_iter();
    Ok((it.next().unwrap().1, it.next().unwrap().1))
}

/// Cosine similarity of two equal-length vectors: `dot(a,b) / (|a| * |b|)`, in `[-1, 1]`. Returns
/// `None` if the lengths differ or either vector has zero magnitude (similarity is undefined).
fn cosine_similarity(a: &[f64], b: &[f64]) -> Option<f64> {
    if a.len() != b.len() || a.is_empty() {
        return None;
    }
    let mut dot = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;
    for (x, y) in a.iter().zip(b) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    if norm_a == 0.0 || norm_b == 0.0 {
        return None;
    }
    Some(dot / (norm_a.sqrt() * norm_b.sqrt()))
}

/// A friendly one-word reading of a cosine similarity (informational only).
fn interpret(similarity: f64) -> &'static str {
    match similarity {
        s if s >= 0.9 => "near-identical",
        s if s >= 0.7 => "very similar",
        s if s >= 0.5 => "similar",
        s if s >= 0.3 => "loosely related",
        s if s >= 0.0 => "unrelated",
        _ => "opposed",
    }
}

/// Format the embeddings `usage` object into a one-line footer. `total` falls back to `prompt`; cost
/// is an llmleaf extension present only when the pricing dataset covers the model (`echo` reports none).
fn usage_footer(usage: &Value) -> Option<String> {
    let n = |k: &str| usage.get(k).and_then(Value::as_u64);
    let prompt = n("prompt_tokens");
    let total = n("total_tokens");
    let cost = usage.get("cost_usd").and_then(Value::as_f64);
    if prompt.is_none() && total.is_none() && cost.is_none() {
        return None;
    }
    let p = prompt.unwrap_or(0);
    let t = total.unwrap_or(p);
    let mut s = format!("  [{p} prompt = {t} tokens");
    if let Some(cost) = cost {
        s.push_str(&format!(", ${cost:.6}"));
    }
    s.push(']');
    Some(s)
}
