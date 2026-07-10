//! Example: rerank a set of documents against a query, against a stood-up llmleaf server
//! (`POST /v1/rerank`).
//!
//! A *consumer* of the proxy's rerank surface (Cohere/Jina/OpenRouter dialect). It sends a `query` and
//! a list of `documents` in a single request, then prints the results the server returns — each a
//! `{ index, relevance_score }` pointing back at one input document — ordered best-match first, with the
//! document text alongside when the server echoes it (`return_documents`).
//!
//! Unlike `embed`, this has NO offline default: the built-in `echo` provider does not implement rerank,
//! so a rerank route must resolve to a REAL rerank provider (Cohere natively, or Together/OpenRouter's
//! `/v1/rerank`, or a self-hosted OpenAI-wire reranker opted in with `settings.rerank_api = true`).
//! `--model` therefore defaults to `rerank-v3.5` — the rerank route in `llmleaf.example.toml` — not to
//! the first configured route the way `embed` falls back. Point it at any rerank route you have.
//!
//! Run it (with the server up via `cargo run -p llmleaf`); `--help` lists every flag:
//!
//! ```text
//! cargo run -p llmleaf --example rerank -- --query "capital of France" \
//!     --document "Paris is the capital of France." --document "Berlin is a city in Germany."
//! cargo run -p llmleaf --example rerank -- --model rerank-v3.5 --top-n 3 \
//!     --query "..." --document "..." --document "..."
//! cargo run -p llmleaf --example rerank -- --help                       # full usage
//! ```
//!
//! `--model` falls back to its `LLMLEAF_MODEL` env var (and `--top-n` to `LLMLEAF_TOP_N`); the model
//! ultimately defaults to `rerank-v3.5`. See also the auth/base-url/config overrides documented in
//! [`common`].

#[path = "common/mod.rs"]
mod common;

use clap::Parser;
use common::BoxError;
use serde_json::{json, Value};
use tokio::io::AsyncWriteExt;

/// Rerank a set of documents against a query on a stood-up llmleaf server and print the ranking.
#[derive(Parser)]
#[command(name = "rerank")]
struct Args {
    /// The search query the documents are ranked against.
    #[arg(short, long)]
    query: String,

    /// A candidate document to rank; repeat the flag to pass several.
    #[arg(short, long = "document")]
    documents: Vec<String>,

    /// Logical model (route) to request — must resolve to a real rerank provider.
    #[arg(short, long, env = "LLMLEAF_MODEL", default_value = "rerank-v3.5")]
    model: String,

    /// Return at most this many ranked results (server-side truncation).
    #[arg(short, long, env = "LLMLEAF_TOP_N")]
    top_n: Option<u32>,
}

#[tokio::main]
async fn main() -> Result<(), BoxError> {
    let args = Args::parse();
    if args.documents.is_empty() {
        return Err("no documents: pass at least one --document VALUE".into());
    }
    let conn = common::connect()?;
    let model = args.model;

    let mut out = tokio::io::stdout();
    out.write_all(format!("rerank → {}  (model: {})\n", conn.base_url, model).as_bytes())
        .await?;
    out.flush().await?;

    let mut body = json!({
        "model": model,
        "query": args.query,
        "documents": args.documents,
        // Ask the server to echo each ranked document back so we can print its text next to the score.
        "return_documents": true,
    });
    if let Some(n) = args.top_n {
        body["top_n"] = json!(n);
    }

    let resp = reqwest::Client::new()
        .post(format!("{}/v1/rerank", conn.base_url))
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
    let results = ranked_results(&payload, &args.documents)?;

    let mut report = format!(
        "\n  query: {:?}\n  {} result(s), best match first:\n",
        args.query,
        results.len(),
    );
    for (rank, r) in results.iter().enumerate() {
        report.push_str(&format!(
            "    {:>2}. input #{:<3} score {:.6}  {:?}\n",
            rank + 1,
            r.index,
            r.score,
            r.text,
        ));
    }
    out.write_all(report.as_bytes()).await?;
    if let Some(footer) = payload.get("usage").and_then(usage_footer) {
        out.write_all(footer.as_bytes()).await?;
        out.write_all(b"\n").await?;
    }
    out.flush().await?;
    Ok(())
}

/// One reranked candidate: its position in the original `documents` list, the relevance score, and the
/// document text (echoed by the server when `return_documents` is set, else recovered from the inputs).
struct Ranked {
    index: u64,
    score: f64,
    text: String,
}

/// Pull the ranked candidates out of a rerank `{ object: "list", results: [...] }` response and order
/// them best-match first. `document` is echoed only when `return_documents` was honored, so fall back
/// to the original input at the same `index` when it is absent.
fn ranked_results(payload: &Value, inputs: &[String]) -> Result<Vec<Ranked>, BoxError> {
    let results = payload
        .get("results")
        .and_then(Value::as_array)
        .ok_or("malformed response: missing `results` array")?;

    let mut ranked: Vec<Ranked> = results
        .iter()
        .map(|item| {
            let index = item
                .get("index")
                .and_then(Value::as_u64)
                .ok_or("malformed response: a result is missing its `index`")?;
            let score = item
                .get("relevance_score")
                .and_then(Value::as_f64)
                .ok_or("malformed response: a result is missing its `relevance_score`")?;
            let text = document_text(item.get("document"))
                .or_else(|| inputs.get(index as usize).cloned())
                .unwrap_or_default();
            Ok::<_, BoxError>(Ranked { index, score, text })
        })
        .collect::<Result<_, _>>()?;

    // Rerank upstreams usually return sorted by score, but not all do — order it ourselves so the
    // printed ranking is authoritative regardless of wire order.
    ranked.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(ranked)
}

/// Extract a result's echoed document text. Providers return it as either a bare string or an object
/// with a `text` field (Cohere v2's shape); `None` when the server omitted it (`return_documents` off).
fn document_text(document: Option<&Value>) -> Option<String> {
    match document? {
        Value::String(s) => Some(s.clone()),
        obj @ Value::Object(_) => obj.get("text").and_then(Value::as_str).map(str::to_string),
        _ => None,
    }
}

/// Format the rerank `usage` object into a one-line footer. Rerank reports `total_tokens`; `cost_usd`
/// is an llmleaf extension present only when the pricing dataset covers the model.
fn usage_footer(usage: &Value) -> Option<String> {
    let total = usage.get("total_tokens").and_then(Value::as_u64);
    let cost = usage.get("cost_usd").and_then(Value::as_f64);
    if total.is_none() && cost.is_none() {
        return None;
    }
    let mut s = format!("  [{} tokens", total.unwrap_or(0));
    if let Some(cost) = cost {
        s.push_str(&format!(", ${cost:.6}"));
    }
    s.push(']');
    Some(s)
}
