//! Evaluate a support ticket using the configured `jev` decisions route.
//!
//! Start llmleaf with a decisions provider configured, then run:
//! `cargo run -p llmleaf --example decisions -- --state "The export button does nothing."`
//! Connection and authentication settings use the same environment variables as the other examples.

#[path = "common/mod.rs"]
mod common;

use clap::Parser;
use common::BoxError;
use serde_json::{json, Value};

#[derive(Parser)]
struct Args {
    /// Ticket text to evaluate.
    #[arg(long)]
    state: String,

    /// Logical route backed by OpenRouter or TypeSafe JEV.
    #[arg(short, long, env = "LLMLEAF_MODEL", default_value = "jev")]
    model: String,
}

#[tokio::main]
async fn main() -> Result<(), BoxError> {
    let args = Args::parse();
    let conn = common::connect()?;
    let response = reqwest::Client::new()
        .post(format!("{}/v1/decisions", conn.base_url))
        .bearer_auth(&conn.bearer)
        .json(&json!({
            "model": args.model,
            "state": args.state,
            "questions": {
                "bug": {
                    "type": "noul",
                    "instructions": "Does this report broken software?"
                },
                "team": {
                    "type": "choice",
                    "instructions": "Which team should handle the ticket?",
                    "criteria": {
                        "engineering": "Software defects",
                        "support": "Questions about using the product",
                        "billing": "Invoices and payment problems"
                    }
                },
                "urgency": {
                    "type": "score",
                    "instructions": "How urgently does the ticket need attention?",
                    "criteria": [
                        "Work can continue without a fix",
                        "Some work is blocked",
                        "All work is blocked"
                    ]
                }
            }
        }))
        .send()
        .await?;
    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await?;
        return Err(format!("{status}: {}", common::error_detail(&text)).into());
    }
    let payload: Value = response.json().await?;
    println!("{}", serde_json::to_string_pretty(&payload)?);
    Ok(())
}
