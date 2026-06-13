//! Runnable example: non-streaming chat, streaming chat, and listing models.
//!
//! Configure with environment variables and run against a live gateway:
//!
//! ```sh
//! export LLMLEAF_BASE_URL="https://gateway.example.com"
//! export LLMLEAF_API_KEY="sk-..."
//! # optional: which model to hit (defaults to gpt-4o-mini)
//! export LLMLEAF_MODEL="gpt-4o-mini"
//! cargo run --example basic
//! ```

use futures::StreamExt;
use llmleaf_client::{ChatMessage, ChatRequest, Client, Error, ModelType};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let base_url = std::env::var("LLMLEAF_BASE_URL")
        .map_err(|_| "set LLMLEAF_BASE_URL to your gateway base URL")?;
    let api_key =
        std::env::var("LLMLEAF_API_KEY").map_err(|_| "set LLMLEAF_API_KEY to your API key")?;
    let model = std::env::var("LLMLEAF_MODEL").unwrap_or_else(|_| "gpt-4o-mini".to_string());

    let client = Client::new(base_url, api_key)?;

    // 1. List models.
    println!("== models ==");
    match client.list_models(Some(ModelType::All), None).await {
        Ok(list) => {
            for m in list.data.iter().take(10) {
                println!("  {}", m.id);
            }
            println!("  ({} total)", list.data.len());
        }
        Err(Error::Api { status, message }) => {
            println!("  api error {status}: {message}");
        }
        Err(e) => return Err(e.into()),
    }

    // 2. Non-streaming chat.
    println!("\n== chat (non-streaming) ==");
    let resp = client
        .chat(ChatRequest::new(
            &model,
            vec![
                ChatMessage::system("You are concise."),
                ChatMessage::user("Say hello in one short sentence."),
            ],
        ))
        .await?;
    println!("{}", resp.first_text().unwrap_or("(no text)"));
    if let Some(usage) = &resp.usage {
        println!(
            "  [tokens prompt={} completion={} total={}]",
            usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
        );
    }

    // 3. Streaming chat — print deltas as they arrive.
    println!("\n== chat (streaming) ==");
    let mut stream = client
        .chat_stream(ChatRequest::new(
            &model,
            vec![ChatMessage::user("Count from 1 to 5.")],
        ))
        .await?;

    use std::io::Write as _;
    let mut stdout = std::io::stdout();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        if let Some(delta) = chunk.first_delta_text() {
            print!("{delta}");
            let _ = stdout.flush();
        }
    }
    println!();

    Ok(())
}
