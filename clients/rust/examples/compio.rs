//! Run with `cargo run --example compio --no-default-features --features compio`.

use llmleaf_client::{ChatMessage, ChatRequest, Client};

#[compio::main]
async fn main() -> Result<(), llmleaf_client::Error> {
    let base_url = std::env::var("LLMLEAF_BASE_URL")
        .unwrap_or_else(|_| "https://gateway.example.com".to_string());
    let api_key = std::env::var("LLMLEAF_API_KEY").unwrap_or_else(|_| "sk-...".to_string());
    let client = Client::new(base_url, api_key)?;
    let response = client
        .chat(ChatRequest::new(
            "gpt-4o-mini",
            vec![ChatMessage::user("Say hi.")],
        ))
        .await?;
    println!("{}", response.first_text().unwrap_or_default());
    Ok(())
}
