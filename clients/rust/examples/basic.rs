//! Runnable example: listing models, non-streaming + streaming chat, the OpenAI
//! Responses dialect (non-streaming + streaming), and a rerank call.
//!
//! Configure with environment variables and run against a live gateway:
//!
//! ```sh
//! export LLMLEAF_BASE_URL="https://gateway.example.com"
//! export LLMLEAF_API_KEY="sk-..."
//! # optional: which model to hit (defaults to gpt-4o-mini)
//! export LLMLEAF_MODEL="gpt-4o-mini"
//! # optional: a rerank model to exercise POST /v1/rerank
//! export LLMLEAF_RERANK_MODEL="rerank-english-v3.0"
//! cargo run --example basic
//! ```

use futures::StreamExt;
use llmleaf_client::{
    ChatMessage, ChatRequest, Client, Error, ModelType, RerankRequest, ResponsesRequest,
};

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

    // 4. Responses dialect (non-streaming). `input` is a bare string here.
    println!("\n== responses (non-streaming) ==");
    let resp = client
        .responses(ResponsesRequest::new(&model, "Say hello in one short sentence."))
        .await?;
    println!("{}", {
        let text = resp.output_text();
        if text.is_empty() { "(no text)".to_string() } else { text }
    });
    if let Some(usage) = &resp.usage {
        println!(
            "  [tokens input={} output={} total={} cached={}]",
            usage.input_tokens,
            usage.output_tokens,
            usage.total_tokens,
            usage.cached_tokens(),
        );
    }

    // 5. Responses dialect (streaming) — accumulate output_text deltas; stops on the
    //    terminal event (there is no [DONE] sentinel).
    println!("\n== responses (streaming) ==");
    let mut events = client
        .responses_stream(ResponsesRequest::new(&model, "Count from 1 to 5."))
        .await?;
    while let Some(event) = events.next().await {
        let event = event?;
        if let Some(delta) = event.output_text_delta() {
            print!("{delta}");
            let _ = stdout.flush();
        }
    }
    println!();

    // 6. Rerank — score documents against a query (needs a rerank-capable model).
    if let Ok(rerank_model) = std::env::var("LLMLEAF_RERANK_MODEL") {
        println!("\n== rerank ==");
        let mut request = RerankRequest::new(
            &rerank_model,
            "What is the capital of France?",
            vec![
                "Paris is the capital of France.",
                "Berlin is the capital of Germany.",
                "The Eiffel Tower is in Paris.",
            ],
        );
        request.top_n = Some(2);
        match client.rerank(request).await {
            Ok(resp) => {
                for r in &resp.results {
                    println!("  index={} score={:.4}", r.index, r.relevance_score);
                }
            }
            Err(Error::Api { status, message }) => {
                println!("  api error {status}: {message}");
            }
            Err(e) => return Err(e.into()),
        }
    }

    Ok(())
}
