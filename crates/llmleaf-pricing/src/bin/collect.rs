use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use llmleaf_pricing::collect::{self, CollectorConfig, CollectorProvider};

#[derive(Debug, Parser)]
#[command(
    name = "llmleaf-pricing-collect",
    about = "Collect provider model metadata into the bundled llmleaf pricing dataset"
)]
struct Args {
    /// TOML collector config. See llmleaf_pricing::collect::CollectorConfig docs.
    #[arg(long)]
    config: Option<PathBuf>,
    /// Override the output prices.json path.
    #[arg(long)]
    output: Option<String>,
    /// Override the seed prices.json path.
    #[arg(long)]
    seed: Option<String>,
    /// Remove seed rows that are not observed in this collector run.
    #[arg(long)]
    prune: bool,
    /// Provider kind to collect. Use KIND:ENV to read its credential from ENV.
    #[arg(long = "provider", value_name = "KIND[:CREDENTIAL_ENV]")]
    providers: Vec<String>,
}

#[tokio::main]
async fn main() -> ExitCode {
    match run().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("fatal: {e}");
            ExitCode::FAILURE
        }
    }
}

async fn run() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args = Args::parse();
    let mut cfg = if let Some(path) = args.config {
        let text = std::fs::read_to_string(&path)?;
        toml::from_str::<CollectorConfig>(&text)?
    } else {
        CollectorConfig::default()
    };

    if let Some(output) = args.output {
        cfg.output = Some(output);
    }
    if let Some(seed) = args.seed {
        cfg.seed = Some(seed);
    }
    if args.prune {
        cfg.prune = true;
    }
    for raw in args.providers {
        cfg.providers.push(provider_arg(&raw)?);
    }
    if cfg.providers.is_empty() {
        return Err(format!(
            "no providers configured; pass --config or --provider. {}",
            collect::describe()
        )
        .into());
    }

    let report = collect::run(cfg).await?;
    for p in &report.providers {
        eprintln!(
            "collected {} model{} from {} ({})",
            p.models,
            if p.models == 1 { "" } else { "s" },
            p.name,
            p.kind
        );
    }
    println!(
        "wrote {} model{} to {}",
        report.model_count,
        if report.model_count == 1 { "" } else { "s" },
        report.output
    );
    Ok(())
}

fn provider_arg(raw: &str) -> Result<CollectorProvider, Box<dyn std::error::Error + Send + Sync>> {
    let (kind, credential_env) = raw
        .split_once(':')
        .map(|(kind, env)| (kind, Some(env.to_string())))
        .unwrap_or((raw, None));
    if kind.trim().is_empty() {
        return Err("provider kind cannot be empty".into());
    }
    Ok(CollectorProvider {
        name: None,
        kind: kind.to_string(),
        endpoint: None,
        credential: None,
        credential_env,
        settings: Default::default(),
        source: Default::default(),
        pricing_url: None,
        list_url: None,
    })
}
