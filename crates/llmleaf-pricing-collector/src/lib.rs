//! Offline generation of the pricing dataset consumed by `llmleaf-pricing`.
//!
//! This crate owns provider fetching, pricing-page parsing, seed merging, and deterministic dataset
//! generation. It is intentionally separate from the read-only runtime pricing-information library.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;

use llmleaf_model::{Modality, ModelInfo};
use llmleaf_pricing::{ModelPricing, PricingDataset};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// Offline collection and generation implementation.
pub mod collect {
    use super::*;

    const COMMENT: &str = "Bundled pricing dataset. Generated OFFLINE by the llmleaf-pricing-collector crate; the runtime llmleaf-pricing library only reads this bundled information. Rates are USD per 1,000,000 tokens.";
    const COMMENT_METADATA: &str = "Each row may carry capability metadata for GET /v1/models: modality (llm|tts|stt|embedding|rerank), published limits max_context/max_output/max_thinking, and supports_reasoning when a provider publishes that capability without a numeric budget. Missing fields mean 'not collected' and must be rendered as unknown, never guessed or zeroed.";
    const COMMENT_PARAMS: &str = "unsupported_parameters lists canonical sampling params the model rejects; default_parameters carries provider- or dataset-recommended defaults. Missing means 'not collected'.";

    /// The default location of the committed dataset when the collector is run from the workspace root.
    pub const DEFAULT_OUTPUT: &str = "crates/llmleaf-pricing/data/prices.json";

    /// Short human-readable summary used by docs/tests and by the collector binary's help.
    pub fn describe() -> &'static str {
        "offline collector: fetch/read provider model catalogs -> write crates/llmleaf-pricing/data/prices.json"
    }

    /// Result of one collector run.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct Report {
        pub output: String,
        pub providers: Vec<ProviderReport>,
        pub model_count: usize,
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct ProviderReport {
        pub name: String,
        pub kind: String,
        pub source: String,
        pub models: usize,
    }

    /// TOML configuration for the offline collector binary.
    ///
    /// Example:
    ///
    /// ```toml
    /// output = "crates/llmleaf-pricing/data/prices.json"
    ///
    /// [[providers]]
    /// kind = "openai"
    ///
    /// [[providers]]
    /// kind = "openrouter"
    /// source = "list-endpoint"
    /// credential_env = "OPENROUTER_API_KEY"
    /// ```
    #[derive(Debug, Clone, Default, Deserialize)]
    #[serde(default, deny_unknown_fields)]
    pub struct CollectorConfig {
        /// Existing dataset to merge into. Defaults to `output` when it already exists.
        pub seed: Option<String>,
        /// Dataset path to write.
        pub output: Option<String>,
        /// If true, remove seed rows that were not observed in this run.
        pub prune: bool,
        pub providers: Vec<CollectorProvider>,
    }

    #[derive(Debug, Clone, Deserialize)]
    #[serde(default, deny_unknown_fields)]
    pub struct CollectorProvider {
        pub name: Option<String>,
        pub kind: String,
        pub endpoint: Option<String>,
        /// Literal credential for private/offline use. Prefer `credential_env` for committed configs.
        pub credential: Option<String>,
        /// Environment variable containing the provider credential.
        pub credential_env: Option<String>,
        pub settings: Map<String, Value>,
        /// `auto` chooses a provider-specific pricing page when one is known, otherwise a priced list
        /// endpoint. `list-endpoint` is only accepted when returned rows include token prices.
        #[serde(default)]
        pub source: CollectorSource,
        /// Override the provider pricing page URL. Used only by `pricing-page` collectors.
        pub pricing_url: Option<String>,
        /// Override the provider model-list URL. Used by `list-endpoint` collectors.
        pub list_url: Option<String>,
    }

    #[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
    #[serde(rename_all = "kebab-case")]
    pub enum CollectorSource {
        #[default]
        Auto,
        ListEndpoint,
        PricingPage,
    }

    impl Default for CollectorProvider {
        fn default() -> Self {
            Self {
                name: None,
                kind: String::new(),
                endpoint: None,
                credential: None,
                credential_env: None,
                settings: Map::new(),
                source: CollectorSource::Auto,
                pricing_url: None,
                list_url: None,
            }
        }
    }

    #[derive(Serialize)]
    struct OutputDataset {
        #[serde(rename = "_comment")]
        comment: &'static str,
        #[serde(rename = "_comment_metadata")]
        comment_metadata: &'static str,
        #[serde(rename = "_comment_params")]
        comment_params: &'static str,
        version: u32,
        models: BTreeMap<String, ModelPricing>,
    }

    /// Read the current dataset file. Unknown/comment fields are ignored.
    pub(crate) fn read_seed(
        path: impl AsRef<Path>,
    ) -> Result<HashMap<String, ModelPricing>, serde_json::Error> {
        let text = std::fs::read_to_string(path).map_err(serde_json::Error::io)?;
        let ds: PricingDataset = serde_json::from_str(&text)?;
        Ok(ds.models)
    }

    /// Merge newly collected rows into the dataset schema.
    ///
    /// Reported `Some` / non-empty fields overwrite the seed. Missing fields preserve seed values, so
    /// pricing-page rows do not erase curated context/capability metadata. Rows absent from this run
    /// stay in the seed unless `prune` is explicitly set.
    pub(crate) fn merge_model_infos<I>(
        mut seed: HashMap<String, ModelPricing>,
        infos: I,
        prune: bool,
    ) -> HashMap<String, ModelPricing>
    where
        I: IntoIterator<Item = ModelInfo>,
    {
        let mut seen = HashSet::new();
        for info in infos {
            seen.insert(info.id.clone());
            let rate = seed.entry(info.id.clone()).or_default();
            apply_model_info(rate, info);
        }
        if prune {
            seed.retain(|id, _| seen.contains(id));
        }
        seed
    }

    fn apply_model_info(rate: &mut ModelPricing, info: ModelInfo) {
        if let Some(v) = info.modality {
            rate.modality = Some(v);
        }
        if let Some(v) = info.max_context {
            rate.max_context = Some(v);
        }
        if let Some(v) = info.max_output {
            rate.max_output = Some(v);
        }
        if let Some(v) = info.max_thinking {
            rate.max_thinking = Some(v);
        }
        if let Some(v) = info.supports_reasoning {
            rate.supports_reasoning = Some(v);
        }
        if let Some(v) = info.input_per_mtok {
            rate.input_per_mtok = Some(v);
        }
        if let Some(v) = info.output_per_mtok {
            rate.output_per_mtok = Some(v);
        }
        if !info.unsupported_parameters.is_empty() {
            rate.unsupported_parameters = Some(info.unsupported_parameters);
        }
        if !info.default_parameters.is_empty() {
            rate.default_parameters = Some(info.default_parameters);
        }
    }

    /// Serialize and write a deterministic `prices.json`.
    pub(crate) fn write_dataset(
        path: impl AsRef<Path>,
        models: HashMap<String, ModelPricing>,
    ) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
        let sorted: BTreeMap<String, ModelPricing> = models.into_iter().collect();
        let count = sorted.len();
        let out = OutputDataset {
            comment: COMMENT,
            comment_metadata: COMMENT_METADATA,
            comment_params: COMMENT_PARAMS,
            version: 5,
            models: sorted,
        };
        let text = serde_json::to_string_pretty(&out)?;
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, format!("{text}\n"))?;
        Ok(count)
    }

    pub async fn run(
        cfg: CollectorConfig,
    ) -> Result<Report, Box<dyn std::error::Error + Send + Sync>> {
        let output = cfg.output.unwrap_or_else(|| DEFAULT_OUTPUT.to_string());
        let seed_path = cfg
            .seed
            .clone()
            .or_else(|| Path::new(&output).exists().then(|| output.clone()));
        let mut models = match seed_path {
            Some(path) => read_seed(path)?,
            None => HashMap::new(),
        };
        let mut reports = Vec::new();
        let mut collected = Vec::new();
        let http = reqwest::Client::builder()
            .user_agent(concat!(
                "llmleaf-pricing-collect/",
                env!("CARGO_PKG_VERSION")
            ))
            .build()?;
        for p in cfg.providers {
            if p.kind.trim().is_empty() {
                return Err("collector provider is missing kind".into());
            }
            let name = p.name.clone().unwrap_or_else(|| p.kind.clone());
            let (source, mut infos) = collect_provider(&http, &p, &name).await?;
            reports.push(ProviderReport {
                name,
                kind: p.kind,
                source,
                models: infos.len(),
            });
            collected.append(&mut infos);
        }
        models = merge_model_infos(models, collected, cfg.prune);
        let model_count = write_dataset(&output, models)?;
        Ok(Report {
            output,
            providers: reports,
            model_count,
        })
    }

    async fn collect_provider(
        http: &reqwest::Client,
        p: &CollectorProvider,
        name: &str,
    ) -> Result<(String, Vec<ModelInfo>), Box<dyn std::error::Error + Send + Sync>> {
        match p.source {
            CollectorSource::PricingPage => collect_pricing_page(http, p, name).await,
            CollectorSource::ListEndpoint => collect_list_endpoint(p, name).await,
            CollectorSource::Auto => {
                if !pricing_page_urls(p).is_empty() {
                    collect_pricing_page(http, p, name).await
                } else {
                    collect_list_endpoint(p, name).await
                }
            }
        }
    }

    async fn collect_list_endpoint(
        p: &CollectorProvider,
        name: &str,
    ) -> Result<(String, Vec<ModelInfo>), Box<dyn std::error::Error + Send + Sync>> {
        let url = list_endpoint_url(p).ok_or_else(|| {
            format!(
                "{name} ({}) has no default priced list endpoint; set list_url or use a pricing-page collector",
                p.kind
            )
        })?;
        let http = reqwest::Client::builder()
            .user_agent(concat!(
                "llmleaf-pricing-collect/",
                env!("CARGO_PKG_VERSION")
            ))
            .build()?;
        let mut req = http.get(&url);
        if let Some(credential) = resolve_credential(p)? {
            req = req.bearer_auth(credential);
        }
        let value: Value = req.send().await?.error_for_status()?.json().await?;
        let priced = parse_priced_list_endpoint(value);
        if priced.is_empty() {
            return Err(format!(
                "{name} ({}) list endpoint returned no token-priced rows; use source = \"pricing-page\" with pricing_url or add a provider-specific page collector",
                p.kind
            )
            .into());
        }
        Ok((format!("list-endpoint:{url}"), priced))
    }

    fn list_endpoint_url(p: &CollectorProvider) -> Option<String> {
        if let Some(url) = &p.list_url {
            return Some(url.clone());
        }
        if let Some(endpoint) = &p.endpoint {
            return Some(format!("{}/models", endpoint.trim_end_matches('/')));
        }
        match normalized_kind(&p.kind).as_str() {
            "openrouter" => Some("https://openrouter.ai/api/v1/models".to_string()),
            "together" => Some("https://api.together.ai/v1/models".to_string()),
            "cerebras" => Some("https://api.cerebras.ai/public/v1/models".to_string()),
            "groq" => Some("https://api.groq.com/openai/v1/models".to_string()),
            "mistral" => Some("https://api.mistral.ai/v1/models".to_string()),
            "moonshot" | "kimi" | "kimi-k2" => {
                Some("https://api.moonshot.ai/v1/models".to_string())
            }
            _ => None,
        }
    }

    async fn collect_pricing_page(
        http: &reqwest::Client,
        p: &CollectorProvider,
        name: &str,
    ) -> Result<(String, Vec<ModelInfo>), Box<dyn std::error::Error + Send + Sync>> {
        let urls = pricing_page_urls(p);
        if urls.is_empty() {
            return Err(format!(
                "{name} ({}) has no built-in pricing page collector; set pricing_url and source = \"pricing-page\" only after adding parser support",
                p.kind
            )
            .into());
        }
        let mut infos = Vec::new();
        for url in &urls {
            let body = http
                .get(url)
                .send()
                .await?
                .error_for_status()?
                .text()
                .await?;
            // Moonshot publishes its tables as MDX. The rendered page omits the client-side table
            // from visible HTML text, while the official `.md` representation contains the rows.
            let lines = if url.ends_with(".md") {
                body.lines()
                    .map(str::trim)
                    .filter(|line| !line.is_empty())
                    .map(str::to_string)
                    .collect()
            } else {
                html_lines(&body)
            };
            let mut page_infos = match normalized_kind(&p.kind).as_str() {
                "cohere" => parse_cohere_pricing_lines(&lines),
                "anthropic" => parse_anthropic_pricing_lines(&lines),
                "mistral" => parse_mistral_pricing_lines(&lines),
                "openai" => parse_openai_pricing_lines(&lines),
                "moonshot" | "kimi" | "kimi-k2" => parse_moonshot_pricing_lines(&lines),
                _ => {
                    return Err(format!(
                        "{name} ({}) has no pricing-page parser; supported page parsers: openai, anthropic, cohere, mistral, moonshot",
                        p.kind
                    )
                    .into());
                }
            };
            infos.append(&mut page_infos);
        }
        let infos = dedup_model_infos(infos);
        if infos.is_empty() {
            return Err(format!(
                "pricing page parser for {name} ({}) found no priced model rows at {}",
                p.kind,
                urls.join(", ")
            )
            .into());
        }
        Ok((format!("pricing-page:{}", urls.join(",")), infos))
    }

    fn pricing_page_urls(p: &CollectorProvider) -> Vec<String> {
        if let Some(url) = &p.pricing_url {
            return vec![url.clone()];
        }
        match normalized_kind(&p.kind).as_str() {
            // OpenAI's former /api/pricing page now serves ChatGPT plan pricing. The API model
            // catalog publishes the current model ids and their per-MTok rates directly.
            "openai" => vec!["https://developers.openai.com/api/docs/models".to_string()],
            // The Claude Platform pricing reference exposes the complete base-price table in its
            // server-rendered HTML (including dated introductory prices).
            "anthropic" => {
                vec!["https://platform.claude.com/docs/en/about-claude/pricing".to_string()]
            }
            "cohere" => vec!["https://cohere.com/pricing".to_string()],
            "mistral" => vec!["https://mistral.ai/pricing/api/".to_string()],
            // Prices are split by model family; query every official MDX source and merge by id.
            "moonshot" | "kimi" | "kimi-k2" => [
                "chat-k3",
                "chat-k27-code",
                "chat-k26",
                "chat-k25",
                "chat-v1",
            ]
            .into_iter()
            .map(|page| format!("https://platform.kimi.ai/docs/pricing/{page}.md"))
            .collect(),
            _ => Vec::new(),
        }
    }

    pub(crate) fn parse_priced_list_endpoint(value: Value) -> Vec<ModelInfo> {
        let items = match value {
            Value::Array(arr) => arr,
            Value::Object(mut obj) => obj
                .remove("data")
                .or_else(|| obj.remove("models"))
                .and_then(|v| match v {
                    Value::Array(arr) => Some(arr),
                    _ => None,
                })
                .unwrap_or_default(),
            _ => Vec::new(),
        };
        items
            .into_iter()
            .filter_map(list_item_to_model_info)
            .filter(|m| m.input_per_mtok.is_some() || m.output_per_mtok.is_some())
            .collect()
    }

    fn list_item_to_model_info(item: Value) -> Option<ModelInfo> {
        let Value::Object(obj) = item else {
            return None;
        };
        let id = obj
            .get("id")
            .or_else(|| obj.get("key"))
            .or_else(|| obj.get("name"))
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())?
            .to_string();
        let mut info = ModelInfo::new(id);
        info.name = obj
            .get("name")
            .and_then(Value::as_str)
            .filter(|s| *s != info.id)
            .map(str::to_string);
        info.modality = list_item_modality(&obj);
        info.max_context = first_u32(
            &obj,
            &["context_window", "context_length", "max_context_length"],
        )
        .or_else(|| nested_u32(&obj, "top_provider", "context_length"))
        .or_else(|| nested_u32(&obj, "limits", "max_context_length"));
        info.max_output = first_u32(&obj, &["max_completion_tokens", "max_output_tokens"])
            .or_else(|| nested_u32(&obj, "top_provider", "max_completion_tokens"))
            .or_else(|| nested_u32(&obj, "limits", "max_completion_tokens"));
        if let Some(p) = obj.get("pricing").and_then(Value::as_object) {
            info.input_per_mtok = per_token_or_mtok_price(p, "prompt", "input");
            info.output_per_mtok = per_token_or_mtok_price(p, "completion", "output");
        }
        if let Some(Value::Array(params)) = obj.get("supported_parameters") {
            info.supported_parameters = Some(
                params
                    .iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect(),
            );
        }
        if let Some(Value::Object(params)) = obj.get("default_parameters") {
            info.default_parameters = params.clone();
        }
        Some(info)
    }

    fn per_token_or_mtok_price(
        p: &Map<String, Value>,
        per_token_key: &str,
        per_mtok_key: &str,
    ) -> Option<f64> {
        if let Some(x) = p.get(per_token_key).and_then(json_f64) {
            return (x >= 0.0).then_some(x * 1_000_000.0);
        }
        p.get(per_mtok_key)
            .and_then(json_f64)
            .and_then(|x| (x >= 0.0).then_some(x))
    }

    fn list_item_modality(obj: &Map<String, Value>) -> Option<Modality> {
        if let Some(t) = obj.get("type").and_then(Value::as_str) {
            match t.to_ascii_lowercase().as_str() {
                "chat" | "language" | "code" | "llm" | "vlm" => return Some(Modality::Llm),
                "embedding" | "embeddings" | "embed" => return Some(Modality::Embedding),
                "rerank" | "reranker" | "rank" => return Some(Modality::Rerank),
                _ => {}
            }
        }
        if let Some(outs) = obj
            .get("architecture")
            .and_then(Value::as_object)
            .and_then(|a| a.get("output_modalities"))
            .and_then(Value::as_array)
        {
            let outs: Vec<&str> = outs.iter().filter_map(Value::as_str).collect();
            if outs.iter().any(|o| *o == "audio" || *o == "speech") {
                return Some(Modality::Tts);
            }
            if outs.contains(&"text") {
                return Some(Modality::Llm);
            }
        }
        None
    }

    fn first_u32(obj: &Map<String, Value>, keys: &[&str]) -> Option<u32> {
        keys.iter().find_map(|k| obj.get(*k).and_then(json_u32))
    }

    fn nested_u32(obj: &Map<String, Value>, parent: &str, key: &str) -> Option<u32> {
        obj.get(parent)
            .and_then(Value::as_object)
            .and_then(|o| o.get(key))
            .and_then(json_u32)
    }

    fn json_u32(v: &Value) -> Option<u32> {
        v.as_u64()
            .or_else(|| v.as_str().and_then(|s| s.parse::<u64>().ok()))
            .map(|n| n.min(u32::MAX as u64) as u32)
    }

    fn json_f64(v: &Value) -> Option<f64> {
        v.as_f64()
            .or_else(|| v.as_str().and_then(|s| s.parse::<f64>().ok()))
    }

    /// Parse Moonshot's official MDX pricing rows. Current families use either three prices
    /// (cache-hit input, cache-miss input, output) or two (input, output); llmleaf's base input rate
    /// is the cache-miss rate. Context windows ride in the same row, so the offline dataset can
    /// enrich explicitly routed models even when no live catalog is queried.
    pub(crate) fn parse_moonshot_pricing_lines(lines: &[String]) -> Vec<ModelInfo> {
        let mut out = Vec::new();
        for line in lines {
            let Some(id) = line
                .strip_prefix("[\"")
                .and_then(|rest| rest.split('"').next())
                .filter(|id| id.starts_with("kimi-") || id.starts_with("moonshot-v1-"))
            else {
                continue;
            };
            let prices = moonshot_mdx_prices(line);
            let (input, output) = match prices.as_slice() {
                [input, output] => (*input, *output),
                [_, cache_miss, output, ..] => (*cache_miss, *output),
                _ => continue,
            };
            let context = line
                .split('"')
                .filter(|field| field.trim_end().ends_with("tokens"))
                .find_map(|field| {
                    field
                        .trim_end_matches("tokens")
                        .trim()
                        .replace(',', "")
                        .parse::<u32>()
                        .ok()
                });

            let mut info = ModelInfo::new(id);
            info.modality = Some(Modality::Llm);
            info.max_context = context;
            info.input_per_mtok = Some(input);
            info.output_per_mtok = Some(output);
            out.push(info);
        }
        out
    }

    fn moonshot_mdx_prices(line: &str) -> Vec<f64> {
        let marker = r#"{"$"}"#;
        let mut rest = line;
        let mut prices = Vec::new();
        while let Some(at) = rest.find(marker) {
            rest = &rest[at + marker.len()..];
            let start = rest
                .char_indices()
                .find_map(|(i, ch)| (ch.is_ascii_digit() || ch == '.').then_some(i));
            let Some(start) = start else { break };
            let end = rest[start..]
                .char_indices()
                .find_map(|(i, ch)| (!(ch.is_ascii_digit() || ch == '.')).then_some(start + i))
                .unwrap_or(rest.len());
            if let Ok(price) = rest[start..end].parse() {
                prices.push(price);
            }
            rest = &rest[end..];
        }
        if prices.is_empty() {
            dollar_prices(line)
        } else {
            prices
        }
    }

    pub(crate) fn parse_openai_pricing_lines(lines: &[String]) -> Vec<ModelInfo> {
        let mut structured = Vec::new();
        for (i, line) in lines.iter().enumerate() {
            if line != "Model ID" {
                continue;
            }
            let Some(id) = lines.get(i + 1).filter(|id| is_openai_api_model_id(id)) else {
                continue;
            };
            let end = lines[i + 2..]
                .iter()
                .position(|line| line == "Model ID")
                .map(|offset| i + 2 + offset)
                .unwrap_or_else(|| lines.len().min(i + 48));
            let window = &lines[i + 2..end];
            let input = labeled_price(window, "Input price");
            let output = labeled_price(window, "Output price");
            if input.is_none() && output.is_none() {
                continue;
            }
            let mut info = ModelInfo::new(id.clone());
            info.modality = Some(Modality::Llm);
            info.input_per_mtok = input;
            info.output_per_mtok = output;
            info.max_context = labeled_token_count(window, "Context window");
            info.max_output = labeled_token_count(window, "Max output");
            info.supports_reasoning = window
                .iter()
                .any(|line| line == "Reasoning")
                .then_some(true);
            structured.push(info.clone());

            if let Some(alias) = labeled_value(window, "Alias").filter(|id| {
                is_openai_api_model_id(id) && !structured.iter().any(|info| info.id == **id)
            }) {
                info.id = alias.clone();
                structured.push(info);
            }
        }
        if !structured.is_empty() {
            return structured;
        }

        // Compatibility with the former pricing page and with explicitly overridden sources.
        let mut out = Vec::new();
        for (i, line) in lines.iter().enumerate() {
            let Some(id) = openai_model_id(line) else {
                continue;
            };
            let end = lines[i + 1..]
                .iter()
                .position(|line| openai_model_id(line).is_some())
                .map(|offset| i + 1 + offset)
                .unwrap_or_else(|| lines.len().min(i + 48));
            let window = &lines[i + 1..end];
            let input =
                labeled_price(window, "Input:").or_else(|| labeled_price(window, "Input price"));
            let output =
                labeled_price(window, "Output:").or_else(|| labeled_price(window, "Output price"));
            if input.is_none() && output.is_none() {
                continue;
            }
            let mut info = ModelInfo::new(id);
            info.modality = Some(Modality::Llm);
            info.input_per_mtok = input;
            info.output_per_mtok = output;
            info.max_context = labeled_token_count(window, "Context window");
            info.max_output = labeled_token_count(window, "Max output");
            out.push(info);
        }
        out
    }

    fn is_openai_api_model_id(id: &str) -> bool {
        let valid_prefix = ["gpt-", "o1", "o3", "o4"]
            .iter()
            .any(|prefix| id.starts_with(prefix));
        valid_prefix
            && id
                .chars()
                .all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || matches!(ch, '-' | '.'))
    }

    pub(crate) fn parse_anthropic_pricing_lines(lines: &[String]) -> Vec<ModelInfo> {
        parse_anthropic_pricing_lines_at(lines, current_utc_date_key())
    }

    pub(crate) fn parse_anthropic_pricing_lines_at(lines: &[String], today: u32) -> Vec<ModelInfo> {
        let mut out = Vec::new();
        for (i, line) in lines.iter().enumerate() {
            let Some(name_end) = line.find('$') else {
                // The Claude Platform docs render each table cell as a separate text node. Base
                // pricing rows have five prices (input, two cache writes, cache read, output);
                // fast-mode and batch tables have only two and must not replace base rates.
                if line.starts_with("Claude ") {
                    let end = lines[i + 1..]
                        .iter()
                        .position(|line| line.starts_with("Claude "))
                        .map(|offset| i + 1 + offset)
                        .unwrap_or_else(|| lines.len().min(i + 9));
                    let window = &lines[i + 1..end];
                    if !anthropic_row_applies(window.first().map(String::as_str), today) {
                        continue;
                    }
                    let prices = prices_in_lines(window, 8);
                    if prices.len() >= 5 {
                        let name = strip_parenthetical(line);
                        let mut info = ModelInfo::new(anthropic_label_to_model_id(&name));
                        info.modality = Some(Modality::Llm);
                        info.input_per_mtok = prices.first().copied();
                        info.output_per_mtok = prices.get(4).copied();
                        info.unsupported_parameters = anthropic_unsupported_parameters(&name);
                        out.push(info);
                    }
                }
                if let Some(label) = anthropic_heading_label(line) {
                    let end = lines[i + 1..]
                        .iter()
                        .position(|line| anthropic_heading_label(line).is_some())
                        .map(|offset| i + 1 + offset)
                        .unwrap_or_else(|| lines.len().min(i + 24));
                    let window = &lines[i + 1..end];
                    let input = anthropic_labeled_price(window, "Input");
                    let output = anthropic_labeled_price(window, "Output");
                    if input.is_some() || output.is_some() {
                        let mut info = ModelInfo::new(anthropic_label_to_model_id(&label));
                        info.modality = Some(Modality::Llm);
                        info.input_per_mtok = input;
                        info.output_per_mtok = output;
                        info.unsupported_parameters = anthropic_unsupported_parameters(&label);
                        out.push(info);
                    }
                }
                continue;
            };
            let name = strip_parenthetical(line[..name_end].trim());
            if !name.starts_with("Claude ") {
                continue;
            }
            let prices = dollar_prices(line);
            if prices.len() < 2 {
                continue;
            }
            let mut info = ModelInfo::new(anthropic_label_to_model_id(&name));
            info.modality = Some(Modality::Llm);
            info.input_per_mtok = prices.first().copied();
            info.output_per_mtok = prices.last().copied();
            info.unsupported_parameters = anthropic_unsupported_parameters(&name);
            out.push(info);
        }
        out
    }

    fn anthropic_row_applies(qualifier: Option<&str>, today: u32) -> bool {
        let Some(qualifier) = qualifier else {
            return true;
        };
        if let Some(raw) = qualifier.strip_prefix("through ") {
            return parse_english_date_key(raw).is_none_or(|last_day| today <= last_day);
        }
        if let Some(raw) = qualifier.strip_prefix("starting ") {
            return parse_english_date_key(raw).is_none_or(|first_day| today >= first_day);
        }
        true
    }

    fn parse_english_date_key(raw: &str) -> Option<u32> {
        let parts = raw.replace(',', "");
        let mut parts = parts.split_whitespace();
        let month = match parts.next()? {
            "January" => 1,
            "February" => 2,
            "March" => 3,
            "April" => 4,
            "May" => 5,
            "June" => 6,
            "July" => 7,
            "August" => 8,
            "September" => 9,
            "October" => 10,
            "November" => 11,
            "December" => 12,
            _ => return None,
        };
        let day = parts.next()?.parse::<u32>().ok()?;
        let year = parts.next()?.parse::<u32>().ok()?;
        (parts.next().is_none() && (1..=31).contains(&day))
            .then_some(year * 10_000 + month * 100 + day)
    }

    fn current_utc_date_key() -> u32 {
        let seconds = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let z = (seconds / 86_400) as i64 + 719_468;
        let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
        let day_of_era = z - era * 146_097;
        let year_of_era =
            (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
        let mut year = year_of_era + era * 400;
        let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
        let month_prime = (5 * day_of_year + 2) / 153;
        let day = day_of_year - (153 * month_prime + 2) / 5 + 1;
        let month = month_prime + if month_prime < 10 { 3 } else { -9 };
        year += i64::from(month <= 2);
        (year as u32) * 10_000 + (month as u32) * 100 + day as u32
    }

    fn anthropic_heading_label(line: &str) -> Option<String> {
        let label = line.trim().trim_start_matches('#').trim();
        let first = label.split_whitespace().next()?;
        matches!(first, "Fable" | "Mythos" | "Opus" | "Sonnet" | "Haiku")
            .then_some(format!("Claude {label}"))
    }

    fn anthropic_labeled_price(lines: &[String], label: &str) -> Option<f64> {
        let pos = lines.iter().position(|line| line.trim() == label)?;
        prices_in_lines(&lines[pos + 1..], 6).into_iter().next()
    }

    fn anthropic_unsupported_parameters(label: &str) -> Vec<String> {
        let mut params = vec![
            "frequency_penalty".to_string(),
            "presence_penalty".to_string(),
        ];
        if anthropic_removes_sampling_parameters(label) {
            params.extend([
                "temperature".to_string(),
                "top_p".to_string(),
                "top_k".to_string(),
            ]);
        }
        params
    }

    fn anthropic_removes_sampling_parameters(label: &str) -> bool {
        let id = anthropic_label_to_model_id(label);
        if id.starts_with("claude-fable-") {
            return true;
        }
        let Some(suffix) = id.strip_prefix("claude-opus-4-") else {
            return false;
        };
        suffix
            .split('-')
            .next()
            .and_then(|n| n.parse::<u32>().ok())
            .is_some_and(|minor| minor >= 7)
    }

    pub(crate) fn parse_cohere_pricing_lines(lines: &[String]) -> Vec<ModelInfo> {
        let mut out = Vec::new();
        for line in lines {
            let lower = line.to_ascii_lowercase();
            if let Some(aya_at) = lower.find("aya expanse models") {
                if let Some((input, output)) = input_output_prices(&line[aya_at..]) {
                    for id in ["aya-expanse-8b", "aya-expanse-32b"] {
                        let mut info = ModelInfo::new(id);
                        info.modality = Some(Modality::Llm);
                        info.input_per_mtok = Some(input);
                        info.output_per_mtok = Some(output);
                        out.push(info);
                    }
                }
                continue;
            }
            let Some(pricing_at) = lower.find(" pricing is ") else {
                continue;
            };
            let label = line[..pricing_at].trim().trim_start_matches("* ").trim();
            if label.is_empty() {
                continue;
            }
            if let Some((input, output)) = input_output_prices(&line[pricing_at..]) {
                let mut info = ModelInfo::new(cohere_label_to_model_id(label));
                info.modality = Some(Modality::Llm);
                info.input_per_mtok = Some(input);
                info.output_per_mtok = Some(output);
                if info.id == "command-r-plus" || info.id.starts_with("command-r-plus-") {
                    info.default_parameters
                        .insert("temperature".into(), Value::from(0.3));
                }
                out.push(info);
            }
        }
        out
    }

    pub(crate) fn parse_mistral_pricing_lines(lines: &[String]) -> Vec<ModelInfo> {
        let mut out = Vec::new();
        for (i, line) in lines.iter().enumerate() {
            if !is_mistral_model_heading(line) {
                continue;
            }
            let end = lines[i + 1..]
                .iter()
                .position(|line| is_mistral_model_heading(line))
                .map(|offset| i + 1 + offset)
                .unwrap_or_else(|| lines.len().min(i + 24));
            let window = &lines[i + 1..end];
            let input = mistral_labeled_price(window, "Input");
            let output = mistral_labeled_price(window, "Output");
            if input.is_none() && output.is_none() {
                continue;
            }
            let mut info = ModelInfo::new(label_to_model_id(line));
            let lower = line.to_ascii_lowercase();
            info.modality = Some(if lower.contains("embed") {
                Modality::Embedding
            } else if lower.contains("voxtral") {
                Modality::Stt
            } else {
                Modality::Llm
            });
            info.input_per_mtok = input;
            info.output_per_mtok = output;
            out.push(info);
        }
        out
    }

    fn is_mistral_model_heading(line: &str) -> bool {
        let lower = line.to_ascii_lowercase();
        (lower.starts_with("mistral ")
            || lower.starts_with("ministral ")
            || lower.starts_with("magistral ")
            || lower.starts_with("devstral ")
            || lower.starts_with("codestral")
            || lower.starts_with("voxtral "))
            && !lower.contains("pricing")
            && !lower.contains("documentation")
    }

    fn mistral_labeled_price(lines: &[String], label: &str) -> Option<f64> {
        let pos = lines.iter().position(|line| {
            line.starts_with(label)
                && (line.contains("/M tokens")
                    || line.contains("/M tok")
                    || line.contains("per M tok"))
        })?;
        let window = next_price_window(&lines[pos + 1..]);
        let prices = prices_in_lines(window, 5);
        if label == "Input"
            && prices.len() >= 2
            && window.iter().take(5).any(|line| {
                let lower = line.to_ascii_lowercase();
                lower.contains("audio") || lower.contains("text")
            })
        {
            prices.last().copied()
        } else {
            prices.first().copied()
        }
    }

    fn labeled_price(lines: &[String], label: &str) -> Option<f64> {
        let pos = lines.iter().position(|line| line == label)?;
        prices_in_lines(&lines[pos + 1..], 4).into_iter().next()
    }

    fn labeled_value<'a>(lines: &'a [String], label: &str) -> Option<&'a String> {
        let pos = lines.iter().position(|line| line == label)?;
        lines.get(pos + 1)
    }

    fn labeled_token_count(lines: &[String], label: &str) -> Option<u32> {
        let pos = lines.iter().position(|line| line == label)?;
        lines[pos + 1..]
            .iter()
            .take(3)
            .find_map(|line| parse_token_count(line))
    }

    fn parse_token_count(line: &str) -> Option<u32> {
        let raw = line
            .trim()
            .trim_end_matches("tokens")
            .trim()
            .replace(',', "");
        let (number, multiplier) = match raw.chars().last()? {
            'K' | 'k' => (&raw[..raw.len() - 1], 1_000.0),
            'M' | 'm' => (&raw[..raw.len() - 1], 1_000_000.0),
            ch if ch.is_ascii_digit() => (raw.as_str(), 1.0),
            _ => return None,
        };
        let tokens = number.parse::<f64>().ok()? * multiplier;
        (tokens.is_finite() && tokens >= 0.0 && tokens <= u32::MAX as f64)
            .then_some(tokens.round() as u32)
    }

    fn next_price_window(lines: &[String]) -> &[String] {
        let end = lines
            .iter()
            .position(|line| line.starts_with("Input") || line.starts_with("Output"))
            .unwrap_or(lines.len());
        &lines[..end]
    }

    fn input_output_prices(line: &str) -> Option<(f64, f64)> {
        let prices = dollar_prices(line);
        if prices.len() < 2 {
            return None;
        }
        Some((prices[0], prices[1]))
    }

    fn dollar_prices(line: &str) -> Vec<f64> {
        let mut prices = Vec::new();
        let bytes = line.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] != b'$' {
                i += 1;
                continue;
            }
            i += 1;
            while i < bytes.len() && bytes[i].is_ascii_whitespace() {
                i += 1;
            }
            let start = i;
            while i < bytes.len()
                && (bytes[i].is_ascii_digit() || bytes[i] == b'.' || bytes[i] == b',')
            {
                i += 1;
            }
            if start < i {
                if let Ok(v) = line[start..i].replace(',', "").parse::<f64>() {
                    prices.push(v);
                }
            }
        }
        prices
    }

    fn prices_in_lines(lines: &[String], max_lines: usize) -> Vec<f64> {
        let mut prices = Vec::new();
        let window = lines.iter().take(max_lines).collect::<Vec<_>>();
        let mut skip_numeric_after_split_dollar = None;

        for (i, line) in window.iter().enumerate() {
            if skip_numeric_after_split_dollar == Some(i) {
                continue;
            }

            let mut found = dollar_prices(line);
            if found.is_empty() && line.trim() == "$" {
                if let Some((offset, price)) = window
                    .iter()
                    .enumerate()
                    .skip(i + 1)
                    .take(3)
                    .find_map(|(j, candidate)| plain_price(candidate).map(|price| (j, price)))
                {
                    found.push(price);
                    skip_numeric_after_split_dollar = Some(offset);
                }
            }
            prices.extend(found);
        }

        prices
    }

    fn plain_price(line: &str) -> Option<f64> {
        let s = line.trim().trim_start_matches('$').replace(',', "");
        if s.is_empty() || !s.chars().all(|ch| ch.is_ascii_digit() || ch == '.') {
            return None;
        }
        s.parse::<f64>().ok()
    }

    fn openai_model_id(label: &str) -> Option<String> {
        let label = label.trim();
        if !(label.starts_with("GPT-")
            || label.starts_with("O1")
            || label.starts_with("O3")
            || label.starts_with("O4"))
        {
            return None;
        }
        let id = label_to_model_id(label);
        (id.starts_with("gpt-")
            || id.starts_with("o1")
            || id.starts_with("o3")
            || id.starts_with("o4"))
        .then_some(id)
    }

    fn anthropic_label_to_model_id(label: &str) -> String {
        label_to_model_id(label).replace('.', "-")
    }

    fn cohere_label_to_model_id(label: &str) -> String {
        label_to_model_id(&label.replace('+', " plus"))
    }

    fn label_to_model_id(label: &str) -> String {
        let mut out = String::new();
        let mut dash = false;
        for ch in strip_parenthetical(label).chars() {
            let ch = ch.to_ascii_lowercase();
            if ch.is_ascii_alphanumeric() || ch == '.' {
                if dash && !out.is_empty() {
                    out.push('-');
                }
                out.push(ch);
                dash = false;
            } else {
                dash = true;
            }
        }
        out.trim_matches('-').to_string()
    }

    fn dedup_model_infos(infos: Vec<ModelInfo>) -> Vec<ModelInfo> {
        let mut by_id = HashMap::new();
        for info in infos {
            by_id.insert(info.id.clone(), info);
        }
        by_id.into_values().collect()
    }

    fn strip_parenthetical(s: &str) -> String {
        let mut out = String::new();
        let mut depth = 0usize;
        for ch in s.chars() {
            match ch {
                '(' => depth += 1,
                ')' => depth = depth.saturating_sub(1),
                _ if depth == 0 => out.push(ch),
                _ => {}
            }
        }
        out.trim().to_string()
    }

    fn html_lines(html: &str) -> Vec<String> {
        let doc = scraper::Html::parse_document(html);
        doc.root_element()
            .text()
            .flat_map(|text| text.split('\n'))
            .map(|line| line.replace('\u{a0}', " "))
            .map(|line| line.trim().to_string())
            .filter(|line| !line.is_empty())
            .collect()
    }

    fn normalized_kind(kind: &str) -> String {
        match kind {
            "claude" => "anthropic",
            other => other,
        }
        .to_ascii_lowercase()
    }

    fn resolve_credential(
        p: &CollectorProvider,
    ) -> Result<Option<String>, Box<dyn std::error::Error + Send + Sync>> {
        if let Some(env) = &p.credential_env {
            return std::env::var(env)
                .map(Some)
                .map_err(|e| format!("reading credential_env {env}: {e}").into());
        }
        Ok(p.credential.clone())
    }
}

pub use collect::{
    describe, run, CollectorConfig, CollectorProvider, CollectorSource, ProviderReport, Report,
    DEFAULT_OUTPUT,
};

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    #[test]
    fn collector_merge_preserves_seed_when_catalog_is_id_only() {
        let mut seed = HashMap::new();
        seed.insert(
            "gpt-4o".to_string(),
            ModelPricing {
                input_per_mtok: Some(2.5),
                output_per_mtok: Some(10.0),
                modality: Some(Modality::Llm),
                max_context: Some(128_000),
                ..ModelPricing::default()
            },
        );

        let out = collect::merge_model_infos(seed, [ModelInfo::new("gpt-4o")], false);
        let rate = out.get("gpt-4o").unwrap();
        assert_eq!(rate.input_per_mtok, Some(2.5));
        assert_eq!(rate.output_per_mtok, Some(10.0));
        assert_eq!(rate.modality, Some(Modality::Llm));
        assert_eq!(rate.max_context, Some(128_000));
    }

    #[test]
    fn collector_merge_updates_reported_fields_and_can_prune() {
        let mut seed = HashMap::new();
        seed.insert(
            "old".to_string(),
            ModelPricing {
                input_per_mtok: Some(99.0),
                ..ModelPricing::default()
            },
        );

        let mut info = ModelInfo::new("fresh");
        info.modality = Some(Modality::Embedding);
        info.max_context = Some(8192);
        info.input_per_mtok = Some(0.02);
        info.output_per_mtok = Some(0.0);

        let out = collect::merge_model_infos(seed, [info], true);
        assert!(!out.contains_key("old"));
        let rate = out.get("fresh").unwrap();
        assert_eq!(rate.modality, Some(Modality::Embedding));
        assert_eq!(rate.max_context, Some(8192));
        assert_eq!(rate.input_per_mtok, Some(0.02));
        assert_eq!(rate.output_per_mtok, Some(0.0));
    }

    #[test]
    fn list_endpoint_parser_keeps_only_token_priced_rows() {
        let rows = collect::parse_priced_list_endpoint(serde_json::json!({
            "data": [
                {
                    "id": "openrouter/model-a",
                    "context_length": 128000,
                    "pricing": { "prompt": "0.000003", "completion": "0.000015" },
                    "architecture": { "output_modalities": ["text"] }
                },
                { "id": "id-only" }
            ]
        }));

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].id, "openrouter/model-a");
        assert_eq!(rows[0].input_per_mtok, Some(3.0));
        assert_eq!(rows[0].output_per_mtok, Some(15.0));
        assert_eq!(rows[0].max_context, Some(128_000));
        assert_eq!(rows[0].modality, Some(Modality::Llm));
    }

    #[test]
    fn openai_pricing_page_parser_reads_input_output_pairs() {
        let lines = vec![
            "GPT-5.5",
            "A new class of intelligence.",
            "Price",
            "Input:",
            "$5.00 / 1M tokens",
            "Cached input:",
            "$0.50 / 1M tokens",
            "Output:",
            "$30.00 / 1M tokens",
        ]
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();

        let rows = collect::parse_openai_pricing_lines(&lines);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].id, "gpt-5.5");
        assert_eq!(rows[0].input_per_mtok, Some(5.0));
        assert_eq!(rows[0].output_per_mtok, Some(30.0));
    }

    #[test]
    fn openai_model_catalog_parser_reads_ids_alias_prices_and_limits() {
        let lines = vec![
            "GPT-5.6 Sol",
            "Frontier model for complex professional work",
            "Model ID",
            "gpt-5.6-sol",
            "Alias",
            "gpt-5.6",
            "Reasoning",
            "none",
            "low",
            "high",
            "Input price",
            "$5 / Input MTok",
            "Output price",
            "$30 / Output MTok",
            "Max output",
            "128K tokens",
            "Context window",
            "1.05M",
            "GPT-5.6 Terra",
            "Model ID",
            "gpt-5.6-terra",
            "Reasoning",
            "none",
            "Input price",
            "$2.50 / Input MTok",
            "Output price",
            "$15 / Output MTok",
            "Max output",
            "128K tokens",
            "Context window",
            "1.05M",
        ]
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();

        let rows = collect::parse_openai_pricing_lines(&lines);
        assert_eq!(rows.len(), 3);
        let sol = rows.iter().find(|row| row.id == "gpt-5.6-sol").unwrap();
        assert_eq!(sol.input_per_mtok, Some(5.0));
        assert_eq!(sol.output_per_mtok, Some(30.0));
        assert_eq!(sol.max_context, Some(1_050_000));
        assert_eq!(sol.max_output, Some(128_000));
        assert_eq!(sol.supports_reasoning, Some(true));
        let alias = rows.iter().find(|row| row.id == "gpt-5.6").unwrap();
        assert_eq!(alias.input_per_mtok, Some(5.0));
        assert_eq!(alias.max_context, Some(1_050_000));
        let terra = rows.iter().find(|row| row.id == "gpt-5.6-terra").unwrap();
        assert_eq!(terra.input_per_mtok, Some(2.5));
        assert_eq!(terra.output_per_mtok, Some(15.0));
    }

    #[test]
    fn moonshot_pricing_parser_reads_cached_and_classic_rows() {
        let lines = vec![
            r#"["kimi-k3", "1M tokens", <>{"$"}0.30</>, <>{"$"}3.00</>, <>{"$"}15.00</>, "1,048,576 tokens"],"#,
            r#"["moonshot-v1-8k", "1M tokens", <>{"$"}0.20</>, <>{"$"}2.00</>, "8,192 tokens"],"#,
        ]
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();

        let rows = collect::parse_moonshot_pricing_lines(&lines);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].id, "kimi-k3");
        assert_eq!(rows[0].input_per_mtok, Some(3.0));
        assert_eq!(rows[0].output_per_mtok, Some(15.0));
        assert_eq!(rows[0].max_context, Some(1_048_576));
        assert_eq!(rows[1].id, "moonshot-v1-8k");
        assert_eq!(rows[1].input_per_mtok, Some(0.2));
        assert_eq!(rows[1].output_per_mtok, Some(2.0));
        assert_eq!(rows[1].max_context, Some(8_192));
    }

    #[test]
    fn anthropic_pricing_page_parser_reads_table_rows() {
        let lines = vec![
            "Claude Opus 4.8$5 / MTok$6.25 / MTok$10 / MTok$0.50 / MTok$25 / MTok",
            "Claude Sonnet 4.6$3 / MTok$3.75 / MTok$6 / MTok$0.30 / MTok$15 / MTok",
        ]
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();

        let rows = collect::parse_anthropic_pricing_lines(&lines);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].id, "claude-opus-4-8");
        assert_eq!(rows[0].input_per_mtok, Some(5.0));
        assert_eq!(rows[0].output_per_mtok, Some(25.0));
        assert!(rows[0]
            .unsupported_parameters
            .contains(&"frequency_penalty".to_string()));
    }

    #[test]
    fn anthropic_pricing_page_parser_reads_split_api_cards() {
        let lines = vec![
            "### Fable 5",
            "Next generation intelligence for long-running agents",
            "Input",
            "$",
            "10",
            "/ MTok",
            "Output",
            "$",
            "50",
            "/ MTok",
            "Prompt caching",
            "Write",
            "$12.50 / MTok",
            "Read",
            "$1 / MTok",
            "### Opus 4.8",
            "Input",
            "$",
            "5",
            "/ MTok",
            "Output",
            "$",
            "25",
            "/ MTok",
        ]
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();

        let rows = collect::parse_anthropic_pricing_lines(&lines);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].id, "claude-fable-5");
        assert_eq!(rows[0].input_per_mtok, Some(10.0));
        assert_eq!(rows[0].output_per_mtok, Some(50.0));
        assert_eq!(rows[1].id, "claude-opus-4-8");
        assert_eq!(rows[1].input_per_mtok, Some(5.0));
        assert_eq!(rows[1].output_per_mtok, Some(25.0));
    }

    #[test]
    fn anthropic_pricing_table_uses_base_and_current_introductory_rates() {
        let lines = vec![
            "Claude Opus 5",
            "$5 / MTok",
            "$6.25 / MTok",
            "$10 / MTok",
            "$0.50 / MTok",
            "$25 / MTok",
            "Claude Sonnet 5",
            "through August 31, 2026",
            "$2 / MTok",
            "$2.50 / MTok",
            "$4 / MTok",
            "$0.20 / MTok",
            "$10 / MTok",
            "Claude Sonnet 5",
            "starting September 1, 2026",
            "$3 / MTok",
            "$3.75 / MTok",
            "$6 / MTok",
            "$0.30 / MTok",
            "$15 / MTok",
            // A later fast-mode table must not overwrite the base rate.
            "Claude Opus 5 / Claude Opus 4.8",
            "$10 / MTok",
            "$50 / MTok",
        ]
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();

        let rows = collect::parse_anthropic_pricing_lines_at(&lines, 20_260_725);
        assert_eq!(rows.len(), 2);
        let opus = rows.iter().find(|row| row.id == "claude-opus-5").unwrap();
        assert_eq!(opus.input_per_mtok, Some(5.0));
        assert_eq!(opus.output_per_mtok, Some(25.0));
        let sonnet = rows.iter().find(|row| row.id == "claude-sonnet-5").unwrap();
        assert_eq!(sonnet.input_per_mtok, Some(2.0));
        assert_eq!(sonnet.output_per_mtok, Some(10.0));

        let later = collect::parse_anthropic_pricing_lines_at(&lines, 20_260_901);
        let sonnet = later
            .iter()
            .find(|row| row.id == "claude-sonnet-5")
            .unwrap();
        assert_eq!(sonnet.input_per_mtok, Some(3.0));
        assert_eq!(sonnet.output_per_mtok, Some(15.0));
    }

    #[test]
    fn cohere_pricing_page_parser_reads_legacy_and_aya_rows() {
        let lines = vec![
            "Command R+ 08-2024 pricing is $2.50/1M tokens for input and $10.00/1M tokens for output",
            // Hydration JSON may put unrelated dollar values before the pricing sentence.
            "$15 unrelated Aya Expanse models (8B and 32B) on the API are charged at $0.50/1M tokens for input and $1.50/1M tokens for output.",
        ]
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();

        let rows = collect::parse_cohere_pricing_lines(&lines);
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].id, "command-r-plus-08-2024");
        assert_eq!(rows[0].input_per_mtok, Some(2.5));
        assert_eq!(rows[0].output_per_mtok, Some(10.0));
        assert_eq!(
            rows[0].default_parameters.get("temperature"),
            Some(&serde_json::json!(0.3))
        );
        assert!(rows.iter().any(|m| m.id == "aya-expanse-8b"));
        assert!(rows.iter().any(|m| m.id == "aya-expanse-32b"));
        let aya = rows.iter().find(|m| m.id == "aya-expanse-8b").unwrap();
        assert_eq!(aya.input_per_mtok, Some(0.5));
        assert_eq!(aya.output_per_mtok, Some(1.5));
    }

    #[test]
    fn mistral_pricing_page_parser_reads_api_cards_only() {
        let lines = vec![
            "Mistral Medium 3.5",
            "Open",
            "Text-to-text",
            "Input (/M tokens)",
            "$1.5",
            "Output (/M tokens)",
            "$7.5",
            "Voxtral TTS",
            "Audio generation",
            "$0.016 per 1k characters",
            "Voxtral Small",
            "Transcription",
            "Text-to-text",
            "Input (per min / per M tok)",
            "$0.004 (audio) / $0.1 (text)",
            "Output (/M tokens)",
            "$0.4",
        ]
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();

        let rows = collect::parse_mistral_pricing_lines(&lines);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].id, "mistral-medium-3.5");
        assert_eq!(rows[0].input_per_mtok, Some(1.5));
        assert_eq!(rows[0].output_per_mtok, Some(7.5));
        assert_eq!(rows[1].id, "voxtral-small");
        assert_eq!(rows[1].input_per_mtok, Some(0.1));
        assert_eq!(rows[1].output_per_mtok, Some(0.4));
        assert_eq!(rows[1].modality, Some(Modality::Stt));

        let embed_lines = ["Codestral Embed", "Embedding", "Input (/M tokens)", "$0.15"]
            .into_iter()
            .map(str::to_string)
            .collect::<Vec<_>>();
        let embed = collect::parse_mistral_pricing_lines(&embed_lines);
        assert_eq!(embed.len(), 1);
        assert_eq!(embed[0].id, "codestral-embed");
        assert_eq!(embed[0].modality, Some(Modality::Embedding));
    }
}
