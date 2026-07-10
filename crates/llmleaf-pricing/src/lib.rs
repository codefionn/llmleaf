//! The one sanctioned sidecar crate: pricing (SOUL.md "One sidecar crate — pricing").
//!
//! Collection of pricing data from providers happens OFFLINE (see [`collect`]). The resulting
//! dataset is bundled into the binary at compile time via `include_str!` and the core only ever
//! *reads* it. Cost reporting on the hot path is a lookup plus a couple of multiplies against the
//! looked-up rate — never a network fetch (principle: "consumption is a lookup, like everything else").

#[cfg(feature = "collect")]
use std::collections::BTreeMap;
use std::collections::HashMap;
#[cfg(any(test, feature = "collect"))]
use std::collections::HashSet;
#[cfg(feature = "collect")]
use std::path::Path;

#[cfg(any(test, feature = "collect"))]
use llmleaf_model::ModelInfo;
use llmleaf_model::{Modality, Usage};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// The dataset that ships with the core.
const BUNDLED: &str = include_str!("../data/prices.json");

#[derive(Debug, Clone, Deserialize)]
struct Dataset {
    #[allow(dead_code)]
    version: u32,
    models: HashMap<String, Rate>,
}

/// One row of the bundled dataset. Beyond the token rates it now carries optional *capability*
/// metadata (modality + published limits) consumed by the model-catalog surface (`GET /v1/models`).
///
/// Every field is optional. Token rates are `Option` because some models are not token-priced
/// (per-character TTS, per-minute STT) — a `None` rate means "not token-priced", never `0`. The
/// metadata fields are `None` when the offline collector has not recorded them; a `None` must be
/// rendered as "unknown", never defaulted to a number (SOUL: no silent magic, no guessing).
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
struct Rate {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    input_per_mtok: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    output_per_mtok: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    modality: Option<Modality>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    max_context: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    max_output: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    max_thinking: Option<u32>,
    /// Canonical sampling parameters this model REJECTS (e.g. a reasoning model's `temperature`, or
    /// Anthropic's `frequency_penalty`). Omitted ⇒ `None` ⇒ "no restriction collected", never a guess.
    #[serde(default, skip_serializing_if = "opt_vec_is_none_or_empty")]
    unsupported_parameters: Option<Vec<String>>,
    /// Recommended default values for sampling parameters (e.g. `{"temperature": 0.3}`). Omitted ⇒
    /// `None` ⇒ "none collected".
    #[serde(default, skip_serializing_if = "opt_map_is_none_or_empty")]
    default_parameters: Option<Map<String, Value>>,
}

fn opt_vec_is_none_or_empty(v: &Option<Vec<String>>) -> bool {
    v.as_ref().is_none_or(Vec::is_empty)
}

fn opt_map_is_none_or_empty(v: &Option<Map<String, Value>>) -> bool {
    v.as_ref().is_none_or(Map::is_empty)
}

/// One catalog entry: the bundled metadata for a single model id, as returned by [`Pricing::card`]
/// and [`Pricing::cards`]. `id` is owned so callers may hold it across `.await` points without
/// borrowing the [`Pricing`] table. Every field beyond `id` is `Option` — `None` means "not
/// collected", NOT a default or zero (consumers must render it as unknown).
#[derive(Debug, Clone, PartialEq)]
pub struct ModelCard {
    pub id: String,
    pub modality: Option<Modality>,
    pub max_context: Option<u32>,
    pub max_output: Option<u32>,
    pub max_thinking: Option<u32>,
    pub input_per_mtok: Option<f64>,
    pub output_per_mtok: Option<f64>,
    /// Canonical sampling parameters this model rejects; `None` ⇒ not collected. See [`Rate`].
    pub unsupported_parameters: Option<Vec<String>>,
    /// Recommended sampling-parameter defaults; `None` ⇒ not collected. See [`Rate`].
    pub default_parameters: Option<Map<String, Value>>,
}

impl Rate {
    /// Build a [`ModelCard`] for the given (owned) id from this row. Takes `&self` (the row carries
    /// owned `Vec`/`Map` fields, so it is no longer `Copy`); the small clones are off the hot path.
    fn to_card(&self, id: String) -> ModelCard {
        ModelCard {
            id,
            modality: self.modality,
            max_context: self.max_context,
            max_output: self.max_output,
            max_thinking: self.max_thinking,
            input_per_mtok: self.input_per_mtok,
            output_per_mtok: self.output_per_mtok,
            unsupported_parameters: self.unsupported_parameters.clone(),
            default_parameters: self.default_parameters.clone(),
        }
    }
}

/// Read-only pricing table. Cheap to clone-share (wrap in `Arc` at the call site if needed).
#[derive(Debug, Clone)]
pub struct Pricing {
    models: HashMap<String, Rate>,
}

impl Pricing {
    /// Load the dataset bundled with the core. Infallible in practice — the bundled file is
    /// validated by the crate's tests — but returns the parse error if a future edit breaks it.
    pub fn bundled() -> Result<Self, serde_json::Error> {
        let ds: Dataset = serde_json::from_str(BUNDLED)?;
        Ok(Self { models: ds.models })
    }

    /// Compute the USD cost of a usage report for `model`, if the model is *token-priced* in the
    /// dataset. A lookup and two multiplies — nothing else.
    ///
    /// Returns `None` when the model is unknown OR when it carries no token rates at all (a
    /// per-character/per-minute audio model): reporting `$0.00` for those would misstate cost, so we
    /// report "no token cost" instead. A model with one rate present (e.g. an embedding model with a
    /// zero output rate) prices the missing side as `0.0`.
    pub fn cost_usd(&self, model: &str, usage: &Usage) -> Option<f64> {
        let rate = self.models.get(model)?;
        if rate.input_per_mtok.is_none() && rate.output_per_mtok.is_none() {
            return None;
        }
        let input = usage.prompt_tokens as f64 / 1_000_000.0 * rate.input_per_mtok.unwrap_or(0.0);
        let output =
            usage.completion_tokens as f64 / 1_000_000.0 * rate.output_per_mtok.unwrap_or(0.0);
        Some(input + output)
    }

    /// Return `usage` with `cost_usd` filled from the dataset when the model is known.
    pub fn price(&self, model: &str, mut usage: Usage) -> Usage {
        usage.cost_usd = self.cost_usd(model, &usage);
        usage
    }

    pub fn knows(&self, model: &str) -> bool {
        self.models.contains_key(model)
    }

    /// Look up the full catalog card (modality + limits + rates) for one model id, if present in the
    /// bundled dataset. Pure lookup — no fetch ("consumption is a lookup").
    pub fn card(&self, model: &str) -> Option<ModelCard> {
        let (id, rate) = self.models.get_key_value(model)?;
        Some(rate.to_card(id.clone()))
    }

    /// Iterate every catalog entry in the bundled dataset. Order is unspecified (`HashMap`); the
    /// caller sorts for stable output.
    pub fn cards(&self) -> impl Iterator<Item = ModelCard> + '_ {
        self.models
            .iter()
            .map(|(id, rate)| rate.to_card(id.clone()))
    }
}

/// Offline collection entry point. The core never calls this module; it only reads [`BUNDLED`].
pub mod collect {
    use super::*;

    #[cfg(feature = "collect")]
    const COMMENT: &str = "Bundled pricing dataset. Collected OFFLINE by the pricing crate's collector (SOUL.md: 'Collection happens offline in the pricing crate; the core only ever reads the bundled dataset'). Rates are USD per 1,000,000 tokens.";
    #[cfg(feature = "collect")]
    const COMMENT_METADATA: &str = "Each row may carry capability metadata for GET /v1/models: modality (llm|tts|stt|embedding) and published limits max_context/max_output/max_thinking. Missing fields mean 'not collected' and must be rendered as unknown, never guessed or zeroed.";
    #[cfg(feature = "collect")]
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

    #[cfg(feature = "collect")]
    #[derive(Serialize)]
    struct OutputDataset {
        #[serde(rename = "_comment")]
        comment: &'static str,
        #[serde(rename = "_comment_metadata")]
        comment_metadata: &'static str,
        #[serde(rename = "_comment_params")]
        comment_params: &'static str,
        version: u32,
        models: BTreeMap<String, Rate>,
    }

    /// Read the current dataset file. Unknown/comment fields are ignored.
    #[cfg(feature = "collect")]
    pub(crate) fn read_seed(
        path: impl AsRef<Path>,
    ) -> Result<HashMap<String, Rate>, serde_json::Error> {
        let text = std::fs::read_to_string(path).map_err(serde_json::Error::io)?;
        let ds: Dataset = serde_json::from_str(&text)?;
        Ok(ds.models)
    }

    /// Merge newly collected rows into the dataset schema.
    ///
    /// Reported `Some` / non-empty fields overwrite the seed. Missing fields preserve seed values, so
    /// pricing-page rows do not erase curated context/capability metadata. Rows absent from this run
    /// stay in the seed unless `prune` is explicitly set.
    #[cfg(any(test, feature = "collect"))]
    pub(crate) fn merge_model_infos<I>(
        mut seed: HashMap<String, Rate>,
        infos: I,
        prune: bool,
    ) -> HashMap<String, Rate>
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

    #[cfg(any(test, feature = "collect"))]
    fn apply_model_info(rate: &mut Rate, info: ModelInfo) {
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
    #[cfg(feature = "collect")]
    pub(crate) fn write_dataset(
        path: impl AsRef<Path>,
        models: HashMap<String, Rate>,
    ) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
        let sorted: BTreeMap<String, Rate> = models.into_iter().collect();
        let count = sorted.len();
        let out = OutputDataset {
            comment: COMMENT,
            comment_metadata: COMMENT_METADATA,
            comment_params: COMMENT_PARAMS,
            version: 4,
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

    #[cfg(feature = "collect")]
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

    #[cfg(feature = "collect")]
    async fn collect_provider(
        http: &reqwest::Client,
        p: &CollectorProvider,
        name: &str,
    ) -> Result<(String, Vec<ModelInfo>), Box<dyn std::error::Error + Send + Sync>> {
        match p.source {
            CollectorSource::PricingPage => collect_pricing_page(http, p, name).await,
            CollectorSource::ListEndpoint => collect_list_endpoint(p, name).await,
            CollectorSource::Auto => {
                if pricing_page_url(p).is_some() {
                    collect_pricing_page(http, p, name).await
                } else {
                    collect_list_endpoint(p, name).await
                }
            }
        }
    }

    #[cfg(feature = "collect")]
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

    #[cfg(feature = "collect")]
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

    #[cfg(feature = "collect")]
    async fn collect_pricing_page(
        http: &reqwest::Client,
        p: &CollectorProvider,
        name: &str,
    ) -> Result<(String, Vec<ModelInfo>), Box<dyn std::error::Error + Send + Sync>> {
        let url = pricing_page_url(p).ok_or_else(|| {
            format!(
                "{name} ({}) has no built-in pricing page collector; set pricing_url and source = \"pricing-page\" only after adding parser support",
                p.kind
            )
        })?;
        let html = http
            .get(&url)
            .send()
            .await?
            .error_for_status()?
            .text()
            .await?;
        let lines = html_lines(&html);
        let infos = match normalized_kind(&p.kind).as_str() {
            "cohere" => parse_cohere_pricing_lines(&lines),
            "anthropic" => parse_anthropic_pricing_lines(&lines),
            "mistral" => parse_mistral_pricing_lines(&lines),
            "openai" => parse_openai_pricing_lines(&lines),
            _ => {
                return Err(format!(
                    "{name} ({}) has no pricing-page parser; supported page parsers: openai, anthropic, cohere, mistral",
                    p.kind
                )
                .into());
            }
        };
        let infos = dedup_model_infos(infos);
        if infos.is_empty() {
            return Err(format!(
                "pricing page parser for {name} ({}) found no priced model rows at {url}",
                p.kind
            )
            .into());
        }
        Ok((format!("pricing-page:{url}"), infos))
    }

    #[cfg(feature = "collect")]
    fn pricing_page_url(p: &CollectorProvider) -> Option<String> {
        if let Some(url) = &p.pricing_url {
            return Some(url.clone());
        }
        match normalized_kind(&p.kind).as_str() {
            "openai" => Some("https://openai.com/api/pricing/".to_string()),
            "anthropic" => Some("https://claude.com/pricing#api".to_string()),
            "cohere" => Some("https://cohere.com/pricing".to_string()),
            "mistral" => Some("https://mistral.ai/pricing/".to_string()),
            _ => None,
        }
    }

    #[cfg(any(test, feature = "collect"))]
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

    #[cfg(any(test, feature = "collect"))]
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

    #[cfg(any(test, feature = "collect"))]
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

    #[cfg(any(test, feature = "collect"))]
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

    #[cfg(any(test, feature = "collect"))]
    fn first_u32(obj: &Map<String, Value>, keys: &[&str]) -> Option<u32> {
        keys.iter().find_map(|k| obj.get(*k).and_then(json_u32))
    }

    #[cfg(any(test, feature = "collect"))]
    fn nested_u32(obj: &Map<String, Value>, parent: &str, key: &str) -> Option<u32> {
        obj.get(parent)
            .and_then(Value::as_object)
            .and_then(|o| o.get(key))
            .and_then(json_u32)
    }

    #[cfg(any(test, feature = "collect"))]
    fn json_u32(v: &Value) -> Option<u32> {
        v.as_u64()
            .or_else(|| v.as_str().and_then(|s| s.parse::<u64>().ok()))
            .map(|n| n.min(u32::MAX as u64) as u32)
    }

    #[cfg(any(test, feature = "collect"))]
    fn json_f64(v: &Value) -> Option<f64> {
        v.as_f64()
            .or_else(|| v.as_str().and_then(|s| s.parse::<f64>().ok()))
    }

    #[cfg(any(test, feature = "collect"))]
    pub(crate) fn parse_openai_pricing_lines(lines: &[String]) -> Vec<ModelInfo> {
        let mut out = Vec::new();
        for (i, line) in lines.iter().enumerate() {
            let Some(id) = openai_model_id(line) else {
                continue;
            };
            let window = &lines[i + 1..lines.len().min(i + 18)];
            let input = labeled_price(window, "Input:");
            let output = labeled_price(window, "Output:");
            if input.is_none() && output.is_none() {
                continue;
            }
            let mut info = ModelInfo::new(id);
            info.modality = Some(Modality::Llm);
            info.input_per_mtok = input;
            info.output_per_mtok = output;
            out.push(info);
        }
        out
    }

    #[cfg(any(test, feature = "collect"))]
    pub(crate) fn parse_anthropic_pricing_lines(lines: &[String]) -> Vec<ModelInfo> {
        let mut out = Vec::new();
        for (i, line) in lines.iter().enumerate() {
            let Some(name_end) = line.find('$') else {
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

    #[cfg(any(test, feature = "collect"))]
    fn anthropic_heading_label(line: &str) -> Option<String> {
        let label = line.trim().trim_start_matches('#').trim();
        let first = label.split_whitespace().next()?;
        matches!(first, "Fable" | "Mythos" | "Opus" | "Sonnet" | "Haiku")
            .then_some(format!("Claude {label}"))
    }

    #[cfg(any(test, feature = "collect"))]
    fn anthropic_labeled_price(lines: &[String], label: &str) -> Option<f64> {
        let pos = lines.iter().position(|line| line.trim() == label)?;
        prices_in_lines(&lines[pos + 1..], 6).into_iter().next()
    }

    #[cfg(any(test, feature = "collect"))]
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

    #[cfg(any(test, feature = "collect"))]
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

    #[cfg(any(test, feature = "collect"))]
    pub(crate) fn parse_cohere_pricing_lines(lines: &[String]) -> Vec<ModelInfo> {
        let mut out = Vec::new();
        for line in lines {
            let lower = line.to_ascii_lowercase();
            if lower.contains("aya expanse models") {
                if let Some((input, output)) = input_output_prices(line) {
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
            if let Some((input, output)) = input_output_prices(line) {
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

    #[cfg(any(test, feature = "collect"))]
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
            info.modality = Some(if line.to_ascii_lowercase().contains("voxtral") {
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

    #[cfg(any(test, feature = "collect"))]
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

    #[cfg(any(test, feature = "collect"))]
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

    #[cfg(any(test, feature = "collect"))]
    fn labeled_price(lines: &[String], label: &str) -> Option<f64> {
        let pos = lines.iter().position(|line| line == label)?;
        prices_in_lines(&lines[pos + 1..], 4).into_iter().next()
    }

    #[cfg(any(test, feature = "collect"))]
    fn next_price_window(lines: &[String]) -> &[String] {
        let end = lines
            .iter()
            .position(|line| line.starts_with("Input") || line.starts_with("Output"))
            .unwrap_or(lines.len());
        &lines[..end]
    }

    #[cfg(any(test, feature = "collect"))]
    fn input_output_prices(line: &str) -> Option<(f64, f64)> {
        let prices = dollar_prices(line);
        if prices.len() < 2 {
            return None;
        }
        Some((prices[0], prices[1]))
    }

    #[cfg(any(test, feature = "collect"))]
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

    #[cfg(any(test, feature = "collect"))]
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

    #[cfg(any(test, feature = "collect"))]
    fn plain_price(line: &str) -> Option<f64> {
        let s = line.trim().trim_start_matches('$').replace(',', "");
        if s.is_empty() || !s.chars().all(|ch| ch.is_ascii_digit() || ch == '.') {
            return None;
        }
        s.parse::<f64>().ok()
    }

    #[cfg(any(test, feature = "collect"))]
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

    #[cfg(any(test, feature = "collect"))]
    fn anthropic_label_to_model_id(label: &str) -> String {
        label_to_model_id(label).replace('.', "-")
    }

    #[cfg(any(test, feature = "collect"))]
    fn cohere_label_to_model_id(label: &str) -> String {
        label_to_model_id(&label.replace('+', " plus"))
    }

    #[cfg(any(test, feature = "collect"))]
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

    #[cfg(feature = "collect")]
    fn dedup_model_infos(infos: Vec<ModelInfo>) -> Vec<ModelInfo> {
        let mut by_id = HashMap::new();
        for info in infos {
            by_id.insert(info.id.clone(), info);
        }
        by_id.into_values().collect()
    }

    #[cfg(any(test, feature = "collect"))]
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

    #[cfg(feature = "collect")]
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

    #[cfg(feature = "collect")]
    fn normalized_kind(kind: &str) -> String {
        match kind {
            "claude" => "anthropic",
            other => other,
        }
        .to_ascii_lowercase()
    }

    #[cfg(feature = "collect")]
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

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    #[test]
    fn bundled_dataset_parses() {
        let p = Pricing::bundled().expect("bundled prices.json must parse");
        assert!(p.knows("gpt-4o"));
    }

    #[test]
    fn cost_is_lookup_times_tokens() {
        let p = Pricing::bundled().unwrap();
        let usage = Usage {
            prompt_tokens: 1_000_000,
            completion_tokens: 1_000_000,
            total_tokens: 2_000_000,
            cost_usd: None,
            cache_read_tokens: 0,
            cache_creation_tokens: 0,
        };
        // gpt-4o: 2.5 in + 10.0 out per Mtok.
        let cost = p.cost_usd("gpt-4o", &usage).unwrap();
        assert!((cost - 12.5).abs() < 1e-9, "got {cost}");
    }

    #[test]
    fn knows_embedding_model() {
        // Embedding/audio models were added to the bundled dataset alongside chat models;
        // a known embedding id must resolve so its usage gets priced like any other.
        let p = Pricing::bundled().unwrap();
        assert!(p.knows("text-embedding-3-small"));
    }

    #[test]
    fn unknown_model_has_no_cost() {
        let p = Pricing::bundled().unwrap();
        assert!(p.cost_usd("does-not-exist", &Usage::default()).is_none());
    }

    #[test]
    fn collector_merge_preserves_seed_when_catalog_is_id_only() {
        let mut seed = HashMap::new();
        seed.insert(
            "gpt-4o".to_string(),
            Rate {
                input_per_mtok: Some(2.5),
                output_per_mtok: Some(10.0),
                modality: Some(Modality::Llm),
                max_context: Some(128_000),
                ..Rate::default()
            },
        );

        let out = collect::merge_model_infos(seed, [ModelInfo::new("gpt-4o")], false);
        let card = out.get("gpt-4o").unwrap().to_card("gpt-4o".to_string());
        assert_eq!(card.input_per_mtok, Some(2.5));
        assert_eq!(card.output_per_mtok, Some(10.0));
        assert_eq!(card.modality, Some(Modality::Llm));
        assert_eq!(card.max_context, Some(128_000));
    }

    #[test]
    fn collector_merge_updates_reported_fields_and_can_prune() {
        let mut seed = HashMap::new();
        seed.insert(
            "old".to_string(),
            Rate {
                input_per_mtok: Some(99.0),
                ..Rate::default()
            },
        );

        let mut info = ModelInfo::new("fresh");
        info.modality = Some(Modality::Embedding);
        info.max_context = Some(8192);
        info.input_per_mtok = Some(0.02);
        info.output_per_mtok = Some(0.0);

        let out = collect::merge_model_infos(seed, [info], true);
        assert!(!out.contains_key("old"));
        let card = out.get("fresh").unwrap().to_card("fresh".to_string());
        assert_eq!(card.modality, Some(Modality::Embedding));
        assert_eq!(card.max_context, Some(8192));
        assert_eq!(card.input_per_mtok, Some(0.02));
        assert_eq!(card.output_per_mtok, Some(0.0));
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
    fn cohere_pricing_page_parser_reads_legacy_and_aya_rows() {
        let lines = vec![
            "Command R+ 08-2024 pricing is $2.50/1M tokens for input and $10.00/1M tokens for output",
            "Aya Expanse models (8B and 32B) on the API are charged at $0.50/1M tokens for input and $1.50/1M tokens for output.",
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
    }

    #[test]
    fn card_exposes_metadata() {
        let p = Pricing::bundled().unwrap();
        let c = p.card("gpt-4o").expect("gpt-4o card");
        assert_eq!(c.id, "gpt-4o");
        assert_eq!(c.modality, Some(Modality::Llm));
        assert_eq!(c.max_context, Some(128_000));
        assert_eq!(c.input_per_mtok, Some(2.5));
    }

    #[test]
    fn embedding_card_has_modality_no_output_limit() {
        let p = Pricing::bundled().unwrap();
        let c = p.card("text-embedding-3-small").expect("embedding card");
        assert_eq!(c.modality, Some(Modality::Embedding));
        // max_output is meaningless for embeddings — left null, never zeroed.
        assert_eq!(c.max_output, None);
        assert_eq!(c.max_thinking, None);
    }

    #[test]
    fn audio_models_have_modality_but_no_rates() {
        let p = Pricing::bundled().unwrap();
        let tts = p.card("tts-1").expect("tts-1 card");
        assert_eq!(tts.modality, Some(Modality::Tts));
        assert_eq!(tts.input_per_mtok, None);
        assert_eq!(tts.output_per_mtok, None);
        let stt = p.card("whisper-1").expect("whisper-1 card");
        assert_eq!(stt.modality, Some(Modality::Stt));
        assert_eq!(stt.input_per_mtok, None);
        assert_eq!(stt.output_per_mtok, None);
    }

    #[test]
    fn audio_model_reports_no_token_cost() {
        // tts-1/whisper-1 are not token-priced: cost must be None, never a misleading $0.00.
        let p = Pricing::bundled().unwrap();
        let usage = Usage {
            prompt_tokens: 1_000_000,
            completion_tokens: 1_000_000,
            total_tokens: 2_000_000,
            cost_usd: None,
            cache_read_tokens: 0,
            cache_creation_tokens: 0,
        };
        assert!(p.cost_usd("tts-1", &usage).is_none());
        assert!(p.cost_usd("whisper-1", &usage).is_none());
    }

    #[test]
    fn cards_iterates_full_catalog() {
        let p = Pricing::bundled().unwrap();
        let ids: Vec<String> = p.cards().map(|c| c.id).collect();
        assert!(
            ids.len() >= 16,
            "expected the full seed catalog, got {}",
            ids.len()
        );
        // Every yielded id is non-empty and re-fetchable via card().
        for id in &ids {
            assert!(!id.is_empty());
            assert!(p.card(id).is_some());
        }
    }

    #[test]
    fn omitted_field_means_unknown_not_zero() {
        // gpt-4o has no max_thinking in the seed → None, NOT Some(0).
        let p = Pricing::bundled().unwrap();
        assert_eq!(p.card("gpt-4o").unwrap().max_thinking, None);
    }

    #[test]
    fn card_carries_param_metadata() {
        let p = Pricing::bundled().unwrap();
        // A reasoning model records the sampling params it rejects + its pinned defaults.
        let g = p.card("gpt-5").expect("gpt-5 card");
        assert_eq!(
            g.unsupported_parameters,
            Some(vec![
                "temperature".to_string(),
                "top_p".to_string(),
                "frequency_penalty".to_string(),
                "presence_penalty".to_string(),
            ])
        );
        assert_eq!(
            g.default_parameters.unwrap().get("temperature"),
            Some(&serde_json::json!(1))
        );
        // Anthropic standard: penalties unsupported, but temperature/top_p remain (not in the list).
        let c = p.card("claude-sonnet-4").unwrap();
        let u = c.unsupported_parameters.unwrap();
        assert!(u.contains(&"frequency_penalty".to_string()));
        assert!(u.contains(&"presence_penalty".to_string()));
        assert!(!u.contains(&"temperature".to_string()));
        // Anthropic 4.7+/Fable generation: sampling params removed → temperature/top_p/top_k also rejected.
        let removed = p
            .card("claude-opus-4-8")
            .unwrap()
            .unsupported_parameters
            .unwrap();
        for name in [
            "frequency_penalty",
            "presence_penalty",
            "temperature",
            "top_p",
            "top_k",
        ] {
            assert!(
                removed.contains(&name.to_string()),
                "{name} must be rejected by claude-opus-4-8"
            );
        }
        assert_eq!(
            p.card("claude-fable-5").unwrap().unsupported_parameters,
            p.card("claude-opus-4-8").unwrap().unsupported_parameters
        );
        // A plain model carries neither — None means "no restriction collected", never an invented default.
        let four = p.card("gpt-4o").unwrap();
        assert_eq!(four.unsupported_parameters, None);
        assert_eq!(four.default_parameters, None);
    }

    #[test]
    fn modality_json_roundtrips_lowercase() {
        for (m, s) in [
            (Modality::Llm, "\"llm\""),
            (Modality::Tts, "\"tts\""),
            (Modality::Stt, "\"stt\""),
            (Modality::Embedding, "\"embedding\""),
            (Modality::Rerank, "\"rerank\""),
        ] {
            assert_eq!(serde_json::to_string(&m).unwrap(), s);
            assert_eq!(serde_json::from_str::<Modality>(s).unwrap(), m);
        }
    }
}
