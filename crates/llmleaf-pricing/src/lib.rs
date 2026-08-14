//! Bundled provider pricing information and read-only lookup.
//!
//! The dataset is generated offline by the separate `llmleaf-pricing-collector` crate and bundled
//! into this library at compile time. Runtime cost reporting is only a lookup plus arithmetic; this
//! crate contains no HTTP client, scraper, configuration parser, or generation code.

use std::collections::HashMap;

use llmleaf_model::{Modality, Usage};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// The pricing dataset that ships with llmleaf.
const BUNDLED: &str = include_str!("../data/prices.json");

/// Serializable pricing dataset shared with the offline collector.
///
/// Consumers normally use [`Pricing::bundled`]. This type is public so the collector can read,
/// merge, and write the same schema without duplicating it.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PricingDataset {
    pub version: u32,
    pub models: HashMap<String, ModelPricing>,
}

/// Pricing and capability information for one model.
///
/// Every field is optional. A missing token rate can mean that a model uses a different billing
/// unit, while missing capability data means it has not been collected. Neither case should be
/// presented as zero.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct ModelPricing {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_per_mtok: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cached_input_per_mtok: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_per_mtok: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub modality: Option<Modality>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_context: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_output: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_thinking: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_reasoning: Option<bool>,
    /// Published input media accepted by the model (`text`, `image`, `file`, `audio`, `video`).
    /// This refines the coarse output-oriented [`Modality`] used by `?type=` filtering.
    #[serde(default, skip_serializing_if = "opt_vec_is_none_or_empty")]
    pub input_modalities: Option<Vec<String>>,
    /// Published output media produced by the model (normally `text` for an LLM).
    #[serde(default, skip_serializing_if = "opt_vec_is_none_or_empty")]
    pub output_modalities: Option<Vec<String>>,
    /// Provider service tier, when the model id selects one (for example `standard` or `contributor`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tier: Option<String>,
    /// Whether prompts and completions may be used to train future provider models.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompts_used_for_training: Option<bool>,
    /// Canonical sampling parameters this model rejects. `None` means none were collected.
    #[serde(default, skip_serializing_if = "opt_vec_is_none_or_empty")]
    pub unsupported_parameters: Option<Vec<String>>,
    /// Provider- or dataset-recommended sampling parameter defaults.
    #[serde(default, skip_serializing_if = "opt_map_is_none_or_empty")]
    pub default_parameters: Option<Map<String, Value>>,
}

fn opt_vec_is_none_or_empty(value: &Option<Vec<String>>) -> bool {
    value.as_ref().is_none_or(Vec::is_empty)
}

fn opt_map_is_none_or_empty(value: &Option<Map<String, Value>>) -> bool {
    value.as_ref().is_none_or(Map::is_empty)
}

/// One catalog entry from the bundled pricing information.
///
/// `id` is owned so callers may retain a card across `.await` points without borrowing the
/// [`Pricing`] table. Every other field is optional; `None` means unknown, not zero.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelCard {
    pub id: String,
    pub modality: Option<Modality>,
    pub max_context: Option<u32>,
    pub max_output: Option<u32>,
    pub max_thinking: Option<u32>,
    pub supports_reasoning: Option<bool>,
    pub input_modalities: Option<Vec<String>>,
    pub output_modalities: Option<Vec<String>>,
    pub input_per_mtok: Option<f64>,
    pub cached_input_per_mtok: Option<f64>,
    pub output_per_mtok: Option<f64>,
    pub tier: Option<String>,
    pub prompts_used_for_training: Option<bool>,
    pub unsupported_parameters: Option<Vec<String>>,
    pub default_parameters: Option<Map<String, Value>>,
}

impl ModelPricing {
    fn to_card(&self, id: String) -> ModelCard {
        ModelCard {
            id,
            modality: self.modality,
            max_context: self.max_context,
            max_output: self.max_output,
            max_thinking: self.max_thinking,
            supports_reasoning: self.supports_reasoning,
            input_modalities: self.input_modalities.clone(),
            output_modalities: self.output_modalities.clone(),
            input_per_mtok: self.input_per_mtok,
            cached_input_per_mtok: self.cached_input_per_mtok,
            output_per_mtok: self.output_per_mtok,
            tier: self.tier.clone(),
            prompts_used_for_training: self.prompts_used_for_training,
            unsupported_parameters: self.unsupported_parameters.clone(),
            default_parameters: self.default_parameters.clone(),
        }
    }
}

/// Read-only pricing table. Cheap to clone-share (wrap in `Arc` at the call site if needed).
#[derive(Debug, Clone)]
pub struct Pricing {
    models: HashMap<String, ModelPricing>,
}

impl Pricing {
    /// Load the dataset bundled with this library.
    pub fn bundled() -> Result<Self, serde_json::Error> {
        let dataset: PricingDataset = serde_json::from_str(BUNDLED)?;
        Ok(Self {
            models: dataset.models,
        })
    }

    /// Compute the USD cost of a usage report for `model`, if it is token-priced.
    ///
    /// Returns `None` for an unknown model or one with no token rates. When only one token rate is
    /// present, the missing side is priced as zero.
    pub fn cost_usd(&self, model: &str, usage: &Usage) -> Option<f64> {
        let rate = self.models.get(model)?;
        if rate.input_per_mtok.is_none()
            && rate.cached_input_per_mtok.is_none()
            && rate.output_per_mtok.is_none()
        {
            return None;
        }
        let cached_tokens = usage.cache_read_tokens.min(usage.prompt_tokens);
        let uncached_tokens = usage.prompt_tokens - cached_tokens;
        let input = uncached_tokens as f64 / 1_000_000.0 * rate.input_per_mtok.unwrap_or(0.0)
            + cached_tokens as f64 / 1_000_000.0
                * rate
                    .cached_input_per_mtok
                    .or(rate.input_per_mtok)
                    .unwrap_or(0.0);
        let output =
            usage.completion_tokens as f64 / 1_000_000.0 * rate.output_per_mtok.unwrap_or(0.0);
        Some(input + output)
    }

    /// Return `usage` with `cost_usd` filled from the dataset when the model is known and
    /// token-priced.
    pub fn price(&self, model: &str, mut usage: Usage) -> Usage {
        usage.cost_usd = self.cost_usd(model, &usage);
        usage
    }

    pub fn knows(&self, model: &str) -> bool {
        self.models.contains_key(model)
    }

    /// Look up the full catalog card for one model id.
    pub fn card(&self, model: &str) -> Option<ModelCard> {
        let (id, rate) = self.models.get_key_value(model)?;
        Some(rate.to_card(id.clone()))
    }

    /// Iterate every catalog entry. Order is unspecified; callers can sort for stable output.
    pub fn cards(&self) -> impl Iterator<Item = ModelCard> + '_ {
        self.models
            .iter()
            .map(|(id, rate)| rate.to_card(id.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundled_dataset_parses() {
        let pricing = Pricing::bundled().expect("bundled prices.json must parse");
        assert!(pricing.knows("gpt-4o"));
    }

    #[test]
    fn bundled_dataset_classifies_meta_muse_models() {
        let pricing = Pricing::bundled().unwrap();
        let v11 = pricing.card("muse-spark-1.1").unwrap();
        assert_eq!(v11.modality, Some(Modality::Llm));
        assert_eq!(v11.max_context, Some(1_048_576));
        assert_eq!(v11.supports_reasoning, Some(true));
        assert_eq!(v11.input_per_mtok, Some(1.25));
        assert_eq!(v11.cached_input_per_mtok, Some(0.15));
        assert_eq!(v11.output_per_mtok, Some(4.25));
        assert_eq!(v11.tier.as_deref(), Some("standard"));
        assert_eq!(v11.prompts_used_for_training, Some(false));
        assert_eq!(
            v11.unsupported_parameters.as_ref().unwrap().join(","),
            "stop"
        );
        assert_eq!(
            v11.input_modalities.as_ref().unwrap().join(","),
            "text,image,video,file"
        );
        assert_eq!(v11.output_modalities.as_ref().unwrap().join(","), "text");

        let v12 = pricing.card("muse-spark-1.2").unwrap();
        assert_eq!(v12.modality, Some(Modality::Llm));
        assert_eq!(v12.max_context, Some(1_048_576));
        assert_eq!(v12.supports_reasoning, Some(true));
        assert_eq!(v12.input_per_mtok, Some(1.25));
        assert_eq!(v12.cached_input_per_mtok, Some(0.15));
        assert_eq!(v12.output_per_mtok, Some(4.25));
        assert_eq!(
            v12.input_modalities.as_ref().unwrap().join(","),
            "text,image,video,file"
        );

        let contributor = pricing.card("muse-spark-1.2-contributor").unwrap();
        assert_eq!(contributor.input_per_mtok, Some(0.10));
        assert_eq!(contributor.cached_input_per_mtok, Some(0.002));
        assert_eq!(contributor.output_per_mtok, Some(0.20));
        assert_eq!(contributor.tier.as_deref(), Some("contributor"));
        assert_eq!(contributor.prompts_used_for_training, Some(true));
    }

    #[test]
    fn bundled_dataset_has_current_openai_gpt_5_6_prices() {
        let pricing = Pricing::bundled().unwrap();
        for (id, input, cached_input, output) in [
            ("gpt-5.6", 5.0, 0.5, 30.0),
            ("gpt-5.6-sol", 5.0, 0.5, 30.0),
            ("gpt-5.6-terra", 2.0, 0.2, 12.0),
            ("gpt-5.6-luna", 0.2, 0.02, 1.2),
        ] {
            let card = pricing.card(id).unwrap();
            assert_eq!(card.input_per_mtok, Some(input), "{id}");
            assert_eq!(card.cached_input_per_mtok, Some(cached_input), "{id}");
            assert_eq!(card.output_per_mtok, Some(output), "{id}");
        }
    }

    #[test]
    fn bundled_dataset_has_documented_provider_gap_catalogs() {
        let pricing = Pricing::bundled().unwrap();

        let zai_newest = pricing.card("glm-5.3").unwrap();
        assert_eq!(zai_newest.max_context, Some(1_000_000));
        assert_eq!(zai_newest.max_output, Some(131_072));
        assert_eq!(zai_newest.input_per_mtok, None);
        assert_eq!(zai_newest.cached_input_per_mtok, None);
        assert_eq!(zai_newest.output_per_mtok, None);
        assert_eq!(zai_newest.supports_reasoning, Some(true));

        let zai = pricing.card("glm-5.2").unwrap();
        assert_eq!(zai.max_context, Some(1_000_000));
        assert_eq!(zai.max_output, Some(131_072));
        assert_eq!(zai.input_per_mtok, Some(1.4));
        assert_eq!(zai.cached_input_per_mtok, Some(0.26));
        assert_eq!(zai.output_per_mtok, Some(4.4));

        let deepseek = pricing.card("deepseek-v4-flash").unwrap();
        assert_eq!(deepseek.cached_input_per_mtok, Some(0.0028));
        assert_eq!(deepseek.output_per_mtok, Some(0.28));

        let minimax = pricing.card("MiniMax-M2.7-highspeed").unwrap();
        assert_eq!(minimax.max_context, Some(204_800));
        assert_eq!(minimax.max_output, None);
        assert_eq!(minimax.input_per_mtok, Some(0.6));
        assert_eq!(minimax.output_per_mtok, Some(2.4));

        let groq = pricing.card("llama-3.3-70b-versatile").unwrap();
        assert_eq!(groq.cached_input_per_mtok, None);
        assert_eq!(groq.input_per_mtok, Some(0.59));
        assert_eq!(groq.output_per_mtok, Some(0.79));
    }

    #[test]
    fn cost_is_lookup_times_tokens() {
        let pricing = Pricing::bundled().unwrap();
        let usage = Usage {
            prompt_tokens: 1_000_000,
            completion_tokens: 1_000_000,
            total_tokens: 2_000_000,
            cost_usd: None,
            cache_read_tokens: 0,
            cache_creation_tokens: 0,
        };
        let cost = pricing.cost_usd("gpt-4o", &usage).unwrap();
        assert!((cost - 12.5).abs() < 1e-9, "got {cost}");
    }

    #[test]
    fn cached_input_uses_the_models_discounted_rate() {
        let pricing = Pricing::bundled().unwrap();
        let usage = Usage {
            prompt_tokens: 1_000_000,
            completion_tokens: 1_000_000,
            total_tokens: 2_000_000,
            cost_usd: None,
            cache_read_tokens: 400_000,
            cache_creation_tokens: 0,
        };
        let cost = pricing
            .cost_usd("muse-spark-1.2-contributor", &usage)
            .unwrap();
        let expected = 0.6 * 0.10 + 0.4 * 0.002 + 0.20;
        assert!((cost - expected).abs() < 1e-9, "got {cost}");
    }

    #[test]
    fn knows_embedding_model() {
        let pricing = Pricing::bundled().unwrap();
        assert!(pricing.knows("text-embedding-3-small"));
    }

    #[test]
    fn unknown_model_has_no_cost() {
        let pricing = Pricing::bundled().unwrap();
        assert!(pricing
            .cost_usd("does-not-exist", &Usage::default())
            .is_none());
    }

    #[test]
    fn card_exposes_metadata() {
        let pricing = Pricing::bundled().unwrap();
        let card = pricing.card("gpt-4o").expect("gpt-4o card");
        assert_eq!(card.id, "gpt-4o");
        assert_eq!(card.modality, Some(Modality::Llm));
        assert_eq!(card.max_context, Some(128_000));
        assert_eq!(card.input_per_mtok, Some(2.5));
    }

    #[test]
    fn moonshot_cards_expose_rates_context_and_reasoning() {
        let pricing = Pricing::bundled().unwrap();
        let k3 = pricing.card("kimi-k3").expect("kimi-k3 card");
        assert_eq!(k3.modality, Some(Modality::Llm));
        assert_eq!(k3.max_context, Some(1_048_576));
        assert_eq!(k3.supports_reasoning, Some(true));
        assert_eq!(k3.input_per_mtok, Some(3.0));
        assert_eq!(k3.output_per_mtok, Some(15.0));

        let classic = pricing.card("moonshot-v1-128k").expect("v1 card");
        assert_eq!(classic.max_context, Some(131_072));
        assert_eq!(classic.input_per_mtok, Some(2.0));
        assert_eq!(classic.output_per_mtok, Some(5.0));
        assert_eq!(classic.supports_reasoning, None);
    }

    #[test]
    fn embedding_card_has_modality_no_output_limit() {
        let pricing = Pricing::bundled().unwrap();
        let card = pricing
            .card("text-embedding-3-small")
            .expect("embedding card");
        assert_eq!(card.modality, Some(Modality::Embedding));
        assert_eq!(card.max_output, None);
        assert_eq!(card.max_thinking, None);
    }

    #[test]
    fn audio_models_have_modality_but_no_rates() {
        let pricing = Pricing::bundled().unwrap();
        let tts = pricing.card("tts-1").expect("tts-1 card");
        assert_eq!(tts.modality, Some(Modality::Tts));
        assert_eq!(tts.input_per_mtok, None);
        assert_eq!(tts.output_per_mtok, None);
        let stt = pricing.card("whisper-1").expect("whisper-1 card");
        assert_eq!(stt.modality, Some(Modality::Stt));
        assert_eq!(stt.input_per_mtok, None);
        assert_eq!(stt.output_per_mtok, None);
    }

    #[test]
    fn audio_model_reports_no_token_cost() {
        let pricing = Pricing::bundled().unwrap();
        let usage = Usage {
            prompt_tokens: 1_000_000,
            completion_tokens: 1_000_000,
            total_tokens: 2_000_000,
            cost_usd: None,
            cache_read_tokens: 0,
            cache_creation_tokens: 0,
        };
        assert!(pricing.cost_usd("tts-1", &usage).is_none());
        assert!(pricing.cost_usd("whisper-1", &usage).is_none());
    }

    #[test]
    fn cards_iterates_full_catalog() {
        let pricing = Pricing::bundled().unwrap();
        let ids: Vec<String> = pricing.cards().map(|card| card.id).collect();
        assert!(
            ids.len() >= 16,
            "expected the full seed catalog, got {}",
            ids.len()
        );
        for id in &ids {
            assert!(!id.is_empty());
            assert!(pricing.card(id).is_some());
        }
    }

    #[test]
    fn omitted_field_means_unknown_not_zero() {
        let pricing = Pricing::bundled().unwrap();
        assert_eq!(pricing.card("gpt-4o").unwrap().max_thinking, None);
    }

    #[test]
    fn card_carries_param_metadata() {
        let pricing = Pricing::bundled().unwrap();
        let gpt = pricing.card("gpt-5").expect("gpt-5 card");
        assert_eq!(
            gpt.unsupported_parameters,
            Some(vec![
                "temperature".to_string(),
                "top_p".to_string(),
                "frequency_penalty".to_string(),
                "presence_penalty".to_string(),
            ])
        );
        assert_eq!(
            gpt.default_parameters.unwrap().get("temperature"),
            Some(&serde_json::json!(1))
        );

        let claude = pricing.card("claude-sonnet-4").unwrap();
        let unsupported = claude.unsupported_parameters.unwrap();
        assert!(unsupported.contains(&"frequency_penalty".to_string()));
        assert!(unsupported.contains(&"presence_penalty".to_string()));
        assert!(!unsupported.contains(&"temperature".to_string()));

        let removed = pricing
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
            assert!(removed.contains(&name.to_string()));
        }
        assert_eq!(
            pricing
                .card("claude-fable-5")
                .unwrap()
                .unsupported_parameters,
            pricing
                .card("claude-opus-4-8")
                .unwrap()
                .unsupported_parameters
        );

        let four = pricing.card("gpt-4o").unwrap();
        assert_eq!(four.unsupported_parameters, None);
        assert_eq!(four.default_parameters, None);
    }

    #[test]
    fn modality_json_roundtrips_lowercase() {
        for (modality, json) in [
            (Modality::Llm, "\"llm\""),
            (Modality::Tts, "\"tts\""),
            (Modality::Stt, "\"stt\""),
            (Modality::Embedding, "\"embedding\""),
            (Modality::Rerank, "\"rerank\""),
        ] {
            assert_eq!(serde_json::to_string(&modality).unwrap(), json);
            assert_eq!(serde_json::from_str::<Modality>(json).unwrap(), modality);
        }
    }
}
