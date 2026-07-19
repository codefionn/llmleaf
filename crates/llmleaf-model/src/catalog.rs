//! Provider-reported model catalog entries.
//!
//! [`ModelInfo`] is what a provider extension returns from `Provider::models` — one upstream model the
//! provider serves, with whatever metadata its list-models API reports. It mirrors the shape of
//! [`VoiceInfo`](crate::VoiceInfo): the `id` is mandatory; every other field is optional and `None`
//! means "the provider's API did not report it" — NEVER a guessed default (SOUL: no silent magic).
//! Gaps are filled downstream from the bundled dataset, not invented here.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::request::Modality;

/// One model an upstream provider serves, as enumerated by `Provider::models`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelInfo {
    /// The upstream model id — the value a consumer would request (before any routing prefix is added).
    pub id: String,
    /// A human-friendly display name, when the provider distinguishes one from the id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// The model's modality, when the provider's catalog makes it inferable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub modality: Option<Modality>,
    /// Maximum context window in tokens, when reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_context: Option<u32>,
    /// Maximum output/completion tokens, when reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_output: Option<u32>,
    /// Maximum reasoning/thinking budget in tokens, when published.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_thinking: Option<u32>,
    /// Whether the provider explicitly reports reasoning/thinking support. Kept separate from
    /// `max_thinking` because some catalogs publish a capability flag but no numeric budget.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_reasoning: Option<bool>,
    /// USD per 1,000,000 input tokens, when the catalog prices the model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_per_mtok: Option<f64>,
    /// USD per 1,000,000 output tokens, when the catalog prices the model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_per_mtok: Option<f64>,
    /// The positive list of canonical parameters the provider's OWN catalog declares supported — e.g.
    /// OpenRouter's `supported_parameters`, captured verbatim when the upstream reports it. `None` means
    /// the provider reported no list; the catalog surface then computes one from the modality baseline
    /// minus [`unsupported_parameters`](Self::unsupported_parameters). Never invented (SOUL: no guessing)
    /// and never filled from the bundled dataset — that side records the negative delta instead.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supported_parameters: Option<Vec<String>>,
    /// Canonical sampling parameters this model REJECTS (returns an error for), as recorded by the
    /// bundled dataset or reported by the provider. Names are canonical (`temperature`, `top_p`, …).
    /// Empty means "no restriction known" — NEVER a guess. The catalog subtracts these from the
    /// advertised `supported_parameters` so a consumer never sends, e.g., a reasoning model a
    /// `temperature` it will 400 on.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub unsupported_parameters: Vec<String>,
    /// Provider- or dataset-recommended default values for sampling parameters (e.g. `{"temperature":
    /// 0.3}` for Cohere, or the fixed `{"temperature": 1}` a reasoning model pins). Surfaced verbatim as
    /// OpenRouter's `default_parameters`. Empty means "none reported".
    #[serde(default, skip_serializing_if = "Map::is_empty")]
    pub default_parameters: Map<String, Value>,
    /// Provider-specific fields preserved verbatim (principle 7).
    #[serde(default, skip_serializing_if = "Map::is_empty")]
    pub extra: Map<String, Value>,
}

impl ModelInfo {
    /// A bare entry carrying only the id — the honest result for a provider whose list-models API
    /// returns ids and nothing else (e.g. OpenAI). Gaps are enhanced downstream.
    pub fn new(id: impl Into<String>) -> Self {
        ModelInfo {
            id: id.into(),
            name: None,
            modality: None,
            max_context: None,
            max_output: None,
            max_thinking: None,
            supports_reasoning: None,
            input_per_mtok: None,
            output_per_mtok: None,
            supported_parameters: None,
            unsupported_parameters: Vec::new(),
            default_parameters: Map::new(),
            extra: Map::new(),
        }
    }
}
