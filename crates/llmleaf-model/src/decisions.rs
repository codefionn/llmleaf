//! Canonical decisions model.
//!
//! Decisions evaluates one state against named, typed questions and returns one structured answer
//! for each question. Questions and answers deliberately remain JSON: their three supported shapes
//! are an edge contract, while providers must receive the full structured instructions and criteria.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::Usage;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DecisionsRequest {
    pub model: String,
    pub state: Value,
    pub questions: Map<String, Value>,
    #[serde(default, skip_serializing_if = "Map::is_empty")]
    pub extra: Map<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DecisionsResponse {
    pub model: String,
    pub answers: Map<String, Value>,
    #[serde(default)]
    pub usage: Usage,
    /// Upstream response fields such as `id` and `provider`.
    #[serde(default, skip_serializing_if = "Map::is_empty")]
    pub extra: Map<String, Value>,
    /// Unknown fields nested under the upstream `usage` object.
    #[serde(default, skip_serializing_if = "Map::is_empty")]
    pub usage_extra: Map<String, Value>,
}
