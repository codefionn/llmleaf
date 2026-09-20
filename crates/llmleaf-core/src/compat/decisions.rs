//! Decisions compatibility surfaces (`/v1/decisions`, OpenRouter alpha, and TypeSafe System One).

use llmleaf_model::{DecisionsRequest, DecisionsResponse, ModelError};
use serde_json::Value;

fn mapping(message: impl Into<String>) -> ModelError {
    ModelError::Mapping(message.into())
}

pub fn parse_decisions_request(value: Value) -> Result<DecisionsRequest, ModelError> {
    let Value::Object(mut obj) = value else {
        return Err(mapping("request body must be a JSON object"));
    };
    let model = match obj.remove("model") {
        Some(Value::String(value)) if !value.is_empty() => value,
        _ => {
            return Err(mapping(
                "`model` is required and must be a non-empty string",
            ))
        }
    };
    let state = match obj.remove("state") {
        Some(value @ (Value::String(_) | Value::Object(_) | Value::Array(_))) => value,
        _ => {
            return Err(mapping(
                "`state` is required and must be a string, object, or array",
            ))
        }
    };
    let questions = match obj.remove("questions") {
        Some(Value::Object(questions)) if !questions.is_empty() => questions,
        _ => {
            return Err(mapping(
                "`questions` is required and must be a non-empty object",
            ))
        }
    };
    for (id, question) in &questions {
        validate_question(id, question)?;
    }
    if let Some(session_id) = obj.get("session_id") {
        match session_id.as_str() {
            Some(session_id) if session_id.chars().count() <= 256 => {}
            _ => {
                return Err(mapping(
                    "`session_id` must be a string of at most 256 characters",
                ))
            }
        }
    }
    Ok(DecisionsRequest {
        model,
        state,
        questions,
        extra: obj,
    })
}

fn is_structured(value: &Value) -> bool {
    matches!(value, Value::String(_) | Value::Object(_) | Value::Array(_))
}

fn validate_question(id: &str, value: &Value) -> Result<(), ModelError> {
    let Value::Object(question) = value else {
        return Err(mapping(format!("`questions.{id}` must be an object")));
    };
    let kind = question
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            mapping(format!(
                "`questions.{id}.type` is required and must be noul, choice, or score"
            ))
        })?;
    if !matches!(kind, "noul" | "choice" | "score") {
        return Err(mapping(format!(
            "`questions.{id}.type` must be noul, choice, or score"
        )));
    }
    match question.get("instructions") {
        Some(instructions) if is_structured(instructions) => {}
        _ => {
            return Err(mapping(format!(
                "`questions.{id}.instructions` is required and must be a string, object, or array"
            )))
        }
    }
    match kind {
        "noul" => {
            if let Some(criteria) = question.get("criteria") {
                let Value::Object(criteria) = criteria else {
                    return Err(mapping(format!(
                        "`questions.{id}.criteria` must be an object"
                    )));
                };
                if criteria.values().any(|value| !is_structured(value)) {
                    return Err(mapping(format!(
                        "`questions.{id}.criteria` values must be strings, objects, or arrays"
                    )));
                }
            }
        }
        "choice" => {
            let Some(Value::Object(criteria)) = question.get("criteria") else {
                return Err(mapping(format!(
                    "`questions.{id}.criteria` is required and must be an object"
                )));
            };
            if criteria.is_empty() {
                return Err(mapping(format!(
                    "`questions.{id}.criteria` must contain at least one option"
                )));
            }
            if criteria
                .values()
                .any(|value| !value.is_null() && !is_structured(value))
            {
                return Err(mapping(format!(
                    "`questions.{id}.criteria` values must be null, strings, objects, or arrays"
                )));
            }
        }
        "score" => {
            let Some(Value::Array(criteria)) = question.get("criteria") else {
                return Err(mapping(format!(
                    "`questions.{id}.criteria` is required and must be an array"
                )));
            };
            if criteria.is_empty() || criteria.iter().any(|value| !is_structured(value)) {
                return Err(mapping(format!("`questions.{id}.criteria` must be a non-empty array of strings, objects, or arrays")));
            }
        }
        _ => unreachable!(),
    }
    Ok(())
}

/// Map the canonical response to the documented Decisions wire shape. Upstream `id` and `provider`
/// stay at top level through `extra`; unknown usage fields remain nested under `usage`.
pub fn response_to_wire(resp: &DecisionsResponse) -> Value {
    let mut response = resp.extra.clone();
    let mut usage = resp.usage_extra.clone();
    usage.insert("input_tokens".into(), Value::from(resp.usage.prompt_tokens));
    usage.insert(
        "output_tokens".into(),
        Value::from(resp.usage.completion_tokens),
    );
    if let Some(cost) = resp.usage.cost_usd {
        usage.insert("cost".into(), Value::from(cost));
    }
    response.insert("model".into(), Value::String(resp.model.clone()));
    response.insert("answers".into(), Value::Object(resp.answers.clone()));
    response.insert("usage".into(), Value::Object(usage));
    Value::Object(response)
}

#[cfg(test)]
mod tests {
    use super::*;
    use llmleaf_model::Usage;
    use serde_json::{json, Map};

    #[test]
    fn accepts_full_structured_questions_and_preserves_extra() {
        let req = parse_decisions_request(json!({
            "model": "jev-latest", "state": {"ticket": "payouts failed"}, "trace": {"trace_id": "t-1"},
            "questions": {
                "urgent": {"type": "noul", "instructions": {"question": "Urgent?", "context": [1]}, "criteria": {"true": {"meaning": "yes"}, "false": ["no"]}},
                "team": {"type": "choice", "instructions": "Who?", "criteria": {"billing": null, "support": {"detail": "help"}}},
                "severity": {"type": "score", "instructions": ["Rate", "it"], "criteria": ["low", {"label": "high"}]}
            }
        })).unwrap();
        assert_eq!(req.extra["trace"]["trace_id"], "t-1");
        assert_eq!(req.questions.len(), 3);
    }

    #[test]
    fn rejects_invalid_question_shapes() {
        let body = json!({"model":"m", "state":"s", "questions":{"x":{"type":"choice", "instructions":"x", "criteria":{}}}});
        assert!(parse_decisions_request(body).is_err());
    }

    #[test]
    fn response_keeps_provider_fields_and_unknown_usage() {
        let response = DecisionsResponse {
            model: "jev-1.13".into(),
            answers: Map::new(),
            usage: Usage {
                prompt_tokens: 4,
                completion_tokens: 2,
                total_tokens: 6,
                cost_usd: Some(0.01),
                ..Usage::default()
            },
            extra: Map::from_iter([
                ("id".into(), json!("gen-1")),
                ("provider".into(), json!("TypeSafe")),
            ]),
            usage_extra: Map::from_iter([("cached_tokens".into(), json!(3))]),
        };
        let wire = response_to_wire(&response);
        assert_eq!(wire["id"], "gen-1");
        assert_eq!(wire["usage"]["input_tokens"], 4);
        assert_eq!(wire["usage"]["cached_tokens"], 3);
    }
}
