//! Wire mapping shared by the two Decisions-capable provider edges.
//!
//! TypeSafe System One and OpenRouter's Decisions router deliberately use the same request and
//! response shape. Keep the lossless JSON mapping here, while each provider owns its endpoint and
//! authentication quirks.

use llmleaf_model::{DecisionsRequest, DecisionsResponse, ModelError, Usage};
use serde_json::{Map, Value};

/// Build the upstream body. Provider-specific request fields ride in `extra` unchanged; canonical
/// fields always win on a collision so the routed model and submitted questions cannot be replaced.
pub(crate) fn decisions_request_to_wire(req: &DecisionsRequest) -> Value {
    let mut body = req.extra.clone();
    body.insert("model".into(), Value::String(req.model.clone()));
    body.insert("state".into(), req.state.clone());
    body.insert("questions".into(), Value::Object(req.questions.clone()));
    Value::Object(body)
}

/// Map a Decisions response without flattening typed answers or discarding fields added by either
/// upstream. `id` and `provider` intentionally remain in `extra`, as do any future response fields.
pub(crate) fn decisions_response_from_wire(
    value: Value,
    fallback_model: &str,
) -> Result<DecisionsResponse, ModelError> {
    let mut object = match value {
        Value::Object(object) => object,
        _ => {
            return Err(ModelError::Mapping(
                "decisions response must be a JSON object".into(),
            ));
        }
    };
    let model = match object.remove("model") {
        Some(Value::String(model)) => model,
        Some(_) => {
            return Err(ModelError::Mapping(
                "decisions response `model` must be a string".into(),
            ));
        }
        None => fallback_model.to_owned(),
    };
    let answers = match object.remove("answers") {
        Some(Value::Object(answers)) => answers,
        _ => {
            return Err(ModelError::Mapping(
                "decisions response `answers` must be an object".into(),
            ));
        }
    };
    let (usage, usage_extra) = match object.remove("usage").map(parse_usage).transpose()? {
        Some(usage) => usage,
        None => {
            return Err(ModelError::Mapping(
                "decisions response `usage` must be an object".into(),
            ));
        }
    };
    Ok(DecisionsResponse {
        model,
        answers,
        usage,
        extra: object,
        usage_extra,
    })
}

fn parse_usage(value: Value) -> Result<(Usage, Map<String, Value>), ModelError> {
    let mut object = match value {
        Value::Object(object) => object,
        _ => {
            return Err(ModelError::Mapping(
                "decisions response `usage` must be an object".into(),
            ));
        }
    };
    let prompt_tokens = take_u64(&mut object, "input_tokens")
        .or_else(|| take_u64(&mut object, "prompt_tokens"))
        .unwrap_or_default();
    let completion_tokens = take_u64(&mut object, "output_tokens")
        .or_else(|| take_u64(&mut object, "completion_tokens"))
        .unwrap_or_default();
    // These APIs document input/output counters. Keep a non-standard upstream `total_tokens` in
    // `usage_extra` and derive the canonical total from the two documented counters.
    let total_tokens = prompt_tokens.saturating_add(completion_tokens);
    let cost_usd = take_f64(&mut object, "cost").or_else(|| take_f64(&mut object, "cost_usd"));
    Ok((
        Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens,
            cost_usd,
            ..Default::default()
        },
        object,
    ))
}

fn take_u64(object: &mut Map<String, Value>, key: &str) -> Option<u64> {
    object.remove(key).and_then(|v| v.as_u64())
}

fn take_f64(object: &mut Map<String, Value>, key: &str) -> Option<f64> {
    object.remove(key).and_then(|v| v.as_f64())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn response_keeps_structured_answers_and_unknown_fields() {
        let response = decisions_response_from_wire(
            json!({
                "id": "gen-1", "provider": "TypeSafe", "model": "jev-1.13",
                "answers": { "triage": { "type": "choice", "choice": "payments", "probabilities": { "payments": 0.8 } } },
                "usage": { "input_tokens": 4, "output_tokens": 2, "total_tokens": 99, "cost": 0.00001, "native": true },
                "trace": { "request": "abc" }
            }),
            "fallback",
        ).unwrap();
        assert_eq!(response.answers["triage"]["probabilities"]["payments"], 0.8);
        assert_eq!(response.usage.prompt_tokens, 4);
        assert_eq!(response.usage.cost_usd, Some(0.00001));
        assert_eq!(response.usage.total_tokens, 6);
        assert_eq!(response.extra["id"], "gen-1");
        assert_eq!(response.usage_extra["native"], true);
        assert_eq!(response.usage_extra["total_tokens"], 99);
    }

    #[test]
    fn rejects_malformed_or_error_envelope_successes() {
        for value in [
            json!(null),
            json!({ "error": { "message": "bad request" } }),
            json!({ "answers": {}, "usage": [] }),
        ] {
            assert!(matches!(
                decisions_response_from_wire(value, "jev"),
                Err(ModelError::Mapping(_))
            ));
        }
    }
}
