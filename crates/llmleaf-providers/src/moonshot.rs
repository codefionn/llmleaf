//! Moonshot AI (Kimi) provider — the OpenAI wire plus Moonshot's "flavored" JSON-schema dialect.
//!
//! Moonshot serves the plain OpenAI chat-completions wire, so endpoint/auth/batch quirks stay where
//! every other compatible vendor's live: the [`crate::Brand`] table rows for `moonshot` and `kimi-coding`,
//! and this provider delegates every operation to the [`OpenAiCompatProvider`] built from them. What
//! makes Moonshot a first-class provider rather than a pure table row is its *request validator*:
//! `tools[].function.parameters` must be a **"moonshot flavored JSON schema"** (MFJS) — a restricted
//! subset of JSON Schema enforced server-side by Moonshot's open-source validator
//! (<https://github.com/MoonshotAI/walle>, `docs/mfjs-spec.zh.md`). MFJS allows only `type` (a
//! single string), `properties`, `required`, `additionalProperties`, `items` (a single schema),
//! `anyOf`, `enum`, root-level `$defs` with internal `#/$defs/…` refs, `description`, and `default`.
//! Anything else a mainstream generator emits (Pydantic, zod) can 400 — e.g. a node declaring both
//! `anyOf` and a sibling `type` fails with `"when using anyOf, type should be defined in anyOf items
//! instead of the parent schema"` — so tools that work against every other OpenAI-wire brand break
//! on Moonshot.
//!
//! The fix is a documented dialect mapping (principle 7 — never silent mutation): [`flavor_schema`]
//! rewrites each tool's parameter schema into the flavored subset before the shared wire mapping
//! sends it. Each rewrite preserves the schema's meaning where MFJS can express it and widens
//! minimally where it cannot (a widened schema still validates every instance the original
//! accepted, so no valid tool call is rejected):
//!
//!   - `type` alongside `anyOf`: the type moves into each branch that lacks one, off the parent.
//!   - `oneOf` → `anyOf` ("exactly one" widens to "at least one"; MFJS has no `oneOf`).
//!   - `allOf` → merged into the parent (`properties` union, `required` union, parent keys win).
//!   - `const: v` → `enum: [v]`.
//!   - a `type` *array* (`["string","null"]`) → an `anyOf` of single-type branches.
//!   - tuple-form `items: […]` / `prefixItems` → one `items` schema of the branches' `anyOf`.
//!   - `$ref` with siblings (Pydantic's `{"$ref": …, "description": …}`) → the ref moves into a
//!     single-branch `anyOf`; the siblings stay on the parent (MFJS refs must stand alone).
//!   - root `definitions` → `$defs`, and `#/definitions/…` refs → `#/$defs/…`.
//!   - a node with no `type`/`anyOf`/`$ref` gets one *inferred from its own keywords* (`properties`
//!     → object, `items` → array, uniform `enum` literals → their type, `pattern` → string, …) —
//!     MFJS rejects a property without an explicit type. Nothing is ever guessed: a node whose
//!     keywords don't pin the type down is left alone.
//!   - `required` names not present in `properties` are dropped (MFJS rejects them).
//!
//! Unsupported *annotations* MFJS merely ignores (`format`, `title`, length/range constraints) ride
//! through verbatim — stripping them would discard information the model can still read. The same
//! rewrite applies to a `response_format.json_schema.schema` riding in `extra`, which the upstream
//! validates the same way. Chat and batch lines both pass through it.
//!
//! Beyond schemas, Moonshot's catalog is chat + files only — no embeddings, TTS, or STT endpoints —
//! so those modalities return `Unsupported` (routing falls past without a health penalty) instead of
//! delegating into an upstream 404 that would read as a provider failure. Known upstream limits this
//! provider deliberately does NOT paper over (an honest upstream 400 beats a silent behavior
//! change): `tool_choice: "required"`/named-function forcing is not served, temperature caps at 1.0,
//! and `n > 1` needs a non-zero temperature.
//!
//! On the hot path the rewrite is in-place on the already-owned request: a conformant schema (the
//! common case) is a read-only walk, zero allocations (principle 1).

use async_trait::async_trait;
use llmleaf_model::{
    AudioStream, BatchHandle, BatchResultStream, BatchSpec, ChatRequest, EmbeddingRequest,
    EmbeddingResponse, ModelError, ModelInfo, RerankRequest, RerankResponse, ResponseStream,
    SpeechRequest, ToolDef, TranscriptionRequest, TranscriptionResponse, VoiceInfo,
};
use llmleaf_provider::{Provider, ProviderCx, RealtimeParams, RealtimePeer};
use serde_json::{Map, Value};

use crate::compat::OpenAiCompatProvider;
use crate::transport::Transports;

/// The Moonshot provider: the OpenAI-compat implementation for the `moonshot`/`kimi-coding` brand
/// rows, with tool parameter schemas rewritten into Moonshot's flavored subset on the way out.
pub struct MoonshotProvider {
    inner: OpenAiCompatProvider,
}

impl MoonshotProvider {
    /// Construct from a Moonshot config `kind` (`moonshot`/`kimi`/`kimi-k2`, or the "Kimi for
    /// Coding" subscription kinds `kimi-coding`/`kimi-for-coding`). `None` for any other kind — the
    /// factory falls through to the plain compat table.
    pub fn for_kind(kind: &str, transports: &Transports) -> Option<Self> {
        if !matches!(
            kind,
            "moonshot" | "kimi" | "kimi-k2" | "kimi-coding" | "kimi-for-coding"
        ) {
            return None;
        }
        let inner = OpenAiCompatProvider::for_kind(kind, transports)?;
        Some(MoonshotProvider { inner })
    }
}

/// Rewrite every tool's parameter schema into the flavored subset, in place.
fn flavor_tools(tools: &mut [ToolDef]) {
    for tool in tools {
        flavor_schema(&mut tool.parameters);
    }
}

/// Rewrite one JSON Schema into Moonshot's flavored subset (MFJS), in place. Returns whether
/// anything changed (a conformant schema — the common case — is untouched: a read-only walk, no
/// allocation). The individual rewrites are documented on the module and on each rule below.
fn flavor_schema(schema: &mut Value) -> bool {
    // Root-only rule first: draft-07 generators emit `definitions`, MFJS only knows `$defs`.
    let mut changed = definitions_to_defs(schema);
    changed |= flavor_node(schema);
    changed
}

/// Root `definitions` → `$defs` (merged if both exist, existing `$defs` entries win). The matching
/// `#/definitions/…` → `#/$defs/…` ref rewrite happens per node in [`flavor_node`]. MFJS only
/// supports `$defs` at the root, so this does not walk deeper.
fn definitions_to_defs(root: &mut Value) -> bool {
    let Some(obj) = root.as_object_mut() else {
        return false;
    };
    let Some(defs) = obj.remove("definitions") else {
        return false;
    };
    match (obj.get_mut("$defs"), defs) {
        (Some(Value::Object(existing)), Value::Object(moved)) => {
            for (k, v) in moved {
                existing.entry(k).or_insert(v);
            }
        }
        (Some(_), _) => {}
        (None, defs) => {
            obj.insert("$defs".into(), defs);
        }
    }
    true
}

/// Rewrite one schema node, then recurse into every position that holds a subschema.
fn flavor_node(schema: &mut Value) -> bool {
    let Some(obj) = schema.as_object_mut() else {
        // A boolean schema (`additionalProperties: false`) or junk — nothing to rewrite.
        return false;
    };

    // Order matters: `allOf` merging can surface a `$ref`/`type`/`anyOf` on this node; `oneOf`
    // folding, type-array splitting, and ref isolation all produce the `anyOf` the type push-down
    // then feeds (isolation MUST precede the push, or a `$ref`+`type` node would keep an illegal
    // `type` beside the fresh `anyOf`); type inference runs last so it sees the node's final shape.
    let mut changed = merge_all_of(obj);
    changed |= const_to_enum(obj);
    changed |= one_of_to_any_of(obj);
    changed |= split_type_array(obj);
    changed |= isolate_ref(obj);
    changed |= push_type_into_any_of(obj);
    changed |= fold_tuple_items(obj);
    changed |= infer_missing_type(obj);
    changed |= prune_required(obj);
    changed |= rewrite_ref_target(obj);

    // Positions holding one subschema (some are MFJS-unknown keywords — recurse anyway; if the
    // upstream ignores them, a well-formed subschema still reads better than a broken one)…
    for key in [
        "items",
        "additionalProperties",
        "not",
        "if",
        "then",
        "else",
        "contains",
        "propertyNames",
    ] {
        if let Some(v) = obj.get_mut(key) {
            changed |= flavor_node(v);
        }
    }
    // …a list of subschemas…
    if let Some(Value::Array(items)) = obj.get_mut("anyOf") {
        for item in items {
            changed |= flavor_node(item);
        }
    }
    // …or a map of subschemas.
    for key in ["properties", "patternProperties", "$defs"] {
        if let Some(Value::Object(map)) = obj.get_mut(key) {
            for (_, v) in map.iter_mut() {
                changed |= flavor_node(v);
            }
        }
    }
    changed
}

/// MFJS has no `allOf`: merge its branches into the parent node. `properties` merge per key and
/// `required` unions; for everything else the parent (or an earlier branch) wins. An intersection
/// can't be expressed in MFJS, so this is the minimal widening that keeps every branch's structure
/// visible to the model.
fn merge_all_of(obj: &mut Map<String, Value>) -> bool {
    let Some(all_of) = obj.remove("allOf") else {
        return false;
    };
    let Value::Array(branches) = all_of else {
        return true; // malformed `allOf` — dropping it is all MFJS would let us do anyway
    };
    for branch in branches {
        let Value::Object(branch) = branch else {
            continue;
        };
        for (k, v) in branch {
            if !obj.contains_key(&k) {
                obj.insert(k, v);
                continue;
            }
            match (obj.get_mut(&k), k.as_str(), v) {
                (
                    Some(Value::Object(existing)),
                    "properties" | "patternProperties",
                    Value::Object(moved),
                ) => {
                    for (pk, pv) in moved {
                        existing.entry(pk).or_insert(pv);
                    }
                }
                (Some(Value::Array(existing)), "required", Value::Array(moved)) => {
                    for name in moved {
                        if !existing.contains(&name) {
                            existing.push(name);
                        }
                    }
                }
                _ => {} // both sides carry the key with unmergeable shapes — the parent wins
            }
        }
    }
    true
}

/// MFJS has no `const`: `const: v` is exactly `enum: [v]`. A coexisting `enum` (pathological) loses
/// to the more specific `const`.
fn const_to_enum(obj: &mut Map<String, Value>) -> bool {
    let Some(c) = obj.remove("const") else {
        return false;
    };
    obj.insert("enum".into(), Value::Array(vec![c]));
    true
}

/// MFJS has no `oneOf`: fold it into `anyOf` ("exactly one" widens to "at least one" — every
/// instance the original accepted still validates).
fn one_of_to_any_of(obj: &mut Map<String, Value>) -> bool {
    let Some(one_of) = obj.remove("oneOf") else {
        return false;
    };
    match (obj.get_mut("anyOf"), one_of) {
        (Some(Value::Array(existing)), Value::Array(moved)) => existing.extend(moved),
        (Some(_), _) => {}
        (None, one_of) => {
            obj.insert("anyOf".into(), one_of);
        }
    }
    true
}

/// MFJS `type` must be a single string: split `type: ["string","null"]` into an `anyOf` of
/// single-type branches. If an `anyOf` already coexists (pathological — the standard reads that as
/// an intersection MFJS can't express), the array type is simply dropped: widening, never rejecting.
fn split_type_array(obj: &mut Map<String, Value>) -> bool {
    let Some(Value::Array(types)) = obj.get("type") else {
        return false;
    };
    if let [Value::String(single)] = types.as_slice() {
        let single = Value::String(single.clone());
        obj.insert("type".into(), single);
        return true;
    }
    let branches: Vec<Value> = types
        .iter()
        .filter(|t| t.is_string())
        .map(|t| {
            let mut b = Map::new();
            b.insert("type".into(), t.clone());
            Value::Object(b)
        })
        .collect();
    obj.remove("type");
    if !obj.contains_key("anyOf") && !branches.is_empty() {
        obj.insert("anyOf".into(), Value::Array(branches));
    }
    true
}

/// The documented rejection this provider exists for: a node must not declare `type` alongside
/// `anyOf` ("when using anyOf, type should be defined in anyOf items instead of the parent
/// schema"). Move the parent type into each branch that lacks one — the standard's semantics for
/// the original — and drop it from the parent. A `$ref` branch (from [`isolate_ref`]) takes no
/// type: a ref must stand alone, and the def it points at declares its own.
fn push_type_into_any_of(obj: &mut Map<String, Value>) -> bool {
    if !obj.contains_key("anyOf") || !obj.contains_key("type") {
        return false;
    }
    let ty = obj.remove("type").expect("checked above");
    if let Some(Value::Array(branches)) = obj.get_mut("anyOf") {
        for branch in branches.iter_mut() {
            if let Some(b) = branch.as_object_mut() {
                if !b.contains_key("$ref") {
                    b.entry("type").or_insert_with(|| ty.clone());
                }
            }
        }
    }
    true
}

/// MFJS refs must stand alone — even a sibling `description` (Pydantic's habitual shape) is
/// rejected. Move the ref into a single-branch `anyOf` and leave the siblings on the parent, where
/// `description`/`default` are legal.
fn isolate_ref(obj: &mut Map<String, Value>) -> bool {
    if obj.len() <= 1 || !obj.contains_key("$ref") {
        return false;
    }
    let reference = obj.remove("$ref").expect("checked above");
    let mut branch = Map::new();
    branch.insert("$ref".into(), reference);
    match obj.get_mut("anyOf") {
        Some(Value::Array(existing)) => existing.push(Value::Object(branch)),
        _ => {
            obj.insert("anyOf".into(), Value::Array(vec![Value::Object(branch)]));
        }
    }
    true
}

/// MFJS `items` must be one schema object ("items must be an object"): fold draft-07 tuple-form
/// `items: […]` and 2020-12 `prefixItems` into a single `items` whose `anyOf` carries the branches
/// (positional validation widens to "each element matches some branch").
fn fold_tuple_items(obj: &mut Map<String, Value>) -> bool {
    let mut branches: Vec<Value> = Vec::new();
    if matches!(obj.get("prefixItems"), Some(Value::Array(_))) {
        if let Some(Value::Array(prefix)) = obj.remove("prefixItems") {
            branches.extend(prefix);
        }
    }
    match obj.get("items") {
        Some(Value::Array(_)) => {
            if let Some(Value::Array(tuple)) = obj.remove("items") {
                branches.extend(tuple);
            }
        }
        // A single-schema `items` next to `prefixItems` covers the elements *after* the prefix —
        // in the folded form it is just one more branch.
        Some(_) if !branches.is_empty() => {
            branches.push(obj.remove("items").expect("checked above"));
        }
        _ => {}
    }
    if branches.is_empty() {
        return false;
    }
    let mut items = Map::new();
    items.insert("anyOf".into(), Value::Array(branches));
    obj.insert("items".into(), Value::Object(items));
    true
}

/// MFJS rejects a node with no explicit type ("At path 'properties.X': type is not defined").
/// When the node's own keywords pin the type down, state it; when they don't, leave the node alone
/// — inferring would be guessing (SOUL: never guess). A node carrying `anyOf`/`$ref` already
/// declares its shape and needs none.
fn infer_missing_type(obj: &mut Map<String, Value>) -> bool {
    if obj.contains_key("type") || obj.contains_key("anyOf") || obj.contains_key("$ref") {
        return false;
    }
    let has = |keys: &[&str]| keys.iter().any(|k| obj.contains_key(*k));
    let inferred = if has(&[
        "properties",
        "patternProperties",
        "required",
        "additionalProperties",
    ]) {
        Some("object")
    } else if has(&["items", "contains", "minItems", "maxItems", "uniqueItems"]) {
        Some("array")
    } else if let Some(Value::Array(values)) = obj.get("enum") {
        enum_literal_type(values)
    } else if has(&["pattern", "format", "minLength", "maxLength"]) {
        Some("string")
    } else if has(&[
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
    ]) {
        Some("number")
    } else {
        None
    };
    match inferred {
        Some(ty) => {
            obj.insert("type".into(), Value::String(ty.into()));
            true
        }
        None => false,
    }
}

/// The single type a uniform `enum` literal list implies, or `None` for an empty/mixed list
/// (mixed-type enums are themselves invalid MFJS, but nothing we could rewrite would fix that).
fn enum_literal_type(values: &[Value]) -> Option<&'static str> {
    let literal = |v: &Value| match v {
        Value::String(_) => Some("string"),
        Value::Number(n) if n.is_i64() || n.is_u64() => Some("integer"),
        Value::Number(_) => Some("number"),
        Value::Bool(_) => Some("boolean"),
        _ => None,
    };
    let first = literal(values.first()?)?;
    values.iter().skip(1).try_fold(first, |ty, v| {
        match (ty, literal(v)?) {
            (a, b) if a == b => Some(a),
            // Ints and floats mix into plain `number`.
            ("integer" | "number", "integer" | "number") => Some("number"),
            _ => None,
        }
    })
}

/// MFJS rejects `required` names that aren't declared under `properties`: drop exactly those
/// (widening — the original may have required a key `additionalProperties` would carry).
fn prune_required(obj: &mut Map<String, Value>) -> bool {
    let Some(Value::Object(properties)) = obj.get("properties") else {
        return false;
    };
    let known: Vec<String> = properties.keys().cloned().collect();
    let Some(Value::Array(required)) = obj.get_mut("required") else {
        return false;
    };
    let before = required.len();
    required.retain(|name| name.as_str().is_some_and(|n| known.iter().any(|k| k == n)));
    if required.is_empty() && before > 0 {
        obj.remove("required");
        return true;
    }
    required.len() != before
}

/// Rewrite a draft-07 `#/definitions/…` ref onto the `$defs` the root rewrite moved it to.
fn rewrite_ref_target(obj: &mut Map<String, Value>) -> bool {
    let Some(Value::String(reference)) = obj.get_mut("$ref") else {
        return false;
    };
    match reference.strip_prefix("#/definitions/") {
        Some(rest) => {
            *reference = format!("#/$defs/{rest}");
            true
        }
        None => false,
    }
}

#[async_trait]
impl Provider for MoonshotProvider {
    fn name(&self) -> &str {
        self.inner.name()
    }

    /// Chat with the tool schemas flavored. The request is already owned, so the rewrite mutates it
    /// in place — no clone, and a conformant schema costs only the walk.
    async fn chat(
        &self,
        mut req: ChatRequest,
        cx: &ProviderCx,
    ) -> Result<ResponseStream, ModelError> {
        flavor_tools(&mut req.tools);
        // A structured-output schema riding in `extra` hits the same upstream validator.
        if let Some(schema) = req
            .extra
            .get_mut("response_format")
            .and_then(|rf| rf.pointer_mut("/json_schema/schema"))
        {
            flavor_schema(schema);
        }
        self.inner.chat(req, cx).await
    }

    /// Batch lines are built by the same wire mapper live chat uses, so they need the same flavoring.
    async fn batch_create(
        &self,
        mut req: BatchSpec,
        cx: &ProviderCx,
    ) -> Result<BatchHandle, ModelError> {
        for item in &mut req.items {
            flavor_tools(&mut item.request.tools);
        }
        self.inner.batch_create(req, cx).await
    }

    // Moonshot's platform serves chat + files only — no embeddings, TTS, or STT endpoints. Declare
    // those honestly instead of delegating into a 404 that would read as a provider *failure*:
    // `Unsupported` lets routing fall past without a health penalty.

    async fn embed(
        &self,
        _req: EmbeddingRequest,
        _cx: &ProviderCx,
    ) -> Result<EmbeddingResponse, ModelError> {
        Err(ModelError::Unsupported(format!(
            "provider '{}' does not support embeddings",
            self.name()
        )))
    }

    async fn speech(
        &self,
        _req: SpeechRequest,
        _cx: &ProviderCx,
    ) -> Result<AudioStream, ModelError> {
        Err(ModelError::Unsupported(format!(
            "provider '{}' does not support speech synthesis",
            self.name()
        )))
    }

    async fn transcribe(
        &self,
        _req: TranscriptionRequest,
        _cx: &ProviderCx,
    ) -> Result<TranscriptionResponse, ModelError> {
        Err(ModelError::Unsupported(format!(
            "provider '{}' does not support transcription",
            self.name()
        )))
    }

    // Everything below is the brand row's business — pure delegation.

    async fn rerank(
        &self,
        req: RerankRequest,
        cx: &ProviderCx,
    ) -> Result<RerankResponse, ModelError> {
        self.inner.rerank(req, cx).await
    }

    async fn voices(&self, model: &str, cx: &ProviderCx) -> Result<Vec<VoiceInfo>, ModelError> {
        self.inner.voices(model, cx).await
    }

    async fn models(&self, cx: &ProviderCx) -> Result<Vec<ModelInfo>, ModelError> {
        self.inner.models(cx).await
    }

    async fn batch_retrieve(
        &self,
        upstream_id: &str,
        cx: &ProviderCx,
    ) -> Result<BatchHandle, ModelError> {
        self.inner.batch_retrieve(upstream_id, cx).await
    }

    async fn batch_results(
        &self,
        upstream_id: &str,
        cx: &ProviderCx,
    ) -> Result<BatchResultStream, ModelError> {
        self.inner.batch_results(upstream_id, cx).await
    }

    async fn batch_cancel(
        &self,
        upstream_id: &str,
        cx: &ProviderCx,
    ) -> Result<BatchHandle, ModelError> {
        self.inner.batch_cancel(upstream_id, cx).await
    }

    fn supports_realtime(&self) -> bool {
        self.inner.supports_realtime()
    }

    async fn realtime(
        &self,
        params: RealtimeParams,
        peer: RealtimePeer,
        cx: &ProviderCx,
    ) -> Result<(), ModelError> {
        self.inner.realtime(params, peer, cx).await
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use llmleaf_model::{collect, Message, Role};
    use serde_json::json;

    use super::*;
    use crate::fake::{FakeHttpTransport, FakeRealtimeTransport, FakeResponse};
    use crate::transport::{HttpBody, HttpRequest, MultipartPart};

    fn http_transports(http: FakeHttpTransport) -> Transports {
        Transports {
            http: Arc::new(http),
            realtime: Arc::new(FakeRealtimeTransport::scripted(Vec::new())),
        }
    }

    fn cx() -> ProviderCx {
        ProviderCx {
            credential: Some("test-key".into()),
            ..Default::default()
        }
    }

    fn chat_with_tools(tools: Vec<ToolDef>) -> ChatRequest {
        ChatRequest {
            model: "kimi-k2-0711-preview".into(),
            messages: vec![Message::text(Role::User, "hi")],
            max_tokens: None,
            temperature: None,
            top_p: None,
            stop: vec![],
            stream: false,
            tools,
            tool_choice: None,
            thinking: None,
            extra: Default::default(),
        }
    }

    // -----------------------------------------------------------------------------------------
    // The flavored-schema rewrite
    // -----------------------------------------------------------------------------------------

    #[test]
    fn parent_type_moves_into_anyof_branches() {
        // The exact shape Moonshot 400s on: `type` alongside `anyOf` at the root ("At path 'root':
        // when using anyOf, type should be defined in anyOf items instead of the parent schema").
        let mut schema = json!({
            "type": "string",
            "anyOf": [{ "const": "a" }, { "type": "null" }]
        });
        assert!(flavor_schema(&mut schema));
        assert_eq!(
            schema,
            json!({
                // A branch without a type inherits the parent's; one with its own keeps it. The
                // branch `const` (also not MFJS) becomes the equivalent one-literal `enum`.
                "anyOf": [{ "enum": ["a"], "type": "string" }, { "type": "null" }]
            })
        );
    }

    #[test]
    fn oneof_folds_into_anyof_and_gets_the_type() {
        // MFJS has no `oneOf`: it widens into `anyOf`, and the sibling `type` then pushes down.
        let mut schema = json!({
            "type": "integer",
            "oneOf": [{ "minimum": 0 }, { "maximum": -10 }]
        });
        assert!(flavor_schema(&mut schema));
        assert_eq!(
            schema,
            json!({
                "anyOf": [
                    { "minimum": 0, "type": "integer" },
                    { "maximum": -10, "type": "integer" }
                ]
            })
        );
    }

    #[test]
    fn type_array_splits_into_anyof_branches() {
        // MFJS `type` is a single string; the classic nullable shape becomes a union.
        let mut schema = json!({ "type": ["string", "null"], "description": "maybe" });
        assert!(flavor_schema(&mut schema));
        assert_eq!(
            schema,
            json!({
                "anyOf": [{ "type": "string" }, { "type": "null" }],
                "description": "maybe"
            })
        );

        // A one-element array unwraps to the plain string.
        let mut schema = json!({ "type": ["boolean"] });
        assert!(flavor_schema(&mut schema));
        assert_eq!(schema, json!({ "type": "boolean" }));
    }

    #[test]
    fn pydantic_ref_with_description_unwraps_end_to_end() {
        // Pydantic's habitual shape: a single-branch `allOf` carrying a draft-07 ref, described on
        // the parent. MFJS wants a bare `#/$defs/…` ref with no siblings: the `allOf` merges, the
        // ref isolates into a one-branch `anyOf`, `definitions` becomes `$defs`, refs follow.
        let mut schema = json!({
            "type": "object",
            "properties": {
                "shape": { "allOf": [{ "$ref": "#/definitions/Shape" }], "description": "d" }
            },
            "definitions": {
                "Shape": { "type": "string", "enum": ["circle", "square"] }
            }
        });
        assert!(flavor_schema(&mut schema));
        assert_eq!(
            schema,
            json!({
                "type": "object",
                "properties": {
                    "shape": { "description": "d", "anyOf": [{ "$ref": "#/$defs/Shape" }] }
                },
                "$defs": {
                    "Shape": { "type": "string", "enum": ["circle", "square"] }
                }
            })
        );
    }

    #[test]
    fn ref_with_sibling_type_isolates_without_leaking_the_type() {
        // `$ref` + `type` on one node: the ref isolates into `anyOf` and the type drops — it may
        // live on neither the parent (illegal beside `anyOf`) nor the ref branch (refs stand
        // alone); the def it points at declares its own.
        let mut schema = json!({ "$ref": "#/$defs/Shape", "type": "object" });
        assert!(flavor_schema(&mut schema));
        assert_eq!(schema, json!({ "anyOf": [{ "$ref": "#/$defs/Shape" }] }));
    }

    #[test]
    fn const_becomes_enum_with_inferred_type() {
        // MFJS has no `const`; `enum: [v]` says the same thing, and the literal pins the type a
        // bare `const` node would otherwise lack.
        let mut schema = json!({ "const": "fixed" });
        assert!(flavor_schema(&mut schema));
        assert_eq!(schema, json!({ "enum": ["fixed"], "type": "string" }));
    }

    #[test]
    fn tuple_items_fold_into_one_schema() {
        // MFJS `items` must be a single object ("items must be an object"): tuple positions widen
        // into an `anyOf` every element may match.
        let mut schema = json!({
            "type": "array",
            "items": [{ "type": "string" }, { "type": "integer" }]
        });
        assert!(flavor_schema(&mut schema));
        assert_eq!(
            schema,
            json!({
                "type": "array",
                "items": { "anyOf": [{ "type": "string" }, { "type": "integer" }] }
            })
        );

        // 2020-12 `prefixItems` + trailing `items` fold the same way.
        let mut schema = json!({
            "type": "array",
            "prefixItems": [{ "type": "string" }],
            "items": { "type": "number" }
        });
        assert!(flavor_schema(&mut schema));
        assert_eq!(
            schema,
            json!({
                "type": "array",
                "items": { "anyOf": [{ "type": "string" }, { "type": "number" }] }
            })
        );
    }

    #[test]
    fn missing_types_are_inferred_never_guessed() {
        // MFJS rejects a property with no explicit type; keywords that pin the type down state it.
        let mut schema = json!({
            "properties": {
                "kind": { "enum": ["a", "b"] },
                "count": { "minimum": 0 },
                "name": { "pattern": "^u" },
                "mixed": { "enum": ["a", 1] }
            }
        });
        assert!(flavor_schema(&mut schema));
        assert_eq!(schema["type"], "object");
        assert_eq!(schema["properties"]["kind"]["type"], "string");
        assert_eq!(schema["properties"]["count"]["type"], "number");
        assert_eq!(schema["properties"]["name"]["type"], "string");
        // A mixed-literal enum pins nothing down — no guess (it is invalid MFJS either way, but a
        // wrong type would misdescribe the tool).
        assert_eq!(schema["properties"]["mixed"], json!({ "enum": ["a", 1] }));

        // A node whose keywords say nothing about its type is left alone entirely.
        let original = json!({ "description": "anything goes" });
        let mut schema = original.clone();
        assert!(!flavor_schema(&mut schema));
        assert_eq!(schema, original);
    }

    #[test]
    fn required_names_prune_to_declared_properties() {
        // MFJS rejects `required` entries that aren't in `properties`.
        let mut schema = json!({
            "type": "object",
            "properties": { "a": { "type": "string" } },
            "required": ["a", "b"]
        });
        assert!(flavor_schema(&mut schema));
        assert_eq!(schema["required"], json!(["a"]));

        // …and an emptied list drops rather than sending `required: []`.
        let mut schema = json!({
            "type": "object",
            "properties": { "a": { "type": "string" } },
            "required": ["b"]
        });
        assert!(flavor_schema(&mut schema));
        assert!(schema.get("required").is_none());
    }

    #[test]
    fn rewrites_nested_subschemas() {
        // The offending union sits under properties/items/$defs — every subschema position walks.
        let mut schema = json!({
            "type": "object",
            "properties": {
                "id": { "type": "string", "anyOf": [{ "pattern": "^u" }, { "type": "null" }] },
                "tags": {
                    "type": "array",
                    "items": { "type": "number", "anyOf": [{ "minimum": 0 }] }
                }
            },
            "$defs": {
                "u": { "type": "string", "anyOf": [{ "const": "x" }] }
            }
        });
        assert!(flavor_schema(&mut schema));
        assert_eq!(
            schema,
            json!({
                "type": "object",
                "properties": {
                    "id": { "anyOf": [{ "pattern": "^u", "type": "string" }, { "type": "null" }] },
                    "tags": {
                        "type": "array",
                        "items": { "anyOf": [{ "minimum": 0, "type": "number" }] }
                    }
                },
                "$defs": {
                    "u": { "anyOf": [{ "enum": ["x"], "type": "string" }] }
                }
            })
        );
    }

    #[test]
    fn conformant_schema_is_untouched() {
        let original = json!({
            "type": "object",
            "properties": {
                "q": { "type": "string", "description": "query" },
                "u": { "anyOf": [{ "type": "string" }, { "type": "null" }] }
            },
            "required": ["q"],
            "additionalProperties": false
        });
        let mut schema = original.clone();
        assert!(!flavor_schema(&mut schema));
        assert_eq!(schema, original);
    }

    #[test]
    fn non_object_schemas_are_tolerated() {
        // Boolean schemas (`additionalProperties: false`-style) and junk pass through unchanged.
        for v in [json!(true), json!(false), json!(null), json!("nonsense")] {
            let mut schema = v.clone();
            assert!(!flavor_schema(&mut schema));
            assert_eq!(schema, v);
        }
    }

    // -----------------------------------------------------------------------------------------
    // The provider edge: what actually goes over the wire
    // -----------------------------------------------------------------------------------------

    /// A fake transport that captures each request's JSON body (or multipart JSONL) for assertions.
    fn capturing_transport(
        captured: Arc<Mutex<Vec<HttpRequest>>>,
        respond: impl Fn(&HttpRequest) -> FakeResponse + Send + Sync + 'static,
    ) -> FakeHttpTransport {
        FakeHttpTransport::new(move |req| {
            captured.lock().unwrap().push(req.clone());
            Ok(respond(req))
        })
    }

    #[tokio::test]
    async fn chat_sends_flavored_tool_schemas() {
        let captured = Arc::new(Mutex::new(Vec::new()));
        let sse = concat!(
            "data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n",
        );
        let transports = http_transports(capturing_transport(captured.clone(), move |_| {
            FakeResponse::ok_bytes("text/event-stream", sse)
        }));
        let provider = MoonshotProvider::for_kind("moonshot", &transports).unwrap();

        let offending = ToolDef {
            name: "lookup".into(),
            description: Some("look a thing up".into()),
            parameters: json!({
                "type": "object",
                "properties": {
                    "id": { "type": "string", "anyOf": [{ "pattern": "^u" }, { "type": "null" }] }
                }
            }),
        };
        let conformant = ToolDef {
            name: "noop".into(),
            description: None,
            parameters: json!({ "type": "object", "properties": {} }),
        };
        let stream = provider
            .chat(chat_with_tools(vec![offending, conformant]), &cx())
            .await
            .expect("chat returns a stream");
        collect(stream).await.expect("stream collects cleanly");

        let reqs = captured.lock().unwrap();
        assert_eq!(reqs.len(), 1);
        let HttpBody::Json(body) = &reqs[0].body else {
            panic!("chat posts JSON");
        };
        // The offending union was rewritten before it hit the wire…
        assert_eq!(
            body["tools"][0]["function"]["parameters"]["properties"]["id"],
            json!({ "anyOf": [{ "pattern": "^u", "type": "string" }, { "type": "null" }] })
        );
        // …the conformant tool rides through verbatim, and nothing else moved (principle 7).
        assert_eq!(
            body["tools"][1]["function"]["parameters"],
            json!({ "type": "object", "properties": {} })
        );
        assert_eq!(body["model"], "kimi-k2-0711-preview");
        assert_eq!(body["messages"][0]["content"], "hi");
    }

    #[tokio::test]
    async fn batch_lines_are_flavored_too() {
        let captured = Arc::new(Mutex::new(Vec::new()));
        let transports = http_transports(capturing_transport(captured.clone(), |req| {
            if req.url.contains("/files") {
                FakeResponse::ok_json(&json!({ "id": "file-1" }))
            } else {
                FakeResponse::ok_json(&json!({ "id": "batch-1", "status": "validating" }))
            }
        }));
        let provider = MoonshotProvider::for_kind("moonshot", &transports).unwrap();

        let mut request = chat_with_tools(vec![ToolDef {
            name: "lookup".into(),
            description: None,
            parameters: json!({ "type": "string", "anyOf": [{ "const": "a" }] }),
        }]);
        request.model = "kimi-latest".into();
        let spec = BatchSpec {
            items: vec![llmleaf_model::BatchItem {
                custom_id: "item-1".into(),
                request,
            }],
        };
        provider
            .batch_create(spec, &cx())
            .await
            .expect("batch create succeeds");

        let reqs = captured.lock().unwrap();
        let HttpBody::Multipart(form) = &reqs[0].body else {
            panic!("batch uploads a multipart file first");
        };
        let jsonl = form
            .parts
            .iter()
            .find_map(|p| match p {
                MultipartPart::Bytes { data, .. } => Some(data.clone()),
                _ => None,
            })
            .expect("the upload carries the JSONL part");
        let line: Value = serde_json::from_slice(&jsonl).expect("one JSONL line");
        assert_eq!(
            line["body"]["tools"][0]["function"]["parameters"],
            json!({ "anyOf": [{ "enum": ["a"], "type": "string" }] })
        );
    }

    #[test]
    fn for_kind_covers_moonshot_kinds_only() {
        let t = Transports::fake();
        for kind in ["moonshot", "kimi", "kimi-k2"] {
            let p = MoonshotProvider::for_kind(kind, &t).expect(kind);
            assert_eq!(p.name(), "moonshot");
        }
        for kind in ["kimi-coding", "kimi-for-coding"] {
            let p = MoonshotProvider::for_kind(kind, &t).expect(kind);
            assert_eq!(p.name(), "kimi-coding");
        }
        // Every other kind falls through to the plain compat table (or a native provider).
        for kind in ["openai", "minimax", "anthropic", "nonsense"] {
            assert!(MoonshotProvider::for_kind(kind, &t).is_none());
        }
    }
}
