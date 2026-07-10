//! The self-describing API contract — the static OpenAPI 3.1 document for the consumer surface.
//!
//! `GET /v1/openapi.json` serves this. Unlike [`super::openai`] et al. this is not a per-request
//! dialect mapping: it is the *description* of those mappings — the fixed shape of every consumer
//! endpoint, with zero runtime or tenant data. It names no provider and reveals no topology (P2), so
//! it is public and transparent by construction (P7): a Swagger UI, Postman import, or
//! `openapi-generator` can fetch it with no credential, exactly as the OpenAPI convention expects.
//!
//! It is built once and cached: the document is independent of config and request, so there is nothing
//! to recompute per call. [`document`] hands back the cached UTF-8 JSON as a `&'static str`; the
//! handler writes it straight to the wire with no per-request allocation.
//!
//! Hand-authored, deliberately. Deriving it from macros would drag a heavy dependency and annotations
//! onto types the core keeps lean; the surface is small and stable enough that a reviewed literal —
//! which the e2e suite parses and checks path-for-path against the served consumer surface — is the
//! lighter, clearer choice. Add or remove a `/v1` endpoint and the test fails until this document
//! matches (the decision filter: dialect-mapping fidelity beats convenience).

use std::sync::LazyLock;

use serde_json::{json, Value};

/// The cached, serialized OpenAPI document. Built on first access from [`build`]; the contents never
/// change at runtime, so one allocation serves the process.
static DOCUMENT: LazyLock<String> =
    LazyLock::new(|| serde_json::to_string(&build()).expect("openapi document serializes"));

/// The OpenAPI 3.1 document for the consumer surface, as a `&'static str` of JSON. Cheap to call
/// repeatedly — it borrows the process-wide cache.
pub fn document() -> &'static str {
    DOCUMENT.as_str()
}

/// A `$ref` into `#/components/schemas/<name>`.
fn schema_ref(name: &str) -> Value {
    json!({ "$ref": format!("#/components/schemas/{name}") })
}

/// A JSON request body of the given component schema (the `application/json` content type).
fn json_body(schema: &str) -> Value {
    json!({
        "required": true,
        "content": { "application/json": { "schema": schema_ref(schema) } },
    })
}

/// A `200 OK` JSON response of the given component schema.
fn json_ok(description: &str, schema: &str) -> Value {
    json!({
        "description": description,
        "content": { "application/json": { "schema": schema_ref(schema) } },
    })
}

/// The standard error responses shared by every authenticated endpoint, keyed by status. `extra` is
/// merged in for endpoint-specific codes (e.g. a `404` for an unrouted model).
fn error_responses(extra: Value) -> Value {
    let mut responses = json!({
        "400": json_err("Malformed request (bad JSON, missing or invalid fields)"),
        "401": json_err("Missing or invalid bearer credential"),
        "403": json_err("Key is blocked, or the model is not allowed for this key"),
        "429": json_err("Key is suspended by a pulled limiter verdict"),
        "502": json_err("All upstream targets failed"),
    });
    if let (Some(base), Some(more)) = (responses.as_object_mut(), extra.as_object()) {
        for (k, v) in more {
            base.insert(k.clone(), v.clone());
        }
    }
    responses
}

/// A JSON error response of the canonical `{ "error": { "message": … } }` envelope.
fn json_err(description: &str) -> Value {
    json!({
        "description": description,
        "content": { "application/json": { "schema": schema_ref("Error") } },
    })
}

/// An error response in the Anthropic-shaped envelope (the `/v1/messages` surface speaks the Anthropic
/// dialect on errors too — `{ "type": "error", "error": { … } }`).
fn anthropic_err(description: &str) -> Value {
    json!({
        "description": description,
        "content": { "application/json": { "schema": schema_ref("AnthropicError") } },
    })
}

/// Build the whole document. Split into paths + components for readability; this runs once.
fn build() -> Value {
    json!({
        "openapi": "3.1.0",
        "info": {
            "title": "llmleaf",
            "description": "The llmleaf consumer surface — an OpenAI/OpenRouter-compatible proxy in \
                front of every model provider. This document describes only the fixed API contract; it \
                carries no provider identity, no routing topology, and no runtime state.",
            "version": env!("CARGO_PKG_VERSION"),
            "license": { "name": "MIT OR Apache-2.0" },
        },
        // Relative server: the spec is served by the same node it describes, so a client resolves
        // paths against whatever origin it fetched this from.
        "servers": [{ "url": "/", "description": "This node" }],
        // Bearer is the default for every operation; public endpoints override with `security: []`.
        "security": [{ "bearerAuth": [] }],
        "tags": [
            { "name": "chat", "description": "Chat completions" },
            { "name": "embeddings", "description": "Vector embeddings" },
            { "name": "rerank", "description": "Document reranking" },
            { "name": "audio", "description": "Text-to-speech, speech-to-text, and voices" },
            { "name": "models", "description": "Model discovery" },
            { "name": "batches", "description": "Asynchronous batch jobs" },
            { "name": "realtime", "description": "Realtime bidirectional sessions" },
            { "name": "ops", "description": "Liveness and self-description" },
        ],
        "paths": paths(),
        "components": components(),
    })
}

fn paths() -> Value {
    json!({
        "/healthz": {
            "get": {
                "tags": ["ops"],
                "operationId": "healthz",
                "summary": "Liveness probe",
                "description": "Always returns `ok`. Unauthenticated.",
                "security": [],
                "responses": {
                    "200": {
                        "description": "The node is alive",
                        "content": { "text/plain": { "schema": { "type": "string", "example": "ok" } } },
                    }
                },
            }
        },
        "/v1/openapi.json": {
            "get": {
                "tags": ["ops"],
                "operationId": "openapi",
                "summary": "This document",
                "description": "The OpenAPI 3.1 description of the consumer surface. Static and \
                    unauthenticated — it contains no runtime, tenant, or provider data.",
                "security": [],
                "responses": {
                    "200": {
                        "description": "The OpenAPI document",
                        "content": { "application/json": { "schema": { "type": "object" } } },
                    }
                },
            }
        },
        "/v1/chat/completions": {
            "post": {
                "tags": ["chat"],
                "operationId": "createChatCompletion",
                "summary": "Create a chat completion",
                "description": "OpenAI-compatible chat completions. With `stream: true` the response is \
                    a `text/event-stream` of `chat.completion.chunk` events terminated by `[DONE]`; \
                    otherwise a single `chat.completion` object (a collected stream — P4).",
                "requestBody": json_body("ChatCompletionRequest"),
                "responses": error_responses(json!({
                    "200": {
                        "description": "A chat completion, or an SSE stream when `stream` is true",
                        "content": {
                            "application/json": { "schema": schema_ref("ChatCompletionResponse") },
                            "text/event-stream": {
                                "schema": { "type": "string" },
                                "description": "Newline-delimited `data:` frames carrying \
                                    `ChatCompletionChunk` objects, ending with `data: [DONE]`.",
                            },
                        },
                    },
                    "404": json_err("No route for the requested model"),
                })),
            }
        },
        "/v1/messages": {
            "post": {
                "tags": ["messages"],
                "operationId": "createMessage",
                "summary": "Create a message (Anthropic-compatible)",
                "description": "Anthropic Messages API compatibility — a second chat dialect on the same \
                    canonical core (P3). With `stream: true` the response is a `text/event-stream` of \
                    named Anthropic events (`message_start`, `content_block_start`/`_delta`/`_stop`, \
                    `message_delta`, `message_stop`); otherwise a single `message` object (a collected \
                    stream — P4). Authenticated with the `x-api-key` header (an `Authorization: Bearer` \
                    token is also accepted); the credential is a consumer virtual key, as everywhere.",
                "security": [{ "apiKeyAuth": [] }, { "bearerAuth": [] }],
                "requestBody": json_body("AnthropicMessagesRequest"),
                "responses": {
                    "200": {
                        "description": "A message, or an SSE stream of message events when `stream` is true",
                        "content": {
                            "application/json": { "schema": schema_ref("AnthropicMessage") },
                            "text/event-stream": {
                                "schema": { "type": "string" },
                                "description": "Named SSE events (`event: message_start` … \
                                    `event: message_stop`) whose `data:` lines carry the Anthropic \
                                    streaming event objects.",
                            },
                        },
                    },
                    "400": anthropic_err("Invalid request (e.g. missing `max_tokens`)"),
                    "401": anthropic_err("Missing or invalid credential"),
                    "403": anthropic_err("Key is blocked, or the model is not allowed for this key"),
                    "404": anthropic_err("No route for the requested model"),
                    "429": anthropic_err("Key is suspended by a pulled limiter verdict"),
                    "502": anthropic_err("All upstream targets failed"),
                },
            }
        },
        "/v1/responses": {
            "post": {
                "tags": ["responses"],
                "operationId": "createResponse",
                "summary": "Create a model response (OpenAI Responses API)",
                "description": "OpenAI Responses API compatibility — a third chat dialect on the same \
                    canonical core (P3), served statelessly. llmleaf stores nothing (the core is not a \
                    database), so `store` is accepted but the response always reports `\"store\": \
                    false`, and `previous_response_id`, `background: true`, and `item_reference` inputs \
                    are rejected. With `stream: true` the response is a `text/event-stream` of named \
                    Responses events (`response.created`, `response.output_text.delta`, …, \
                    `response.completed`); otherwise a single `response` object (a collected stream — P4).",
                "requestBody": json_body("ResponsesRequest"),
                "responses": error_responses(json!({
                    "200": {
                        "description": "A response object, or an SSE stream of response events when `stream` is true",
                        "content": {
                            "application/json": { "schema": schema_ref("ResponseObject") },
                            "text/event-stream": {
                                "schema": { "type": "string" },
                                "description": "Named SSE events (`event: response.created` … \
                                    `event: response.completed`) whose `data:` lines carry the \
                                    Responses streaming event objects. No `[DONE]` sentinel.",
                            },
                        },
                    },
                    "404": json_err("No route for the requested model"),
                })),
            }
        },
        "/v1/responses/{id}": {
            "get": {
                "tags": ["responses"],
                "operationId": "getResponse",
                "summary": "Retrieve a stored response (always 404 — stateless)",
                "description": "Always returns 404: llmleaf is stateless and stores no responses \
                    (`store` is always false), so retrieval is unsupported by design. This is P7 \
                    transparency — a client that ignored `\"store\": false` is told exactly why.",
                "parameters": [{
                    "name": "id",
                    "in": "path",
                    "required": true,
                    "schema": { "type": "string" },
                    "description": "A response id. Never resolvable — nothing is stored.",
                }],
                "responses": {
                    "404": json_err("No stored response — llmleaf is stateless (`store` is always false)"),
                },
            }
        },
        "/v1/embeddings": {
            "post": {
                "tags": ["embeddings"],
                "operationId": "createEmbedding",
                "summary": "Create embeddings",
                "requestBody": json_body("EmbeddingRequest"),
                "responses": error_responses(json!({
                    "200": json_ok("The embedding vectors", "EmbeddingResponse"),
                    "404": json_err("No route for the requested model"),
                })),
            }
        },
        "/v1/rerank": {
            "post": {
                "tags": ["rerank"],
                "operationId": "createRerank",
                "summary": "Rerank documents by relevance to a query",
                "description": "The Cohere/Jina/OpenRouter rerank dialect: a query plus candidate \
                    documents in, a relevance-ordered `results` list out. Documents may be plain \
                    strings or structured `{ text?, image? }` objects (multimodal rerankers).",
                "requestBody": json_body("RerankRequest"),
                "responses": error_responses(json!({
                    "200": json_ok("The ranked results", "RerankResponse"),
                    "404": json_err("No route for the requested model"),
                })),
            }
        },
        "/v1/audio/speech": {
            "post": {
                "tags": ["audio"],
                "operationId": "createSpeech",
                "summary": "Generate speech (text-to-speech)",
                "description": "Returns the raw audio body (not SSE). The `Content-Type` reflects the \
                    requested `response_format` (e.g. `audio/mpeg`).",
                "requestBody": json_body("SpeechRequest"),
                "responses": error_responses(json!({
                    "200": {
                        "description": "The synthesized audio bytes",
                        "content": {
                            "application/octet-stream": {
                                "schema": { "type": "string", "format": "binary" }
                            }
                        },
                    },
                    "404": json_err("No route for the requested model"),
                })),
            }
        },
        "/v1/audio/voices": {
            "get": {
                "tags": ["audio"],
                "operationId": "listVoices",
                "summary": "List the voices a speech model can use",
                "parameters": [{
                    "name": "model",
                    "in": "query",
                    "required": true,
                    "schema": { "type": "string" },
                    "description": "The logical model whose voices to list; routed like a speech request.",
                }],
                "responses": error_responses(json!({
                    "200": json_ok("The voice catalog for the model", "VoicesResponse"),
                    "404": json_err("No route for the requested model"),
                })),
            }
        },
        "/v1/audio/transcriptions": {
            "post": {
                "tags": ["audio"],
                "operationId": "createTranscription",
                "summary": "Transcribe audio (speech-to-text)",
                "description": "Multipart upload. The response shape follows `response_format`: a JSON \
                    object (`json`/`verbose_json`) or `text/plain` (`text`).",
                "requestBody": {
                    "required": true,
                    "content": {
                        "multipart/form-data": { "schema": schema_ref("TranscriptionForm") }
                    },
                },
                "responses": error_responses(json!({
                    "200": {
                        "description": "The transcription",
                        "content": {
                            "application/json": { "schema": schema_ref("TranscriptionResponse") },
                            "text/plain": { "schema": { "type": "string" } },
                        },
                    },
                    "404": json_err("No route for the requested model"),
                })),
            }
        },
        "/v1/models": {
            "get": {
                "tags": ["models"],
                "operationId": "listModels",
                "summary": "List the models this node serves",
                "description": "OpenRouter-shaped (`{ \"data\": [ … ] }`). The public view carries only \
                    ids, capabilities, and pricing. Presenting a valid `x-admin-token` on top of the \
                    bearer adds a per-model `endpoints` chain with provider names and node-local health.",
                "parameters": [
                    {
                        "name": "type",
                        "in": "query",
                        "required": false,
                        "schema": { "type": "string", "enum": ["all", "llm", "tts", "stt", "embedding", "rerank"] },
                        "description": "Modality filter (llmleaf extension). An unknown value is a 400.",
                    },
                    {
                        "name": "search",
                        "in": "query",
                        "required": false,
                        "schema": { "type": "string" },
                        "description": "Case-insensitive substring over the model id (llmleaf extension).",
                    },
                    {
                        "name": "x-admin-token",
                        "in": "header",
                        "required": false,
                        "schema": { "type": "string" },
                        "description": "Optional. When it matches the configured admin token, the \
                            response gains the admin-only `endpoints` chain. A wrong/absent token \
                            silently yields the public view (never an error).",
                    },
                ],
                "responses": error_responses(json!({
                    "200": json_ok("The served-model catalog", "ModelList"),
                })),
            }
        },
        "/v1/batches": {
            "post": {
                "tags": ["batches"],
                "operationId": "createBatch",
                "summary": "Submit an inline batch",
                "requestBody": json_body("BatchCreateRequest"),
                "responses": error_responses(json!({
                    "200": json_ok("The batch handle", "Batch"),
                })),
            }
        },
        "/v1/batches/{id}": {
            "get": {
                "tags": ["batches"],
                "operationId": "retrieveBatch",
                "summary": "Poll a batch's status",
                "parameters": [batch_id_param()],
                "responses": error_responses(json!({
                    "200": json_ok("The batch handle", "Batch"),
                    "404": json_err("No such batch"),
                })),
            }
        },
        "/v1/batches/{id}/cancel": {
            "post": {
                "tags": ["batches"],
                "operationId": "cancelBatch",
                "summary": "Request cancellation of a batch",
                "parameters": [batch_id_param()],
                "responses": error_responses(json!({
                    "200": json_ok("The updated batch handle", "Batch"),
                    "404": json_err("No such batch"),
                })),
            }
        },
        "/v1/batches/{id}/results": {
            "get": {
                "tags": ["batches"],
                "operationId": "batchResults",
                "summary": "Stream a completed batch's results as JSONL",
                "parameters": [batch_id_param()],
                "responses": error_responses(json!({
                    "200": {
                        "description": "One JSON result object per line (`application/x-ndjson`)",
                        "content": { "application/x-ndjson": { "schema": { "type": "string" } } },
                    },
                    "404": json_err("No such batch"),
                })),
            }
        },
        "/v1/realtime": {
            "get": {
                "tags": ["realtime"],
                "operationId": "realtime",
                "summary": "Open a realtime session (WebSocket)",
                "description": "A WebSocket upgrade carrying the OpenAI Realtime dialect. Authenticate \
                    with `Authorization: Bearer` or the `openai-insecure-api-key.<token>` subprotocol, \
                    and select the model with the `?model=` query parameter. (OpenAPI does not model \
                    WebSocket framing; the request/response below describe only the upgrade handshake.)",
                "parameters": [{
                    "name": "model",
                    "in": "query",
                    "required": true,
                    "schema": { "type": "string" },
                    "description": "The logical model to route the session to.",
                }],
                "responses": {
                    "101": { "description": "Switching Protocols — the WebSocket session is established" },
                    "400": json_err("Missing `?model=`"),
                    "401": json_err("Missing or invalid realtime credential"),
                },
            }
        },
    })
}

/// The shared `{id}` path parameter of the batch sub-resources.
fn batch_id_param() -> Value {
    json!({
        "name": "id",
        "in": "path",
        "required": true,
        "schema": { "type": "string" },
        "description": "The opaque batch id returned by `POST /v1/batches`.",
    })
}

/// The `ResponsesRequest` component schema (factored out of [`components`] to keep that macro under the
/// recursion limit — see the call site).
fn responses_request_schema() -> Value {
    json!({
        "type": "object",
        "description": "OpenAI Responses API request, mapped into llmleaf's canonical request at the \
            edge (P3). Served statelessly: `store` is accepted but ignored (the response always reports \
            false); `previous_response_id`, `background: true`, and `item_reference` inputs are \
            rejected. Unlisted fields (`text`, `truncation`, `include`, `metadata`, \
            `parallel_tool_calls`, …) are accepted and ride through verbatim.",
        "properties": {
            "model": { "type": "string", "description": "The logical model id to route." },
            "input": {
                "description": "A string, or an ordered array of input items (messages, function \
                    calls, function-call outputs, reasoning).",
                "oneOf": [
                    { "type": "string" },
                    { "type": "array", "items": { "type": "object" } },
                ],
            },
            "instructions": { "type": "string", "description": "Hoisted to a leading system message." },
            "max_output_tokens": { "type": "integer" },
            "temperature": { "type": "number" },
            "top_p": { "type": "number" },
            "stream": { "type": "boolean", "default": false },
            "tools": {
                "type": "array",
                "items": { "type": "object" },
                "description": "Flat function tools; hosted tools are rejected.",
            },
            "tool_choice": {},
            "reasoning": { "type": "object" },
            "store": {
                "type": "boolean",
                "description": "Accepted but ignored; the response always reports false.",
            },
        },
        "required": ["model"],
        "additionalProperties": true,
    })
}

/// The `ResponseObject` component schema (factored out of [`components`], as above).
fn response_object_schema() -> Value {
    json!({
        "type": "object",
        "description": "An OpenAI Responses object. With `stream: true` the response is a \
            `text/event-stream` of Responses events instead of this object. `store` is always false \
            (llmleaf is stateless).",
        "properties": {
            "id": { "type": "string" },
            "object": { "type": "string", "const": "response" },
            "created_at": { "type": "integer" },
            "status": { "type": "string", "enum": ["completed", "incomplete", "failed"] },
            "model": { "type": "string" },
            "output": {
                "type": "array",
                "items": { "type": "object" },
                "description": "Output items: `reasoning`, `message`, `function_call`.",
            },
            "store": { "type": "boolean", "const": false },
            "reasoning": { "type": "object" },
            "usage": {
                "type": "object",
                "properties": {
                    "input_tokens": { "type": "integer" },
                    "output_tokens": { "type": "integer" },
                    "total_tokens": { "type": "integer" },
                },
            },
        },
    })
}

fn components() -> Value {
    // The Responses schemas are inserted *after* the base document is built rather than inlined here:
    // this `json!` is already near the macro recursion limit, and adding keys inline tips it over
    // (each object key deepens the expansion). Merging post-construction — like `error_responses` does
    // for its extra status codes — keeps the macro at its original size.
    let mut components = json!({
        "securitySchemes": {
            "bearerAuth": {
                "type": "http",
                "scheme": "bearer",
                "description": "A consumer virtual key. Static keys are presented as \
                    `base64(id:password)`; when the `[oauth]` surface is configured, a JWT bearer is \
                    accepted in its place.",
            },
            "apiKeyAuth": {
                "type": "apiKey",
                "in": "header",
                "name": "x-api-key",
                "description": "The same consumer virtual key as `bearerAuth`, presented as the \
                    Anthropic `x-api-key` header. Accepted by the `/v1/messages` surface.",
            },
        },
        "schemas": {
            "Error": {
                "type": "object",
                "description": "The canonical error envelope returned by every endpoint.",
                "properties": {
                    "error": {
                        "type": "object",
                        "properties": { "message": { "type": "string" } },
                        "required": ["message"],
                    }
                },
                "required": ["error"],
            },
            "Usage": {
                "type": "object",
                "properties": {
                    "prompt_tokens": { "type": "integer" },
                    "completion_tokens": { "type": "integer" },
                    "total_tokens": { "type": "integer" },
                    "cost_usd": { "type": ["number", "null"], "description": "llmleaf cost enrichment." },
                },
            },
            "ChatMessage": {
                "type": "object",
                "description": "A single conversation message. `content` may be a string or an array of \
                    content parts (text/image), per the OpenAI dialect.",
                "properties": {
                    "role": { "type": "string", "enum": ["system", "user", "assistant", "tool"] },
                    "content": {},
                    "name": { "type": "string" },
                    "tool_calls": { "type": "array", "items": { "type": "object" } },
                    "tool_call_id": { "type": "string" },
                    "reasoning": {
                        "type": "string",
                        "description": "Open reasoning text (OpenRouter-style extension). Emitted on \
                            responses when the model produced thinking; omitted otherwise.",
                    },
                    "reasoning_details": {
                        "type": "array",
                        "items": schema_ref("ReasoningDetail"),
                        "description": "Structured reasoning blocks (OpenRouter-style extension), \
                            emitted alongside `reasoning` when the thinking is signed or encrypted.",
                    },
                },
                "required": ["role"],
                "additionalProperties": true,
            },
            "ReasoningDetail": {
                "type": "object",
                "description": "One OpenRouter-style `reasoning_details[]` entry: `reasoning.text` \
                    carries open thinking text and, when the upstream signs it, a `signature`; \
                    `reasoning.encrypted` carries an opaque `data` block, replayed verbatim.",
                "properties": {
                    "type": { "type": "string", "enum": ["reasoning.text", "reasoning.encrypted"] },
                    "text": { "type": "string" },
                    "signature": { "type": "string" },
                    "data": { "type": "string" },
                },
                "required": ["type"],
            },
            "ChatCompletionRequest": {
                "type": "object",
                "description": "OpenAI-compatible chat request. Unlisted OpenAI fields are accepted and \
                    mapped where supported.",
                "properties": {
                    "model": { "type": "string", "description": "The logical model id to route." },
                    "messages": { "type": "array", "items": schema_ref("ChatMessage") },
                    "stream": { "type": "boolean", "default": false },
                    "temperature": { "type": "number" },
                    "top_p": { "type": "number" },
                    "max_tokens": { "type": "integer" },
                    "max_completion_tokens": { "type": "integer" },
                    "stop": {},
                    "n": { "type": "integer" },
                    "seed": { "type": "integer" },
                    "frequency_penalty": { "type": "number" },
                    "presence_penalty": { "type": "number" },
                    "tools": { "type": "array", "items": { "type": "object" } },
                    "tool_choice": {},
                    "response_format": { "type": "object" },
                    "reasoning_effort": { "type": "string" },
                },
                "required": ["model", "messages"],
                "additionalProperties": true,
            },
            "ChatCompletionResponse": {
                "type": "object",
                "properties": {
                    "id": { "type": "string" },
                    "object": { "type": "string", "const": "chat.completion" },
                    "created": { "type": "integer" },
                    "model": { "type": "string" },
                    "choices": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "index": { "type": "integer" },
                                "message": schema_ref("ChatMessage"),
                                "finish_reason": { "type": ["string", "null"] },
                            },
                        },
                    },
                    "usage": schema_ref("Usage"),
                },
            },
            "ChatCompletionChunk": {
                "type": "object",
                "description": "One streamed frame (`stream: true`). Delivered as an SSE `data:` line.",
                "properties": {
                    "id": { "type": "string" },
                    "object": { "type": "string", "const": "chat.completion.chunk" },
                    "created": { "type": "integer" },
                    "model": { "type": "string" },
                    "choices": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "index": { "type": "integer" },
                                "delta": {
                                    "type": "object",
                                    "description": "The incremental change: `role`/`content` and \
                                        `tool_calls` per the OpenAI dialect, plus the OpenRouter-style \
                                        reasoning extensions — `reasoning` (open thinking text delta) \
                                        and `reasoning_details` (signed/encrypted blocks, see \
                                        ReasoningDetail).",
                                    "properties": {
                                        "role": { "type": "string" },
                                        "content": { "type": ["string", "null"] },
                                        "tool_calls": { "type": "array", "items": { "type": "object" } },
                                        "reasoning": { "type": "string" },
                                        "reasoning_details": {
                                            "type": "array",
                                            "items": schema_ref("ReasoningDetail"),
                                        },
                                    },
                                    "additionalProperties": true,
                                },
                                "finish_reason": { "type": ["string", "null"] },
                            },
                        },
                    },
                },
            },
            "AnthropicMessagesRequest": {
                "type": "object",
                "description": "Anthropic Messages API request. Mapped into llmleaf's canonical request \
                    at the edge (P3). Unlisted fields (`top_k`, `metadata`, `thinking`, …) are accepted \
                    and preserved verbatim.",
                "properties": {
                    "model": { "type": "string", "description": "The logical model id to route." },
                    "max_tokens": {
                        "type": "integer",
                        "description": "Required, exactly as in the Anthropic API.",
                    },
                    "messages": {
                        "type": "array",
                        "items": { "type": "object" },
                        "description": "Anthropic messages: `role` (`user`/`assistant`) and `content` \
                            (a string or an array of typed blocks).",
                    },
                    "system": {
                        "description": "A string or an array of text blocks; hoisted to a system message.",
                        "oneOf": [
                            { "type": "string" },
                            { "type": "array", "items": { "type": "object" } },
                        ],
                    },
                    "stream": { "type": "boolean", "default": false },
                    "temperature": { "type": "number" },
                    "top_p": { "type": "number" },
                    "top_k": { "type": "integer" },
                    "stop_sequences": { "type": "array", "items": { "type": "string" } },
                    "tools": { "type": "array", "items": { "type": "object" } },
                    "tool_choice": { "type": "object" },
                    "thinking": { "type": "object" },
                },
                "required": ["model", "max_tokens", "messages"],
                "additionalProperties": true,
            },
            "AnthropicMessage": {
                "type": "object",
                "description": "An Anthropic Messages response. With `stream: true` the response is a \
                    `text/event-stream` of Anthropic message events instead of this object.",
                "properties": {
                    "id": { "type": "string" },
                    "type": { "type": "string", "const": "message" },
                    "role": { "type": "string", "const": "assistant" },
                    "model": { "type": "string" },
                    "content": {
                        "type": "array",
                        "items": { "type": "object" },
                        "description": "Content blocks: `text` and `tool_use`.",
                    },
                    "stop_reason": { "type": ["string", "null"] },
                    "stop_sequence": { "type": ["string", "null"] },
                    "usage": {
                        "type": "object",
                        "properties": {
                            "input_tokens": { "type": "integer" },
                            "output_tokens": { "type": "integer" },
                            "cost_usd": { "type": ["number", "null"], "description": "llmleaf cost enrichment." },
                        },
                    },
                },
            },
            "AnthropicError": {
                "type": "object",
                "description": "The Anthropic-shaped error envelope returned by the `/v1/messages` surface.",
                "properties": {
                    "type": { "type": "string", "const": "error" },
                    "error": {
                        "type": "object",
                        "properties": {
                            "type": { "type": "string" },
                            "message": { "type": "string" },
                        },
                        "required": ["type", "message"],
                    },
                },
                "required": ["type", "error"],
            },
            "EmbeddingRequest": {
                "type": "object",
                "properties": {
                    "model": { "type": "string" },
                    "input": {
                        "description": "A string or an array of strings to embed.",
                        "oneOf": [
                            { "type": "string" },
                            { "type": "array", "items": { "type": "string" } },
                        ],
                    },
                    "encoding_format": { "type": "string", "enum": ["float", "base64"] },
                    "dimensions": { "type": "integer" },
                },
                "required": ["model", "input"],
                "additionalProperties": true,
            },
            "EmbeddingResponse": {
                "type": "object",
                "properties": {
                    "object": { "type": "string", "const": "list" },
                    "model": { "type": "string" },
                    "data": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "object": { "type": "string", "const": "embedding" },
                                "index": { "type": "integer" },
                                "embedding": {
                                    "description": "A float array, or a base64 string when \
                                        `encoding_format` is `base64`.",
                                    "oneOf": [
                                        { "type": "array", "items": { "type": "number" } },
                                        { "type": "string" },
                                    ],
                                },
                            },
                        },
                    },
                    "usage": schema_ref("Usage"),
                },
            },
            "RerankRequest": {
                "type": "object",
                "properties": {
                    "model": { "type": "string" },
                    "query": { "type": "string", "description": "The query documents are ranked against." },
                    "documents": {
                        "type": "array",
                        "description": "Candidate documents: plain strings, or `{ text?, image? }` \
                            objects for multimodal rerankers.",
                        "items": {
                            "oneOf": [
                                { "type": "string" },
                                { "type": "object", "additionalProperties": true },
                            ],
                        },
                    },
                    "top_n": { "type": "integer", "description": "Return only the top N results." },
                    "return_documents": {
                        "type": "boolean",
                        "description": "Echo each ranked document back in the results.",
                    },
                },
                "required": ["model", "query", "documents"],
                "additionalProperties": true,
            },
            "RerankResponse": {
                "type": "object",
                "properties": {
                    "object": { "type": "string", "const": "list" },
                    "model": { "type": "string" },
                    "results": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "index": {
                                    "type": "integer",
                                    "description": "Position of the document in the request list.",
                                },
                                "relevance_score": {
                                    "type": "number",
                                    "description": "Relevance score; higher is more relevant.",
                                },
                                "document": {
                                    "description": "The ranked document, present when \
                                        `return_documents` was set (string or object).",
                                },
                            },
                            "required": ["index", "relevance_score"],
                        },
                    },
                    "usage": schema_ref("Usage"),
                },
            },
            "SpeechRequest": {
                "type": "object",
                "properties": {
                    "model": { "type": "string" },
                    "input": { "type": "string", "description": "The text to synthesize." },
                    "voice": { "type": "string" },
                    "response_format": {
                        "type": "string",
                        "description": "e.g. mp3, opus, aac, flac, wav, pcm.",
                    },
                    "speed": { "type": "number" },
                },
                "required": ["model", "input"],
                "additionalProperties": true,
            },
            "TranscriptionForm": {
                "type": "object",
                "properties": {
                    "file": { "type": "string", "format": "binary", "description": "The audio file." },
                    "model": { "type": "string" },
                    "language": { "type": "string" },
                    "prompt": { "type": "string" },
                    "response_format": {
                        "type": "string",
                        "enum": ["json", "text", "verbose_json"],
                    },
                    "temperature": { "type": "number" },
                },
                "required": ["file", "model"],
            },
            "TranscriptionResponse": {
                "type": "object",
                "properties": {
                    "text": { "type": "string" },
                    "language": { "type": ["string", "null"] },
                    "duration": { "type": ["number", "null"] },
                },
                "required": ["text"],
            },
            "VoicesResponse": {
                "type": "object",
                "properties": {
                    "model": { "type": "string" },
                    "voices": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": { "type": "string" },
                                "name": { "type": "string" },
                                "languages": { "type": "array", "items": { "type": "string" } },
                            },
                            "required": ["id"],
                        },
                    },
                },
            },
            "ModelList": {
                "type": "object",
                "description": "OpenRouter-shaped catalog of the models this node serves.",
                "properties": {
                    "data": { "type": "array", "items": schema_ref("Model") },
                },
                "required": ["data"],
            },
            "Model": {
                "type": "object",
                "description": "One served model. `endpoints` is present only in the admin view.",
                "properties": {
                    "id": { "type": "string" },
                    "canonical_slug": { "type": "string" },
                    "name": { "type": "string" },
                    "created": { "type": "integer" },
                    "description": { "type": "string" },
                    "context_length": { "type": ["integer", "null"] },
                    "architecture": { "type": "object" },
                    "pricing": { "type": ["object", "null"] },
                    "top_provider": { "type": "object" },
                    "supported_parameters": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Canonical parameters this model accepts. A provider-reported positive list (e.g. OpenRouter) passes through verbatim; otherwise computed from the modality baseline minus `unsupported_parameters`.",
                    },
                    "unsupported_parameters": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Canonical parameters this model REJECTS (e.g. a reasoning model's `temperature`). Present only when known; the exact complement of `supported_parameters` within the modality baseline.",
                    },
                    "default_parameters": {
                        "type": "object",
                        "description": "Recommended/pinned default values for sampling parameters (e.g. `{\"temperature\": 0.3}`). Present only when known.",
                    },
                    "endpoints": {
                        "type": "array",
                        "description": "Admin-only: the provider/fallback chain with node-local health.",
                        "items": { "type": "object" },
                    },
                },
                "required": ["id"],
                "additionalProperties": true,
            },
            "BatchCreateRequest": {
                "type": "object",
                "description": "An inline batch of OpenAI-shaped requests.",
                "additionalProperties": true,
            },
            "Batch": {
                "type": "object",
                "description": "A batch handle (OpenAI-shaped): an opaque id plus status.",
                "properties": {
                    "id": { "type": "string" },
                    "object": { "type": "string", "const": "batch" },
                    "status": { "type": "string" },
                },
                "additionalProperties": true,
            },
        },
    });
    if let Some(schemas) = components["schemas"].as_object_mut() {
        schemas.insert("ResponsesRequest".into(), responses_request_schema());
        schemas.insert("ResponseObject".into(), response_object_schema());
    }
    components
}
