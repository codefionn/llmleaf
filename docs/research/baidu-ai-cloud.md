# Baidu AI Cloud (Qianfan) provider research

Research date: 2026-08-10. Sources are Baidu's official documentation only.

## Recommendation

Implement Baidu Qianfan's model inference service as an OpenAI-compatible provider whose default base URL is `https://qianfan.baidubce.com/v2` and whose credentials use `Authorization: Bearer <API Key>`. Sensible config kinds are `baidu`, `qianfan`, and `baidu-qianfan`, with one canonical brand name such as `baidu-qianfan`.

The initial provider can safely reuse llmleaf's Chat Completions, embeddings, rerank, and model-list wire mappings. Responses and batch need Baidu-specific handling described below; they should not be enabled by merely selecting the stock OpenAI flavors.

## Authentication and endpoint

- Baidu documents the OpenAI-compatible base URL as `https://qianfan.baidubce.com/v2`, and its examples initialize the OpenAI SDK with that base URL. ([Development libraries](https://cloud.baidu.com/doc/qianfan-docs/s/1m9l6eex1), [OpenAI-compatible chat example](https://cloud.baidu.com/doc/qianfan-docs/s/Fm9l6ocai))
- Model inference uses a Qianfan API Key as a bearer token: `Authorization: Bearer <API Key>`. The key is created for Qianfan ModelBuilder and can be scoped to all or selected V2 application resources. ([Authentication](https://cloud.baidu.com/doc/qianfan-api/s/ym9chdsy5))
- This is the API Key flow, not the older OAuth access-token URL and not BCE AK/SK request signing used by some Qianfan control-plane operations. The provider only needs bearer auth for the inference endpoints in scope.

## Chat Completions compatibility

`POST https://qianfan.baidubce.com/v2/chat/completions` is OpenAI-shaped and is the safest default chat API. It supports collected and SSE streaming responses, terminating streams with `data: [DONE]`; `stream_options.include_usage=true` adds final usage. Requests and responses use OpenAI-style messages, tools, tool calls, `tool_choice`, finish reasons, and usage. ([Text generation API](https://cloud.baidu.com/doc/qianfan-api/s/3m7of64lb))

Actionable mapping facts:

- Bearer auth, standard `<base>/chat/completions` URL, real SSE streaming, and `stream_options.include_usage` all match the current `OpenAiCompatProvider` path.
- Tool calls and parallel tool calls use the OpenAI shapes. The stock tool request/response mapping is appropriate. ([Text generation API](https://cloud.baidu.com/doc/qianfan-api/s/3m7of64lb))
- Vision uses OpenAI content parts: `{"type":"image_url","image_url":{"url":...,"detail":...}}`; URLs and `data:image/...;base64,...` are accepted. This matches llmleaf's current image-content mapping. Qianfan also has a `video_url` part, which llmleaf's canonical chat model does not currently represent. ([Vision API](https://cloud.baidu.com/doc/qianfan-api/s/rm7u7qdiq))
- Reasoning text is returned as `message.reasoning_content`, which the shared OpenAI parser already recognizes. Qianfan documents `reasoning_effort` values `high` and `max`; older `low`/`medium` map to `high`, and `xhigh` maps to `max`. ([Text generation API](https://cloud.baidu.com/doc/qianfan-api/s/3m7of64lb))
- Both `max_tokens` and `max_completion_tokens` are accepted, but their meanings differ: `max_tokens` limits final answer content, while `max_completion_tokens` limits reasoning plus final answer and wins when both are present. A conservative first-class Chat Completions row should use `max_tokens`, matching the canonical field's answer-output meaning; callers can pass Qianfan-specific controls in `extra`. ([Text generation API](https://cloud.baidu.com/doc/qianfan-api/s/3m7of64lb))
- Qianfan-specific fields such as `enable_thinking`, `thinking_budget`, `thinking_strategy`, `web_search`, and `chunk_include_usage` can ride through `ChatRequest.extra`; no core model changes are required for basic support. ([Text generation API](https://cloud.baidu.com/doc/qianfan-api/s/3m7of64lb))
- Error bodies use ordinary HTTP status codes and OpenAI-like `type`/`code`/message information; documented examples include 400 validation errors, 429 rate limits, and 500 timeouts. The shared HTTP error classification is suitable. ([Error codes](https://cloud.baidu.com/doc/qianfan/s/Qmh4su56e))

## Embeddings and rerank

- `POST <base>/embeddings` accepts `model`, a string or string array `input`, optional `user`, and `encoding_format` (currently only `float`). The response has the standard `data[].embedding`, `data[].index`, and `usage` fields, so llmleaf's shared embedding mapping applies. Text models generally accept at most 16 input strings, with model-specific token/character limits. Qianfan also supports a provider-specific multimodal input object, which llmleaf's current canonical text embedding request cannot express. ([Embeddings API](https://cloud.baidu.com/doc/qianfan-api/s/Fm7u3ropn))
- `POST <base>/rerank` accepts `model`, `query`, string `documents`, optional `top_n`, and optional `user`; it returns `results[].{document,relevance_score,index}`. That is the shared Cohere/Jina-style rerank shape already parsed by `openai_to_rerank`, so the brand should set `rerank_api = true`. Baidu notes that this endpoint is not callable through the OpenAI SDK only because OpenAI itself has no rerank SDK method. ([Rerank API](https://cloud.baidu.com/doc/qianfan-api/s/2m7u4zt74))

## Live model catalog: enable it, but normalize it

`GET <base>/models` is authenticated and returns an OpenAI list envelope (`object: "list"`, `data: [...]`). Entries contain an id, owner, type, context length, architecture input/output modalities, output limits, and pricing. This should make `models_api = true`. ([Get models](https://cloud.baidu.com/doc/qianfan-api/s/Dmba8k71y))

Do not pass this response through the current tolerant parser without a Baidu edge normalization:

1. Qianfan spells the total reasoning-plus-answer cap `max_completions_tokens` (plural `completions`), while the shared parser recognizes `max_completion_tokens`. Qianfan separately publishes `max_tokens` as the final-answer cap. For canonical `ModelInfo.max_output`, prefer the reported `max_tokens`; preserve `max_completions_tokens` and `prompt_tokens` in `extra`.
2. Qianfan's `pricing.prompt` and `pricing.completion` strings are **Chinese yuan per 1,000 tokens**, including tiered arrays for some models. `ModelInfo` requires USD per 1,000,000 tokens, while the shared parser assumes `prompt`/`completion` strings are USD per token and multiplies them by 1,000,000. Reusing that logic would be wrong by units, scale, and currency. Preserve Qianfan's raw `pricing` object in `extra`, but leave `input_per_mtok` and `output_per_mtok` unset unless a separately sourced, explicitly converted USD price is available. Do not bake a fluctuating exchange rate into the provider hot path. ([Get models](https://cloud.baidu.com/doc/qianfan-api/s/Dmba8k71y))
3. Catalog `type` values include `chat`, `multimodal`, `embeddings`, `rerank`, and image/video types. The existing architecture fallback correctly identifies text-output multimodal models as LLMs, and direct `embeddings`/`rerank` values are already recognized.

The clean implementation is a thin `BaiduProvider` wrapper around `OpenAiCompatProvider`, analogous to the existing provider-edge wrappers: delegate requests, but post-process `models()` to correct limits and suppress misinterpreted prices. A table-only row is adequate for requests but would advertise incorrect live pricing.

## Responses API is related but not the stock llmleaf flavor

Qianfan has `POST <base>/responses`, supports streaming, tools, plaintext reasoning items, and OpenAI SDK calls, but only for a documented subset of text models. Baidu describes it as its newer agent-oriented API. ([Responses guide](https://cloud.baidu.com/doc/qianfan-docs/s/4mi400l1m), [Create response](https://cloud.baidu.com/doc/qianfan-api/s/vmhejnuy8), [Streaming responses](https://cloud.baidu.com/doc/qianfan-api/s/9mhsu5bhi))

Keep Chat Completions as the provider default. Do not treat the endpoint as `ResponsesFlavor::OpenAi` without tests and a dedicated flavor:

- Qianfan reasoning items contain plaintext `content[].{type:"reasoning_text",text}`, closer to the open-reasoning variants than stock OpenAI encrypted reasoning.
- Its documented request uses `thinking: {type: "enabled"|"disabled"}`, not the stock `reasoning.effort` mapping.
- `store` defaults to `true` and `store:false` is documented, so llmleaf's stateless `store:false` policy is valid. However, Baidu does not document `include: ["reasoning.encrypted_content"]`; the stock flavor would inject that unsupported/undocumented field.
- The official request reference does not document `max_output_tokens` even though the rest of the wire is OpenAI-like.

A future `ResponsesFlavor::Baidu` should omit the encrypted-reasoning `include`, map canonical thinking to Qianfan's `thinking.type`, replay plaintext reasoning only if the API accepts it as input, and be tested against the published event names.

## Batch requires a Baidu flavor

Qianfan implements the OpenAI file/batch workflow at `<base>/files` and `<base>/batches`: upload multipart JSONL with `purpose=batch`, create a job, retrieve it, cancel it at `/batches/{id}/cancel`, and download `/files/{file_id}/content`. ([Upload file](https://cloud.baidu.com/doc/qianfan-api/s/9mi6szj38), [Create batch](https://cloud.baidu.com/doc/qianfan-api/s/Vmimtsj8f), [Retrieve batch](https://cloud.baidu.com/doc/qianfan-api/s/ymir8ggss), [Cancel batch](https://cloud.baidu.com/doc/qianfan-api/s/emir8fods), [Download file](https://cloud.baidu.com/doc/qianfan-api/s/Bmi6vu310))

It cannot use llmleaf's stock `BatchFlavor::OpenAi` unchanged:

- Each JSONL line's `url` and the create request's `endpoint` must be `/v2/chat/completions`; llmleaf's stock flavor hardcodes `/v1/chat/completions`.
- One batch may contain at most 5,000 requests and all requests must use one model. The provider should reject mixed-model `BatchSpec`s rather than submit a job that fails asynchronously.
- `completion_window` supports 1–672 hours, so llmleaf's existing `24h` value is valid.
- Status strings include `Running`, `Done`, `Stopping`/`Stopped`, `Failed`, and `Expired`. Map `Done` to completed, `Stopping` to canceling, and `Stopped` to canceled; the current shared status mapper does not recognize `Done` or `Stopped`.
- Counts and output/error file ids are otherwise close to the OpenAI shape, and result JSONL can reuse the OpenAI chat result mapping. ([Batch guide](https://cloud.baidu.com/doc/qianfan-docs/s/rm8r4tl9u))

Add a `BatchFlavor::Baidu` only if batch support is included in this change. Otherwise leave batch explicitly unsupported; setting the stock OpenAI flavor would send invalid per-line paths.

## Explicitly unsupported through the generic OpenAI wire

Qianfan exposes speech recognition and synthesis, but its documented short TTS endpoint is `https://tsn.baidu.com/text2audio` with URL-encoded form fields, and speech recognition uses Qianfan/Baidu-specific endpoints and request bodies. These are not OpenAI `/audio/speech` or multipart `/audio/transcriptions` dialects. A first-class Baidu wrapper should return `Unsupported` for llmleaf TTS/STT rather than delegate to nonexistent OpenAI-style paths. ([Qianfan quick start](https://cloud.baidu.com/doc/qianfan-docs/s/qm8qxemze))

Native realtime WebSocket chat and an OpenAI-style voice catalog are not documented for this inference base, so leave `realtime_native` and `voices_api` false.

## Suggested first implementation scope

1. Add `baidu`/`qianfan`/`baidu-qianfan` kinds and the Qianfan base URL/bearer auth row.
2. Wrap it with a small Baidu provider that delegates chat, embeddings, and rerank; returns unsupported for TTS/STT; and normalizes the live catalog without converting CNY prices to USD.
3. Keep Chat Completions as the default; do not advertise the stock Responses flavor.
4. Either add the dedicated Baidu batch flavor above, or deliberately leave batch unsupported.
5. Document an example provider block using `credential = "env:QIANFAN_API_KEY"` and a catalog prefix such as `qianfan`.
