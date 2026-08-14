# Provider pricing-catalog gaps

Research date: 2026-08-14.  This is a recommendation for the offline
`llmleaf-pricing-collector`, not a change to runtime routing.

## What the current schema can represent

`ModelPricing` is keyed only by upstream model id and contains three token
rates: `input_per_mtok`, `cached_input_per_mtok`, and `output_per_mtok`.
`Pricing::cost_usd` can therefore exactly price only:

```
(prompt - cache_read) * input + cache_read * cached_input + completion * output
```

It cannot express a request fee, tool invocation/search fee, cache-storage
hour, cache-write price, an input-length threshold, a service-tier multiplier,
or non-token billing.  A static row must be added only when that formula is the
whole charge for the supported direct API path.  The global model-id key is
also a constraint: never put a provider-specific price under an id that could
be used at a different provider for a different price.

The collector currently has dedicated pricing-page parsers only for OpenAI,
Anthropic, Cohere, Mistral, and Moonshot; its generic list collector accepts a
listing only when it itself contains token prices.  The current 96-row bundled
dataset has no DeepSeek, MiniMax, xAI, Perplexity, Groq, or Gemini rows.

## Recommended static documented catalogs

### 1. DeepSeek — add now

DeepSeek is a particularly good fit.  Its documented `GET /models` response
contains only `id`, `object`, and `owned_by`; pricing is intentionally on the
separate official Models & Pricing page.  The current list example contains
only `deepseek-v4-flash` and `deepseek-v4-pro`.
[DeepSeek model-list reference](https://api-docs.deepseek.com/api/list-models/)

Both models use the schema's three rates exactly.  The OpenAI-wire mapper also
maps DeepSeek's `usage.prompt_cache_hit_tokens` to canonical
`cache_read_tokens`, so cache-hit billing is observable.  Add only these live
V4 IDs; `deepseek-chat` and `deepseek-reasoner` were retired on 2026-07-24.
[DeepSeek Models & Pricing](https://api-docs.deepseek.com/quick_start/pricing)
[DeepSeek V4 release and retirement notice](https://api-docs.deepseek.com/news/news260424/)

| ID | Context | Max output | Input/cache miss | Cache hit | Output |
| --- | ---: | ---: | ---: | ---: | ---: |
| `deepseek-v4-flash` | 1,000,000 | 384,000 | $0.1400 | $0.0028 | $0.2800 |
| `deepseek-v4-pro` | 1,000,000 | 384,000 | $0.4350 | $0.003625 | $0.8700 |

All rates are USD / 1M tokens.  Both models support thinking and
non-thinking modes, but that does not alter the published token price.  Do not
add web-search-assisted integrations as separately priced products: their
extra work results in additional model-token requests, rather than a distinct
per-request charge in the published direct model table.

### 2. MiniMax — add the fixed-rate M2 rows now

The provider intentionally stays non-enumerable: its `GET /models` path is
documented in the codebase as 404.  Its official pay-as-you-go page is the
authoritative catalog and price table.  The MiniMax text-model documentation
publishes a 204,800-token *combined* context window for every M2 row below;
it does **not** publish a model-specific independent maximum output for these
rows, so leave `max_output` absent rather than inventing one.
[MiniMax text models](https://platform.minimax.io/docs/guides/text-generation)
[MiniMax pay-as-you-go pricing](https://platform.minimax.io/docs/guides/pricing-paygo)

| ID | Context | Max output | Input | Cache read | Output |
| --- | ---: | ---: | ---: | ---: | ---: |
| `MiniMax-M2.7` | 204,800 total | not published separately | $0.30 | $0.06 | $1.20 |
| `MiniMax-M2.7-highspeed` | 204,800 total | not published separately | $0.60 | $0.06 | $2.40 |
| `MiniMax-M2.5` | 204,800 total | not published separately | $0.30 | $0.03 | $1.20 |
| `MiniMax-M2.5-highspeed` | 204,800 total | not published separately | $0.60 | $0.03 | $2.40 |
| `MiniMax-M2.1` | 204,800 total | not published separately | $0.30 | $0.03 | $1.20 |
| `MiniMax-M2.1-highspeed` | 204,800 total | not published separately | $0.60 | $0.03 | $2.40 |
| `MiniMax-M2` | 204,800 total | not published separately | $0.30 | $0.03 | $1.20 |
| `M2-her` | 64K | 2,048 | $0.30 | not supported | $1.20 |

All rates are USD / 1M tokens.  These are exact for ordinary standard,
pay-as-you-go text requests whose usage reports cache reads.  MiniMax's
Anthropic-compatible *explicit* cache operation additionally charges $0.375 /M
cache-write tokens, a field absent from the schema; do not claim exact total
cost for that alternate API.  It is not the llmleaf MiniMax OpenAI-compatible
path.  The official caching documentation describes the separate read/write
prices and usage counters.
[MiniMax explicit prompt caching](https://platform.minimax.io/docs/api-reference/anthropic-api-compatible-cache)

The dedicated pay-as-you-go table is the source used for these rows: unlike
the Anthropic-cache guide, it prices the two highspeed input rows at $0.60/M.
That discrepancy is an additional reason not to treat the alternate Anthropic
cache surface as covered by this static catalog.

`M2-her` is safe and its initial omission was an error.  MiniMax documents it
on the same OpenAI-compatible `https://api.minimax.io/v1` Chat Completions
surface, with `model="M2-her"`, a 64K context, and
`max_completion_tokens` up to 2,048.  Its pricing table explicitly shows no
prompt-cache read or write price, so leave `cached_input_per_mtok` absent;
there is no cache counter/rate to model.
[MiniMax M2-her Chat Completions guide](https://platform.minimax.io/docs/guides/text-chat)

Exclude `MiniMax-M3`: it charges one set of rates at <=512K input tokens and a
second, doubled set above that threshold.  Exclude the priority service tier
(1.5x) and all MiniMax audio, video, image, music, and voice products: they
are billed by characters, seconds, images, songs, or a multiplier rather than
the three supported token counters.  The same official page documents both
the M3 threshold and these non-token units.
[MiniMax pay-as-you-go pricing](https://platform.minimax.io/docs/guides/pricing-paygo)

## Considered, but do not add as a static token catalog

### xAI / Grok — improve the live collector instead

xAI is no longer a provider lacking a rich API.  `GET /v1/models` has model
ids and prices, while `GET /v1/language-models` gives full language-model
metadata: modalities, normal and cached text-token prices, normal and cached
long-context prices, threshold, and search price.
[xAI Models API reference](https://docs.x.ai/developers/rest-api-reference/inference/models)

The present schema cannot select the long-context price once the prompt reaches
200K tokens, nor add search/tool fees.  For example, `grok-4.6` is $2/$0.50/$6
(input/cache/output) below 200K, and $4/$1/$12 at or above it.  xAI also has
per-tool fees, priority's 2x multiplier, and model-specific batch discounts.
[xAI pricing](https://docs.x.ai/developers/pricing)

Recommendation: make `xai` a priced live-list collector only after extending
the data/cost schema to model thresholds, modalities, tools, and service tier.
Do not freeze a static standard-rate table: it would silently understate valid
long-context and tool-enabled calls.

### Perplexity — exclude

No three-rate row is exact for the supported Sonar API.  Sonar, Sonar Pro, and
Sonar Reasoning Pro add a context-size-dependent request fee ($5/$8/$12 or
$6/$10/$14 per 1K, depending on model), in addition to token costs.  Sonar
Deep Research also bills citation tokens, search queries, and reasoning tokens.
[Perplexity API pricing](https://docs.perplexity.ai/docs/getting-started/pricing)

Thus even `sonar`'s $1 input / $1 output token rates would report an incomplete
total.  Exclude all Sonar static rows until usage and the pricing schema expose
those counters and request settings.

### Google Gemini — exclude

Gemini does have a live model catalog (`GET /v1beta/models` in the native
provider) and the official model page publishes current ids.  However its
pricing is not a single text-token triplet: input prices can vary by text,
image/video, and audio; explicit context caching adds a recurring
token-hour storage charge; grounding adds per-search charges; and some models
have image/audio output prices or priority tiers.
[Gemini models](https://ai.google.dev/gemini-api/docs/models)
[Gemini Developer API pricing](https://ai.google.dev/gemini-api/docs/pricing)

The native mapper can report cache-read tokens, but not cache storage,
grounding query count, or media-token categories.  A static `gemini-*` catalog
would therefore be precise only for a restricted, undocumented-by-schema slice
of requests, so it is not safe.

### Groq — optional restricted catalog, not a full provider catalog

Groq's authenticated `/openai/v1/models` list supplies active IDs but not
prices, so it cannot feed the generic priced-list collector.  Its official
model page does publish a safe subset of direct token-only LLM rows:

| ID | Context | Max output | Input | Cache read | Output |
| --- | ---: | ---: | ---: | ---: | ---: |
| `llama-3.1-8b-instant` | 131,072 | 131,072 | $0.050 | not published | $0.080 |
| `llama-3.3-70b-versatile` | 131,072 | 32,768 | $0.590 | not published | $0.790 |
| `openai/gpt-oss-120b` | 131,072 | 65,536 | $0.150 | not published | $0.600 |
| `openai/gpt-oss-20b` | 131,072 | 65,536 | $0.075 | not published | $0.300 |
| `meta-llama/llama-prompt-guard-2-22m` | 512 | 512 | $0.030 | not published | $0.030 |
| `meta-llama/llama-prompt-guard-2-86m` | 512 | 512 | $0.040 | not published | $0.040 |
| `openai/gpt-oss-safeguard-20b` | 131,072 | 65,536 | $0.075 | not published | $0.300 |
| `qwen/qwen3.6-27b` | 131,072 | 16,384 | $0.600 | not published | $3.000 |

All rates are USD / 1M tokens.  Groq does not publish a cache-discount column
or cache-hit accounting for these rows.  Leave `cached_input_per_mtok` absent;
the runtime falls back to the ordinary input rate only if an upstream ever
reports cache-read tokens.  The rows and limits are documented on the current
models page.
[Groq supported models](https://console.groq.com/docs/models)

This is optional because a full static catalog would be wrong.  Exclude
`groq/compound` and `groq/compound-mini`: the final price depends on underlying
model and web-search/code-execution usage.  Exclude Whisper ($/hour), Orpheus
($/character), enterprise/contact-sales offerings, and performance-tier
provisioned throughput.  Groq states that Compound's tool and underlying model
use determines the cost, and its performance tier is not per-token billing.
[Groq Compound pricing](https://console.groq.com/docs/compound/systems/compound)
[Groq performance tier](https://console.groq.com/docs/performance-tier)

Implementation decision: only `llama-3.1-8b-instant` and
`llama-3.3-70b-versatile` are bundled.  The slash-qualified rows above can be
served by other providers under the same upstream id, and llmleaf's current
dataset is keyed globally by model id rather than `(provider, model)`.  Giving
those shared ids Groq's rate would violate the no-cross-provider-collision
constraint described at the start of this note.

## Implementation order

1. Add a provider-specific static/doc parser for the two current DeepSeek V4
   IDs and its documented limits/rates.
2. Add the seven fixed-rate MiniMax M2 rows, retaining no `max_output`; clearly
   scope cost accuracy to the normal OpenAI-compatible pay-as-you-go path.
3. If desired, add only Groq's direct text-token subset above, with the
   exclusions enforced in the parser.
4. Do not make xAI, Perplexity, Gemini, MiniMax M3, or Groq Compound look
   token-exact until the pricing model includes their missing billing
   dimensions.
