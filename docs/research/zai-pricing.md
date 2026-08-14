# Z.AI public general-API pricing audit

**Checked:** 2026-08-14.  This is a research note for the offline pricing
collector, not a claim that Z.AI exposes a pricing/list-models API.

## Authoritative sources and scope

* The [published OpenAPI document](https://docs.z.ai/openapi.json) is the
  authoritative source of exact request IDs.  Its `/paas/v4/chat/completions`
  request has separate text and vision schemas, each with a closed `model`
  enum.
* The [official pricing page](https://docs.z.ai/guides/overview/pricing) says
  prices are USD and shows input, cached-input, cached-input-storage, and
  output rates **per 1M tokens**.  It is a web document, not a structured
  model/pricing list endpoint.
* Context and modality use the current [model matrix](https://docs.z.ai/guides/overview/overview), while output ceilings and thinking behaviour use
  [core parameters](https://docs.z.ai/guides/overview/concept-param) and the
  [OpenAPI schema](https://docs.z.ai/openapi.json).  Z.AI defines context as
  input plus generated tokens, so `max_context` must not be treated as a
  prompt-only limit.

The table below deliberately covers the 20 models currently callable through
the *general* chat-completions endpoint.  It excludes the Coding Plan: it has
a different `/api/coding/paas/v4` endpoint and Z.AI says it is for supported
tools, not general-purpose API access ([quick start](https://docs.z.ai/guides/overview/quick-start), [subscription terms](https://docs.z.ai/legal-agreement/subscription-terms)).

## Collector-ready chat catalog

Prices are USD/MTok. `0` means the pricing page explicitly says **Free**;
`—` means no rate is published, not zero.  All model IDs are lowercase exact
wire values, not the title-cased names used in the pricing table.

| Model ID | Input | Cached input | Output | Input -> output modalities | Context | Max output | Reasoning / thinking |
| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| `glm-5.2` | 1.40 | 0.26 | 4.40 | text -> text | 1M | 128K | yes, automatic |
| `glm-5.1` | 1.40 | 0.26 | 4.40 | text -> text | 200K | 128K | yes, automatic |
| `glm-5` | 1.00 | 0.20 | 3.20 | text -> text | 200K | 128K | yes, automatic |
| `glm-5-turbo` | 1.20 | 0.24 | 4.00 | text -> text | 200K | 128K | yes, automatic |
| `glm-4.7` | 0.60 | 0.11 | 2.20 | text -> text | 200K | 128K | yes, forced when enabled |
| `glm-4.7-flashx` | 0.07 | 0.01 | 0.40 | text -> text | 200K | 128K | yes, forced when enabled |
| `glm-4.7-flash` | 0 | 0 | 0 | text -> text | 200K | 128K | yes, forced when enabled |
| `glm-4.6` | 0.60 | 0.11 | 2.20 | text -> text | 200K | 128K | yes, automatic |
| `glm-4.5` | 0.60 | 0.11 | 2.20 | text -> text | 128K | 96K | yes, automatic |
| `glm-4.5-x` | 2.20 | 0.45 | 8.90 | text -> text | 128K | 96K | yes, automatic |
| `glm-4.5-air` | 0.20 | 0.03 | 1.10 | text -> text | 128K | 96K | yes, automatic |
| `glm-4.5-airx` | 1.10 | 0.22 | 4.50 | text -> text | 128K | 96K | yes, automatic |
| `glm-4.5-flash` | 0 | 0 | 0 | text -> text | 200K | 96K | yes, automatic |
| `glm-4-32b-0414-128k` | 0.10 | — | 0.10 | text -> text | 128K | 16K | no exposed `thinking` support published |
| `glm-5v-turbo` | 1.20 | 0.24 | 4.00 | video/image/text/file -> text | 200K | 128K | yes, automatic |
| `glm-4.6v` | 0.30 | 0.05 | 0.90 | video/image/text/file -> text | 128K | 32K | yes, automatic |
| `glm-4.6v-flashx` | 0.04 | 0.004 | 0.40 | video/image/text/file -> text | 128K | 32K | yes, automatic |
| `glm-4.6v-flash` | 0 | 0 | 0 | video/image/text/file -> text | 128K | 32K | yes, automatic |
| `glm-4.5v` | 0.60 | 0.11 | 1.80 | video/image/text/file -> text | 64K | 16K | yes, forced when enabled |
| `autoglm-phone-multilingual` | — | — | — | task instruction -> task action | — | 4K | not published |

The listed rates come verbatim from the [text and vision pricing tables](https://docs.z.ai/guides/overview/pricing).  The text IDs and vision IDs are
separately enumerated in [OpenAPI](https://docs.z.ai/openapi.json); their
modalities/contexts are documented by the [model matrix](https://docs.z.ai/guides/overview/overview).  The output limits are explicitly published in
[core parameters](https://docs.z.ai/guides/overview/concept-param).  That
source also says `thinking` is supported by GLM-4.5 and later, with automatic
thinking for GLM-5.2/5.1/5/5-Turbo/5V-Turbo/4.6/4.5 and forced thinking for
GLM-4.7 and 4.5V.  `reasoning_effort` is separately documented only for
GLM-5.2 and above; it is not a token budget and Z.AI publishes no maximum
thinking-token count.

## Important reconciliation and schema caveats

1. **Do not add `glm-5.3` to the general API dataset.**  Its guide explicitly
   says “API is coming soon”; it is currently a Coding Plan offering, absent
   from both the general chat enum and the pricing table ([GLM-5.3 guide](https://docs.z.ai/guides/llm/glm-5.3)).  Therefore it has no published general-API rate.

2. **The lists do not line up perfectly.**  `autoglm-phone-multilingual` is
   accepted by the vision chat schema, but has neither price nor context on
   the pricing/model-matrix pages.  Retain it only if the collector wants
   unpriced rows; leave rate/context fields absent.  Conversely, the pricing
   table lists `GLM-OCR`, but OpenAPI serves it only at
   `/paas/v4/layout_parsing`, not chat completions.

3. **Non-chat general API models need a different price schema.**  The
   OpenAPI document also names `glm-ocr` (input/output $0.03/MTok, no cached
   rate), `glm-asr-2512` ($0.03/MTok), image models `glm-image` and
   `cogview-4-250304`, and video models `cogvideox-3`, `viduq1-text`,
   `viduq1-image`, `viduq1-start-end`, `vidu2-image`, `vidu2-start-end`, and
   `vidu2-reference`.  The pricing page bills image/video per image/video,
   and ASR is transcription rather than a chat model.  Do not force them
   into the three chat token-rate fields.  The page labels `CogView-4`, but
   the exact OpenAPI ID is `cogview-4-250304`—this is a display-name/ID
   mismatch, not evidence of an alias.

4. **Cached input is a cache-read rate, not cache creation/storage.**  The
   table has a separate “Cached Input Storage” column, currently
   “Limited-time Free” for the paid cache-capable models.  The collector’s
   `cached_input_per_mtok` should take only the `Cached Input` column;
   it has no field for storage cost.  For `glm-4-32b-0414-128k` the table uses
   `-`, so preserve `None`, rather than copying the normal input rate.

5. **Free is an advertised current price, not a permanent contract.**  Store
   the explicit rates as `0.0` if zero-cost accounting is desired, and retain
   the source/retrieval date so a future offline refresh can change them.

6. **No upstream aliases were published.**  Use the exact OpenAPI enum IDs.
   Names such as `GLM-4.7-FlashX` on the marketing/pricing page are display
   labels; the wire ID is `glm-4.7-flashx`.  Repository provider-kind aliases
   (`zai`, `z.ai`, `glm`) are local routing aliases, not Z.AI model aliases.

7. **There is one context discrepancy to preserve deliberately.**  The model
   matrix gives `glm-4.5-flash` 200K context, while the GLM-4.5 family guide
   describes its named 4.5/Air pair as 128K.  The matrix is the page that
   assigns a value to Flash specifically, so use 200K but treat it as
   documented-not-inferred.  Its 96K output ceiling is unambiguous in core
   parameters.
