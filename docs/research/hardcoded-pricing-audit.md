# Hardcoded pricing audit — Meta/Muse and seed-only rows

**Audit date:** 2026-08-14
**Scope:** the production literal-rate rows in
`crates/llmleaf-pricing-collector/src/lib.rs`, and rows in the committed
`crates/llmleaf-pricing/data/prices.json` that cannot be produced by a current
collector. **Snapshot:** committed baseline `0f126511`, before concurrent
working-tree additions. This is deliberately not an audit of prices parsed at
collection time from a vendor page or model-list API.

## Provenance classification

`merge_model_infos` starts from `prices.json`, overwrites only fields supplied
by a collector, and retains unobserved rows unless `prune` is requested. Thus
the file is a **seed plus generated updates**; it has no per-row provenance.
It would be incorrect to call the other 92 rows independently “seed-only” from
the JSON file alone: many can be refreshed by the pricing-page or list-endpoint
collectors, but any may also be carried forward by a partial run.

In that committed baseline there is one production literal pricing catalogue:
`parse_meta_model_api`
(`lib.rs:480–531`). Its three Muse cards contain literal IDs, rates, context,
capabilities, tier, and data-use flags. The three matching JSON cards were
introduced in the same 2026-08-10 commit and are exact output mirrors, rather
than independently maintained seed cards. The other apparent fixed IDs are
not static price rows: for example Cohere's two Aya IDs receive their rates
from parsed current page text.

The sole **unambiguously seed-only** committed rate is `echo`:
`prices.json:241–245` gives both token rates as zero, while the collector has
no `echo` source. This is correct by construction, not a vendor-price claim:
the local `EchoProvider` describes itself as a zero-dependency offline mock
provider ([source](../../crates/llmleaf-providers/src/mock.rs)). It needs no
external pricing citation and should remain $0/$0 unless the provider stops
being local and free.

## Primary-source results

The public primary evidence is recent and complete for Muse Spark 1.2's
standard token prices, but it is not complete for every hardcoded metadata
field. Meta's authenticated [Models documentation](https://dev.meta.ai/docs/models)
is the proper per-model authority; an unauthenticated request on the audit date
returns the sign-in application, not the cards. This audit therefore does not
silently treat third-party registries or search snippets as a substitute.

| Static row | Official evidence as of 2026-08-14 | Verdict |
| --- | --- | --- |
| `muse-spark-1.1` | Meta's [launch post](https://ai.meta.com/blog/introducing-muse-spark-meta-model-api/) names 1.1, calls it a multimodal reasoning model, says it is available through the Meta Model API, and describes a one-million-token context. It describes images, video, PDFs, visual/audio inspection, tool use, and reasoning at a product level. | **Model and broad reasoning/multimodal status confirmed.** The public primary post does **not** publish the literal `$1.25/$0.15/$4.25` standard rates, an exact `1,048,576` limit, `stop` support, exact API modality enumeration, or the `false` data-training policy. Those fields are **unverified**, not proven current. |
| `muse-spark-1.2` | Meta's [2026-08-05 developer post](https://developer.meta.com/ai/resources/blog/build-with-muse-code/) explicitly names `muse-spark-1.2`, calls it a coding-optimized reasoning model, says it is available in the Model API, describes a 1M-token window, and explicitly states standard pay-as-you-go prices: **$0.15/1M cached input, $1.25/1M input, $4.25/1M output**. | **Rates confirmed current.** `standard` is confirmed. Exact `1,048,576`, `stop` unsupported, exact `text/image/video/file -> text` API modalities, and `prompts_used_for_training: false` are **not published by this public source**. Do not label those verified without an authenticated model card. |
| `muse-spark-1.2-contributor` | The same [developer post](https://developer.meta.com/ai/resources/blog/build-with-muse-code/) explicitly names this ID, calls it the contributor tier, says it is token-rate-limited and available only in select countries, and says its use **may** improve Meta products. | **ID/tier/limited availability confirmed.** The public primary source does **not** state the literal `$0.10/$0.002/$0.20` rates or an exact context/capability matrix. `prompts_used_for_training: true` overstates the published conditional “may be used to improve”; it should be **unknown/conditional** unless the authenticated terms/card make that unconditional. |

## Stale/unsafe static behavior

1. **High priority — catalog freshness failure.** The parser only checks whether
   *any* returned ID starts with `muse-spark-`, then always emits its old,
   fixed three-card array. A live catalogue containing `muse-spark-1.3`, a
   retired 1.1, or no Contributor entitlement therefore still writes the three
   hardcoded cards and cannot collect the new one. This contradicts its role as
   availability-driven collection. It is an immediate staleness vector even
   where today's three values happen to match.

2. **High priority — unsupported contributor dollars are persisted as facts.**
   The contributor ID is real, but none of the public official material found
   publishes its three token rates. The literal card should either cite a
   directly accessible authenticated primary source and be re-audited from it,
   or omit those dollar fields until one exists.

3. **Medium priority — training-data policy is too strong.** The official post
   says Contributor use *may* improve products; that does not substantiate an
   unconditional Boolean `true`. Conversely, the public post's mention that
   zero data retention is requested through sales does not substantiate the
   standard rows' unconditional `false` either.

4. **Medium priority — exact capability fields lack a source.** Public Meta
   material supports the broad “reasoning, multimodal, about 1M context”
   description, but not the precise `1_048_576` value, `stop` rejection, or
   schema-level modality list. `file` is a reasonable internal representation
   for publicly mentioned PDFs, but the same 1.1 post also discusses audio;
   that product-level statement is insufficient to add `audio` to the Model API
   card. Keep the API fields unknown rather than guessing.

## Recommended collector direction

Use the authenticated `/v1/models` result strictly for availability and emit
only IDs actually returned. Put documented pricing/capabilities in a separately
audited source keyed by the exact documented ID, with a source URL and
last-verified date per row. Do not manufacture cards for unreturned tiers.
When the authenticated model documentation is available, re-audit the five
currently unverified dimensions: Contributor price, exact context, API
modalities, `stop`, and data-retention/training policy.
