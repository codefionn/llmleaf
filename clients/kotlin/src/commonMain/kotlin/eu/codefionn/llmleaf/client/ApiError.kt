package eu.codefionn.llmleaf.client

/**
 * A typed error raised for any non-2xx response. Parsed from the canonical envelope
 * `{"error":{"message":...}}` (SPEC.md "Errors"); when the body is missing or unparseable the
 * message falls back to a generic description of the status.
 *
 * Status codes the gateway uses:
 *   400 bad request · 401 missing/invalid key · 403 blocked or model-not-allowed ·
 *   404 no route for model · 429 key suspended (limiter) · 502 all upstreams failed.
 */
public class ApiError(
    public val status: Int,
    override val message: String,
) : Exception("llmleaf API error ($status): $message")
