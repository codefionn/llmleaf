package eu.codefionn.llmleaf.client

import kotlin.time.Duration
import kotlin.time.Duration.Companion.seconds

/**
 * Tunables for an [LlmleafClient]. The core is fully operable with defaults; only [adminToken]
 * (which adds the per-model `endpoints` array on `GET /v1/models`) and a non-default
 * [requestTimeout] are commonly overridden.
 */
public class LlmleafConfig private constructor(
    public val requestTimeout: Duration,
    public val adminToken: String?,
) {
    public class Builder {
        /** Per-request timeout. Streaming calls disable it so long-lived streams are not cut. */
        public var requestTimeout: Duration = 60.seconds

        /** Optional `x-admin-token`; when set, model listings include admin-only endpoints. */
        public var adminToken: String? = null

        public fun build(): LlmleafConfig = LlmleafConfig(requestTimeout, adminToken)
    }

    public companion object {
        public val DEFAULT: LlmleafConfig = Builder().build()

        public operator fun invoke(block: Builder.() -> Unit): LlmleafConfig =
            Builder().apply(block).build()
    }
}
