package eu.codefionn.llmleaf.client

import io.ktor.client.engine.HttpClientEngineFactory

/**
 * The platform's default Ktor HTTP engine. JVM and native use CIO; JS uses the JS engine.
 * The `actual`s live in each platform source set (jvmMain, linuxX64Main, jsMain).
 *
 * Callers who want to inject their own engine (e.g. a mock in tests) can pass one to the
 * [LlmleafClient] secondary constructor instead.
 */
public expect fun defaultHttpClientEngine(): HttpClientEngineFactory<*>
