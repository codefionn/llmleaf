package eu.codefionn.llmleaf.client

import io.ktor.client.engine.HttpClientEngineFactory
import io.ktor.client.engine.cio.CIO

/** JVM uses the CIO engine (pure-Kotlin, coroutine-native, no extra deps). */
public actual fun defaultHttpClientEngine(): HttpClientEngineFactory<*> = CIO
