package eu.codefionn.llmleaf.client

import io.ktor.client.engine.HttpClientEngineFactory
import io.ktor.client.engine.cio.CIO

/**
 * linuxX64 uses the CIO engine (Ktor's multiplatform-native engine). To add another native
 * target (macosArm64, mingwX64, …), either copy this `actual` into that target's source set or
 * introduce a shared `nativeMain` source set with the CIO dependency and place the `actual`
 * there. We keep it per-target here so the build does not depend on an intermediate source set.
 */
public actual fun defaultHttpClientEngine(): HttpClientEngineFactory<*> = CIO
