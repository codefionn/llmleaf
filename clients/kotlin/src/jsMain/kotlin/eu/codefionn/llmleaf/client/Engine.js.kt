package eu.codefionn.llmleaf.client

import io.ktor.client.engine.HttpClientEngineFactory
import io.ktor.client.engine.js.Js

/** JS targets (browser + Node) use the Ktor JS engine (fetch / node-fetch). */
public actual fun defaultHttpClientEngine(): HttpClientEngineFactory<*> = Js
