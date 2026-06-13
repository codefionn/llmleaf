package eu.codefionn.llmleaf.client.model

import eu.codefionn.llmleaf.client.RawJson
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject

/**
 * Merges the keys of an `extra` raw-JSON object into [base] at the top level (SPEC.md:
 * "`extra` keys are merged at the top level of the request object").
 *
 * Explicit struct fields win over `extra` on key collisions, so passthrough cannot silently
 * override a typed parameter the caller also set.
 */
internal fun mergeExtra(json: Json, base: JsonObject, extra: RawJson?): JsonElement {
    if (extra == null) return base
    val parsed = json.parseToJsonElement(extra.value)
    require(parsed is JsonObject) { "extra must be a JSON object" }
    if (parsed.isEmpty()) return base
    // base entries take precedence over extra entries.
    val merged = LinkedHashMap<String, JsonElement>(parsed)
    merged.putAll(base)
    return JsonObject(merged)
}
