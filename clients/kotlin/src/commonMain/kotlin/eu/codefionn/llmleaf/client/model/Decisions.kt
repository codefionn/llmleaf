package eu.codefionn.llmleaf.client.model

import eu.codefionn.llmleaf.client.RawJson
import kotlinx.serialization.KSerializer
import kotlinx.serialization.ExperimentalSerializationApi
import kotlinx.serialization.Serializable
import kotlinx.serialization.descriptors.SerialDescriptor
import kotlinx.serialization.encoding.Decoder
import kotlinx.serialization.encoding.Encoder
import kotlinx.serialization.json.JsonDecoder
import kotlinx.serialization.json.JsonEncoder
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.contentOrNull

/** A Decisions request. State and every question are raw JSON, never JSON strings. */
@Serializable(with = DecisionsRequestSerializer::class)
public data class DecisionsRequest(
    val model: String,
    val state: RawJson,
    val questions: Map<String, RawJson>,
    /** Extra top-level JSON fields. Explicit fields take precedence. */
    val extra: RawJson? = null,
)

public object DecisionsRequestSerializer : KSerializer<DecisionsRequest> {
    override val descriptor: SerialDescriptor = JsonObject.serializer().descriptor

    override fun serialize(encoder: Encoder, value: DecisionsRequest) {
        val out = encoder as? JsonEncoder ?: error("DecisionsRequest requires a JSON encoder")
        val json = out.json
        val questions = JsonObject(value.questions.mapValues { (_, raw) -> json.parseToJsonElement(raw.value) })
        val base = JsonObject(mapOf(
            "model" to JsonPrimitive(value.model),
            "state" to json.parseToJsonElement(value.state.value),
            "questions" to questions,
        ))
        out.encodeJsonElement(mergeExtra(json, base, value.extra))
    }

    override fun deserialize(decoder: Decoder): DecisionsRequest {
        val input = decoder as? JsonDecoder ?: error("DecisionsRequest requires a JSON decoder")
        val obj = input.decodeJsonElement() as? JsonObject ?: error("DecisionsRequest must be an object")
        val model = (obj["model"] as? JsonPrimitive)?.content ?: error("DecisionsRequest.model is required")
        val state = obj["state"]?.let(RawJson::of) ?: error("DecisionsRequest.state is required")
        val questionObject = obj["questions"] as? JsonObject ?: error("DecisionsRequest.questions is required")
        val questions = questionObject.mapValues { (_, element) -> RawJson.of(element) }
        val extraFields = obj.filterKeys { it !in setOf("model", "state", "questions") }
        return DecisionsRequest(model, state, questions, extraFields.takeIf { it.isNotEmpty() }?.let { RawJson.of(JsonObject(it)) })
    }
}

/** Provider token accounting for a Decisions response. `cost = 0.0` remains distinct from absent. */
@Serializable(with = DecisionsUsageSerializer::class)
public data class DecisionsUsage(
    val inputTokens: ULong = 0u,
    val outputTokens: ULong = 0u,
    val cost: Double? = null,
    val extra: RawJson? = null,
)

@OptIn(ExperimentalSerializationApi::class)
public object DecisionsUsageSerializer : KSerializer<DecisionsUsage> {
    override val descriptor: SerialDescriptor = JsonObject.serializer().descriptor
    override fun serialize(encoder: Encoder, value: DecisionsUsage) {
        val out = encoder as? JsonEncoder ?: error("DecisionsUsage requires a JSON encoder")
        val base = buildMap<String, JsonElement> {
            put("input_tokens", JsonPrimitive(value.inputTokens))
            put("output_tokens", JsonPrimitive(value.outputTokens))
            value.cost?.let { put("cost", JsonPrimitive(it)) }
        }
        out.encodeJsonElement(mergeExtra(out.json, JsonObject(base), value.extra))
    }
    override fun deserialize(decoder: Decoder): DecisionsUsage {
        val input = decoder as? JsonDecoder ?: error("DecisionsUsage requires a JSON decoder")
        val obj = input.decodeJsonElement() as? JsonObject ?: error("DecisionsUsage must be an object")
        fun token(name: String): ULong = (obj[name] as? JsonPrimitive)?.content?.toULongOrNull() ?: 0u
        val extra = obj.filterKeys { it !in setOf("input_tokens", "output_tokens", "cost") }
        return DecisionsUsage(token("input_tokens"), token("output_tokens"),
            (obj["cost"] as? JsonPrimitive)?.content?.toDoubleOrNull(),
            extra.takeIf { it.isNotEmpty() }?.let { RawJson.of(JsonObject(it)) })
    }
}

/** A Decisions response. Answers and unknown fields remain raw JSON for forward compatibility. */
@Serializable(with = DecisionsResponseSerializer::class)
public data class DecisionsResponse(
    val model: String,
    val answers: Map<String, RawJson>,
    val usage: DecisionsUsage? = null,
    val id: String? = null,
    val provider: String? = null,
    val extra: RawJson? = null,
)

public object DecisionsResponseSerializer : KSerializer<DecisionsResponse> {
    override val descriptor: SerialDescriptor = JsonObject.serializer().descriptor
    override fun serialize(encoder: Encoder, value: DecisionsResponse) {
        val out = encoder as? JsonEncoder ?: error("DecisionsResponse requires a JSON encoder")
        val json = out.json
        val base = buildMap<String, JsonElement> {
            put("model", JsonPrimitive(value.model))
            put("answers", JsonObject(value.answers.mapValues { (_, raw) -> json.parseToJsonElement(raw.value) }))
            value.usage?.let { put("usage", json.encodeToJsonElement(DecisionsUsageSerializer, it)) }
            value.id?.let { put("id", JsonPrimitive(it)) }
            value.provider?.let { put("provider", JsonPrimitive(it)) }
        }
        out.encodeJsonElement(mergeExtra(json, JsonObject(base), value.extra))
    }
    override fun deserialize(decoder: Decoder): DecisionsResponse {
        val input = decoder as? JsonDecoder ?: error("DecisionsResponse requires a JSON decoder")
        val obj = input.decodeJsonElement() as? JsonObject ?: error("DecisionsResponse must be an object")
        val answers = (obj["answers"] as? JsonObject ?: JsonObject(emptyMap())).mapValues { (_, element) -> RawJson.of(element) }
        val extra = obj.filterKeys { it !in setOf("model", "answers", "usage", "id", "provider") }
        return DecisionsResponse(
            model = (obj["model"] as? JsonPrimitive)?.content ?: "",
            answers = answers,
            usage = obj["usage"]?.let { input.json.decodeFromJsonElement(DecisionsUsageSerializer, it) },
            id = (obj["id"] as? JsonPrimitive)?.contentOrNull,
            provider = (obj["provider"] as? JsonPrimitive)?.contentOrNull,
            extra = extra.takeIf { it.isNotEmpty() }?.let { RawJson.of(JsonObject(it)) },
        )
    }
}
