package eu.codefionn.llmleaf.client.model

import eu.codefionn.llmleaf.client.RawJson
import kotlinx.serialization.KSerializer
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.descriptors.SerialDescriptor
import kotlinx.serialization.descriptors.buildClassSerialDescriptor
import kotlinx.serialization.encoding.Decoder
import kotlinx.serialization.encoding.Encoder
import kotlinx.serialization.json.JsonDecoder
import kotlinx.serialization.json.JsonEncoder
import kotlinx.serialization.json.JsonObject

/**
 * A text-to-speech request. Returns raw audio bytes (see [eu.codefionn.llmleaf.client.SpeechResult]).
 * `extra` keys merge at the top level, so a custom serializer is used.
 */
@Serializable(with = SpeechRequestSerializer::class)
public data class SpeechRequest(
    val model: String,
    val input: String,
    val voice: String,
    val responseFormat: String? = null, // mp3|opus|aac|flac|wav|pcm
    val speed: Float? = null,
    val extra: RawJson? = null,
)

@Serializable
internal data class SpeechRequestWire(
    @SerialName("model") val model: String,
    @SerialName("input") val input: String,
    @SerialName("voice") val voice: String,
    @SerialName("response_format") val responseFormat: String? = null,
    @SerialName("speed") val speed: Float? = null,
)

public object SpeechRequestSerializer : KSerializer<SpeechRequest> {
    override val descriptor: SerialDescriptor = buildClassSerialDescriptor("SpeechRequest")

    override fun serialize(encoder: Encoder, value: SpeechRequest) {
        val jsonEncoder = encoder as? JsonEncoder
            ?: error("SpeechRequest requires kotlinx.serialization JSON")
        val json = jsonEncoder.json
        val wire = SpeechRequestWire(
            model = value.model,
            input = value.input,
            voice = value.voice,
            responseFormat = value.responseFormat,
            speed = value.speed,
        )
        val base = json.encodeToJsonElement(SpeechRequestWire.serializer(), wire) as JsonObject
        jsonEncoder.encodeJsonElement(mergeExtra(json, base, value.extra))
    }

    override fun deserialize(decoder: Decoder): SpeechRequest {
        val jsonDecoder = decoder as? JsonDecoder
            ?: error("SpeechRequest requires kotlinx.serialization JSON")
        val obj = jsonDecoder.decodeJsonElement() as JsonObject
        val wire = jsonDecoder.json.decodeFromJsonElement(SpeechRequestWire.serializer(), obj)
        return SpeechRequest(wire.model, wire.input, wire.voice, wire.responseFormat, wire.speed)
    }
}

@Serializable
public data class Voice(
    @SerialName("id") val id: String,
    @SerialName("name") val name: String? = null,
    @SerialName("languages") val languages: List<String> = emptyList(),
)

@Serializable
public data class VoicesResponse(
    @SerialName("model") val model: String,
    @SerialName("voices") val voices: List<Voice> = emptyList(),
)

/**
 * The non-`file` form fields of a transcription (speech-to-text) request. The audio bytes are
 * sent as the multipart `file` part by the client, not via this struct.
 */
public data class TranscriptionRequest(
    val model: String,
    val language: String? = null, // ISO-639-1 hint
    val prompt: String? = null,
    val responseFormat: String? = null, // json|text|verbose_json|srt|vtt
    val temperature: Float? = null,
)

/** For `response_format` json/verbose_json; text/srt/vtt return a plain-text body instead. */
@Serializable
public data class TranscriptionResponse(
    @SerialName("text") val text: String,
    @SerialName("task") val task: String? = null,
    @SerialName("language") val language: String? = null,
    @SerialName("duration") val duration: Float? = null,
    @SerialName("usage") val usage: Usage? = null,
)
