package eu.codefionn.llmleaf.client

/**
 * The result of a text-to-speech call: the raw audio [bytes] plus the [contentType] the gateway
 * reported (e.g. `audio/mpeg` for mp3, `audio/wav` for wav — SPEC.md `POST /v1/audio/speech`).
 */
public class SpeechResult(
    public val bytes: ByteArray,
    public val contentType: String?,
)
