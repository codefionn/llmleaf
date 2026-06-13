package eu.codefionn.llmleaf.client.model

import kotlinx.serialization.KSerializer
import kotlinx.serialization.descriptors.PrimitiveKind
import kotlinx.serialization.descriptors.PrimitiveSerialDescriptor
import kotlinx.serialization.descriptors.SerialDescriptor
import kotlinx.serialization.encoding.Decoder
import kotlinx.serialization.encoding.Encoder

/**
 * Every closed-set enum on the wire carries a lowercase string token (SPEC.md "Enum ⇄ wire
 * mapping"): `TOOL_CALLS` ⇄ `"tool_calls"`, `ASSISTANT` ⇄ `"assistant"`, etc. Enums implement
 * this interface so a single serializer pair ([enumToWire] / [enumFromWire]) handles them all.
 */
public interface WireEnum {
    /** The lowercase token this value serialises to / from. */
    public val wire: String
}

/** The lowercase wire token for [value] — the one place encode happens. */
public fun enumToWire(value: WireEnum): String = value.wire

/**
 * Resolves a wire token back to its enum value, or null when absent / unknown (the
 * `*_UNSPECIFIED ⇔ field absent` rule plus forward-compatibility for new tokens).
 */
public inline fun <reified E> enumFromWire(token: String?): E?
    where E : Enum<E>, E : WireEnum {
    if (token == null) return null
    return enumValues<E>().firstOrNull { it.wire == token }
}

/**
 * Generic serializer for a [WireEnum]. Subclass it once per enum with the value array so the
 * mapping stays mechanical. Unknown tokens decode to null is impossible for a non-null field,
 * so unknown tokens fall back to the type's [unspecified] value.
 */
public abstract class WireEnumSerializer<E>(
    private val name: String,
    private val values: Array<E>,
    private val unspecified: E,
) : KSerializer<E> where E : Enum<E>, E : WireEnum {
    override val descriptor: SerialDescriptor = PrimitiveSerialDescriptor(name, PrimitiveKind.STRING)

    override fun serialize(encoder: Encoder, value: E) {
        encoder.encodeString(value.wire)
    }

    override fun deserialize(decoder: Decoder): E {
        val token = decoder.decodeString()
        return values.firstOrNull { it.wire == token } ?: unspecified
    }
}
