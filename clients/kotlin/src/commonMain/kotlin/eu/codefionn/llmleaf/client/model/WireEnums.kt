package eu.codefionn.llmleaf.client.model

import kotlinx.serialization.Serializable

/**
 * Message author role. Mirrors `enum Role` in the proto; `ROLE_UNSPECIFIED` means the field
 * is absent on the wire (it never serialises a token).
 */
@Serializable(with = RoleSerializer::class)
public enum class Role(override val wire: String) : WireEnum {
    ROLE_UNSPECIFIED(""),
    SYSTEM("system"),
    USER("user"),
    ASSISTANT("assistant"),
    TOOL("tool"),
}

public object RoleSerializer :
    WireEnumSerializer<Role>("Role", Role.entries.toTypedArray(), Role.ROLE_UNSPECIFIED)

/** Why generation stopped. Mirrors `enum FinishReason`. */
@Serializable(with = FinishReasonSerializer::class)
public enum class FinishReason(override val wire: String) : WireEnum {
    FINISH_REASON_UNSPECIFIED(""),
    STOP("stop"),
    LENGTH("length"),
    TOOL_CALLS("tool_calls"),
    CONTENT_FILTER("content_filter"),
}

public object FinishReasonSerializer :
    WireEnumSerializer<FinishReason>(
        "FinishReason",
        FinishReason.entries.toTypedArray(),
        FinishReason.FINISH_REASON_UNSPECIFIED,
    )

/** Lifecycle state of a batch. Mirrors `enum BatchStatus`. */
@Serializable(with = BatchStatusSerializer::class)
public enum class BatchStatus(override val wire: String) : WireEnum {
    BATCH_STATUS_UNSPECIFIED(""),
    VALIDATING("validating"),
    IN_PROGRESS("in_progress"),
    FINALIZING("finalizing"),
    COMPLETED("completed"),
    FAILED("failed"),
    EXPIRED("expired"),
    CANCELING("canceling"),
    CANCELED("canceled"),
}

public object BatchStatusSerializer :
    WireEnumSerializer<BatchStatus>(
        "BatchStatus",
        BatchStatus.entries.toTypedArray(),
        BatchStatus.BATCH_STATUS_UNSPECIFIED,
    )
