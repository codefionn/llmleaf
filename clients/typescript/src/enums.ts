// Enum <-> wire-token mapping (SPEC.md "Enum ⇄ wire mapping").
//
// Every closed-set enum maps to its wire token by LOWERCASING the value name:
//   TOOL_CALLS -> "tool_calls", ASSISTANT -> "assistant", IN_PROGRESS -> "in_progress".
// The `*_UNSPECIFIED` zero value <-> field absent on the wire (undefined here).
//
// The generated protobuf-es file emits these as TypeScript string-keyed numeric enums
// (e.g. `Role.ASSISTANT`). A TS numeric enum is bidirectional at runtime: indexing by
// the numeric value yields the member NAME, which is exactly the proto value name. We
// reuse that to derive the wire token mechanically — one helper pair for every enum,
// no per-enum hand mapping (SPEC.md).

import { Role, FinishReason, BatchStatus } from "./gen/llmleaf/v1/llmleaf_pb.js";

export { Role, FinishReason, BatchStatus };

/** A TS numeric enum object: name<->value reverse-mappable at runtime. */
type NumericEnum = Record<string, string | number>;

function unspecifiedName(e: NumericEnum): string | undefined {
  // The zero value is the `*_UNSPECIFIED` member; its name is what 0 maps to.
  const name = e[0];
  return typeof name === "string" ? name : undefined;
}

/**
 * Encode an enum value to its wire token, or `undefined` for the unspecified zero
 * value (which means "field absent on the wire").
 */
export function enumToWire<E extends number>(
  enumObj: NumericEnum,
  value: E | undefined,
): string | undefined {
  if (value === undefined) return undefined;
  const name = enumObj[value as number];
  if (typeof name !== "string") return undefined;
  if (name === unspecifiedName(enumObj)) return undefined;
  return name.toLowerCase();
}

/**
 * Decode a wire token back into the enum value. An absent/empty token, or a token
 * that matches no member, maps to the unspecified zero value (0).
 */
export function enumFromWire<E extends number>(
  enumObj: NumericEnum,
  token: string | null | undefined,
): E {
  if (token === null || token === undefined || token === "") return 0 as E;
  const want = token.toLowerCase();
  for (const [name, value] of Object.entries(enumObj)) {
    if (typeof value !== "number") continue;
    if (name.toLowerCase() === want) return value as E;
  }
  return 0 as E;
}

// Convenience typed wrappers for the three enums on the surface.

export const roleToWire = (v: Role | undefined): string | undefined =>
  enumToWire<Role>(Role as unknown as NumericEnum, v);
export const roleFromWire = (t: string | null | undefined): Role =>
  enumFromWire<Role>(Role as unknown as NumericEnum, t);

export const finishReasonToWire = (v: FinishReason | undefined): string | undefined =>
  enumToWire<FinishReason>(FinishReason as unknown as NumericEnum, v);
export const finishReasonFromWire = (t: string | null | undefined): FinishReason =>
  enumFromWire<FinishReason>(FinishReason as unknown as NumericEnum, t);

export const batchStatusToWire = (v: BatchStatus | undefined): string | undefined =>
  enumToWire<BatchStatus>(BatchStatus as unknown as NumericEnum, v);
export const batchStatusFromWire = (t: string | null | undefined): BatchStatus =>
  enumFromWire<BatchStatus>(BatchStatus as unknown as NumericEnum, t);
