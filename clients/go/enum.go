package llmleaf

import "strings"

// enumToWire maps a generated enum value-name (e.g. "TOOL_CALLS", "ASSISTANT",
// "IN_PROGRESS") to its wire token by lowercasing it, per SPEC.md. The
// "*_UNSPECIFIED" zero value (and the empty name) maps to "" (field absent).
//
// nameMap is the generated enum's `_name` map (int32 -> NAME); value is the
// concrete enum's int32. Implemented once and reused for every closed-set enum.
func enumToWire(nameMap map[int32]string, value int32) string {
	name := nameMap[value]
	if name == "" || strings.HasSuffix(name, "_UNSPECIFIED") {
		return ""
	}
	return strings.ToLower(name)
}

// enumFromWire maps a wire token back to the generated enum's int32 by
// uppercasing it and looking it up in the generated `_value` map. An empty or
// unknown token yields the zero value (UNSPECIFIED) and ok=false, so callers can
// leave the field absent.
func enumFromWire(valueMap map[string]int32, token string) (int32, bool) {
	if token == "" {
		return 0, false
	}
	if v, ok := valueMap[strings.ToUpper(token)]; ok {
		return v, true
	}
	return 0, false
}
