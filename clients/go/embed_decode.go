package llmleaf

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"math"
)

// decodeEmbeddingVector decodes the wire `embedding` value, which is either a
// JSON array of float32 (encoding_format "float") or a base64 string of
// little-endian f32 bytes (encoding_format "base64"). Per SPEC.md the base64
// form is decoded into the float vector before being returned to the caller.
func decodeEmbeddingVector(raw json.RawMessage) ([]float32, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil
	}
	switch raw[0] {
	case '[':
		var vec []float32
		if err := json.Unmarshal(raw, &vec); err != nil {
			return nil, err
		}
		return vec, nil
	case '"':
		var b64 string
		if err := json.Unmarshal(raw, &b64); err != nil {
			return nil, err
		}
		return decodeBase64F32(b64)
	default:
		return nil, fmt.Errorf("llmleaf: unexpected embedding encoding: %q", string(raw))
	}
}

func decodeBase64F32(s string) ([]float32, error) {
	data, err := base64.StdEncoding.DecodeString(s)
	if err != nil {
		return nil, fmt.Errorf("llmleaf: decode base64 embedding: %w", err)
	}
	if len(data)%4 != 0 {
		return nil, fmt.Errorf("llmleaf: base64 embedding byte length %d is not a multiple of 4", len(data))
	}
	vec := make([]float32, len(data)/4)
	for i := range vec {
		bits := uint32(data[i*4]) |
			uint32(data[i*4+1])<<8 |
			uint32(data[i*4+2])<<16 |
			uint32(data[i*4+3])<<24
		vec[i] = math.Float32frombits(bits)
	}
	return vec, nil
}
