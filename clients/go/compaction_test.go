package llmleaf

import (
	"encoding/json"
	"testing"

	pb "github.com/codefionn/llmleaf/clients/go/llmleafpb"
)

func TestCompactionWireRoundTrip(t *testing.T) {
	var item wireResponseItem
	input := []byte(`{"type":"compaction","id":"cmp_1","encrypted_content":"opaque"}`)
	if err := json.Unmarshal(input, &item); err != nil {
		t.Fatal(err)
	}
	if got := item.toPB().GetCompaction().GetEncryptedContent(); got != "opaque" {
		t.Fatalf("encrypted content = %q", got)
	}
	encoded, err := json.Marshal(responseItemFromPB(item.toPB()))
	if err != nil {
		t.Fatal(err)
	}
	var got map[string]any
	if err := json.Unmarshal(encoded, &got); err != nil {
		t.Fatal(err)
	}
	if got["id"] != "cmp_1" || got["encrypted_content"] != "opaque" {
		t.Fatalf("replayed item = %s", encoded)
	}

	message := &pb.ChatMessage{Compaction: []*pb.CompactionBlock{{Content: ptr("summary"), Signature: ptr("signed")}}}
	wire, err := chatMessageToWire(message)
	if err != nil {
		t.Fatal(err)
	}
	encoded, err = json.Marshal(wire)
	if err != nil {
		t.Fatal(err)
	}
	var decoded wireChatMessage
	if err := json.Unmarshal(encoded, &decoded); err != nil {
		t.Fatal(err)
	}
	back, err := chatMessageFromWire(decoded)
	if err != nil {
		t.Fatal(err)
	}
	if got := back.GetCompaction()[0].GetSignature(); got != "signed" {
		t.Fatalf("signature = %q", got)
	}
}
