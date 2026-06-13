# Regenerate every client's typed model from the single proto source of truth.
#
#   make proto-check   # validate the schema compiles (uses protoc)
#   make gen           # regenerate all five clients
#   make gen-go gen-ts gen-rust gen-kotlin gen-zig   # one language at a time
#
# Each client owns its own generator invocation under clients/<lang>/scripts/gen.sh so the
# language's native build stays self-contained; the targets here just delegate.

PROTO_ROOT := proto
PROTO_FILES := $(shell find $(PROTO_ROOT) -name '*.proto')

.PHONY: proto-check gen gen-go gen-ts gen-rust gen-kotlin gen-zig gen-csharp clean-desc

proto-check:
	protoc --proto_path=$(PROTO_ROOT) --descriptor_set_out=/tmp/llmleaf.desc $(PROTO_FILES)
	@echo "proto OK"

gen: gen-go gen-ts gen-rust gen-kotlin gen-zig gen-csharp

gen-go:
	cd go && ./scripts/gen.sh

gen-ts:
	cd typescript && ./scripts/gen.sh

gen-rust:
	cd rust && ./scripts/gen.sh

gen-kotlin:
	cd kotlin && ./scripts/gen.sh

gen-zig:
	cd zig && ./scripts/gen.sh

gen-csharp:
	cd csharp && ./scripts/gen.sh

clean-desc:
	rm -f /tmp/llmleaf.desc
