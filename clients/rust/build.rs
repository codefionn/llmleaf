//! Build script: compile the single-source-of-truth proto with `prost-build`.
//!
//! This is the real codegen step. It runs every `cargo build`, invokes `protoc`
//! (which must be on PATH; libprotoc 35 is what the rest of `clients/` targets),
//! and emits `$OUT_DIR/llmleaf.v1.rs`. `src/pb.rs` then `include!`s that file so the
//! prost-generated messages are genuinely part of the crate — proof the schema
//! compiles and a typed mirror available to consumers under `llmleaf_client::pb`.
//!
//! The public SDK types and the HTTP transport are hand-written serde structs (see
//! `src/types.rs`): prost's generated types do not serialise to the OpenAI/OpenRouter
//! JSON wire (oneofs, enum casing, snake_case content unions), so prost compiles the
//! proto while serde drives the wire.

use std::path::PathBuf;

fn main() {
    let proto_root = PathBuf::from("../proto");
    let proto_file = proto_root.join("llmleaf/v1/llmleaf.proto");

    // Re-run codegen whenever the contract changes.
    println!("cargo:rerun-if-changed={}", proto_file.display());
    println!("cargo:rerun-if-changed=build.rs");

    let mut config = prost_build::Config::new();
    // Keep the generated module self-contained inside OUT_DIR.
    config.compile_protos(&[proto_file], &[proto_root]).expect(
        "failed to compile llmleaf.proto with prost-build; \
             ensure `protoc` (libprotoc 35) is installed and on PATH",
    );
}
