//! Prost-generated typed model, compiled from `proto/llmleaf/v1/llmleaf.proto` by
//! `build.rs`. This is the "real codegen" proof: the proto genuinely compiles into the
//! crate and the generated messages are available here as a typed mirror.
//!
//! These types are NOT what the SDK serialises onto the wire — prost emits
//! protobuf-binary-shaped types (oneofs, `i32` enums, no OpenAI JSON casing). The wire
//! uses the hand-written serde structs in [`crate::types`]. Use this module when you
//! want the canonical proto representation (e.g. to interop with a protobuf toolchain).
#![allow(clippy::all)]
#![allow(missing_docs)]
#![allow(rustdoc::all)]

include!(concat!(env!("OUT_DIR"), "/llmleaf.v1.rs"));
