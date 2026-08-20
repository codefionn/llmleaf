//! Prost-generated typed model from `proto/llmleaf/v1/llmleaf.proto`.
//!
//! These types are NOT what the SDK serialises onto the wire — prost emits
//! protobuf-binary-shaped types (oneofs, `i32` enums, no OpenAI JSON casing). The wire
//! uses the hand-written serde structs in [`crate::types`]. Use this module when you
//! want the canonical proto representation (e.g. to interop with a protobuf toolchain).
#![allow(clippy::all)]
#![allow(missing_docs)]
#![allow(rustdoc::all)]

include!("gen/llmleaf/v1/llmleaf.v1.rs");
