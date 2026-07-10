//! llmleaf — official Zig client SDK for the llmleaf LLM proxy.
//!
//! The wire is OpenAI/OpenRouter-shaped JSON over HTTP (see ../SPEC.md), not
//! protobuf-binary. The typed model in `gen/llmleaf.zig` mirrors the proto at
//! `clients/proto/llmleaf/v1/llmleaf.proto`.
//!
//! Quick start:
//! ```zig
//! const llmleaf = @import("llmleaf");
//! var gpa: std.heap.GeneralPurposeAllocator(.{}) = .init;
//! const a = gpa.allocator();
//! const client = try llmleaf.Client.init(a, "https://gateway.example.com", "sk-...", .{});
//! defer client.deinit();
//!
//! const resp = try client.chat(.{
//!     .model = "gpt-4o-mini",
//!     .messages = &.{ .{ .role = .user, .content = .{ .text = "hi" } } },
//! }, null);
//! defer resp.deinit();
//! std.debug.print("{s}\n", .{resp.value.choices[0].message.content.?.text});
//! ```

const std = @import("std");

// The typed model (proto mirror), namespaced and also re-exported flat so
// callers can write either `llmleaf.gen.ChatRequest` or `llmleaf.ChatRequest`.
// (`usingnamespace` was removed in Zig 0.16, so the flat names are listed.)
pub const gen = @import("gen/llmleaf.zig");

// Common
pub const Usage = gen.Usage;
pub const PromptTokensDetails = gen.PromptTokensDetails;
pub const ErrorBody = gen.ErrorBody;
// Enums
pub const Role = gen.Role;
pub const FinishReason = gen.FinishReason;
pub const BatchStatus = gen.BatchStatus;
pub const ModelType = gen.ModelType;
// Chat
pub const TextPart = gen.TextPart;
pub const ImageUrlPart = gen.ImageUrlPart;
pub const ContentPart = gen.ContentPart;
pub const Content = gen.Content;
pub const FunctionCall = gen.FunctionCall;
pub const ToolCall = gen.ToolCall;
pub const FunctionCallDelta = gen.FunctionCallDelta;
pub const ToolCallDelta = gen.ToolCallDelta;
pub const ReasoningDetail = gen.ReasoningDetail;
pub const ChatMessage = gen.ChatMessage;
pub const FunctionDef = gen.FunctionDef;
pub const ToolDef = gen.ToolDef;
pub const FunctionName = gen.FunctionName;
pub const NamedToolChoice = gen.NamedToolChoice;
pub const ToolChoice = gen.ToolChoice;
pub const ResponseFormat = gen.ResponseFormat;
pub const ChatRequest = gen.ChatRequest;
pub const Choice = gen.Choice;
pub const ChatResponse = gen.ChatResponse;
pub const Delta = gen.Delta;
pub const ChunkChoice = gen.ChunkChoice;
pub const ChatCompletionChunk = gen.ChatCompletionChunk;
// Responses (POST /v1/responses)
pub const ResponseTextPart = gen.ResponseTextPart;
pub const ResponseInputImagePart = gen.ResponseInputImagePart;
pub const ResponseContentPart = gen.ResponseContentPart;
pub const ResponseContent = gen.ResponseContent;
pub const ResponseMessageItem = gen.ResponseMessageItem;
pub const ResponseFunctionCallItem = gen.ResponseFunctionCallItem;
pub const ResponseFunctionCallOutputItem = gen.ResponseFunctionCallOutputItem;
pub const ResponseReasoningText = gen.ResponseReasoningText;
pub const ResponseReasoningItem = gen.ResponseReasoningItem;
pub const ResponseItem = gen.ResponseItem;
pub const ResponsesInput = gen.ResponsesInput;
pub const ResponsesToolDef = gen.ResponsesToolDef;
pub const ResponsesNamedToolChoice = gen.ResponsesNamedToolChoice;
pub const ResponsesToolChoice = gen.ResponsesToolChoice;
pub const ResponsesReasoning = gen.ResponsesReasoning;
pub const ResponsesRequest = gen.ResponsesRequest;
pub const ResponsesInputTokensDetails = gen.ResponsesInputTokensDetails;
pub const ResponsesOutputTokensDetails = gen.ResponsesOutputTokensDetails;
pub const ResponsesUsage = gen.ResponsesUsage;
pub const ResponsesIncompleteDetails = gen.ResponsesIncompleteDetails;
pub const ResponsesResponse = gen.ResponsesResponse;
pub const ResponsesStreamEvent = gen.ResponsesStreamEvent;
// Embeddings
pub const EmbeddingRequest = gen.EmbeddingRequest;
pub const Embedding = gen.Embedding;
pub const EmbeddingResponse = gen.EmbeddingResponse;
// Rerank
pub const RerankRequest = gen.RerankRequest;
pub const RerankResult = gen.RerankResult;
pub const RerankResponse = gen.RerankResponse;
// Audio
pub const SpeechRequest = gen.SpeechRequest;
pub const Speech = gen.Speech;
pub const Voice = gen.Voice;
pub const VoicesResponse = gen.VoicesResponse;
pub const TranscriptionRequest = gen.TranscriptionRequest;
pub const TranscriptionResponse = gen.TranscriptionResponse;
// Models
pub const Architecture = gen.Architecture;
pub const Pricing = gen.Pricing;
pub const TopProvider = gen.TopProvider;
pub const ModelEndpoint = gen.ModelEndpoint;
pub const ModelEntry = gen.ModelEntry;
pub const ListModelsResponse = gen.ListModelsResponse;
// Batches
pub const BatchRequestItem = gen.BatchRequestItem;
pub const BatchCreateRequest = gen.BatchCreateRequest;
pub const BatchCounts = gen.BatchCounts;
pub const BatchHandle = gen.BatchHandle;
pub const BatchResponse = gen.BatchResponse;
pub const BatchError = gen.BatchError;
pub const BatchResultLine = gen.BatchResultLine;
// Mechanical enum<->wire helpers
pub const enumToWire = gen.enumToWire;
pub const enumFromWire = gen.enumFromWire;

// The client + its supporting types.
const client_mod = @import("client.zig");
pub const Client = client_mod.Client;
pub const Options = client_mod.Options;
pub const Owned = client_mod.Owned;
pub const ChatStream = client_mod.ChatStream;
pub const ResponsesStream = client_mod.ResponsesStream;
pub const BatchResults = client_mod.BatchResults;
pub const TranscriptionResult = client_mod.Client.TranscriptionResult;

const errors_mod = @import("errors.zig");
pub const ApiError = errors_mod.ApiError;
pub const Error = errors_mod.Error;

// Lower-level pieces, exported for advanced use / testing.
pub const wire = @import("wire.zig");
pub const http = @import("http.zig");

test {
    // Pull in every module's unit tests.
    std.testing.refAllDecls(@This());
    _ = @import("gen/llmleaf.zig");
    _ = @import("wire.zig");
    _ = @import("errors.zig");
    _ = @import("client.zig");
    _ = @import("http.zig");
}
