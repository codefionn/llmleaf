// @llmleaf/client — official TypeScript/JavaScript client for the llmleaf LLM proxy.
//
// The wire is OpenAI/OpenRouter-shaped JSON over HTTP (see clients/SPEC.md). The proto
// (clients/proto/llmleaf/v1/llmleaf.proto) is the typed source of truth; the protobuf-es
// codegen artifact lives under ./gen and the hand-written transport here maps the public
// model to/from the wire JSON.

export { LlmleafClient } from "./client.js";
export type {
  LlmleafClientOptions,
  FetchLike,
  TranscriptionFile,
} from "./client.js";

export { ApiError } from "./error.js";

// Enums (re-exported straight from the generated descriptor file) + the generic
// wire-token mapping helpers.
export {
  Role,
  FinishReason,
  BatchStatus,
  enumToWire,
  enumFromWire,
} from "./enums.js";

// The public typed model.
export type {
  Usage,
  TextPart,
  ImageUrlPart,
  ContentPart,
  MessageContent,
  FunctionCall,
  ToolCall,
  ChatMessage,
  FunctionDef,
  ToolDef,
  ToolChoice,
  ResponseFormat,
  ChatRequest,
  Choice,
  ChatResponse,
  FunctionCallDelta,
  ToolCallDelta,
  Delta,
  ChunkChoice,
  ChatCompletionChunk,
  EmbeddingRequest,
  Embedding,
  EmbeddingResponse,
  SpeechRequest,
  SpeechResult,
  Voice,
  VoicesResponse,
  TranscriptionRequest,
  TranscriptionResponse,
  Architecture,
  Pricing,
  TopProvider,
  ModelEndpoint,
  ModelEntry,
  ListModelsResponse,
  ModelType,
  ListModelsOptions,
  BatchRequestItem,
  BatchCreateRequest,
  BatchCounts,
  BatchHandle,
  BatchResponse,
  BatchError,
  BatchResultLine,
} from "./types.js";

// The generated protobuf-es descriptors/types remain importable for advanced users
// who want the codegen artifact directly (e.g. binary/JSON reflection via @bufbuild/protobuf).
export * as gen from "./gen/llmleaf/v1/llmleaf_pb.js";
