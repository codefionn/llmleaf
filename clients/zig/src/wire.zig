//! The edge mapping between the typed model (`gen/llmleaf.zig`) and the
//! OpenAI/OpenRouter-shaped JSON the llmleaf core speaks (SPEC.md).
//!
//! Encoding uses `std.json.Stringify` directly so we control every wire
//! quirk: raw-JSON splicing (`extra`, `parameters`, `json_schema`, ...),
//! string-or-array `content`/`stop`, `max_completion_tokens` precedence, and
//! the bare-string-vs-object `tool_choice`. Decoding goes through
//! `std.json.Value` and pulls fields out by hand, so optional/absent fields and
//! the flexible `content` shape are handled explicitly rather than by reflection.
//!
//! Allocations made while decoding land in a caller-provided arena; the caller
//! frees the arena (see `client.zig` + the `Parsed`/`*Owned` wrappers).

const std = @import("std");
const gen = @import("gen/llmleaf.zig");

const Allocator = std.mem.Allocator;
const Writer = std.Io.Writer;
const Value = std.json.Value;
const Stringify = std.json.Stringify;

// ===========================================================================
// Encoding (typed model -> OpenAI/OpenRouter JSON request body)
// ===========================================================================

// `std.json.Stringify` writes raw bytes via `print`, so to splice a free-form
// JSON field we validate it (against a throwaway allocator) then print it
// verbatim. A malformed `extra`/`parameters`/... becomes a clear error rather
// than corrupt output on the wire.
fn validateRaw(gpa: Allocator, raw: []const u8) bool {
    return std.json.validate(gpa, raw) catch false;
}

/// Emit a field whose value is a raw-JSON object string, validating it first.
fn rawField(s: *Stringify, gpa: Allocator, key: []const u8, raw: []const u8) !void {
    if (!validateRaw(gpa, raw)) return error.InvalidRawJson;
    try s.objectField(key);
    // `Stringify.print` writes bytes verbatim into the underlying writer.
    try s.print("{s}", .{raw});
}

fn writeContent(s: *Stringify, content: gen.Content) !void {
    switch (content) {
        .text => |t| try s.write(t),
        .parts => |parts| {
            try s.beginArray();
            for (parts) |part| {
                try s.beginObject();
                switch (part) {
                    .text => |tp| {
                        try s.objectField("type");
                        try s.write("text");
                        try s.objectField("text");
                        try s.write(tp.text);
                    },
                    .image_url => |ip| {
                        try s.objectField("type");
                        try s.write("image_url");
                        try s.objectField("image_url");
                        try s.beginObject();
                        try s.objectField("url");
                        try s.write(ip.url);
                        if (ip.detail) |d| {
                            try s.objectField("detail");
                            try s.write(d);
                        }
                        try s.endObject();
                    },
                }
                try s.endObject();
            }
            try s.endArray();
        },
    }
}

/// Write one `reasoning_details[]` entry as an object on `s`. Optional fields are omitted
/// when absent; `signature`/`data` are opaque and round-tripped verbatim.
fn writeReasoningDetail(s: *Stringify, rd: gen.ReasoningDetail) !void {
    try s.beginObject();
    try s.objectField("type");
    try s.write(rd.type);
    if (rd.text) |v| {
        try s.objectField("text");
        try s.write(v);
    }
    if (rd.summary) |v| {
        try s.objectField("summary");
        try s.write(v);
    }
    if (rd.data) |v| {
        try s.objectField("data");
        try s.write(v);
    }
    if (rd.signature) |v| {
        try s.objectField("signature");
        try s.write(v);
    }
    if (rd.id) |v| {
        try s.objectField("id");
        try s.write(v);
    }
    if (rd.format) |v| {
        try s.objectField("format");
        try s.write(v);
    }
    if (rd.index) |v| {
        try s.objectField("index");
        try s.write(v);
    }
    try s.endObject();
}

fn writeMessage(s: *Stringify, gpa: Allocator, m: gen.ChatMessage) !void {
    try s.beginObject();
    try s.objectField("role");
    try s.write(gen.enumToWire(m.role));
    if (m.content) |c| {
        try s.objectField("content");
        try writeContent(s, c);
    }
    if (m.name) |n| {
        try s.objectField("name");
        try s.write(n);
    }
    if (m.tool_calls.len > 0) {
        try s.objectField("tool_calls");
        try s.beginArray();
        for (m.tool_calls) |tc| {
            try s.beginObject();
            try s.objectField("id");
            try s.write(tc.id);
            try s.objectField("type");
            try s.write(tc.type);
            try s.objectField("function");
            try s.beginObject();
            try s.objectField("name");
            try s.write(tc.function.name);
            try s.objectField("arguments");
            try s.write(tc.function.arguments); // OpenAI: a JSON string, escaped
            try s.endObject();
            try s.endObject();
        }
        try s.endArray();
    }
    if (m.tool_call_id) |id| {
        try s.objectField("tool_call_id");
        try s.write(id);
    }
    // Echo a prior assistant turn's reasoning back verbatim (preserves signed reasoning).
    if (m.reasoning) |r| {
        try s.objectField("reasoning");
        try s.write(r);
    }
    if (m.reasoning_details.len > 0) {
        try s.objectField("reasoning_details");
        try s.beginArray();
        for (m.reasoning_details) |rd| try writeReasoningDetail(s, rd);
        try s.endArray();
    }
    _ = gpa;
    try s.endObject();
}

fn writeTools(s: *Stringify, gpa: Allocator, tools: []const gen.ToolDef) !void {
    try s.objectField("tools");
    try s.beginArray();
    for (tools) |t| {
        try s.beginObject();
        try s.objectField("type");
        try s.write(t.type);
        try s.objectField("function");
        try s.beginObject();
        try s.objectField("name");
        try s.write(t.function.name);
        if (t.function.description) |d| {
            try s.objectField("description");
            try s.write(d);
        }
        if (t.function.parameters) |p| try rawField(s, gpa, "parameters", p);
        try s.endObject();
        try s.endObject();
    }
    try s.endArray();
}

fn writeStop(s: *Stringify, stop: []const []const u8) !void {
    try s.objectField("stop");
    // SPEC.md: a single element MAY be a bare string; an array is always fine.
    if (stop.len == 1) {
        try s.write(stop[0]);
    } else {
        try s.beginArray();
        for (stop) |x| try s.write(x);
        try s.endArray();
    }
}

/// Write the body of a ChatRequest into an already-open object on `s`. Split
/// out so batches can embed it under `body`.
fn writeChatBody(s: *Stringify, gpa: Allocator, req: gen.ChatRequest) !void {
    try s.objectField("model");
    try s.write(req.model);

    try s.objectField("messages");
    try s.beginArray();
    for (req.messages) |m| try writeMessage(s, gpa, m);
    try s.endArray();

    if (req.stream) |v| {
        try s.objectField("stream");
        try s.write(v);
    }
    if (req.temperature) |v| {
        try s.objectField("temperature");
        try s.write(v);
    }
    if (req.top_p) |v| {
        try s.objectField("top_p");
        try s.write(v);
    }
    // Prefer the modern name; still emit the legacy one if only it was set.
    if (req.max_completion_tokens) |v| {
        try s.objectField("max_completion_tokens");
        try s.write(v);
    } else if (req.max_tokens) |v| {
        try s.objectField("max_tokens");
        try s.write(v);
    }
    if (req.stop.len > 0) try writeStop(s, req.stop);
    if (req.n) |v| {
        try s.objectField("n");
        try s.write(v);
    }
    if (req.seed) |v| {
        try s.objectField("seed");
        try s.write(v);
    }
    if (req.frequency_penalty) |v| {
        try s.objectField("frequency_penalty");
        try s.write(v);
    }
    if (req.presence_penalty) |v| {
        try s.objectField("presence_penalty");
        try s.write(v);
    }
    if (req.tools.len > 0) try writeTools(s, gpa, req.tools);
    if (req.tool_choice) |tc| {
        try s.objectField("tool_choice");
        switch (tc) {
            .mode => |mode| try s.write(mode),
            .named => |n| {
                try s.beginObject();
                try s.objectField("type");
                try s.write(n.type);
                try s.objectField("function");
                try s.beginObject();
                try s.objectField("name");
                try s.write(n.function.name);
                try s.endObject();
                try s.endObject();
            },
        }
    }
    if (req.response_format) |rf| {
        try s.objectField("response_format");
        try s.beginObject();
        try s.objectField("type");
        try s.write(rf.type);
        if (rf.json_schema) |js| try rawField(s, gpa, "json_schema", js);
        try s.endObject();
    }
    if (req.reasoning_effort) |v| {
        try s.objectField("reasoning_effort");
        try s.write(v);
    }
    // `extra` keys are merged at the top level: splice each member as a field.
    if (req.extra) |raw| try mergeExtra(s, gpa, raw);
}

/// Merge a raw-JSON object's members into the current (open) object, so the
/// caller's passthrough keys appear at the top level (SPEC.md).
fn mergeExtra(s: *Stringify, gpa: Allocator, raw: []const u8) !void {
    const parsed = std.json.parseFromSlice(Value, gpa, raw, .{}) catch return error.InvalidRawJson;
    defer parsed.deinit();
    switch (parsed.value) {
        .object => |obj| {
            var it = obj.iterator();
            while (it.next()) |entry| {
                try s.objectField(entry.key_ptr.*);
                try s.write(entry.value_ptr.*);
            }
        },
        else => return error.InvalidRawJson,
    }
}

/// Serialise a ChatRequest to its OpenAI/OpenRouter JSON body. Caller owns and
/// frees the returned slice (allocated with `gpa`).
pub fn encodeChatRequest(gpa: Allocator, req: gen.ChatRequest) ![]u8 {
    var aw: Writer.Allocating = .init(gpa);
    errdefer aw.deinit();
    var s: Stringify = .{ .writer = &aw.writer, .options = .{} };
    try s.beginObject();
    try writeChatBody(&s, gpa, req);
    try s.endObject();
    return aw.toOwnedSlice();
}

// ---------------------------------------------------------------------------
// Responses dialect encoding (POST /v1/responses)
// ---------------------------------------------------------------------------

/// One content part of a Responses message. The active union field's name is the
/// wire `"type"` token; an `output_text` part additionally emits `annotations:[]`.
fn writeResponseContentPart(s: *Stringify, part: gen.ResponseContentPart) !void {
    try s.beginObject();
    switch (part) {
        .input_text => |p| {
            try s.objectField("type");
            try s.write("input_text");
            try s.objectField("text");
            try s.write(p.text);
        },
        .input_image => |p| {
            try s.objectField("type");
            try s.write("input_image");
            try s.objectField("image_url");
            try s.write(p.image_url); // plain string, not a nested {url} object
            if (p.detail) |d| {
                try s.objectField("detail");
                try s.write(d);
            }
        },
        .output_text => |p| {
            try s.objectField("type");
            try s.write("output_text");
            try s.objectField("text");
            try s.write(p.text);
            try s.objectField("annotations");
            try s.beginArray();
            try s.endArray();
        },
    }
    try s.endObject();
}

fn writeResponseContent(s: *Stringify, content: gen.ResponseContent) !void {
    switch (content) {
        .text => |t| try s.write(t),
        .parts => |parts| {
            try s.beginArray();
            for (parts) |p| try writeResponseContentPart(s, p);
            try s.endArray();
        },
    }
}

/// Write a reasoning entry list (`summary` or `content`). `wire_type` is the
/// per-entry `"type"` token the enclosing list dictates ("summary_text" for a
/// `summary[]` entry, "reasoning_text" for a `content[]` entry).
fn writeReasoningTexts(s: *Stringify, key: []const u8, wire_type: []const u8, entries: []const gen.ResponseReasoningText) !void {
    try s.objectField(key);
    try s.beginArray();
    for (entries) |e| {
        try s.beginObject();
        try s.objectField("type");
        try s.write(wire_type);
        try s.objectField("text");
        try s.write(e.text);
        try s.endObject();
    }
    try s.endArray();
}

/// One `input`/`output` item. A message is emitted role-keyed WITHOUT a `"type"`
/// (a bare role-keyed object implies `"message"`); every other variant carries
/// its `"type"` discriminator.
fn writeResponseItem(s: *Stringify, item: gen.ResponseItem) !void {
    try s.beginObject();
    switch (item) {
        .message => |m| {
            if (m.id) |id| {
                try s.objectField("id");
                try s.write(id);
            }
            try s.objectField("role");
            try s.write(m.role);
            if (m.content) |c| {
                try s.objectField("content");
                try writeResponseContent(s, c);
            }
            if (m.status) |st| {
                try s.objectField("status");
                try s.write(st);
            }
        },
        .function_call => |fc| {
            try s.objectField("type");
            try s.write("function_call");
            if (fc.id) |id| {
                try s.objectField("id");
                try s.write(id);
            }
            try s.objectField("call_id");
            try s.write(fc.call_id);
            try s.objectField("name");
            try s.write(fc.name);
            try s.objectField("arguments");
            try s.write(fc.arguments); // raw JSON string, escaped
            if (fc.status) |st| {
                try s.objectField("status");
                try s.write(st);
            }
        },
        .function_call_output => |fo| {
            try s.objectField("type");
            try s.write("function_call_output");
            if (fo.id) |id| {
                try s.objectField("id");
                try s.write(id);
            }
            try s.objectField("call_id");
            try s.write(fo.call_id);
            try s.objectField("output");
            try s.write(fo.output);
        },
        .reasoning => |r| {
            try s.objectField("type");
            try s.write("reasoning");
            if (r.id) |id| {
                try s.objectField("id");
                try s.write(id);
            }
            try writeReasoningTexts(s, "summary", "summary_text", r.summary);
            try writeReasoningTexts(s, "content", "reasoning_text", r.content);
            if (r.encrypted_content) |ec| {
                try s.objectField("encrypted_content");
                try s.write(ec);
            }
        },
    }
    try s.endObject();
}

/// The Responses dialect's FLAT tool list (`type`/`name`/`parameters` at the top
/// level, no nested `function` object).
fn writeResponsesTools(s: *Stringify, gpa: Allocator, tools: []const gen.ResponsesToolDef) !void {
    try s.objectField("tools");
    try s.beginArray();
    for (tools) |t| {
        try s.beginObject();
        try s.objectField("type");
        try s.write(t.type);
        try s.objectField("name");
        try s.write(t.name);
        if (t.description) |d| {
            try s.objectField("description");
            try s.write(d);
        }
        if (t.parameters) |p| try rawField(s, gpa, "parameters", p);
        if (t.strict) |st| {
            try s.objectField("strict");
            try s.write(st);
        }
        try s.endObject();
    }
    try s.endArray();
}

/// Serialise a ResponsesRequest to its OpenAI Responses JSON body. Caller owns
/// and frees the returned slice (allocated with `gpa`).
pub fn encodeResponsesRequest(gpa: Allocator, req: gen.ResponsesRequest) ![]u8 {
    var aw: Writer.Allocating = .init(gpa);
    errdefer aw.deinit();
    var s: Stringify = .{ .writer = &aw.writer, .options = .{} };
    try s.beginObject();

    try s.objectField("model");
    try s.write(req.model);

    try s.objectField("input");
    switch (req.input) {
        .text => |t| try s.write(t), // bare string: one user message
        .items => |items| {
            try s.beginArray();
            for (items) |it| try writeResponseItem(&s, it);
            try s.endArray();
        },
    }

    if (req.instructions) |v| {
        try s.objectField("instructions");
        try s.write(v);
    }
    if (req.stream) |v| {
        try s.objectField("stream");
        try s.write(v);
    }
    if (req.temperature) |v| {
        try s.objectField("temperature");
        try s.write(v);
    }
    if (req.top_p) |v| {
        try s.objectField("top_p");
        try s.write(v);
    }
    if (req.max_output_tokens) |v| {
        try s.objectField("max_output_tokens");
        try s.write(v);
    }
    if (req.tools.len > 0) try writeResponsesTools(&s, gpa, req.tools);
    if (req.tool_choice) |tc| {
        try s.objectField("tool_choice");
        switch (tc) {
            .mode => |mode| try s.write(mode),
            .named => |n| {
                // FLAT named choice: {"type","name"} — no nested `function`.
                try s.beginObject();
                try s.objectField("type");
                try s.write(n.type);
                try s.objectField("name");
                try s.write(n.name);
                try s.endObject();
            },
        }
    }
    if (req.reasoning) |r| {
        try s.objectField("reasoning");
        try s.beginObject();
        if (r.effort) |e| {
            try s.objectField("effort");
            try s.write(e);
        }
        if (r.summary) |sm| {
            try s.objectField("summary");
            try s.write(sm);
        }
        try s.endObject();
    }
    if (req.store) |v| {
        try s.objectField("store");
        try s.write(v);
    }
    // `extra` keys are merged at the top level, exactly like ChatRequest.extra.
    if (req.extra) |raw| try mergeExtra(&s, gpa, raw);

    try s.endObject();
    return aw.toOwnedSlice();
}

pub fn encodeEmbeddingRequest(gpa: Allocator, req: gen.EmbeddingRequest) ![]u8 {
    var aw: Writer.Allocating = .init(gpa);
    errdefer aw.deinit();
    var s: Stringify = .{ .writer = &aw.writer, .options = .{} };
    try s.beginObject();
    try s.objectField("model");
    try s.write(req.model);
    try s.objectField("input");
    if (req.input.len == 1) {
        try s.write(req.input[0]); // string-or-array: single -> bare string
    } else {
        try s.beginArray();
        for (req.input) |x| try s.write(x);
        try s.endArray();
    }
    if (req.dimensions) |v| {
        try s.objectField("dimensions");
        try s.write(v);
    }
    if (req.encoding_format) |v| {
        try s.objectField("encoding_format");
        try s.write(v);
    }
    if (req.extra) |raw| try mergeExtra(&s, gpa, raw);
    try s.endObject();
    return aw.toOwnedSlice();
}

pub fn encodeSpeechRequest(gpa: Allocator, req: gen.SpeechRequest) ![]u8 {
    var aw: Writer.Allocating = .init(gpa);
    errdefer aw.deinit();
    var s: Stringify = .{ .writer = &aw.writer, .options = .{} };
    try s.beginObject();
    try s.objectField("model");
    try s.write(req.model);
    try s.objectField("input");
    try s.write(req.input);
    try s.objectField("voice");
    try s.write(req.voice);
    if (req.response_format) |v| {
        try s.objectField("response_format");
        try s.write(v);
    }
    if (req.speed) |v| {
        try s.objectField("speed");
        try s.write(v);
    }
    if (req.extra) |raw| try mergeExtra(&s, gpa, raw);
    try s.endObject();
    return aw.toOwnedSlice();
}

pub fn encodeBatchCreateRequest(gpa: Allocator, req: gen.BatchCreateRequest) ![]u8 {
    var aw: Writer.Allocating = .init(gpa);
    errdefer aw.deinit();
    var s: Stringify = .{ .writer = &aw.writer, .options = .{} };
    try s.beginObject();
    try s.objectField("requests");
    try s.beginArray();
    for (req.requests) |item| {
        try s.beginObject();
        try s.objectField("custom_id");
        try s.write(item.custom_id);
        try s.objectField("body");
        try s.beginObject();
        try writeChatBody(&s, gpa, item.body);
        try s.endObject();
        try s.endObject();
    }
    try s.endArray();
    try s.endObject();
    return aw.toOwnedSlice();
}

// ===========================================================================
// Decoding (OpenAI/OpenRouter JSON -> typed model), via std.json.Value.
// Everything allocated here uses `arena`; the caller frees the arena.
// ===========================================================================

fn getStr(obj: Value, key: []const u8) ?[]const u8 {
    const v = objGet(obj, key) orelse return null;
    return switch (v) {
        .string => |s| s,
        else => null,
    };
}

fn objGet(obj: Value, key: []const u8) ?Value {
    return switch (obj) {
        .object => |m| m.get(key),
        else => null,
    };
}

fn getInt(comptime T: type, obj: Value, key: []const u8) ?T {
    const v = objGet(obj, key) orelse return null;
    return switch (v) {
        .integer => |i| std.math.cast(T, i),
        .number_string => |ns| std.fmt.parseInt(T, ns, 10) catch null,
        else => null,
    };
}

fn getFloat(obj: Value, key: []const u8) ?f64 {
    const v = objGet(obj, key) orelse return null;
    return switch (v) {
        .float => |f| f,
        .integer => |i| @floatFromInt(i),
        .number_string => |ns| std.fmt.parseFloat(f64, ns) catch null,
        else => null,
    };
}

fn getBool(obj: Value, key: []const u8) ?bool {
    const v = objGet(obj, key) orelse return null;
    return switch (v) {
        .bool => |b| b,
        else => null,
    };
}

/// Re-serialise a sub-value back to a raw-JSON string owned by `arena` (for the
/// free-form fields captured on decode, e.g. `default_parameters`).
fn captureRaw(arena: Allocator, v: Value) ![]const u8 {
    return std.json.Stringify.valueAlloc(arena, v, .{});
}

fn dupStrArray(arena: Allocator, v: ?Value) ![]const []const u8 {
    const arr = switch (v orelse return &.{}) {
        .array => |a| a,
        else => return &.{},
    };
    var out = try arena.alloc([]const u8, arr.items.len);
    for (arr.items, 0..) |it, i| {
        out[i] = switch (it) {
            .string => |s| s,
            else => "",
        };
    }
    return out;
}

fn parseUsage(obj: Value, key: []const u8) ?gen.Usage {
    const u = objGet(obj, key) orelse return null;
    if (u != .object) return null;
    // Prompt-cache hit accounting; absent when the upstream reported no caching.
    var details: ?gen.PromptTokensDetails = null;
    if (objGet(u, "prompt_tokens_details")) |d| {
        if (d == .object) details = .{ .cached_tokens = getInt(u32, d, "cached_tokens") };
    }
    return gen.Usage{
        .prompt_tokens = getInt(u32, u, "prompt_tokens") orelse 0,
        .completion_tokens = getInt(u32, u, "completion_tokens") orelse 0,
        .total_tokens = getInt(u32, u, "total_tokens") orelse 0,
        .cost_usd = getFloat(u, "cost_usd"),
        .prompt_tokens_details = details,
        .cache_creation_tokens = getInt(u32, u, "cache_creation_tokens"),
    };
}

fn parseContent(arena: Allocator, v: Value) !?gen.Content {
    switch (v) {
        .string => |s| return gen.Content{ .text = s },
        .array => |arr| {
            var parts = try arena.alloc(gen.ContentPart, arr.items.len);
            var n: usize = 0;
            for (arr.items) |it| {
                if (it != .object) continue;
                const ty = getStr(it, "type") orelse continue;
                if (std.mem.eql(u8, ty, "text")) {
                    parts[n] = .{ .text = .{ .text = getStr(it, "text") orelse "" } };
                    n += 1;
                } else if (std.mem.eql(u8, ty, "image_url")) {
                    const iu = objGet(it, "image_url") orelse continue;
                    parts[n] = .{ .image_url = .{
                        .url = getStr(iu, "url") orelse "",
                        .detail = getStr(iu, "detail"),
                    } };
                    n += 1;
                }
            }
            return gen.Content{ .parts = parts[0..n] };
        },
        .null => return null,
        else => return null,
    }
}

fn parseToolCalls(arena: Allocator, v: ?Value) ![]const gen.ToolCall {
    const arr = switch (v orelse return &.{}) {
        .array => |a| a,
        else => return &.{},
    };
    var out = try arena.alloc(gen.ToolCall, arr.items.len);
    var n: usize = 0;
    for (arr.items) |it| {
        if (it != .object) continue;
        const fnobj = objGet(it, "function") orelse Value{ .null = {} };
        out[n] = .{
            .id = getStr(it, "id") orelse "",
            .type = getStr(it, "type") orelse "function",
            .function = .{
                .name = getStr(fnobj, "name") orelse "",
                .arguments = getStr(fnobj, "arguments") orelse "",
            },
        };
        n += 1;
    }
    return out[0..n];
}

/// Parse a `reasoning_details[]` array. Each entry's `type` discriminates open vs hidden;
/// the opaque `signature`/`data` are captured as-is (slices into the parsed `Value`, which
/// the caller's arena owns) so they replay verbatim. Non-object entries are skipped.
fn parseReasoningDetails(arena: Allocator, v: ?Value) ![]const gen.ReasoningDetail {
    const arr = switch (v orelse return &.{}) {
        .array => |a| a,
        else => return &.{},
    };
    var out = try arena.alloc(gen.ReasoningDetail, arr.items.len);
    var n: usize = 0;
    for (arr.items) |it| {
        if (it != .object) continue;
        out[n] = .{
            .type = getStr(it, "type") orelse "",
            .text = getStr(it, "text"),
            .summary = getStr(it, "summary"),
            .data = getStr(it, "data"),
            .signature = getStr(it, "signature"),
            .id = getStr(it, "id"),
            .format = getStr(it, "format"),
            .index = getInt(u32, it, "index"),
        };
        n += 1;
    }
    return out[0..n];
}

fn parseMessage(arena: Allocator, v: Value) !gen.ChatMessage {
    return gen.ChatMessage{
        .role = if (getStr(v, "role")) |r| (gen.enumFromWire(gen.Role, r) orelse .assistant) else .assistant,
        .content = if (objGet(v, "content")) |c| try parseContent(arena, c) else null,
        .name = getStr(v, "name"),
        .tool_calls = try parseToolCalls(arena, objGet(v, "tool_calls")),
        .tool_call_id = getStr(v, "tool_call_id"),
        .reasoning = getStr(v, "reasoning"),
        .reasoning_details = try parseReasoningDetails(arena, objGet(v, "reasoning_details")),
    };
}

fn parseFinish(v: Value, key: []const u8) ?gen.FinishReason {
    const s = getStr(v, key) orelse return null;
    return gen.enumFromWire(gen.FinishReason, s);
}

pub fn decodeChatResponse(arena: Allocator, root: Value) !gen.ChatResponse {
    const choices_v = objGet(root, "choices") orelse return error.MalformedResponse;
    const choices_arr = switch (choices_v) {
        .array => |a| a,
        else => return error.MalformedResponse,
    };
    var choices = try arena.alloc(gen.Choice, choices_arr.items.len);
    for (choices_arr.items, 0..) |c, i| {
        choices[i] = .{
            .index = getInt(u32, c, "index") orelse 0,
            .message = try parseMessage(arena, objGet(c, "message") orelse Value{ .null = {} }),
            .finish_reason = parseFinish(c, "finish_reason"),
        };
    }
    return gen.ChatResponse{
        .id = getStr(root, "id") orelse "",
        .object = getStr(root, "object") orelse "chat.completion",
        .created = getInt(i64, root, "created") orelse 0,
        .model = getStr(root, "model") orelse "",
        .choices = choices,
        .usage = parseUsage(root, "usage"),
    };
}

/// Decode one streaming chunk's already-parsed JSON `Value`.
pub fn decodeChunk(arena: Allocator, root: Value) !gen.ChatCompletionChunk {
    const choices_arr = switch (objGet(root, "choices") orelse Value{ .null = {} }) {
        .array => |a| a,
        else => return error.MalformedResponse,
    };
    var choices = try arena.alloc(gen.ChunkChoice, choices_arr.items.len);
    for (choices_arr.items, 0..) |c, i| {
        const delta_v = objGet(c, "delta") orelse Value{ .null = {} };
        var delta: gen.Delta = .{};
        if (delta_v == .object) {
            delta.role = if (getStr(delta_v, "role")) |r| gen.enumFromWire(gen.Role, r) else null;
            delta.content = getStr(delta_v, "content");
            delta.tool_calls = try parseToolCallDeltas(arena, objGet(delta_v, "tool_calls"));
            delta.reasoning = getStr(delta_v, "reasoning");
            delta.reasoning_details = try parseReasoningDetails(arena, objGet(delta_v, "reasoning_details"));
        }
        choices[i] = .{
            .index = getInt(u32, c, "index") orelse 0,
            .delta = delta,
            .finish_reason = parseFinish(c, "finish_reason"),
        };
    }
    return gen.ChatCompletionChunk{
        .id = getStr(root, "id") orelse "",
        .object = getStr(root, "object") orelse "chat.completion.chunk",
        .created = getInt(i64, root, "created") orelse 0,
        .model = getStr(root, "model") orelse "",
        .choices = choices,
        .usage = parseUsage(root, "usage"),
    };
}

fn parseToolCallDeltas(arena: Allocator, v: ?Value) ![]const gen.ToolCallDelta {
    const arr = switch (v orelse return &.{}) {
        .array => |a| a,
        else => return &.{},
    };
    var out = try arena.alloc(gen.ToolCallDelta, arr.items.len);
    for (arr.items, 0..) |it, i| {
        const fnobj = objGet(it, "function");
        var fnd: ?gen.FunctionCallDelta = null;
        if (fnobj) |fo| {
            if (fo == .object) fnd = .{ .name = getStr(fo, "name"), .arguments = getStr(fo, "arguments") };
        }
        out[i] = .{
            .index = getInt(u32, it, "index") orelse 0,
            .id = getStr(it, "id"),
            .type = getStr(it, "type"),
            .function = fnd,
        };
    }
    return out;
}

// ---------------------------------------------------------------------------
// Responses dialect decoding (POST /v1/responses)
// ---------------------------------------------------------------------------

/// One content part; the `"type"` token selects the variant. `input_image`'s
/// `image_url` is a plain string here. Unknown types yield `null` (skipped).
fn parseResponseContentPart(v: Value) ?gen.ResponseContentPart {
    if (v != .object) return null;
    const ty = getStr(v, "type") orelse return null;
    if (std.mem.eql(u8, ty, "input_text")) {
        return .{ .input_text = .{ .text = getStr(v, "text") orelse "" } };
    } else if (std.mem.eql(u8, ty, "output_text")) {
        return .{ .output_text = .{ .text = getStr(v, "text") orelse "" } };
    } else if (std.mem.eql(u8, ty, "input_image")) {
        return .{ .input_image = .{
            .image_url = getStr(v, "image_url") orelse "",
            .detail = getStr(v, "detail"),
        } };
    }
    return null;
}

fn parseResponseContent(arena: Allocator, v: Value) !?gen.ResponseContent {
    switch (v) {
        .string => |s| return gen.ResponseContent{ .text = s },
        .array => |arr| {
            var parts = try arena.alloc(gen.ResponseContentPart, arr.items.len);
            var n: usize = 0;
            for (arr.items) |it| {
                if (parseResponseContentPart(it)) |p| {
                    parts[n] = p;
                    n += 1;
                }
            }
            return gen.ResponseContent{ .parts = parts[0..n] };
        },
        .null => return null,
        else => return null,
    }
}

/// Parse a reasoning `summary[]`/`content[]` list — both are `[{...,"text"}]`;
/// only the `text` is retained (the entry's list, not its `type`, decides intent).
fn parseReasoningTexts(arena: Allocator, v: ?Value) ![]const gen.ResponseReasoningText {
    const arr = switch (v orelse return &.{}) {
        .array => |a| a,
        else => return &.{},
    };
    var out = try arena.alloc(gen.ResponseReasoningText, arr.items.len);
    var n: usize = 0;
    for (arr.items) |it| {
        if (getStr(it, "text")) |t| {
            out[n] = .{ .text = t };
            n += 1;
        }
    }
    return out[0..n];
}

fn parseResponseMessageItem(arena: Allocator, v: Value) !gen.ResponseMessageItem {
    return gen.ResponseMessageItem{
        .id = getStr(v, "id"),
        .role = getStr(v, "role") orelse "assistant",
        .content = if (objGet(v, "content")) |c| try parseResponseContent(arena, c) else null,
        .status = getStr(v, "status"),
    };
}

/// Parse one `input`/`output` item. A missing or `"message"` `type` is a message
/// (role-keyed). An unrecognised `type` yields `null` so the caller skips it.
fn parseResponseItem(arena: Allocator, v: Value) !?gen.ResponseItem {
    if (v != .object) return null;
    const ty = getStr(v, "type");
    if (ty == null or std.mem.eql(u8, ty.?, "message")) {
        return gen.ResponseItem{ .message = try parseResponseMessageItem(arena, v) };
    }
    const t = ty.?;
    if (std.mem.eql(u8, t, "function_call")) {
        return gen.ResponseItem{ .function_call = .{
            .id = getStr(v, "id"),
            .call_id = getStr(v, "call_id") orelse "",
            .name = getStr(v, "name") orelse "",
            .arguments = getStr(v, "arguments") orelse "",
            .status = getStr(v, "status"),
        } };
    } else if (std.mem.eql(u8, t, "function_call_output")) {
        return gen.ResponseItem{ .function_call_output = .{
            .id = getStr(v, "id"),
            .call_id = getStr(v, "call_id") orelse "",
            .output = getStr(v, "output") orelse "",
        } };
    } else if (std.mem.eql(u8, t, "reasoning")) {
        return gen.ResponseItem{ .reasoning = .{
            .id = getStr(v, "id"),
            .summary = try parseReasoningTexts(arena, objGet(v, "summary")),
            .content = try parseReasoningTexts(arena, objGet(v, "content")),
            .encrypted_content = getStr(v, "encrypted_content"),
        } };
    }
    return null; // unknown item type — ignore
}

fn parseResponseOutput(arena: Allocator, v: ?Value) ![]const gen.ResponseItem {
    const arr = switch (v orelse return &.{}) {
        .array => |a| a,
        else => return &.{},
    };
    var out = try arena.alloc(gen.ResponseItem, arr.items.len);
    var n: usize = 0;
    for (arr.items) |it| {
        if (try parseResponseItem(arena, it)) |item| {
            out[n] = item;
            n += 1;
        }
    }
    return out[0..n];
}

fn parseResponsesUsage(obj: Value, key: []const u8) ?gen.ResponsesUsage {
    const u = objGet(obj, key) orelse return null;
    if (u != .object) return null;
    var in_details: ?gen.ResponsesInputTokensDetails = null;
    if (objGet(u, "input_tokens_details")) |d| {
        if (d == .object) in_details = .{ .cached_tokens = getInt(u32, d, "cached_tokens") };
    }
    var out_details: ?gen.ResponsesOutputTokensDetails = null;
    if (objGet(u, "output_tokens_details")) |d| {
        if (d == .object) out_details = .{ .reasoning_tokens = getInt(u32, d, "reasoning_tokens") };
    }
    return gen.ResponsesUsage{
        .input_tokens = getInt(u32, u, "input_tokens") orelse 0,
        .input_tokens_details = in_details,
        .output_tokens = getInt(u32, u, "output_tokens") orelse 0,
        .output_tokens_details = out_details,
        .total_tokens = getInt(u32, u, "total_tokens") orelse 0,
    };
}

fn parseErrorBody(v: Value) ?gen.ErrorBody {
    if (v != .object) return null;
    const msg = getStr(v, "message") orelse return null;
    return gen.ErrorBody{ .message = msg, .type = getStr(v, "type"), .code = getStr(v, "code") };
}

fn parseResponsesReasoning(v: Value) ?gen.ResponsesReasoning {
    if (v != .object) return null;
    return gen.ResponsesReasoning{ .effort = getStr(v, "effort"), .summary = getStr(v, "summary") };
}

pub fn decodeResponsesResponse(arena: Allocator, root: Value) !gen.ResponsesResponse {
    var incomplete: ?gen.ResponsesIncompleteDetails = null;
    if (objGet(root, "incomplete_details")) |d| {
        if (d == .object) {
            if (getStr(d, "reason")) |r| incomplete = .{ .reason = r };
        }
    }
    var err: ?gen.ErrorBody = null;
    if (objGet(root, "error")) |e| err = parseErrorBody(e);
    var reasoning: ?gen.ResponsesReasoning = null;
    if (objGet(root, "reasoning")) |r| reasoning = parseResponsesReasoning(r);
    return gen.ResponsesResponse{
        .id = getStr(root, "id") orelse "",
        .object = getStr(root, "object") orelse "response",
        .created_at = getInt(i64, root, "created_at") orelse 0,
        .status = getStr(root, "status") orelse "",
        .incomplete_details = incomplete,
        .@"error" = err,
        .model = getStr(root, "model") orelse "",
        .output = try parseResponseOutput(arena, objGet(root, "output")),
        .usage = parseResponsesUsage(root, "usage"),
        .store = getBool(root, "store"),
        .instructions = getStr(root, "instructions"),
        .max_output_tokens = getInt(u32, root, "max_output_tokens"),
        .temperature = if (getFloat(root, "temperature")) |f| @floatCast(f) else null,
        .top_p = if (getFloat(root, "top_p")) |f| @floatCast(f) else null,
        .reasoning = reasoning,
    };
}

/// Decode one streaming SSE event's already-parsed JSON `Value`. Every event
/// type decodes into the same flat superset; `type` says which fields matter.
pub fn decodeResponsesStreamEvent(arena: Allocator, root: Value) !gen.ResponsesStreamEvent {
    var response: ?gen.ResponsesResponse = null;
    if (objGet(root, "response")) |r| {
        if (r == .object) response = try decodeResponsesResponse(arena, r);
    }
    var item: ?gen.ResponseItem = null;
    if (objGet(root, "item")) |it| item = try parseResponseItem(arena, it);
    var part: ?gen.ResponseContentPart = null;
    if (objGet(root, "part")) |p| part = parseResponseContentPart(p);
    return gen.ResponsesStreamEvent{
        .type = getStr(root, "type") orelse "",
        .sequence_number = getInt(u64, root, "sequence_number") orelse 0,
        .response = response,
        .output_index = getInt(u32, root, "output_index"),
        .item_id = getStr(root, "item_id"),
        .content_index = getInt(u32, root, "content_index"),
        .item = item,
        .part = part,
        .delta = getStr(root, "delta"),
        .text = getStr(root, "text"),
        .arguments = getStr(root, "arguments"),
        .message = getStr(root, "message"),
    };
}

pub fn decodeEmbeddingResponse(arena: Allocator, root: Value) !gen.EmbeddingResponse {
    const data_arr = switch (objGet(root, "data") orelse Value{ .null = {} }) {
        .array => |a| a,
        else => return error.MalformedResponse,
    };
    var data = try arena.alloc(gen.Embedding, data_arr.items.len);
    for (data_arr.items, 0..) |d, i| {
        const emb_v = objGet(d, "embedding") orelse Value{ .null = {} };
        const floats = try decodeEmbeddingVector(arena, emb_v);
        data[i] = .{
            .object = getStr(d, "object") orelse "embedding",
            .index = getInt(u32, d, "index") orelse 0,
            .embedding = floats,
        };
    }
    return gen.EmbeddingResponse{
        .object = getStr(root, "object") orelse "list",
        .data = data,
        .model = getStr(root, "model") orelse "",
        .usage = parseUsage(root, "usage"),
    };
}

/// An embedding is either a float array (`encoding_format:"float"`) or a base64
/// string of little-endian f32 bytes (`encoding_format:"base64"`). Decode both
/// to `[]f32`.
fn decodeEmbeddingVector(arena: Allocator, v: Value) ![]const f32 {
    switch (v) {
        .array => |a| {
            var out = try arena.alloc(f32, a.items.len);
            for (a.items, 0..) |it, i| {
                out[i] = switch (it) {
                    .float => |f| @floatCast(f),
                    .integer => |n| @floatFromInt(n),
                    .number_string => |ns| std.fmt.parseFloat(f32, ns) catch 0,
                    else => 0,
                };
            }
            return out;
        },
        .string => |b64| return decodeBase64Floats(arena, b64),
        else => return &.{},
    }
}

/// base64 -> little-endian f32 vector.
pub fn decodeBase64Floats(arena: Allocator, b64: []const u8) ![]const f32 {
    const dec = std.base64.standard.Decoder;
    const n = try dec.calcSizeForSlice(b64);
    const bytes = try arena.alloc(u8, n);
    try dec.decode(bytes, b64);
    if (bytes.len % 4 != 0) return error.MalformedResponse;
    const count = bytes.len / 4;
    var out = try arena.alloc(f32, count);
    var i: usize = 0;
    while (i < count) : (i += 1) {
        const word = std.mem.readInt(u32, bytes[i * 4 ..][0..4], .little);
        out[i] = @bitCast(word);
    }
    return out;
}

pub fn decodeVoicesResponse(arena: Allocator, root: Value) !gen.VoicesResponse {
    const arr = switch (objGet(root, "voices") orelse Value{ .null = {} }) {
        .array => |a| a,
        else => return error.MalformedResponse,
    };
    var voices = try arena.alloc(gen.Voice, arr.items.len);
    for (arr.items, 0..) |it, i| {
        voices[i] = .{
            .id = getStr(it, "id") orelse "",
            .name = getStr(it, "name"),
            .languages = try dupStrArray(arena, objGet(it, "languages")),
        };
    }
    return gen.VoicesResponse{
        .model = getStr(root, "model") orelse "",
        .voices = voices,
    };
}

pub fn decodeTranscription(root: Value) gen.TranscriptionResponse {
    return gen.TranscriptionResponse{
        .text = getStr(root, "text") orelse "",
        .task = getStr(root, "task"),
        .language = getStr(root, "language"),
        .duration = if (getFloat(root, "duration")) |f| @floatCast(f) else null,
        .usage = parseUsage(root, "usage"),
    };
}

fn decodeModelEntry(arena: Allocator, v: Value) !gen.ModelEntry {
    var arch: ?gen.Architecture = null;
    if (objGet(v, "architecture")) |a| {
        if (a == .object) arch = .{
            .input_modalities = try dupStrArray(arena, objGet(a, "input_modalities")),
            .output_modalities = try dupStrArray(arena, objGet(a, "output_modalities")),
            .modality = getStr(a, "modality"),
            .tokenizer = getStr(a, "tokenizer") orelse "",
            .instruct_type = getStr(a, "instruct_type"),
        };
    }
    var pricing: ?gen.Pricing = null;
    if (objGet(v, "pricing")) |p| {
        if (p == .object) pricing = .{
            .prompt = getStr(p, "prompt") orelse "",
            .completion = getStr(p, "completion") orelse "",
        };
    }
    var top: ?gen.TopProvider = null;
    if (objGet(v, "top_provider")) |tp| {
        if (tp == .object) top = .{
            .context_length = getInt(u32, tp, "context_length"),
            .max_completion_tokens = getInt(u32, tp, "max_completion_tokens"),
            .is_moderated = getBool(tp, "is_moderated") orelse false,
            .max_thinking_tokens = getInt(u32, tp, "max_thinking_tokens"),
        };
    }
    var endpoints: []const gen.ModelEndpoint = &.{};
    if (objGet(v, "endpoints")) |e| {
        if (e == .array) {
            var out = try arena.alloc(gen.ModelEndpoint, e.array.items.len);
            for (e.array.items, 0..) |it, i| {
                out[i] = .{
                    .provider = getStr(it, "provider") orelse "",
                    .model = getStr(it, "model") orelse "",
                    .down = getBool(it, "down") orelse false,
                    .source = getStr(it, "source") orelse "",
                };
            }
            endpoints = out;
        }
    }
    var default_params: ?[]const u8 = null;
    if (objGet(v, "default_parameters")) |dp| {
        if (dp != .null) default_params = try captureRaw(arena, dp);
    }
    return gen.ModelEntry{
        .id = getStr(v, "id") orelse "",
        .canonical_slug = getStr(v, "canonical_slug") orelse "",
        .name = getStr(v, "name") orelse "",
        .created = getInt(i64, v, "created") orelse 0,
        .description = getStr(v, "description") orelse "",
        .context_length = getInt(u32, v, "context_length"),
        .architecture = arch,
        .pricing = pricing,
        .top_provider = top,
        .supported_parameters = try dupStrArray(arena, objGet(v, "supported_parameters")),
        .unsupported_parameters = try dupStrArray(arena, objGet(v, "unsupported_parameters")),
        .default_parameters = default_params,
        .endpoints = endpoints,
    };
}

pub fn decodeListModels(arena: Allocator, root: Value) !gen.ListModelsResponse {
    const arr = switch (objGet(root, "data") orelse Value{ .null = {} }) {
        .array => |a| a,
        else => return error.MalformedResponse,
    };
    var data = try arena.alloc(gen.ModelEntry, arr.items.len);
    for (arr.items, 0..) |it, i| data[i] = try decodeModelEntry(arena, it);
    return gen.ListModelsResponse{ .data = data };
}

fn parseStatus(v: Value, key: []const u8) ?gen.BatchStatus {
    const s = getStr(v, key) orelse return null;
    return gen.enumFromWire(gen.BatchStatus, s);
}

pub fn decodeBatchHandle(root: Value) gen.BatchHandle {
    var counts: gen.BatchCounts = .{};
    if (objGet(root, "counts")) |c| {
        if (c == .object) counts = .{
            .total = getInt(u64, c, "total") orelse 0,
            .processing = getInt(u64, c, "processing") orelse 0,
            .succeeded = getInt(u64, c, "succeeded") orelse 0,
            .errored = getInt(u64, c, "errored") orelse 0,
            .canceled = getInt(u64, c, "canceled") orelse 0,
            .expired = getInt(u64, c, "expired") orelse 0,
        };
    }
    return gen.BatchHandle{
        .id = getStr(root, "id") orelse "",
        .status = parseStatus(root, "status"),
        .counts = counts,
        .created_at = getInt(i64, root, "created_at"),
        .expires_at = getInt(i64, root, "expires_at"),
        .ended_at = getInt(i64, root, "ended_at"),
        .endpoint = getStr(root, "endpoint"),
    };
}

pub fn decodeBatchResultLine(arena: Allocator, root: Value) !gen.BatchResultLine {
    var response: ?gen.BatchResponse = null;
    if (objGet(root, "response")) |r| {
        if (r == .object) {
            const body_v = objGet(r, "body") orelse Value{ .null = {} };
            response = .{
                .status_code = getInt(u32, r, "status_code") orelse 0,
                .body = try decodeChatResponse(arena, body_v),
            };
        }
    }
    var err: ?gen.BatchError = null;
    if (objGet(root, "error")) |e| {
        if (e == .object) err = .{
            .code = getStr(e, "code") orelse "",
            .message = getStr(e, "message") orelse "",
        };
    }
    return gen.BatchResultLine{
        .custom_id = getStr(root, "custom_id") orelse "",
        .response = response,
        .@"error" = err,
    };
}

/// Parse the error envelope `{"error":{"message":...}}`. Returns the message
/// (a slice into `root`'s arena) or `null` if the body wasn't that shape.
pub fn decodeErrorMessage(root: Value) ?[]const u8 {
    const e = objGet(root, "error") orelse return null;
    return getStr(e, "message");
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;

test "encode minimal chat request" {
    const req = gen.ChatRequest{
        .model = "gpt-4o-mini",
        .messages = &.{gen.ChatMessage.textMsg(.user, "hi")},
    };
    const body = try encodeChatRequest(testing.allocator, req);
    defer testing.allocator.free(body);
    try testing.expectEqualStrings(
        \\{"model":"gpt-4o-mini","messages":[{"role":"user","content":"hi"}]}
    , body);
}

test "encode chat request prefers max_completion_tokens and merges extra" {
    const req = gen.ChatRequest{
        .model = "m",
        .messages = &.{gen.ChatMessage.textMsg(.system, "be terse")},
        .max_tokens = 10,
        .max_completion_tokens = 20,
        .stop = &.{ "a", "b" },
        .extra = "{\"provider\":{\"order\":[\"x\"]}}",
    };
    const body = try encodeChatRequest(testing.allocator, req);
    defer testing.allocator.free(body);
    try testing.expect(std.mem.indexOf(u8, body, "\"max_completion_tokens\":20") != null);
    try testing.expect(std.mem.indexOf(u8, body, "max_tokens") == null);
    try testing.expect(std.mem.indexOf(u8, body, "\"stop\":[\"a\",\"b\"]") != null);
    // `extra` is spliced (not stringified) and merged at top level.
    try testing.expect(std.mem.indexOf(u8, body, "\"provider\":{\"order\":[\"x\"]}") != null);
}

test "single stop emits bare string" {
    const req = gen.ChatRequest{
        .model = "m",
        .messages = &.{gen.ChatMessage.textMsg(.user, "x")},
        .stop = &.{"END"},
    };
    const body = try encodeChatRequest(testing.allocator, req);
    defer testing.allocator.free(body);
    try testing.expect(std.mem.indexOf(u8, body, "\"stop\":\"END\"") != null);
}

test "tool_choice mode and named" {
    {
        const req = gen.ChatRequest{
            .model = "m",
            .messages = &.{gen.ChatMessage.textMsg(.user, "x")},
            .tool_choice = .{ .mode = "auto" },
        };
        const body = try encodeChatRequest(testing.allocator, req);
        defer testing.allocator.free(body);
        try testing.expect(std.mem.indexOf(u8, body, "\"tool_choice\":\"auto\"") != null);
    }
    {
        const req = gen.ChatRequest{
            .model = "m",
            .messages = &.{gen.ChatMessage.textMsg(.user, "x")},
            .tool_choice = .{ .named = .{ .function = .{ .name = "get_weather" } } },
        };
        const body = try encodeChatRequest(testing.allocator, req);
        defer testing.allocator.free(body);
        try testing.expect(std.mem.indexOf(u8, body, "\"tool_choice\":{\"type\":\"function\",\"function\":{\"name\":\"get_weather\"}}") != null);
    }
}

test "embedding request single input is bare string" {
    const req = gen.EmbeddingRequest{ .model = "e", .input = &.{"hello"} };
    const body = try encodeEmbeddingRequest(testing.allocator, req);
    defer testing.allocator.free(body);
    try testing.expectEqualStrings("{\"model\":\"e\",\"input\":\"hello\"}", body);
}

test "decode chat response" {
    const json =
        \\{"id":"x","object":"chat.completion","created":1,"model":"m",
        \\ "choices":[{"index":0,"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}],
        \\ "usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3,"cost_usd":0.001}}
    ;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const parsed = try std.json.parseFromSlice(Value, a, json, .{});
    const resp = try decodeChatResponse(a, parsed.value);
    try testing.expectEqualStrings("x", resp.id);
    try testing.expectEqual(@as(usize, 1), resp.choices.len);
    try testing.expectEqualStrings("hello", resp.choices[0].message.content.?.text);
    try testing.expectEqual(gen.FinishReason.stop, resp.choices[0].finish_reason.?);
    try testing.expectEqual(@as(u32, 3), resp.usage.?.total_tokens);
    try testing.expectEqual(@as(f64, 0.001), resp.usage.?.cost_usd.?);
}

test "decode base64 embedding" {
    // Two f32s: 1.0 and 2.0, little-endian, base64-encoded.
    var bytes: [8]u8 = undefined;
    std.mem.writeInt(u32, bytes[0..4], @bitCast(@as(f32, 1.0)), .little);
    std.mem.writeInt(u32, bytes[4..8], @bitCast(@as(f32, 2.0)), .little);
    var b64buf: [16]u8 = undefined;
    const b64 = std.base64.standard.Encoder.encode(&b64buf, &bytes);
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const floats = try decodeBase64Floats(arena.allocator(), b64);
    try testing.expectEqual(@as(usize, 2), floats.len);
    try testing.expectEqual(@as(f32, 1.0), floats[0]);
    try testing.expectEqual(@as(f32, 2.0), floats[1]);
}

test "decode error envelope" {
    const json = "{\"error\":{\"message\":\"model not allowed\"}}";
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const parsed = try std.json.parseFromSlice(Value, arena.allocator(), json, .{});
    try testing.expectEqualStrings("model not allowed", decodeErrorMessage(parsed.value).?);
}

test "decode streaming chunk" {
    const json =
        \\{"id":"c1","object":"chat.completion.chunk","created":2,"model":"m",
        \\ "choices":[{"index":0,"delta":{"role":"assistant","content":"Hel"},"finish_reason":null}]}
    ;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const parsed = try std.json.parseFromSliceLeaky(Value, a, json, .{});
    const chunk = try decodeChunk(a, parsed);
    try testing.expectEqualStrings("c1", chunk.id);
    try testing.expectEqual(@as(usize, 1), chunk.choices.len);
    try testing.expectEqual(gen.Role.assistant, chunk.choices[0].delta.role.?);
    try testing.expectEqualStrings("Hel", chunk.choices[0].delta.content.?);
}

test "encode tools with raw parameters" {
    const req = gen.ChatRequest{
        .model = "m",
        .messages = &.{gen.ChatMessage.textMsg(.user, "weather?")},
        .tools = &.{.{ .function = .{
            .name = "get_weather",
            .description = "Get weather",
            .parameters = "{\"type\":\"object\",\"properties\":{}}",
        } }},
    };
    const body = try encodeChatRequest(testing.allocator, req);
    defer testing.allocator.free(body);
    // parameters spliced verbatim (object, not a JSON string).
    try testing.expect(std.mem.indexOf(u8, body, "\"parameters\":{\"type\":\"object\",\"properties\":{}}") != null);
    try testing.expect(std.mem.indexOf(u8, body, "\"name\":\"get_weather\"") != null);
}

test "decode batch result line (success and error)" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    {
        const json =
            \\{"custom_id":"req-1","response":{"status_code":200,"body":
            \\ {"id":"x","object":"chat.completion","model":"m","choices":[
            \\ {"index":0,"message":{"role":"assistant","content":"ok"}}]}}}
        ;
        const parsed = try std.json.parseFromSliceLeaky(Value, a, json, .{});
        const line = try decodeBatchResultLine(a, parsed);
        try testing.expectEqualStrings("req-1", line.custom_id);
        try testing.expectEqual(@as(u32, 200), line.response.?.status_code);
        try testing.expectEqualStrings("ok", line.response.?.body.choices[0].message.content.?.text);
        try testing.expect(line.@"error" == null);
    }
    {
        const json =
            \\{"custom_id":"req-2","error":{"code":"bad","message":"boom"}}
        ;
        const parsed = try std.json.parseFromSliceLeaky(Value, a, json, .{});
        const line = try decodeBatchResultLine(a, parsed);
        try testing.expect(line.response == null);
        try testing.expectEqualStrings("boom", line.@"error".?.message);
    }
}

test "encode message echoes reasoning and reasoning_details" {
    const details = [_]gen.ReasoningDetail{
        .{ .type = "reasoning.text", .text = "let me think", .signature = "sig123" },
        .{ .type = "reasoning.encrypted", .data = "OPAQUE==" },
    };
    const req = gen.ChatRequest{
        .model = "m",
        .messages = &.{.{
            .role = .assistant,
            .content = .{ .text = "answer" },
            .reasoning = "thinking out loud",
            .reasoning_details = &details,
        }},
    };
    const body = try encodeChatRequest(testing.allocator, req);
    defer testing.allocator.free(body);
    try testing.expect(std.mem.indexOf(u8, body, "\"reasoning\":\"thinking out loud\"") != null);
    try testing.expect(std.mem.indexOf(u8, body, "\"type\":\"reasoning.text\"") != null);
    try testing.expect(std.mem.indexOf(u8, body, "\"text\":\"let me think\"") != null);
    try testing.expect(std.mem.indexOf(u8, body, "\"signature\":\"sig123\"") != null);
    // hidden block: opaque data round-tripped verbatim.
    try testing.expect(std.mem.indexOf(u8, body, "\"type\":\"reasoning.encrypted\"") != null);
    try testing.expect(std.mem.indexOf(u8, body, "\"data\":\"OPAQUE==\"") != null);
}

test "decode chat response with reasoning and usage cache metadata" {
    const json =
        \\{"id":"x","object":"chat.completion","created":1,"model":"m",
        \\ "choices":[{"index":0,"message":{"role":"assistant","content":"hi",
        \\   "reasoning":"because",
        \\   "reasoning_details":[
        \\     {"type":"reasoning.text","text":"step 1","signature":"sig","index":0},
        \\     {"type":"reasoning.encrypted","data":"BLOB=="}]},
        \\   "finish_reason":"stop"}],
        \\ "usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12,
        \\   "prompt_tokens_details":{"cached_tokens":7},"cache_creation_tokens":3}}
    ;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const parsed = try std.json.parseFromSliceLeaky(Value, a, json, .{});
    const resp = try decodeChatResponse(a, parsed);
    const msg = resp.choices[0].message;
    try testing.expectEqualStrings("because", msg.reasoning.?);
    try testing.expectEqual(@as(usize, 2), msg.reasoning_details.len);
    try testing.expectEqualStrings("step 1", msg.reasoning_details[0].text.?);
    try testing.expectEqualStrings("sig", msg.reasoning_details[0].signature.?);
    try testing.expectEqual(@as(u32, 0), msg.reasoning_details[0].index.?);
    try testing.expect(!msg.reasoning_details[0].isHidden());
    try testing.expectEqualStrings("BLOB==", msg.reasoning_details[1].data.?);
    try testing.expect(msg.reasoning_details[1].isHidden());
    // usage cache metadata
    try testing.expectEqual(@as(u32, 7), resp.usage.?.cachedTokens());
    try testing.expectEqual(@as(u32, 3), resp.usage.?.cacheWrites());
}

test "decode usage without cache metadata leaves fields absent" {
    const json =
        \\{"id":"x","object":"chat.completion","created":1,"model":"m",
        \\ "choices":[{"index":0,"message":{"role":"assistant","content":"hi"}}],
        \\ "usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}
    ;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const parsed = try std.json.parseFromSliceLeaky(Value, a, json, .{});
    const resp = try decodeChatResponse(a, parsed);
    try testing.expect(resp.usage.?.prompt_tokens_details == null);
    try testing.expect(resp.usage.?.cache_creation_tokens == null);
    try testing.expectEqual(@as(u32, 0), resp.usage.?.cachedTokens());
    try testing.expectEqual(@as(u32, 0), resp.usage.?.cacheWrites());
}

test "decode streaming chunk with reasoning delta" {
    const json =
        \\{"id":"c1","object":"chat.completion.chunk","created":2,"model":"m",
        \\ "choices":[{"index":0,"delta":{"reasoning":"hmm",
        \\   "reasoning_details":[{"type":"reasoning.text","text":"a"}]},"finish_reason":null}]}
    ;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const parsed = try std.json.parseFromSliceLeaky(Value, a, json, .{});
    const chunk = try decodeChunk(a, parsed);
    try testing.expectEqualStrings("hmm", chunk.choices[0].delta.reasoning.?);
    try testing.expectEqual(@as(usize, 1), chunk.choices[0].delta.reasoning_details.len);
    try testing.expectEqualStrings("a", chunk.choices[0].delta.reasoning_details[0].text.?);
}

// --- Responses dialect (POST /v1/responses) --------------------------------

test "encode responses request with bare string input" {
    const req = gen.ResponsesRequest{
        .model = "gpt-5",
        .input = .{ .text = "hi" },
    };
    const body = try encodeResponsesRequest(testing.allocator, req);
    defer testing.allocator.free(body);
    try testing.expectEqualStrings(
        \\{"model":"gpt-5","input":"hi"}
    , body);
}

test "encode responses request flat tools, flat tool_choice, reasoning, store, extra" {
    const req = gen.ResponsesRequest{
        .model = "gpt-5",
        .input = .{ .text = "weather?" },
        .max_output_tokens = 256,
        .tools = &.{.{
            .name = "get_weather",
            .description = "Get weather",
            .parameters = "{\"type\":\"object\"}",
            .strict = false,
        }},
        .tool_choice = .{ .named = .{ .name = "get_weather" } },
        .reasoning = .{ .effort = "low", .summary = "auto" },
        .store = false,
        .extra = "{\"service_tier\":\"flex\"}",
    };
    const body = try encodeResponsesRequest(testing.allocator, req);
    defer testing.allocator.free(body);
    // FLAT tool: type/name/parameters at the top level, no nested "function".
    try testing.expect(std.mem.indexOf(u8, body, "\"tools\":[{\"type\":\"function\",\"name\":\"get_weather\",\"description\":\"Get weather\",\"parameters\":{\"type\":\"object\"},\"strict\":false}]") != null);
    // FLAT named tool_choice: {"type","name"} — no nested "function".
    try testing.expect(std.mem.indexOf(u8, body, "\"tool_choice\":{\"type\":\"function\",\"name\":\"get_weather\"}") != null);
    try testing.expect(std.mem.indexOf(u8, body, "\"reasoning\":{\"effort\":\"low\",\"summary\":\"auto\"}") != null);
    try testing.expect(std.mem.indexOf(u8, body, "\"max_output_tokens\":256") != null);
    try testing.expect(std.mem.indexOf(u8, body, "\"store\":false") != null);
    // `extra` spliced (not stringified) and merged at the top level.
    try testing.expect(std.mem.indexOf(u8, body, "\"service_tier\":\"flex\"") != null);
}

test "encode responses request item array replay (message, reasoning, tool call, output)" {
    const in_parts = [_]gen.ResponseContentPart{
        .{ .input_text = .{ .text = "look:" } },
        .{ .input_image = .{ .image_url = "https://x/y.png", .detail = "low" } },
    };
    const summary = [_]gen.ResponseReasoningText{.{ .text = "planning" }};
    const rcontent = [_]gen.ResponseReasoningText{.{ .text = "step 1" }};
    const out_parts = [_]gen.ResponseContentPart{.{ .output_text = .{ .text = "Sunny" } }};
    const items = [_]gen.ResponseItem{
        .{ .message = .{ .role = "user", .content = .{ .parts = &in_parts } } },
        .{ .reasoning = .{ .summary = &summary, .content = &rcontent, .encrypted_content = "OPAQUE==" } },
        .{ .function_call = .{ .call_id = "call_1", .name = "get_weather", .arguments = "{\"city\":\"SF\"}" } },
        .{ .function_call_output = .{ .call_id = "call_1", .output = "72F" } },
        .{ .message = .{ .role = "assistant", .content = .{ .parts = &out_parts } } },
    };
    const req = gen.ResponsesRequest{ .model = "gpt-5", .input = .{ .items = &items } };
    const body = try encodeResponsesRequest(testing.allocator, req);
    defer testing.allocator.free(body);
    // A message serialises role-keyed with NO "type" discriminator.
    try testing.expect(std.mem.indexOf(u8, body, "{\"role\":\"user\",\"content\":[") != null);
    try testing.expect(std.mem.indexOf(u8, body, "\"type\":\"message\"") == null);
    // input_image.image_url is a PLAIN string (not the chat dialect's nested {url}).
    try testing.expect(std.mem.indexOf(u8, body, "{\"type\":\"input_image\",\"image_url\":\"https://x/y.png\",\"detail\":\"low\"}") != null);
    // Reasoning entries take their wire token from the list they live in.
    try testing.expect(std.mem.indexOf(u8, body, "\"summary\":[{\"type\":\"summary_text\",\"text\":\"planning\"}]") != null);
    try testing.expect(std.mem.indexOf(u8, body, "\"content\":[{\"type\":\"reasoning_text\",\"text\":\"step 1\"}]") != null);
    try testing.expect(std.mem.indexOf(u8, body, "\"encrypted_content\":\"OPAQUE==\"") != null);
    // function_call / function_call_output carry their "type"; arguments stays a raw JSON string.
    try testing.expect(std.mem.indexOf(u8, body, "{\"type\":\"function_call\",\"call_id\":\"call_1\",\"name\":\"get_weather\",\"arguments\":\"{\\\"city\\\":\\\"SF\\\"}\"}") != null);
    try testing.expect(std.mem.indexOf(u8, body, "{\"type\":\"function_call_output\",\"call_id\":\"call_1\",\"output\":\"72F\"}") != null);
    // A constructed output_text part emits annotations:[].
    try testing.expect(std.mem.indexOf(u8, body, "{\"type\":\"output_text\",\"text\":\"Sunny\",\"annotations\":[]}") != null);
}

test "decode responses response (usage cached/reasoning tokens, store:false, output items)" {
    const json =
        \\{"id":"resp_1","object":"response","created_at":1720000000,"status":"completed",
        \\ "model":"gpt-5","store":false,
        \\ "output":[
        \\   {"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text","text":"thinking"}],
        \\     "content":[],"encrypted_content":"BLOB=="},
        \\   {"type":"function_call","id":"fc_1","call_id":"call_1","name":"get_weather",
        \\     "arguments":"{\"city\":\"SF\"}","status":"completed"},
        \\   {"type":"message","id":"msg_1","role":"assistant","status":"completed",
        \\     "content":[{"type":"output_text","text":"Sunny","annotations":[]}]}],
        \\ "usage":{"input_tokens":20,"input_tokens_details":{"cached_tokens":8},
        \\   "output_tokens":5,"output_tokens_details":{"reasoning_tokens":3},"total_tokens":25}}
    ;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const parsed = try std.json.parseFromSliceLeaky(Value, a, json, .{});
    const resp = try decodeResponsesResponse(a, parsed);
    try testing.expectEqualStrings("resp_1", resp.id);
    try testing.expectEqualStrings("response", resp.object);
    try testing.expectEqualStrings("completed", resp.status);
    try testing.expectEqual(@as(i64, 1720000000), resp.created_at);
    try testing.expectEqual(false, resp.store.?);
    try testing.expectEqual(@as(usize, 3), resp.output.len);
    // reasoning item (summary_text entry retained as text)
    try testing.expectEqualStrings("thinking", resp.output[0].reasoning.summary[0].text);
    try testing.expectEqualStrings("BLOB==", resp.output[0].reasoning.encrypted_content.?);
    // function_call item
    try testing.expectEqualStrings("get_weather", resp.output[1].function_call.name);
    try testing.expectEqualStrings("call_1", resp.output[1].function_call.call_id);
    try testing.expectEqualStrings("{\"city\":\"SF\"}", resp.output[1].function_call.arguments);
    // message item with an output_text part
    try testing.expectEqualStrings("assistant", resp.output[2].message.role);
    try testing.expectEqualStrings("Sunny", resp.output[2].message.content.?.parts[0].output_text.text);
    // usage in the dialect's own names + cached/reasoning helpers
    try testing.expectEqual(@as(u32, 25), resp.usage.?.total_tokens);
    try testing.expectEqual(@as(u32, 8), resp.usage.?.cachedTokens());
    try testing.expectEqual(@as(u32, 3), resp.usage.?.reasoningTokens());
}

test "decode responses stream event carries nested response snapshot" {
    const json =
        \\{"type":"response.completed","sequence_number":6,
        \\ "response":{"id":"resp_1","object":"response","status":"completed","model":"gpt-5",
        \\   "output":[{"type":"message","role":"assistant",
        \\     "content":[{"type":"output_text","text":"Hi","annotations":[]}]}],
        \\   "usage":{"input_tokens":9,"output_tokens":3,"total_tokens":12}}}
    ;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const parsed = try std.json.parseFromSliceLeaky(Value, a, json, .{});
    const event = try decodeResponsesStreamEvent(a, parsed);
    try testing.expectEqualStrings("response.completed", event.type);
    try testing.expectEqual(@as(u64, 6), event.sequence_number);
    try testing.expect(event.isTerminal());
    try testing.expectEqualStrings("completed", event.response.?.status);
    try testing.expectEqual(@as(u32, 12), event.response.?.usage.?.total_tokens);
    try testing.expectEqualStrings("Hi", event.response.?.output[0].message.content.?.parts[0].output_text.text);
}

test "encode multimodal content parts" {
    const parts = [_]gen.ContentPart{
        .{ .text = .{ .text = "look:" } },
        .{ .image_url = .{ .url = "http://x/y.png", .detail = "low" } },
    };
    const req = gen.ChatRequest{
        .model = "m",
        .messages = &.{.{ .role = .user, .content = .{ .parts = &parts } }},
    };
    const body = try encodeChatRequest(testing.allocator, req);
    defer testing.allocator.free(body);
    try testing.expect(std.mem.indexOf(u8, body, "{\"type\":\"text\",\"text\":\"look:\"}") != null);
    try testing.expect(std.mem.indexOf(u8, body, "\"image_url\":{\"url\":\"http://x/y.png\",\"detail\":\"low\"}") != null);
}
