//! The llmleaf client: one `Client` bound to a base URL + API key, exposing
//! every endpoint in SPEC.md over the OpenAI/OpenRouter HTTP wire.
//!
//! Construction & ownership
//! ------------------------
//! `Client.init` HEAP-ALLOCATES the client (the embedded HTTP transport holds
//! self-referential pointers and must not move) and borrows the allocator and
//! the `base_url`/`api_key`/`admin_token` slices — those must outlive the
//! client. Always pair with `defer client.deinit()`.
//!
//! Every call that returns parsed data returns it wrapped in an `Owned(T)` (or a
//! type with a `deinit`): the value borrows from an arena the wrapper owns. Call
//! `.deinit()` when done. Raw byte results (speech) and plain-text results
//! (transcription text/srt/vtt) are returned as allocator-owned slices you free.
//!
//! Errors
//! ------
//! Non-2xx responses raise a value from `errors.Error` AND, when you pass a
//! `*?ApiError` out-pointer, write the structured `{status,message}` there
//! (message duplicated into your allocator — free it with `gpa.free(message)`).
//! Pass `null` to ignore the structured form.

const std = @import("std");

const gen = @import("gen/llmleaf.zig");
const wire = @import("wire.zig");
const transport = @import("http.zig");
const errmod = @import("errors.zig");

const Allocator = std.mem.Allocator;
const Value = std.json.Value;

pub const ApiError = errmod.ApiError;
pub const Error = errmod.Error;

/// Owns an arena; the wrapped value borrows from it. `deinit` frees everything.
pub fn Owned(comptime T: type) type {
    return struct {
        const Self = @This();
        arena: *std.heap.ArenaAllocator,
        value: T,

        pub fn deinit(self: Self) void {
            const gpa = self.arena.child_allocator;
            self.arena.deinit();
            gpa.destroy(self.arena);
        }
    };
}

pub const Options = struct {
    /// Optional `x-admin-token` header value (enables admin-only fields, e.g.
    /// per-model `endpoints`). Borrowed; must outlive the client.
    admin_token: ?[]const u8 = null,
    /// Per-request connect+read timeout in milliseconds. 0 means no timeout.
    /// NOTE: `std.http.Client` (0.16) does not yet expose a socket timeout knob,
    /// so this is currently advisory only — see README caveats.
    timeout_ms: u32 = 0,
    /// Overridable User-Agent. Borrowed.
    user_agent: []const u8 = "llmleaf-zig/0.1.0",
};

pub const Client = struct {
    gpa: Allocator,
    base_url: []const u8, // normalised: no trailing '/'
    api_key: []const u8,
    options: Options,
    tp: transport.Transport,

    // Owned, '\0'-terminated header value buffers (built once at init).
    auth_header: []u8,

    /// Heap-allocate and initialise a client. `base_url`, `api_key`, and any
    /// `options.admin_token` are BORROWED and must outlive the returned client.
    pub fn init(gpa: Allocator, base_url: []const u8, api_key: []const u8, options: Options) !*Client {
        const self = try gpa.create(Client);
        errdefer gpa.destroy(self);

        const trimmed = std.mem.trimEnd(u8, base_url, "/");
        const auth = try std.fmt.allocPrint(gpa, "Bearer {s}", .{api_key});
        errdefer gpa.free(auth);

        self.* = .{
            .gpa = gpa,
            .base_url = trimmed,
            .api_key = api_key,
            .options = options,
            .tp = undefined,
            .auth_header = auth,
        };
        self.tp.initInPlace(gpa);
        return self;
    }

    pub fn deinit(self: *Client) void {
        self.tp.deinit();
        self.gpa.free(self.auth_header);
        const gpa = self.gpa;
        gpa.destroy(self);
    }

    // --- internal helpers --------------------------------------------------

    fn url(self: *Client, comptime fmt: []const u8, args: anytype) ![]u8 {
        return std.fmt.allocPrint(self.gpa, "{s}" ++ fmt, .{self.base_url} ++ args);
    }

    /// The standard header set (Authorization, User-Agent, optional admin token).
    /// Returns a slice into `buf`. Pass extra headers via `extra`.
    fn stdHeaders(self: *Client, buf: []transport.Header, extra: []const transport.Header) []transport.Header {
        var n: usize = 0;
        buf[n] = .{ .name = "Authorization", .value = self.auth_header };
        n += 1;
        buf[n] = .{ .name = "User-Agent", .value = self.options.user_agent };
        n += 1;
        if (self.options.admin_token) |tok| {
            buf[n] = .{ .name = "x-admin-token", .value = tok };
            n += 1;
        }
        for (extra) |h| {
            buf[n] = h;
            n += 1;
        }
        return buf[0..n];
    }

    /// Turn a non-2xx `Result` into a typed error, optionally filling `*?ApiError`.
    /// `result.body` is consumed (freed) here.
    fn fail(self: *Client, result: *transport.Result, out_err: ?*?ApiError) Error {
        defer result.deinit(self.gpa);
        return self.failBody(result.status, result.body, out_err);
    }

    fn failBody(self: *Client, status: u16, body: []const u8, out_err: ?*?ApiError) Error {
        // Parse the documented envelope `{"error":{"message":...}}`.
        var msg_owned: ?[]u8 = null;
        if (std.json.parseFromSlice(Value, self.gpa, body, .{}) catch null) |parsed| {
            defer parsed.deinit();
            if (wire.decodeErrorMessage(parsed.value)) |msg| {
                msg_owned = self.gpa.dupe(u8, msg) catch null;
            }
        }
        if (out_err) |slot| {
            if (msg_owned) |m| {
                slot.* = .{ .status = status, .message = m };
            } else {
                slot.* = .{ .status = status, .message = self.gpa.dupe(u8, "") catch "" };
            }
        } else if (msg_owned) |m| {
            self.gpa.free(m); // caller doesn't want it
        }
        return if (msg_owned != null) errmod.fromStatus(status) else error.UnexpectedStatus;
    }

    /// Parse a 2xx JSON body into `T` via `decodeFn(arena, root_value)`.
    fn decodeOwned(
        self: *Client,
        comptime T: type,
        body: []const u8,
        decodeFn: anytype,
    ) !Owned(T) {
        const arena = try self.gpa.create(std.heap.ArenaAllocator);
        errdefer self.gpa.destroy(arena);
        arena.* = std.heap.ArenaAllocator.init(self.gpa);
        errdefer arena.deinit();
        const a = arena.allocator();
        const parsed = try std.json.parseFromSliceLeaky(Value, a, body, .{});
        const value = try decodeFn(a, parsed);
        return .{ .arena = arena, .value = value };
    }

    // =======================================================================
    // POST /v1/chat/completions  (non-streaming)
    // =======================================================================

    pub fn chat(self: *Client, req: gen.ChatRequest, out_err: ?*?ApiError) !Owned(gen.ChatResponse) {
        var r = req;
        r.stream = false;
        const body = try wire.encodeChatRequest(self.gpa, r);
        defer self.gpa.free(body);

        const u = try self.url("/v1/chat/completions", .{});
        defer self.gpa.free(u);

        var hbuf: [4]transport.Header = undefined;
        var result = try self.tp.request(.{
            .method = .POST,
            .url = u,
            .body = body,
            .headers = self.stdHeaders(&hbuf, &.{}),
        });
        if (result.status < 200 or result.status >= 300) return self.fail(&result, out_err);
        defer result.deinit(self.gpa);
        return self.decodeOwned(gen.ChatResponse, result.body, wire.decodeChatResponse);
    }

    // =======================================================================
    // POST /v1/chat/completions  (streaming, SSE)
    // =======================================================================

    /// Begin a streaming chat completion. Returns a `ChatStream` you pull chunks
    /// from with `next()`. The `[DONE]` sentinel ends the stream and is never
    /// JSON-parsed. The stream MUST be `deinit`'d. On a non-2xx open, `next`
    /// returns the typed error (and fills `out_err` if given).
    pub fn chatStream(self: *Client, req: gen.ChatRequest, out_err: ?*?ApiError) !*ChatStream {
        var r = req;
        r.stream = true;
        const body = try wire.encodeChatRequest(self.gpa, r);
        defer self.gpa.free(body);

        const u = try self.url("/v1/chat/completions", .{});
        defer self.gpa.free(u);

        var hbuf: [4]transport.Header = undefined;
        const stream = try self.tp.open(.{
            .method = .POST,
            .url = u,
            .body = body,
            .headers = self.stdHeaders(&hbuf, &.{}),
        });
        errdefer stream.deinit();

        if (stream.status < 200 or stream.status >= 300) {
            const e = self.failBody(stream.status, stream.err_body orelse "", out_err);
            stream.deinit();
            return e;
        }

        const cs = try self.gpa.create(ChatStream);
        cs.* = .{ .gpa = self.gpa, .stream = stream, .arena = std.heap.ArenaAllocator.init(self.gpa) };
        return cs;
    }

    /// Convenience callback form: stream a chat completion and invoke `cb` for
    /// each chunk. Stops at `[DONE]`. Returns the assembled text (owned by
    /// `self.gpa`; free it) gathered from `choices[0].delta.content`.
    pub fn chatStreamCallback(
        self: *Client,
        req: gen.ChatRequest,
        out_err: ?*?ApiError,
        ctx: anytype,
        comptime cb: fn (@TypeOf(ctx), gen.ChatCompletionChunk) anyerror!void,
    ) ![]u8 {
        var cs = try self.chatStream(req, out_err);
        defer cs.deinit();
        var acc: std.ArrayList(u8) = .empty;
        errdefer acc.deinit(self.gpa);
        while (try cs.next()) |chunk| {
            if (chunk.choices.len > 0) {
                if (chunk.choices[0].delta.content) |c| try acc.appendSlice(self.gpa, c);
            }
            try cb(ctx, chunk);
        }
        return acc.toOwnedSlice(self.gpa);
    }

    // =======================================================================
    // POST /v1/embeddings
    // =======================================================================

    pub fn embeddings(self: *Client, req: gen.EmbeddingRequest, out_err: ?*?ApiError) !Owned(gen.EmbeddingResponse) {
        const body = try wire.encodeEmbeddingRequest(self.gpa, req);
        defer self.gpa.free(body);

        const u = try self.url("/v1/embeddings", .{});
        defer self.gpa.free(u);

        var hbuf: [4]transport.Header = undefined;
        var result = try self.tp.request(.{
            .method = .POST,
            .url = u,
            .body = body,
            .headers = self.stdHeaders(&hbuf, &.{}),
        });
        if (result.status < 200 or result.status >= 300) return self.fail(&result, out_err);
        defer result.deinit(self.gpa);
        return self.decodeOwned(gen.EmbeddingResponse, result.body, wire.decodeEmbeddingResponse);
    }

    // =======================================================================
    // GET /v1/models
    // =======================================================================

    pub fn listModels(
        self: *Client,
        model_type: gen.ModelType,
        search: ?[]const u8,
        out_err: ?*?ApiError,
    ) !Owned(gen.ListModelsResponse) {
        var qbuf: std.ArrayList(u8) = .empty;
        defer qbuf.deinit(self.gpa);
        try qbuf.appendSlice(self.gpa, "/v1/models?type=");
        try qbuf.appendSlice(self.gpa, gen.enumToWire(model_type));
        if (search) |s| {
            try qbuf.appendSlice(self.gpa, "&search=");
            try percentEncode(&qbuf, self.gpa, s);
        }
        const u = try self.url("{s}", .{qbuf.items});
        defer self.gpa.free(u);

        var hbuf: [4]transport.Header = undefined;
        var result = try self.tp.request(.{
            .method = .GET,
            .url = u,
            .content_type = null,
            .headers = self.stdHeaders(&hbuf, &.{}),
        });
        if (result.status < 200 or result.status >= 300) return self.fail(&result, out_err);
        defer result.deinit(self.gpa);
        return self.decodeOwned(gen.ListModelsResponse, result.body, wire.decodeListModels);
    }

    // =======================================================================
    // POST /v1/audio/speech  (TTS -> raw bytes)
    // =======================================================================

    /// Returns raw audio bytes + the Content-Type the server set. Both owned by
    /// `self.gpa`; free with `gpa.free(speech.bytes)` and `gpa.free(speech.content_type)`.
    pub fn speech(self: *Client, req: gen.SpeechRequest, out_err: ?*?ApiError) !gen.Speech {
        const body = try wire.encodeSpeechRequest(self.gpa, req);
        defer self.gpa.free(body);

        const u = try self.url("/v1/audio/speech", .{});
        defer self.gpa.free(u);

        var hbuf: [4]transport.Header = undefined;
        var result = try self.tp.request(.{
            .method = .POST,
            .url = u,
            .body = body,
            .headers = self.stdHeaders(&hbuf, &.{}),
        });
        if (result.status < 200 or result.status >= 300) return self.fail(&result, out_err);
        // Hand both owned slices to the caller; do not free here.
        return .{ .bytes = result.body, .content_type = result.content_type };
    }

    // =======================================================================
    // GET /v1/audio/voices?model=<id>
    // =======================================================================

    pub fn voices(self: *Client, model: []const u8, out_err: ?*?ApiError) !Owned(gen.VoicesResponse) {
        var qbuf: std.ArrayList(u8) = .empty;
        defer qbuf.deinit(self.gpa);
        try qbuf.appendSlice(self.gpa, "/v1/audio/voices?model=");
        try percentEncode(&qbuf, self.gpa, model);
        const u = try self.url("{s}", .{qbuf.items});
        defer self.gpa.free(u);

        var hbuf: [4]transport.Header = undefined;
        var result = try self.tp.request(.{
            .method = .GET,
            .url = u,
            .content_type = null,
            .headers = self.stdHeaders(&hbuf, &.{}),
        });
        if (result.status < 200 or result.status >= 300) return self.fail(&result, out_err);
        defer result.deinit(self.gpa);
        return self.decodeOwned(gen.VoicesResponse, result.body, wire.decodeVoicesResponse);
    }

    // =======================================================================
    // POST /v1/audio/transcriptions  (STT, multipart/form-data)
    // =======================================================================

    pub const TranscriptionResult = union(enum) {
        /// For response_format json / verbose_json.
        json: Owned(gen.TranscriptionResponse),
        /// For response_format text / srt / vtt — the plain-text body (owned by gpa).
        text: []u8,

        pub fn deinit(self: TranscriptionResult, gpa: Allocator) void {
            switch (self) {
                .json => |o| o.deinit(),
                .text => |t| gpa.free(t),
            }
        }
    };

    /// `file_bytes` is the raw audio; `filename` is the multipart filename. The
    /// other fields come from `req`. Result is `.text` for text/srt/vtt and
    /// `.json` otherwise.
    pub fn transcribe(
        self: *Client,
        req: gen.TranscriptionRequest,
        file_bytes: []const u8,
        filename: []const u8,
        out_err: ?*?ApiError,
    ) !TranscriptionResult {
        const boundary = "----llmleafZigBoundary7MA4YWxkTrZu0gW";
        const body = try buildMultipart(self.gpa, boundary, req, file_bytes, filename);
        defer self.gpa.free(body);

        const ct = "multipart/form-data; boundary=" ++ boundary;
        const u = try self.url("/v1/audio/transcriptions", .{});
        defer self.gpa.free(u);

        var hbuf: [4]transport.Header = undefined;
        var result = try self.tp.request(.{
            .method = .POST,
            .url = u,
            .content_type = ct,
            .body = body,
            .headers = self.stdHeaders(&hbuf, &.{}),
        });
        if (result.status < 200 or result.status >= 300) return self.fail(&result, out_err);

        // text/srt/vtt -> plain text; json/verbose_json -> structured.
        const is_text = std.mem.indexOf(u8, result.content_type, "application/json") == null;
        if (is_text) {
            self.gpa.free(result.content_type);
            return .{ .text = result.body }; // hand owned body to caller
        }
        defer result.deinit(self.gpa);
        const owned = try self.decodeOwnedTranscription(result.body);
        return .{ .json = owned };
    }

    fn decodeOwnedTranscription(self: *Client, body: []const u8) !Owned(gen.TranscriptionResponse) {
        const arena = try self.gpa.create(std.heap.ArenaAllocator);
        errdefer self.gpa.destroy(arena);
        arena.* = std.heap.ArenaAllocator.init(self.gpa);
        errdefer arena.deinit();
        const a = arena.allocator();
        const parsed = try std.json.parseFromSliceLeaky(Value, a, body, .{});
        return .{ .arena = arena, .value = wire.decodeTranscription(parsed) };
    }

    // =======================================================================
    // Batches
    // =======================================================================

    pub fn createBatch(self: *Client, req: gen.BatchCreateRequest, out_err: ?*?ApiError) !Owned(gen.BatchHandle) {
        const body = try wire.encodeBatchCreateRequest(self.gpa, req);
        defer self.gpa.free(body);
        const u = try self.url("/v1/batches", .{});
        defer self.gpa.free(u);
        return self.batchCall(.POST, u, body, out_err);
    }

    pub fn getBatch(self: *Client, id: []const u8, out_err: ?*?ApiError) !Owned(gen.BatchHandle) {
        const u = try self.url("/v1/batches/{s}", .{id});
        defer self.gpa.free(u);
        return self.batchCall(.GET, u, null, out_err);
    }

    pub fn cancelBatch(self: *Client, id: []const u8, out_err: ?*?ApiError) !Owned(gen.BatchHandle) {
        const u = try self.url("/v1/batches/{s}/cancel", .{id});
        defer self.gpa.free(u);
        return self.batchCall(.POST, u, null, out_err);
    }

    fn batchCall(
        self: *Client,
        method: transport.Method,
        u: []const u8,
        body: ?[]const u8,
        out_err: ?*?ApiError,
    ) !Owned(gen.BatchHandle) {
        var hbuf: [4]transport.Header = undefined;
        var result = try self.tp.request(.{
            .method = method,
            .url = u,
            .content_type = if (body != null) "application/json" else null,
            .body = body,
            .headers = self.stdHeaders(&hbuf, &.{}),
        });
        if (result.status < 200 or result.status >= 300) return self.fail(&result, out_err);
        defer result.deinit(self.gpa);
        const Decode = struct {
            fn d(_: Allocator, root: Value) !gen.BatchHandle {
                return wire.decodeBatchHandle(root);
            }
        };
        return self.decodeOwned(gen.BatchHandle, result.body, Decode.d);
    }

    /// GET /v1/batches/{id}/results — line-delimited NDJSON. Returns a
    /// `BatchResults` iterator; call `next()` until it yields `null`. MUST be
    /// `deinit`'d.
    pub fn batchResults(self: *Client, id: []const u8, out_err: ?*?ApiError) !*BatchResults {
        const u = try self.url("/v1/batches/{s}/results", .{id});
        defer self.gpa.free(u);

        var hbuf: [4]transport.Header = undefined;
        const stream = try self.tp.open(.{
            .method = .GET,
            .url = u,
            .content_type = null,
            .headers = self.stdHeaders(&hbuf, &.{}),
        });
        errdefer stream.deinit();

        if (stream.status < 200 or stream.status >= 300) {
            const e = self.failBody(stream.status, stream.err_body orelse "", out_err);
            stream.deinit();
            return e;
        }

        const br = try self.gpa.create(BatchResults);
        br.* = .{ .gpa = self.gpa, .stream = stream, .arena = std.heap.ArenaAllocator.init(self.gpa) };
        return br;
    }
};

// ---------------------------------------------------------------------------
// Streaming iterators
// ---------------------------------------------------------------------------

/// Pull-based iterator over streaming chat chunks. `next()` returns the next
/// `ChatCompletionChunk` or `null` at `[DONE]`/end. Each returned chunk borrows
/// from an internal arena that is RESET on the following `next()` — copy out any
/// fields you need to keep past the next call. Free via `deinit`.
pub const ChatStream = struct {
    gpa: Allocator,
    stream: *transport.Stream,
    arena: std.heap.ArenaAllocator,

    pub fn next(self: *ChatStream) !?gen.ChatCompletionChunk {
        while (true) {
            const line = (try self.stream.nextLine()) orelse return null;
            // SSE data lines: `data: <json>`. Ignore other field lines (event:, id:, :comment).
            const payload = stripSsePrefix(line) orelse continue;
            if (std.mem.eql(u8, payload, "[DONE]")) return null; // sentinel — never parse it
            _ = self.arena.reset(.retain_capacity);
            const a = self.arena.allocator();
            const parsed = std.json.parseFromSliceLeaky(Value, a, payload, .{}) catch continue;
            return try wire.decodeChunk(a, parsed);
        }
    }

    pub fn deinit(self: *ChatStream) void {
        self.arena.deinit();
        self.stream.deinit();
        self.gpa.destroy(self);
    }
};

/// Iterator over NDJSON batch result lines. Same arena-reset semantics as
/// `ChatStream`. Free via `deinit`.
pub const BatchResults = struct {
    gpa: Allocator,
    stream: *transport.Stream,
    arena: std.heap.ArenaAllocator,

    pub fn next(self: *BatchResults) !?gen.BatchResultLine {
        const line = (try self.stream.nextLine()) orelse return null;
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        const parsed = try std.json.parseFromSliceLeaky(Value, a, line, .{});
        return try wire.decodeBatchResultLine(a, parsed);
    }

    pub fn deinit(self: *BatchResults) void {
        self.arena.deinit();
        self.stream.deinit();
        self.gpa.destroy(self);
    }
};

/// `data: <payload>` -> `<payload>` (trimmed); `null` for non-data lines.
fn stripSsePrefix(line: []const u8) ?[]const u8 {
    if (std.mem.startsWith(u8, line, "data:")) {
        return std.mem.trimStart(u8, line["data:".len..], " ");
    }
    return null;
}

// ---------------------------------------------------------------------------
// Multipart + URL encoding helpers
// ---------------------------------------------------------------------------

fn buildMultipart(
    gpa: Allocator,
    boundary: []const u8,
    req: gen.TranscriptionRequest,
    file_bytes: []const u8,
    filename: []const u8,
) ![]u8 {
    var buf: std.ArrayList(u8) = .empty;
    errdefer buf.deinit(gpa);
    const w = &buf;

    try formField(w, gpa, boundary, "model", req.model);
    if (req.language) |v| try formField(w, gpa, boundary, "language", v);
    if (req.prompt) |v| try formField(w, gpa, boundary, "prompt", v);
    if (req.response_format) |v| try formField(w, gpa, boundary, "response_format", v);
    if (req.temperature) |v| {
        var numbuf: [32]u8 = undefined;
        const s = try std.fmt.bufPrint(&numbuf, "{d}", .{v});
        try formField(w, gpa, boundary, "temperature", s);
    }

    // The file part.
    try appendFmt(w, gpa, "--{s}\r\n", .{boundary});
    try appendFmt(w, gpa, "Content-Disposition: form-data; name=\"file\"; filename=\"{s}\"\r\n", .{filename});
    try w.appendSlice(gpa, "Content-Type: application/octet-stream\r\n\r\n");
    try w.appendSlice(gpa, file_bytes);
    try w.appendSlice(gpa, "\r\n");

    try appendFmt(w, gpa, "--{s}--\r\n", .{boundary});
    return buf.toOwnedSlice(gpa);
}

fn formField(w: *std.ArrayList(u8), gpa: Allocator, boundary: []const u8, name: []const u8, value: []const u8) !void {
    try appendFmt(w, gpa, "--{s}\r\n", .{boundary});
    try appendFmt(w, gpa, "Content-Disposition: form-data; name=\"{s}\"\r\n\r\n", .{name});
    try w.appendSlice(gpa, value);
    try w.appendSlice(gpa, "\r\n");
}

fn appendFmt(w: *std.ArrayList(u8), gpa: Allocator, comptime fmt: []const u8, args: anytype) !void {
    const s = try std.fmt.allocPrint(gpa, fmt, args);
    defer gpa.free(s);
    try w.appendSlice(gpa, s);
}

/// Percent-encode a query-string component (RFC 3986 unreserved kept as-is).
fn percentEncode(out: *std.ArrayList(u8), gpa: Allocator, s: []const u8) !void {
    for (s) |c| {
        const unreserved = std.ascii.isAlphanumeric(c) or c == '-' or c == '_' or c == '.' or c == '~';
        if (unreserved) {
            try out.append(gpa, c);
        } else {
            try appendFmt(out, gpa, "%{X:0>2}", .{c});
        }
    }
}

// ---------------------------------------------------------------------------
// Tests (pure, no network)
// ---------------------------------------------------------------------------

const testing = std.testing;

test "strip SSE prefix and detect sentinel" {
    try testing.expectEqualStrings("{}", stripSsePrefix("data: {}").?);
    try testing.expectEqualStrings("{}", stripSsePrefix("data:{}").?);
    try testing.expect(stripSsePrefix("event: ping") == null);
    try testing.expectEqualStrings("[DONE]", stripSsePrefix("data: [DONE]").?);
}

test "percent encode" {
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(testing.allocator);
    try percentEncode(&buf, testing.allocator, "a b/c");
    try testing.expectEqualStrings("a%20b%2Fc", buf.items);
}

test "multipart contains file and fields" {
    const body = try buildMultipart(
        testing.allocator,
        "BNDRY",
        .{ .model = "whisper-1", .response_format = "text" },
        "RAWAUDIO",
        "a.mp3",
    );
    defer testing.allocator.free(body);
    try testing.expect(std.mem.indexOf(u8, body, "name=\"model\"") != null);
    try testing.expect(std.mem.indexOf(u8, body, "whisper-1") != null);
    try testing.expect(std.mem.indexOf(u8, body, "name=\"file\"; filename=\"a.mp3\"") != null);
    try testing.expect(std.mem.indexOf(u8, body, "RAWAUDIO") != null);
    try testing.expect(std.mem.endsWith(u8, body, "--BNDRY--\r\n"));
}

test "generic chatStreamCallback type-checks" {
    // `chatStreamCallback` is generic over the callback, so its body is only
    // type-checked when instantiated. Referencing a wrapper that calls it forces
    // that instantiation at compile time (the wrapper is never executed).
    const Sink = struct {
        seen: usize = 0,
        fn onChunk(self: *@This(), _: gen.ChatCompletionChunk) anyerror!void {
            self.seen += 1;
        }
        fn drive(client: *Client, self: *@This()) ![]u8 {
            return client.chatStreamCallback(.{
                .model = "m",
                .messages = &.{gen.ChatMessage.textMsg(.user, "x")},
            }, null, self, onChunk);
        }
    };
    // Referencing the wrapper compiles the whole generic chain without running it.
    const drive_ref = &Sink.drive;
    _ = drive_ref;
    // Ensure the transcription union result type also type-checks.
    _ = Client.TranscriptionResult;
}
