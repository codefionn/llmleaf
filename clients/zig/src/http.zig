//! HTTP transport over `std.http.Client` (Zig 0.16.0).
//!
//! Zig 0.16 routes all I/O through an `Io` instance; we own a blocking
//! `std.Io.Threaded` and the `std.http.Client` bound to it. Everything here is
//! synchronous, matching the SDK's synchronous surface.
//!
//! Two response shapes are supported:
//!   * `request`  — send + receive head + read the WHOLE body into an owned
//!     buffer (used by every non-streaming call; also captures Content-Type).
//!   * `open`     — send + receive head, then hand back a live `Response` whose
//!     body the caller drains line-by-line (SSE chat, NDJSON batch results).
//!
//! Ownership: `Transport.init` borrows the allocator; `deinit` frees the client
//! pool. Each `Result.body` is owned by the caller (allocated with the passed
//! allocator) and must be freed. A `Stream` MUST be `deinit`'d to return the
//! connection to the pool.

const std = @import("std");

const Allocator = std.mem.Allocator;
const http = std.http;
const Writer = std.Io.Writer;
const Reader = std.Io.Reader;

/// Caps to keep a misbehaving / hostile server from exhausting memory.
const max_body_bytes: usize = 256 * 1024 * 1024; // 256 MiB (batch results can be large)
/// Per-SSE/NDJSON line buffer. A single chat chunk or result line must fit here.
pub const line_buffer_size: usize = 1024 * 1024; // 1 MiB

pub const Method = enum { GET, POST };

pub const Header = http.Header;

/// A fully-read response.
pub const Result = struct {
    status: u16,
    content_type: []u8, // owned (duped); "" if the server omitted it
    body: []u8, // owned

    pub fn deinit(self: *Result, gpa: Allocator) void {
        gpa.free(self.content_type);
        gpa.free(self.body);
        self.* = undefined;
    }
};

pub const RequestArgs = struct {
    method: Method,
    url: []const u8,
    /// `application/json` unless overridden (e.g. multipart).
    content_type: ?[]const u8 = "application/json",
    body: ?[]const u8 = null,
    /// Extra headers (Authorization, x-admin-token, ...). Must outlive the call.
    headers: []const Header = &.{},
};

pub const Transport = struct {
    gpa: Allocator,
    threaded: std.Io.Threaded,
    client: http.Client,

    /// Initialise into a STABLE address: `client.io` and the thread pool
    /// reference the embedded `threaded`, so the `Transport` must not be moved
    /// afterwards. `Client.init` heap-allocates the owning struct for exactly
    /// this reason.
    pub fn initInPlace(self: *Transport, gpa: Allocator) void {
        self.gpa = gpa;
        self.threaded = std.Io.Threaded.init(gpa, .{});
        self.client = .{ .allocator = gpa, .io = self.threaded.io() };
    }

    pub fn deinit(self: *Transport) void {
        self.client.deinit();
        self.threaded.deinit();
        self.* = undefined;
    }

    fn toHttpMethod(m: Method) http.Method {
        return switch (m) {
            .GET => .GET,
            .POST => .POST,
        };
    }

    /// Build + send a request and read the entire response body into memory.
    /// The returned `Result` is owned by the caller (free via `Result.deinit`).
    pub fn request(self: *Transport, args: RequestArgs) !Result {
        const uri = try std.Uri.parse(args.url);
        const method = toHttpMethod(args.method);

        var headers: http.Client.Request.Headers = .{};
        if (args.content_type) |ct| headers.content_type = .{ .override = ct };

        var req = try self.client.request(method, uri, .{
            .headers = headers,
            .extra_headers = args.headers,
            .keep_alive = true,
        });
        defer req.deinit();

        if (args.body) |body| {
            req.transfer_encoding = .{ .content_length = body.len };
            var send_buf: [4096]u8 = undefined;
            var bw = try req.sendBodyUnflushed(&send_buf);
            try bw.writer.writeAll(body);
            try bw.end();
            try req.connection.?.flush();
        } else {
            try req.sendBodiless();
        }

        var redirect_buf: [8 * 1024]u8 = undefined;
        var response = try req.receiveHead(&redirect_buf);

        const status: u16 = @intFromEnum(response.head.status);
        const content_type = try self.gpa.dupe(u8, response.head.content_type orelse "");
        errdefer self.gpa.free(content_type);

        // Read the whole body. Use a transfer buffer; decompress if negotiated.
        var transfer_buf: [16 * 1024]u8 = undefined;
        var decompress: http.Decompress = undefined;
        var decompress_buf: [std.compress.flate.max_window_len]u8 = undefined;
        const body_reader = response.readerDecompressing(&transfer_buf, &decompress, &decompress_buf);

        const body = body_reader.allocRemaining(self.gpa, .limited(max_body_bytes)) catch |err| switch (err) {
            error.ReadFailed => return response.bodyErr().?,
            else => |e| return e,
        };

        return .{ .status = status, .content_type = content_type, .body = body };
    }

    /// Open a request and return a live `Stream` whose body is read
    /// incrementally line-by-line. Used for SSE (`text/event-stream`) and
    /// NDJSON batch results (`application/x-ndjson`). The caller MUST call
    /// `Stream.deinit`.
    ///
    /// On a non-2xx status the body is drained into `err_body` (owned by the
    /// caller) and the status is returned in the `Stream`; the caller inspects
    /// `Stream.status` before iterating.
    pub fn open(self: *Transport, args: RequestArgs) !*Stream {
        const stream = try self.gpa.create(Stream);
        errdefer self.gpa.destroy(stream);

        const uri = try std.Uri.parse(args.url);
        const method = toHttpMethod(args.method);

        var headers: http.Client.Request.Headers = .{};
        if (args.content_type) |ct| headers.content_type = .{ .override = ct };

        stream.* = .{
            .gpa = self.gpa,
            .req = try self.client.request(method, uri, .{
                .headers = headers,
                .extra_headers = args.headers,
                .keep_alive = true,
            }),
            .reader = undefined,
            .transfer_buf = undefined,
            .status = 0,
            .err_body = null,
            .response = undefined,
        };
        errdefer stream.req.deinit();

        if (args.body) |body| {
            stream.req.transfer_encoding = .{ .content_length = body.len };
            var send_buf: [4096]u8 = undefined;
            var bw = try stream.req.sendBodyUnflushed(&send_buf);
            try bw.writer.writeAll(body);
            try bw.end();
            try stream.req.connection.?.flush();
        } else {
            try stream.req.sendBodiless();
        }

        var redirect_buf: [8 * 1024]u8 = undefined;
        stream.response = try stream.req.receiveHead(&redirect_buf);
        stream.status = @intFromEnum(stream.response.head.status);

        // Body reader over a heap-allocated line buffer so whole lines fit.
        stream.transfer_buf = try self.gpa.alloc(u8, line_buffer_size);
        errdefer self.gpa.free(stream.transfer_buf);
        stream.reader = stream.response.reader(stream.transfer_buf);

        if (stream.status < 200 or stream.status >= 300) {
            // Drain the (small) error body so the caller can parse the envelope.
            stream.err_body = stream.reader.allocRemaining(self.gpa, .limited(1024 * 1024)) catch |err| switch (err) {
                error.ReadFailed => return stream.response.bodyErr().?,
                else => |e| return e,
            };
        }

        return stream;
    }
};

/// A live response whose body is consumed line-by-line. Returned by
/// `Transport.open`. Must be `deinit`'d.
pub const Stream = struct {
    gpa: Allocator,
    req: http.Client.Request,
    response: http.Client.Response,
    reader: *Reader,
    transfer_buf: []u8,
    status: u16,
    /// On a non-2xx open, the whole error body (owned). `null` on success.
    err_body: ?[]u8,

    /// Read the next line (without the trailing '\n'), or `null` at end of
    /// stream. The returned slice is borrowed from the internal buffer and is
    /// invalidated by the next `nextLine`/`deinit`; copy or parse it before
    /// calling again.
    pub fn nextLine(self: *Stream) !?[]const u8 {
        while (true) {
            // `takeDelimiter` advances PAST the '\n' (unlike the exclusive
            // variant, which would leave it and spin on blank lines), and
            // returns `null` at end of stream.
            const line = self.reader.takeDelimiter('\n') catch |err| switch (err) {
                error.ReadFailed => return self.response.bodyErr() orelse error.ReadFailed,
                error.StreamTooLong => return error.StreamTooLong,
            } orelse return null;
            // Skip blank lines (SSE separates events with a blank line).
            const trimmed = std.mem.trimEnd(u8, line, "\r");
            if (trimmed.len == 0) continue;
            return trimmed;
        }
    }

    pub fn deinit(self: *Stream) void {
        if (self.err_body) |b| self.gpa.free(b);
        self.gpa.free(self.transfer_buf);
        self.req.deinit();
        self.gpa.destroy(self);
    }
};
