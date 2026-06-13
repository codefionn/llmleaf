using System.Net;
using System.Text;

namespace Llmleaf.Client.Tests;

/// <summary>
/// A tiny in-process HTTP server (HttpListener) that captures the last request and replies with a
/// canned response. Lets the tests exercise the real HttpClient + System.Text.Json transport
/// end-to-end without a live gateway.
/// </summary>
public sealed class TestServer : IDisposable
{
    private readonly HttpListener _listener;
    private readonly Func<CapturedRequest, CannedResponse> _handler;
    private readonly Task _loop;

    public string BaseUrl { get; }

    /// <summary>The most recently received request (path, headers, body).</summary>
    public CapturedRequest? LastRequest { get; private set; }

    public TestServer(Func<CapturedRequest, CannedResponse> handler)
    {
        _handler = handler;
        var port = GetFreePort();
        BaseUrl = $"http://127.0.0.1:{port}";
        _listener = new HttpListener();
        _listener.Prefixes.Add(BaseUrl + "/");
        _listener.Start();
        _loop = Task.Run(LoopAsync);
    }

    private async Task LoopAsync()
    {
        while (_listener.IsListening)
        {
            HttpListenerContext ctx;
            try
            {
                ctx = await _listener.GetContextAsync();
            }
            catch (HttpListenerException)
            {
                return; // listener stopped
            }
            catch (ObjectDisposedException)
            {
                return;
            }

            try
            {
                using var ms = new MemoryStream();
                await ctx.Request.InputStream.CopyToAsync(ms);
                var rawBytes = ms.ToArray();
                // Latin-1 is byte-preserving, so multipart boundaries/headers survive even when a
                // part carries raw binary bytes that aren't valid UTF-8.
                var body = Encoding.Latin1.GetString(rawBytes);
                var headers = ctx.Request.Headers.AllKeys
                    .Where(k => k is not null)
                    .ToDictionary(k => k!, k => ctx.Request.Headers[k] ?? "", StringComparer.OrdinalIgnoreCase);
                var captured = new CapturedRequest(
                    ctx.Request.HttpMethod,
                    ctx.Request.Url!.AbsolutePath,
                    ctx.Request.Url!.Query,
                    headers,
                    body,
                    ctx.Request.ContentType ?? "");
                LastRequest = captured;

                var resp = _handler(captured);
                ctx.Response.StatusCode = resp.StatusCode;
                ctx.Response.ContentType = resp.ContentType;
                var buf = Encoding.UTF8.GetBytes(resp.Body);
                ctx.Response.ContentLength64 = buf.Length;
                await ctx.Response.OutputStream.WriteAsync(buf);
            }
            catch
            {
                // ignore — the test will fail on the assertion side.
            }
            finally
            {
                ctx.Response.Close();
            }
        }
    }

    private static int GetFreePort()
    {
        var l = new System.Net.Sockets.TcpListener(IPAddress.Loopback, 0);
        l.Start();
        var port = ((IPEndPoint)l.LocalEndpoint).Port;
        l.Stop();
        return port;
    }

    public void Dispose()
    {
        try
        {
            _listener.Stop();
            _listener.Close();
        }
        catch
        {
            // ignore
        }
    }
}

public sealed record CapturedRequest(
    string Method,
    string Path,
    string Query,
    IReadOnlyDictionary<string, string> Headers,
    string Body,
    string ContentType);

public sealed record CannedResponse(int StatusCode, string ContentType, string Body)
{
    public static CannedResponse Json(string body, int status = 200) => new(status, "application/json", body);
    public static CannedResponse Sse(string body) => new(200, "text/event-stream", body);
    public static CannedResponse Ndjson(string body) => new(200, "application/x-ndjson", body);
}
