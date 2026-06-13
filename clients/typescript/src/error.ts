// Typed error surface. Any non-2xx response carries the envelope
//   {"error":{"message":"...", "type"?:"...", "code"?:"..."}}
// (SPEC.md "Errors"). We parse it into ApiError and throw.

/**
 * Thrown for any non-2xx HTTP response from the gateway.
 *
 * Status codes (SPEC.md): 400 bad request · 401 missing/invalid key ·
 * 403 blocked or model-not-allowed · 404 no route for model ·
 * 429 key suspended (limiter) · 502 all upstreams failed.
 */
export class ApiError extends Error {
  readonly status: number;
  /** present on some dialects; absent on the llmleaf core envelope. */
  readonly type?: string;
  readonly code?: string;

  constructor(
    status: number,
    message: string,
    opts?: { type?: string; code?: string },
  ) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.type = opts?.type;
    this.code = opts?.code;
    // Restore prototype chain for instanceof across transpile targets.
    Object.setPrototypeOf(this, ApiError.prototype);
  }
}

interface WireErrorBody {
  message?: unknown;
  type?: unknown;
  code?: unknown;
}

/**
 * Build an {@link ApiError} from a non-2xx response. Reads the body once. Falls back
 * to the HTTP status text when the body is missing or not the expected envelope.
 */
export async function apiErrorFromResponse(res: Response): Promise<ApiError> {
  const fallback = res.statusText || `HTTP ${res.status}`;
  let text: string;
  try {
    text = await res.text();
  } catch {
    return new ApiError(res.status, fallback);
  }
  if (!text) return new ApiError(res.status, fallback);

  try {
    const parsed = JSON.parse(text) as { error?: WireErrorBody } | WireErrorBody;
    const body: WireErrorBody | undefined =
      parsed && typeof parsed === "object" && "error" in parsed
        ? (parsed as { error?: WireErrorBody }).error
        : (parsed as WireErrorBody);
    if (body && typeof body === "object") {
      const message =
        typeof body.message === "string" && body.message ? body.message : fallback;
      return new ApiError(res.status, message, {
        type: typeof body.type === "string" ? body.type : undefined,
        code: typeof body.code === "string" ? body.code : undefined,
      });
    }
  } catch {
    // Not JSON — fall through and surface the raw text.
  }
  return new ApiError(res.status, text.slice(0, 2048));
}
