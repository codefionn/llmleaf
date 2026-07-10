//! The async [`Client`] and its [`ClientBuilder`], plus every endpoint in SPEC.md.

use crate::error::{Error, ErrorEnvelope, Result};
use crate::stream::{ndjson_lines, sse_chunks, sse_responses};
use crate::types::*;
use crate::wire::{speech_content_type, RawEmbeddingResponse};
use bytes::Bytes;
use futures::Stream;
use reqwest::header::{HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use reqwest::{Method, RequestBuilder, Response};
use std::time::Duration;

const ADMIN_TOKEN_HEADER: &str = "x-admin-token";

/// Builder for [`Client`]. Obtain via [`Client::builder`].
#[derive(Debug, Clone)]
pub struct ClientBuilder {
    base_url: String,
    api_key: String,
    admin_token: Option<String>,
    timeout: Option<Duration>,
    http: Option<reqwest::Client>,
}

impl ClientBuilder {
    fn new(base_url: impl Into<String>, api_key: impl Into<String>) -> Self {
        ClientBuilder {
            base_url: base_url.into(),
            api_key: api_key.into(),
            admin_token: None,
            timeout: None,
            http: None,
        }
    }

    /// Set the per-request HTTP timeout. Ignored if a custom [`reqwest::Client`] is
    /// supplied via [`ClientBuilder::http_client`] (configure the timeout there instead).
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Set the optional `x-admin-token`, sent on every request (it only affects
    /// `GET /v1/models`, where it adds the per-model `endpoints` array).
    pub fn admin_token(mut self, token: impl Into<String>) -> Self {
        self.admin_token = Some(token.into());
        self
    }

    /// Supply a fully pre-configured [`reqwest::Client`] (connection pools, proxies,
    /// custom TLS, …). When set, the builder's `timeout` is not applied — configure it on
    /// the supplied client.
    pub fn http_client(mut self, client: reqwest::Client) -> Self {
        self.http = Some(client);
        self
    }

    /// Build the [`Client`].
    pub fn build(self) -> Result<Client> {
        // Normalise the base URL: trim a single trailing slash so path joining is simple.
        let base_url = self.base_url.trim_end_matches('/').to_string();
        if base_url.is_empty() {
            return Err(Error::InvalidBaseUrl("base url is empty".to_string()));
        }

        let http = match self.http {
            Some(c) => c,
            None => {
                let mut b = reqwest::Client::builder();
                if let Some(t) = self.timeout {
                    b = b.timeout(t);
                }
                b.build()?
            }
        };

        let bearer = HeaderValue::from_str(&format!("Bearer {}", self.api_key))
            .map_err(|_| Error::InvalidBaseUrl("api key contains invalid header bytes".into()))?;
        let admin = self
            .admin_token
            .map(|t| HeaderValue::from_str(&t))
            .transpose()
            .map_err(|_| {
                Error::InvalidBaseUrl("admin token contains invalid header bytes".into())
            })?;

        Ok(Client {
            http,
            base_url,
            bearer,
            admin_token: admin,
        })
    }
}

/// Async client for the llmleaf gateway.
///
/// Construct with [`Client::new`] for the common case, or [`Client::builder`] to set a
/// timeout, an admin token, or supply your own [`reqwest::Client`]. The client is cheap
/// to clone (it wraps an `Arc`-backed `reqwest::Client`).
#[derive(Debug, Clone)]
pub struct Client {
    http: reqwest::Client,
    base_url: String,
    bearer: HeaderValue,
    admin_token: Option<HeaderValue>,
}

impl Client {
    /// Construct a client from a base URL and API key with default settings.
    pub fn new(base_url: impl Into<String>, api_key: impl Into<String>) -> Result<Self> {
        ClientBuilder::new(base_url, api_key).build()
    }

    /// Start a [`ClientBuilder`].
    pub fn builder(base_url: impl Into<String>, api_key: impl Into<String>) -> ClientBuilder {
        ClientBuilder::new(base_url, api_key)
    }

    // -- internal request plumbing ------------------------------------------

    fn url(&self, path: &str) -> String {
        format!("{}{}", self.base_url, path)
    }

    /// Start a request with the bearer (and admin, if set) headers applied.
    fn request(&self, method: Method, path: &str) -> RequestBuilder {
        let mut req = self
            .http
            .request(method, self.url(path))
            .header(AUTHORIZATION, self.bearer.clone());
        if let Some(admin) = &self.admin_token {
            req = req.header(ADMIN_TOKEN_HEADER, admin.clone());
        }
        req
    }

    /// Send a request and, on a non-2xx status, parse the error envelope into
    /// [`Error::Api`].
    async fn send(&self, req: RequestBuilder) -> Result<Response> {
        let resp = req.send().await?;
        check_status(resp).await
    }

    // -- chat ----------------------------------------------------------------

    /// `POST /v1/chat/completions` (non-streaming).
    ///
    /// `stream` is forced to `false` (or absent) so a JSON `ChatResponse` comes back.
    pub async fn chat(&self, mut request: ChatRequest) -> Result<ChatResponse> {
        request.stream = None;
        let resp = self
            .send(
                self.request(Method::POST, "/v1/chat/completions")
                    .json(&request),
            )
            .await?;
        Ok(resp.json::<ChatResponse>().await?)
    }

    /// `POST /v1/chat/completions` (streaming, SSE).
    ///
    /// Forces `stream:true`, then yields decoded [`ChatCompletionChunk`]s, stopping on the
    /// `data: [DONE]` sentinel (which is never parsed). Accumulate
    /// `choices[].delta.content` for the assembled text.
    pub async fn chat_stream(
        &self,
        mut request: ChatRequest,
    ) -> Result<impl Stream<Item = Result<ChatCompletionChunk>>> {
        request.stream = Some(true);
        let resp = self
            .send(
                self.request(Method::POST, "/v1/chat/completions")
                    .json(&request),
            )
            .await?;
        Ok(sse_chunks(resp.bytes_stream()))
    }

    // -- responses -----------------------------------------------------------

    /// `POST /v1/responses` (non-streaming) — the OpenAI Responses dialect.
    ///
    /// `stream` is forced to `false` (or absent) so a JSON [`ResponsesResponse`] comes
    /// back. llmleaf serves this dialect statelessly, so the response always reports
    /// `"store": false` (SPEC.md).
    pub async fn responses(&self, mut request: ResponsesRequest) -> Result<ResponsesResponse> {
        request.stream = None;
        let resp = self
            .send(self.request(Method::POST, "/v1/responses").json(&request))
            .await?;
        Ok(resp.json::<ResponsesResponse>().await?)
    }

    /// `POST /v1/responses` (streaming, typed SSE).
    ///
    /// Forces `stream:true`, then yields decoded [`ResponsesStreamEvent`]s. Unlike chat
    /// there is no `[DONE]` sentinel: the stream ends after the terminal
    /// `response.completed` / `response.incomplete` / `response.failed` event. Unrecognised
    /// event types are skipped; the `"error"` event surfaces as an [`Error::Api`], the same
    /// way the chat stream surfaces a mid-stream failure. Accumulate
    /// [`ResponsesStreamEvent::output_text_delta`] for the assembled text.
    pub async fn responses_stream(
        &self,
        mut request: ResponsesRequest,
    ) -> Result<impl Stream<Item = Result<ResponsesStreamEvent>>> {
        request.stream = Some(true);
        let resp = self
            .send(self.request(Method::POST, "/v1/responses").json(&request))
            .await?;
        Ok(sse_responses(resp.bytes_stream()))
    }

    // -- embeddings ----------------------------------------------------------

    /// `POST /v1/embeddings`. Base64 payloads (`encoding_format:"base64"`) are decoded
    /// into float vectors before returning.
    pub async fn embeddings(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse> {
        let resp = self
            .send(self.request(Method::POST, "/v1/embeddings").json(&request))
            .await?;
        let raw = resp.json::<RawEmbeddingResponse>().await?;
        raw.into_public()
    }

    // -- rerank ----------------------------------------------------------------

    /// `POST /v1/rerank`. Scores each of `request.documents` against `request.query`
    /// and returns the results ordered by relevance. `documents` may be plain strings or
    /// structured multimodal objects; when `return_documents` is set the originals are
    /// echoed back on each [`RerankResult`].
    pub async fn rerank(&self, request: RerankRequest) -> Result<RerankResponse> {
        let resp = self
            .send(self.request(Method::POST, "/v1/rerank").json(&request))
            .await?;
        Ok(resp.json::<RerankResponse>().await?)
    }

    // -- models --------------------------------------------------------------

    /// `GET /v1/models`. `type` filters the catalog; `search` is a substring match. The
    /// per-model `endpoints` array appears only when the client was built with an admin
    /// token.
    pub async fn list_models(
        &self,
        model_type: Option<ModelType>,
        search: Option<&str>,
    ) -> Result<ListModelsResponse> {
        let mut req = self.request(Method::GET, "/v1/models");
        let mut query: Vec<(&str, String)> = Vec::new();
        if let Some(t) = model_type {
            query.push(("type", t.as_str().to_string()));
        }
        if let Some(s) = search {
            query.push(("search", s.to_string()));
        }
        if !query.is_empty() {
            req = req.query(&query);
        }
        let resp = self.send(req).await?;
        Ok(resp.json::<ListModelsResponse>().await?)
    }

    // -- audio: speech (TTS) -------------------------------------------------

    /// `POST /v1/audio/speech`. Returns the raw audio bytes and the resolved
    /// `Content-Type` (from the response header, falling back to the SPEC.md table for the
    /// requested `response_format`).
    pub async fn speech(&self, request: SpeechRequest) -> Result<(Bytes, String)> {
        let fallback = speech_content_type(request.response_format.as_deref());
        let resp = self
            .send(
                self.request(Method::POST, "/v1/audio/speech")
                    .json(&request),
            )
            .await?;
        let content_type = resp
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string())
            .unwrap_or_else(|| fallback.to_string());
        let bytes = resp.bytes().await?;
        Ok((bytes, content_type))
    }

    /// `GET /v1/audio/voices?model=<id>`.
    pub async fn voices(&self, model: &str) -> Result<VoicesResponse> {
        let resp = self
            .send(
                self.request(Method::GET, "/v1/audio/voices")
                    .query(&[("model", model)]),
            )
            .await?;
        Ok(resp.json::<VoicesResponse>().await?)
    }

    // -- audio: transcriptions (STT) -----------------------------------------

    /// `POST /v1/audio/transcriptions` (multipart). `file` carries the audio bytes under
    /// `file_name`. Returns a structured [`Transcription::Json`] for json/verbose_json, or
    /// a plain-text [`Transcription::Text`] for text/srt/vtt (SPEC.md).
    pub async fn transcribe(
        &self,
        request: TranscriptionRequest,
        file_name: impl Into<String>,
        audio: impl Into<Bytes>,
    ) -> Result<Transcription> {
        let audio: Bytes = audio.into();
        let part = reqwest::multipart::Part::bytes(audio.to_vec()).file_name(file_name.into());

        let mut form = reqwest::multipart::Form::new()
            .text("model", request.model.clone())
            .part("file", part);
        if let Some(language) = &request.language {
            form = form.text("language", language.clone());
        }
        if let Some(prompt) = &request.prompt {
            form = form.text("prompt", prompt.clone());
        }
        if let Some(rf) = &request.response_format {
            form = form.text("response_format", rf.clone());
        }
        if let Some(t) = request.temperature {
            form = form.text("temperature", t.to_string());
        }

        let resp = self
            .send(
                self.request(Method::POST, "/v1/audio/transcriptions")
                    .multipart(form),
            )
            .await?;

        // json/verbose_json → structured body; text/srt/vtt → plain text.
        let structured = matches!(
            request.response_format.as_deref(),
            None | Some("json") | Some("verbose_json")
        );
        if structured {
            Ok(Transcription::Json(
                resp.json::<TranscriptionResponse>().await?,
            ))
        } else {
            Ok(Transcription::Text(resp.text().await?))
        }
    }

    // -- batches -------------------------------------------------------------

    /// `POST /v1/batches`.
    pub async fn create_batch(&self, request: BatchCreateRequest) -> Result<BatchHandle> {
        let resp = self
            .send(self.request(Method::POST, "/v1/batches").json(&request))
            .await?;
        Ok(resp.json::<BatchHandle>().await?)
    }

    /// `GET /v1/batches/{id}`.
    pub async fn get_batch(&self, id: &str) -> Result<BatchHandle> {
        let resp = self
            .send(self.request(Method::GET, &format!("/v1/batches/{id}")))
            .await?;
        Ok(resp.json::<BatchHandle>().await?)
    }

    /// `POST /v1/batches/{id}/cancel`.
    pub async fn cancel_batch(&self, id: &str) -> Result<BatchHandle> {
        let resp = self
            .send(self.request(Method::POST, &format!("/v1/batches/{id}/cancel")))
            .await?;
        Ok(resp.json::<BatchHandle>().await?)
    }

    /// `GET /v1/batches/{id}/results` (`application/x-ndjson`). Yields one
    /// [`BatchResultLine`] per line.
    pub async fn batch_results(
        &self,
        id: &str,
    ) -> Result<impl Stream<Item = Result<BatchResultLine>>> {
        let resp = self
            .send(self.request(Method::GET, &format!("/v1/batches/{id}/results")))
            .await?;
        Ok(ndjson_lines(resp.bytes_stream()))
    }
}

/// Inspect the response status: on non-2xx, drain the body and parse the
/// `{"error":{"message":...}}` envelope into [`Error::Api`].
async fn check_status(resp: Response) -> Result<Response> {
    let status = resp.status();
    if status.is_success() {
        return Ok(resp);
    }
    let code = status.as_u16();
    let body = resp.bytes().await.unwrap_or_default();
    let message = parse_error_message(&body).unwrap_or_else(|| {
        status
            .canonical_reason()
            .unwrap_or("request failed")
            .to_string()
    });
    Err(Error::Api {
        status: code,
        message,
    })
}

/// Best-effort extraction of `error.message` from an error body.
fn parse_error_message(body: &[u8]) -> Option<String> {
    serde_json::from_slice::<ErrorEnvelope>(body)
        .ok()
        .map(|e| e.error.message)
        .filter(|m| !m.is_empty())
}
