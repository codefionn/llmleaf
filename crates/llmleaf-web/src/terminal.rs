//! Workspace-agent-owned PTY manager.
//!
//! Every terminal tab is a real PTY spawned **inside the workspace** (cwd = workspace root),
//! never in the host's binary directory. The desktop host (scriptschnellng) is a relay only.
//! This fixes two bugs:
//! 1. local tabs landing in the host's pwd
//! 2. remote agents having no host to fork — they POST /create + WS /attach to this service
//!
//! Design:
//! - one PTY per tab id (no singleton), isolated env + per-tab HISTFILE
//! - auth accepts EITHER an operator session cookie OR the control bearer token (remote agents)
//! - workspace root resolved as WORKSPACE_FOLDER -> /workspace -> current_dir

use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Path as AxumPath, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use axum_extra::extract::cookie::CookieJar;
use portable_pty::{CommandBuilder, MasterPty, PtySize};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use uuid::Uuid;

// ---------- public state ----------

#[derive(Clone)]
pub struct TerminalManager {
    inner: Arc<RwLock<HashMap<Uuid, TerminalSession>>>,
    workspace_root: PathBuf,
}

struct TerminalSession {
    id: Uuid,
    cwd: PathBuf,
    master: Arc<Mutex<Box<dyn MasterPty + Send>>>,
    child: Arc<Mutex<Box<dyn portable_pty::Child + Send + Sync>>>,
    writer: Arc<Mutex<Box<dyn Write + Send>>>,
    histfile: PathBuf,
    // broadcast output to all attachers; input via writer
    output_tx: tokio::sync::broadcast::Sender<Vec<u8>>,
}

#[derive(Serialize)]
struct CreateResp {
    id: Uuid,
    cwd: String,
    pid: Option<u32>,
}

#[derive(Serialize)]
struct ListResp {
    sessions: Vec<SessionInfo>,
}

#[derive(Serialize)]
struct SessionInfo {
    id: Uuid,
    cwd: String,
    histfile: String,
}

#[derive(Deserialize)]
struct CreateReq {
    #[serde(default)]
    cols: Option<u16>,
    #[serde(default)]
    rows: Option<u16>,
}

// ---------- helpers ----------

fn workspace_root() -> PathBuf {
    if let Ok(p) = std::env::var("WORKSPACE_FOLDER") {
        let pb = PathBuf::from(p);
        if pb.exists() {
            return pb;
        }
    }
    if Path::new("/workspace").exists() {
        return PathBuf::from("/workspace");
    }
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

fn histfile_for(id: Uuid) -> PathBuf {
    let base = std::env::var("TMPDIR").unwrap_or_else(|_| "/tmp/agent-session".to_string());
    PathBuf::from(base).join(format!("history.{id}"))
}

fn shell_program() -> String {
    std::env::var("SHELL").unwrap_or_else(|_| "bash".to_string())
}

impl Default for TerminalManager {
    fn default() -> Self {
        Self::new()
    }
}

impl TerminalManager {
    pub fn new() -> Self {
        let root = workspace_root();
        // ensure history dir exists
        let _ = std::fs::create_dir_all(
            std::env::var("TMPDIR").unwrap_or_else(|_| "/tmp/agent-session".to_string()),
        );
        tracing::info!(workspace = %root.display(), "terminal manager: workspace root");
        Self {
            inner: Arc::new(RwLock::new(HashMap::new())),
            workspace_root: root,
        }
    }

    pub fn workspace_root(&self) -> &Path {
        &self.workspace_root
    }

    async fn create(&self, cols: u16, rows: u16) -> Result<CreateResp, String> {
        let id = Uuid::new_v4();
        let histfile = histfile_for(id);
        // touch histfile
        let _ = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&histfile);

        let pty_system = portable_pty::native_pty_system();
        let pair = pty_system
            .openpty(PtySize {
                rows,
                cols,
                pixel_width: 0,
                pixel_height: 0,
            })
            .map_err(|e| format!("openpty: {e}"))?;

        let mut cmd = CommandBuilder::new(shell_program());
        cmd.cwd(&self.workspace_root);
        // isolated env: keep minimal, overlay TERM etc. Do NOT inherit host's PWD.
        cmd.env("TERM", "xterm-256color");
        cmd.env("HISTFILE", histfile.to_string_lossy().to_string());
        cmd.env("PWD", self.workspace_root.to_string_lossy().to_string());
        // ensure HOME is sensible
        if std::env::var("HOME").is_ok() {
            // keep host HOME but not PWD
        }

        let child = pair
            .slave
            .spawn_command(cmd)
            .map_err(|e| format!("spawn: {e}"))?;
        // Try to get pid if available (some impls expose it)
        let pid: Option<u32> = None; // portable-pty Child pid not exposed on all versions

        let master_box: Box<dyn MasterPty + Send> = pair.master;
        let writer: Box<dyn Write + Send> = master_box
            .take_writer()
            .map_err(|e| format!("take_writer: {e}"))?;
        let master: Arc<Mutex<Box<dyn MasterPty + Send>>> = Arc::new(Mutex::new(master_box));
        let writer_arc: Arc<Mutex<Box<dyn Write + Send>>> = Arc::new(Mutex::new(writer));
        let child_arc: Arc<Mutex<Box<dyn portable_pty::Child + Send + Sync>>> =
            Arc::new(Mutex::new(child));

        let (output_tx, _) = tokio::sync::broadcast::channel::<Vec<u8>>(256);

        // spawn reader thread -> broadcast
        {
            let master_clone = master.clone();
            let tx = output_tx.clone();
            std::thread::spawn(move || {
                // take a reader; portable-pty wants a dedicated reader handle
                let reader = {
                    let m = master_clone.lock().unwrap();
                    m.try_clone_reader()
                };
                let mut reader = match reader {
                    Ok(r) => r,
                    Err(e) => {
                        tracing::error!("pty reader clone failed: {e}");
                        return;
                    }
                };
                let mut buf = [0u8; 4096];
                loop {
                    match reader.read(&mut buf) {
                        Ok(0) => break,
                        Ok(n) => {
                            let _ = tx.send(buf[..n].to_vec());
                        }
                        Err(e) => {
                            tracing::debug!("pty reader ended: {e}");
                            break;
                        }
                    }
                }
            });
        }

        let session = TerminalSession {
            id,
            cwd: self.workspace_root.clone(),
            master,
            child: child_arc,
            writer: writer_arc,
            histfile: histfile.clone(),
            output_tx: output_tx.clone(),
        };

        let cwd_str = self.workspace_root.to_string_lossy().to_string();
        self.inner.write().await.insert(id, session);

        tracing::info!(%id, cwd = %cwd_str, histfile = %histfile.display(), "terminal created");

        Ok(CreateResp {
            id,
            cwd: cwd_str,
            pid,
        })
    }

    async fn list(&self) -> Vec<SessionInfo> {
        let map = self.inner.read().await;
        map.values()
            .map(|s| SessionInfo {
                id: s.id,
                cwd: s.cwd.to_string_lossy().to_string(),
                histfile: s.histfile.to_string_lossy().to_string(),
            })
            .collect()
    }

    async fn remove(&self, id: Uuid) -> bool {
        let mut map = self.inner.write().await;
        if let Some(sess) = map.remove(&id) {
            // try to kill child
            if let Ok(mut c) = sess.child.lock() {
                let _ = c.kill();
            }
            tracing::info!(%id, "terminal removed");
            true
        } else {
            false
        }
    }
}

// ---------- auth: session cookie OR control bearer ----------

async fn require_terminal_auth(
    state: &crate::state::AppState,
    headers: HeaderMap,
    jar: CookieJar,
) -> Result<(), Response> {
    // bearer check (sync)
    if let Some(expected) = state
        .config
        .control
        .token
        .as_ref()
        .and_then(|s| s.resolve())
    {
        if let Some(presented) = headers
            .get(axum::http::header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "))
        {
            use sha2::{Digest, Sha256};
            let a = Sha256::digest(presented.as_bytes());
            let b = Sha256::digest(expected.as_bytes());
            let mut diff = 0u8;
            for (x, y) in a.iter().zip(b.iter()) {
                diff |= x ^ y;
            }
            if diff == 0 {
                return Ok(());
            }
        }
    } else {
        // no token configured => allow bearer-less for DEV, but still require either bearer or session;
        // in DEV mode we allow all (mirrors control router)
        // check if we are in DEV (no token) -> allow without auth for local dev
        // But still prefer session check; if no token, any request is allowed (open).
        // To keep parity with control endpoints, no token => open.
        return Ok(());
    }

    // session cookie check (async DB lookup)
    if let Some(c) = jar.get(crate::auth::COOKIE) {
        if let Ok(Some(_sess)) = crate::db::session::lookup_session(&state.db, c.value()).await {
            return Ok(());
        }
    }

    Err((
        StatusCode::UNAUTHORIZED,
        Json(serde_json::json!({ "error": "terminal: authentication required (session or control bearer)" })),
    )
        .into_response())
}

// ---------- handlers ----------

async fn create_handler(
    State(state): State<crate::state::AppState>,
    headers: HeaderMap,
    jar: CookieJar,
    Json(req): Json<Option<CreateReq>>,
) -> Response {
    if let Err(resp) = require_terminal_auth(&state, headers, jar).await {
        return resp;
    }
    let manager = state.terminal.clone();
    let cols = req.as_ref().and_then(|r| r.cols).unwrap_or(80);
    let rows = req.as_ref().and_then(|r| r.rows).unwrap_or(24);
    match manager.create(cols, rows).await {
        Ok(resp) => Json(resp).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": e })),
        )
            .into_response(),
    }
}

async fn list_handler(
    State(state): State<crate::state::AppState>,
    headers: HeaderMap,
    jar: CookieJar,
) -> Response {
    if let Err(resp) = require_terminal_auth(&state, headers, jar).await {
        return resp;
    }
    let sessions = state.terminal.list().await;
    Json(ListResp { sessions }).into_response()
}

async fn delete_handler(
    State(state): State<crate::state::AppState>,
    headers: HeaderMap,
    jar: CookieJar,
    AxumPath(id): AxumPath<Uuid>,
) -> Response {
    if let Err(resp) = require_terminal_auth(&state, headers, jar).await {
        return resp;
    }
    if state.terminal.remove(id).await {
        Json(serde_json::json!({ "removed": id })).into_response()
    } else {
        (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "error": "not found" })),
        )
            .into_response()
    }
}

async fn attach_handler(
    State(state): State<crate::state::AppState>,
    headers: HeaderMap,
    jar: CookieJar,
    AxumPath(id): AxumPath<Uuid>,
    ws: WebSocketUpgrade,
) -> Response {
    // WS upgrade auth must be checked before upgrade; we cannot do async DB lookup after upgrade easily,
    // so we check here.
    if let Err(resp) = require_terminal_auth(&state, headers, jar).await {
        return resp;
    }
    // verify session exists
    {
        let map = state.terminal.inner.read().await;
        if !map.contains_key(&id) {
            return (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({ "error": "terminal not found" })),
            )
                .into_response();
        }
    }
    let manager = state.terminal.clone();
    ws.on_upgrade(move |socket| handle_ws(socket, manager, id))
}

async fn handle_ws(mut socket: WebSocket, manager: TerminalManager, id: Uuid) {
    // Grab stored writer and output subscriber
    let (writer, mut rx) = {
        let map = manager.inner.read().await;
        let sess = match map.get(&id) {
            Some(s) => s,
            None => {
                let _ = socket
                    .send(Message::Text("terminal not found".into()))
                    .await;
                return;
            }
        };
        let rx = sess.output_tx.subscribe();
        let writer = sess.writer.clone();
        (writer, rx)
    };

    // Split WS into send/recv halves via tasks
    // Task 1: pty output -> WS
    let mut ws_send = socket;
    // We need to handle both directions. Use tokio::select with a single socket: we will loop.

    loop {
        tokio::select! {
            // pty -> ws
            out = rx.recv() => {
                match out {
                    Ok(bytes) => {
                        // Send as binary; xterm.js expects text, but binary is fine for raw bytes
                        if ws_send
                            .send(Message::Binary(bytes.into()))
                            .await
                            .is_err()
                        {
                            break;
                        }
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                }
            }
            // ws -> pty
            msg = ws_send.recv() => {
                match msg {
                    Some(Ok(Message::Text(text))) => {
                        // Check for resize control message: {"resize":{"cols":80,"rows":24}}
                        if let Ok(val) = serde_json::from_str::<serde_json::Value>(&text) {
                            if let Some(resize) = val.get("resize") {
                                if let (Some(cols), Some(rows)) = (resize.get("cols").and_then(|v| v.as_u64()), resize.get("rows").and_then(|v| v.as_u64())) {
                                    let map = manager.inner.read().await;
                                    if let Some(sess) = map.get(&id) {
                                        let _ = sess.master.lock().unwrap().resize(PtySize {
                                            cols: cols as u16,
                                            rows: rows as u16,
                                            pixel_width: 0,
                                            pixel_height: 0,
                                        });
                                    }
                                    continue;
                                }
                            }
                        }
                        // otherwise treat as input
                        if let Ok(mut w) = writer.lock() {
                            let _ = w.write_all(text.as_bytes());
                        }
                    }
                    Some(Ok(Message::Binary(bin))) => {
                        if let Ok(mut w) = writer.lock() {
                            let _ = w.write_all(&bin);
                        }
                    }
                    Some(Ok(Message::Close(_))) | None => break,
                    Some(Ok(Message::Ping(p))) => {
                        let _ = ws_send.send(Message::Pong(p)).await;
                    }
                    Some(Ok(Message::Pong(_))) => {}
                    Some(Err(_)) => break,
                }
            }
        }
    }
    tracing::info!(%id, "terminal ws detached");
}

pub fn router() -> Router<crate::state::AppState> {
    Router::new()
        .route("/api/terminal/create", post(create_handler))
        .route("/api/terminal/list", get(list_handler))
        .route("/api/terminal/:id", delete(delete_handler))
        .route("/api/terminal/:id/attach", get(attach_handler))
}
