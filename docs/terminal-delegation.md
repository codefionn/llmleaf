# Terminal delegation — workspace agent owns PTYs

The bug: host (scriptschnellng) forked bash in its own pwd (binary dir) and
shared one global PTY. Fix: host is a relay, workspace agent (llmleaf-web)
owns every PTY.

## Host config (scriptschnellng)

```yaml
pty:
  host: "workspace-agent"           # do not fork locally
  createEndpoint: "http://127.0.0.1:3000/api/terminal/create"
  attachEndpoint: "ws://127.0.0.1:3000/api/terminal/{id}/attach"
  listEndpoint: "http://127.0.0.1:3000/api/terminal/list"
  singleton: false                  # one PTY per tab
  cwd: "${workspaceFolder}"        # resolved inside bwrap, never host pwd
  inheritEnv: false
  historyPerTab: true
  fallbackShell: "/workspace/scripts/isolated-shell.sh"
```

Until host is reconfigured, open each new tab with:

`bash /workspace/scripts/isolated-shell.sh`

or

`bash /tmp/agent-session/isolated-shell.sh`

## Workspace agent contract (llmleaf-web)

- `POST /api/terminal/create  {"cols":80,"rows":24} -> {"id","cwd":"/workspace","pid"}`
- `GET  /api/terminal/list   -> {"sessions":[{"id","cwd","histfile"}]}`
- `WS   /api/terminal/:id/attach  <-> raw PTY bytes, JSON resize: {"resize":{"cols":80,"rows":24}}`
- `DELETE /api/terminal/:id  -> {"removed": id}`

Auth: EITHER operator session cookie OR `Authorization: Bearer <control.token>`.
Remote agents use the bearer path — they have no host to fork, so they POST
to the workspace agent directly. Local UI uses the session cookie.

CWD is always `${WORKSPACE_FOLDER} -> /workspace -> current_dir`, per-tab
HISTFILE is `/tmp/agent-session/history.{id}`, env is isolated.

