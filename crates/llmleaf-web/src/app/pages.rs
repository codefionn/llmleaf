//! The routed pages: Dashboard, Keys (+ verdict controls), Accounting, Core (routes/health), Events,
//! and Login. Each loads via a server-function `Resource` and renders inside a `Suspense`.

use leptos::prelude::*;
use leptos_router::hooks::use_query_map;

use super::server;
use super::{fmt_cost, fmt_hms, fmt_int, fmt_tokens};
use crate::dto::{
    Accounting, CoreStatus, EventRow, KeyRow, NewKey, Overview, UsageBy, UsageTotals,
};

// ---------------------------------------------------------------------------------------------
// Small shared building blocks
// ---------------------------------------------------------------------------------------------

#[component]
fn ErrorBox(msg: String) -> impl IntoView {
    view! { <div class="error-box">{msg}</div> }
}

#[component]
fn Loading() -> impl IntoView {
    view! { <p class="muted">"Loading…"</p> }
}

#[component]
fn Stat(
    label: &'static str,
    value: String,
    #[prop(optional)] sub: Option<String>,
) -> impl IntoView {
    view! {
        <div class="stat">
            <div class="stat-label">{label}</div>
            <div class="stat-value">{value}</div>
            {sub.map(|s| view! { <div class="stat-sub">{s}</div> })}
        </div>
    }
}

/// A dependency-free SVG bar chart. Stretches to its container; bars scale to the series max.
#[component]
fn BarChart(values: Vec<f64>) -> impl IntoView {
    let max = values.iter().copied().fold(0.0_f64, f64::max).max(1.0);
    let n = values.len().max(1);
    let slot = 100.0 / n as f64;
    let bars = values
        .into_iter()
        .enumerate()
        .map(|(i, v)| {
            let h = (v / max) * 100.0;
            let x = i as f64 * slot;
            view! {
                <rect
                    x=format!("{:.3}", x + slot * 0.1)
                    y=format!("{:.3}", 100.0 - h)
                    width=format!("{:.3}", slot * 0.8)
                    height=format!("{:.3}", h)
                    class="bar"
                />
            }
        })
        .collect_view();
    view! {
        <svg class="chart" viewBox="0 0 100 100" preserveAspectRatio="none">
            {bars}
        </svg>
    }
}

#[component]
fn TotalsRow(totals: UsageTotals) -> impl IntoView {
    view! {
        <div class="stats">
            <Stat label="Requests" value=fmt_int(totals.requests)/>
            <Stat label="Prompt tok" value=fmt_tokens(totals.prompt_tokens)/>
            <Stat label="Completion tok" value=fmt_tokens(totals.completion_tokens)/>
            <Stat label="Total tok" value=fmt_tokens(totals.total_tokens)/>
            <Stat label="Cost" value=fmt_cost(totals.cost)/>
        </div>
    }
}

// ---------------------------------------------------------------------------------------------
// Dashboard
// ---------------------------------------------------------------------------------------------

#[component]
pub fn Dashboard() -> impl IntoView {
    let data = Resource::new(|| (), |_| async move { server::get_overview().await });
    view! {
        <h1>"Dashboard"</h1>
        <Suspense fallback=|| view! { <Loading/> }>
            {move || data.get().map(|res| match res {
                Ok(o) => view! { <DashboardView overview=o/> }.into_any(),
                Err(e) => view! { <ErrorBox msg=e.to_string()/> }.into_any(),
            })}
        </Suspense>
    }
}

#[component]
fn DashboardView(overview: Overview) -> impl IntoView {
    let series: Vec<f64> = overview
        .series_24h
        .iter()
        .map(|p| p.total_tokens as f64)
        .collect();
    let have_series = !series.is_empty();
    let top = overview.top_models.clone();
    view! {
        <section class="cards">
            <div class="card">
                <h2>"Last 24 hours"</h2>
                <TotalsRow totals=overview.totals_24h/>
                {have_series.then(|| view! {
                    <div class="chart-wrap">
                        <BarChart values=series/>
                        <div class="muted small">"tokens / hour"</div>
                    </div>
                })}
            </div>
            <div class="card">
                <h2>"Keys"</h2>
                <div class="stats">
                    <Stat label="Total keys" value=fmt_int(overview.active_keys)/>
                    <Stat label="Blocked" value=fmt_int(overview.blocked_keys)/>
                </div>
                <h2 class="mt">"All-time cost"</h2>
                <div class="stat-value big">{fmt_cost(overview.totals_all.cost)}</div>
                <div class="muted">{format!("{} requests all-time", fmt_int(overview.totals_all.requests))}</div>
            </div>
        </section>
        <div class="card">
            <h2>"Top models (24h)"</h2>
            <UsageTable rows=top empty="No usage in the last 24 hours."/>
        </div>
    }
}

/// A label + totals table, reused by the dashboard and accounting.
#[component]
fn UsageTable(rows: Vec<UsageBy>, empty: &'static str) -> impl IntoView {
    if rows.is_empty() {
        return view! { <p class="muted">{empty}</p> }.into_any();
    }
    let body = rows
        .into_iter()
        .map(|r| {
            view! {
                <tr>
                    <td class="mono">{r.label}</td>
                    <td class="num">{fmt_int(r.totals.requests)}</td>
                    <td class="num">{fmt_tokens(r.totals.total_tokens)}</td>
                    <td class="num">{fmt_cost(r.totals.cost)}</td>
                </tr>
            }
        })
        .collect_view();
    view! {
        <table class="grid">
            <thead>
                <tr><th>"Name"</th><th class="num">"Requests"</th><th class="num">"Tokens"</th><th class="num">"Cost"</th></tr>
            </thead>
            <tbody>{body}</tbody>
        </table>
    }
    .into_any()
}

// ---------------------------------------------------------------------------------------------
// Keys + verdict controls
// ---------------------------------------------------------------------------------------------

#[component]
pub fn KeysPage() -> impl IntoView {
    let reload = RwSignal::new(0u32);
    let keys = Resource::new(
        move || reload.get(),
        |_| async move { server::list_keys().await },
    );

    // Mutations. Each refetches the roster on completion.
    let create = Action::new(|input: &NewKey| {
        let input = input.clone();
        async move { server::create_key(input).await }
    });
    let delete = Action::new(|id: &String| {
        let id = id.clone();
        async move { server::delete_key(id).await }
    });
    let block = Action::new(|(id, b): &(String, bool)| {
        let (id, b) = (id.clone(), *b);
        async move { server::block_key(id, b).await }
    });
    let suspend = Action::new(|(id, secs): &(String, u64)| {
        let (id, secs) = (id.clone(), *secs);
        async move { server::suspend_key(id, secs).await }
    });

    // Any completed mutation triggers a roster refetch.
    Effect::new(move |_| {
        create.version().track();
        delete.version().track();
        block.version().track();
        suspend.version().track();
        reload.update(|n| *n = n.wrapping_add(1));
    });

    view! {
        <h1>"Keys"</h1>
        <div class="card">
            <h2>"Issue a key"</h2>
            <CreateKeyForm action=create/>
        </div>
        <div class="card">
            <h2>"Roster"</h2>
            <Suspense fallback=|| view! { <Loading/> }>
                {move || keys.get().map(|res| match res {
                    Ok(rows) if rows.is_empty() => view! { <p class="muted">"No keys yet. Issue one above."</p> }.into_any(),
                    Ok(rows) => {
                        let body = rows.into_iter().map(|k| view! {
                            <KeyRowView k=k delete=delete block=block suspend=suspend/>
                        }).collect_view();
                        view! {
                            <table class="grid">
                                <thead>
                                    <tr>
                                        <th>"ID"</th><th>"Name"</th><th>"Status"</th>
                                        <th>"Allowed models"</th><th>"Last seen"</th><th>"Actions"</th>
                                    </tr>
                                </thead>
                                <tbody>{body}</tbody>
                            </table>
                        }.into_any()
                    }
                    Err(e) => view! { <ErrorBox msg=e.to_string()/> }.into_any(),
                })}
            </Suspense>
        </div>
    }
}

#[component]
fn CreateKeyForm(
    action: Action<NewKey, Result<crate::dto::NewKeyResult, ServerFnError>>,
) -> impl IntoView {
    let id = RwSignal::new(String::new());
    let name = RwSignal::new(String::new());
    let models = RwSignal::new(String::new());
    let password = RwSignal::new(String::new());

    let submit = move |_| {
        action.dispatch(NewKey {
            id: id.get(),
            name: {
                let n = name.get();
                (!n.trim().is_empty()).then_some(n)
            },
            allowed_models: models.get(),
            password: password.get(),
        });
    };

    view! {
        <div class="form-row">
            <input placeholder="key id (e.g. team-alpha)" prop:value=move || id.get()
                on:input=move |e| id.set(event_target_value(&e))/>
            <input placeholder="name (optional)" prop:value=move || name.get()
                on:input=move |e| name.set(event_target_value(&e))/>
            <input placeholder="allowed models (comma sep; blank = all)" prop:value=move || models.get()
                on:input=move |e| models.set(event_target_value(&e))/>
            <input placeholder="password (blank = generate)" prop:value=move || password.get()
                on:input=move |e| password.set(event_target_value(&e))/>
            <button class="btn" on:click=submit disabled=move || action.pending().get()>"Create"</button>
        </div>
        {move || action.value().get().map(|res| match res {
            Ok(result) => view! {
                <div class="reveal">
                    <strong>"Key issued — copy the bearer token now (shown once):"</strong>
                    <code class="token">{format!("Authorization: Bearer {}", result.bearer_token)}</code>
                    <div class="muted small">{format!("id: {} · password: {}", result.id, result.password)}</div>
                </div>
            }.into_any(),
            Err(e) => view! { <ErrorBox msg=e.to_string()/> }.into_any(),
        })}
    }
}

#[component]
fn KeyRowView(
    k: KeyRow,
    delete: Action<String, Result<(), ServerFnError>>,
    block: Action<(String, bool), Result<(), ServerFnError>>,
    suspend: Action<(String, u64), Result<(), ServerFnError>>,
) -> impl IntoView {
    let id = k.id.clone();
    let suspended = k.suspended_until.is_some();
    let status = if k.blocked {
        view! { <span class="badge danger">"blocked"</span> }.into_any()
    } else if suspended {
        view! { <span class="badge warn">"suspended"</span> }.into_any()
    } else {
        view! { <span class="badge ok">"active"</span> }.into_any()
    };

    let allowed = if k.allowed_models.is_empty() {
        "all".to_string()
    } else {
        k.allowed_models.join(", ")
    };
    let narrowed = (!k.verdict_allowed_models.is_empty())
        .then(|| format!(" → {}", k.verdict_allowed_models.join(", ")));

    let last_seen = k
        .last_seen_ms
        .map(fmt_hms)
        .unwrap_or_else(|| "—".to_string());

    let (id_b, id_s, id_d, id_c) = (id.clone(), id.clone(), id.clone(), id.clone());
    let blocked_now = k.blocked;

    view! {
        <tr>
            <td class="mono">{k.id.clone()}</td>
            <td>{k.name.clone().unwrap_or_default()}</td>
            <td>{status}</td>
            <td class="small">{allowed}{narrowed}</td>
            <td class="num small">{last_seen}</td>
            <td class="actions">
                <button class="btn-sm" on:click=move |_| { block.dispatch((id_b.clone(), !blocked_now)); }>
                    {if blocked_now { "Unblock" } else { "Block" }}
                </button>
                <button class="btn-sm" on:click=move |_| { suspend.dispatch((id_s.clone(), 3600)); }>"Suspend 1h"</button>
                {suspended.then({
                    let id_c = id_c.clone();
                    move || view! {
                        <button class="btn-sm" on:click=move |_| { suspend.dispatch((id_c.clone(), 0)); }>"Resume"</button>
                    }
                })}
                <button class="btn-sm danger" on:click=move |_| { delete.dispatch(id_d.clone()); }>"Delete"</button>
            </td>
        </tr>
    }
}

// ---------------------------------------------------------------------------------------------
// Accounting
// ---------------------------------------------------------------------------------------------

#[component]
pub fn AccountingPage() -> impl IntoView {
    let days = RwSignal::new(30u32);
    let data = Resource::new(
        move || days.get(),
        |d| async move { server::get_accounting(d).await },
    );

    view! {
        <h1>"Accounting"</h1>
        <div class="form-row">
            <label class="muted">"Window:"</label>
            {[1u32, 7, 30, 90].into_iter().map(|d| {
                view! {
                    <button class="btn-sm"
                        class:active=move || days.get() == d
                        on:click=move |_| days.set(d)>
                        {format!("{d}d")}
                    </button>
                }
            }).collect_view()}
        </div>
        <Suspense fallback=|| view! { <Loading/> }>
            {move || data.get().map(|res| match res {
                Ok(acc) => view! { <AccountingView acc=acc/> }.into_any(),
                Err(e) => view! { <ErrorBox msg=e.to_string()/> }.into_any(),
            })}
        </Suspense>
    }
}

#[component]
fn AccountingView(acc: Accounting) -> impl IntoView {
    view! {
        <div class="card">
            <h2>{format!("Totals · last {} days", acc.window_days)}</h2>
            <TotalsRow totals=acc.totals/>
        </div>
        <section class="cards">
            <div class="card">
                <h2>"By key"</h2>
                <UsageTable rows=acc.by_key empty="No usage attributed to keys in this window."/>
            </div>
            <div class="card">
                <h2>"By model"</h2>
                <UsageTable rows=acc.by_model empty="No model usage in this window."/>
            </div>
        </section>
    }
}

// ---------------------------------------------------------------------------------------------
// Core (routes + health) — the read-only admin-GET mirror
// ---------------------------------------------------------------------------------------------

#[component]
pub fn RoutesPage() -> impl IntoView {
    let data = Resource::new(|| (), |_| async move { server::get_core_status().await });
    view! {
        <h1>"Core"</h1>
        <p class="muted">"A read-only mirror of the core's admin surface. The web app only ever observes the core."</p>
        <Suspense fallback=|| view! { <Loading/> }>
            {move || data.get().map(|res| match res {
                Ok(s) => view! { <CoreStatusView status=s/> }.into_any(),
                Err(e) => view! { <ErrorBox msg=e.to_string()/> }.into_any(),
            })}
        </Suspense>
    }
}

#[component]
fn CoreStatusView(status: CoreStatus) -> impl IntoView {
    if !status.reachable {
        let why = status.error.unwrap_or_else(|| "unreachable".to_string());
        return view! {
            <div class="card">
                <span class="badge danger">"unreachable"</span>
                <p class="muted">{why}</p>
            </div>
        }
        .into_any();
    }

    let models = status.routes.models.clone();
    let prefixes = status.routes.prefixes.clone();
    let health = status.health.clone();

    let model_items = models
        .into_iter()
        .map(|m| view! { <li class="mono">{m}</li> })
        .collect_view();
    let prefix_items = prefixes
        .into_iter()
        .map(|p| view! { <li><span class="mono">{format!("{}/*", p.prefix)}</span>" → "<span class="muted">{p.provider}</span></li> })
        .collect_view();
    let health_items = if health.is_empty() {
        view! { <li class="muted">"all providers healthy"</li> }.into_any()
    } else {
        health
            .into_iter()
            .map(|h| {
                let badge = if h.down {
                    view! { <span class="badge danger">"down"</span> }.into_any()
                } else {
                    view! { <span class="badge ok">"up"</span> }.into_any()
                };
                view! { <li><span class="mono">{h.provider}</span>" "{badge}</li> }
            })
            .collect_view()
            .into_any()
    };

    view! {
        <span class="badge ok">"reachable"</span>
        <section class="cards">
            <div class="card"><h2>"Models"</h2><ul class="list">{model_items}</ul></div>
            <div class="card"><h2>"Prefix routes"</h2><ul class="list">{prefix_items}</ul></div>
            <div class="card"><h2>"Provider health"</h2><ul class="list">{health_items}</ul></div>
        </section>
    }
    .into_any()
}

// ---------------------------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------------------------

#[component]
pub fn EventsPage() -> impl IntoView {
    let reload = RwSignal::new(0u32);
    let data = Resource::new(
        move || reload.get(),
        |_| async move { server::recent_events(200).await },
    );
    view! {
        <h1>"Events"</h1>
        <div class="form-row">
            <button class="btn-sm" on:click=move |_| reload.update(|n| *n += 1)>"Refresh"</button>
            <span class="muted">"The most recent lifecycle/usage events pushed by the core."</span>
        </div>
        <div class="card">
            <Suspense fallback=|| view! { <Loading/> }>
                {move || data.get().map(|res| match res {
                    Ok(rows) if rows.is_empty() => view! { <p class="muted">"No events received yet."</p> }.into_any(),
                    Ok(rows) => view! { <EventsTable rows=rows/> }.into_any(),
                    Err(e) => view! { <ErrorBox msg=e.to_string()/> }.into_any(),
                })}
            </Suspense>
        </div>
    }
}

#[component]
fn EventsTable(rows: Vec<EventRow>) -> impl IntoView {
    let body = rows
        .into_iter()
        .map(|e| {
            view! {
                <tr>
                    <td class="num small mono">{fmt_hms(e.ts_ms)}</td>
                    <td><span class=format!("badge kind-{}", e.kind)>{e.kind.clone()}</span></td>
                    <td class="mono small">{e.request_id}</td>
                    <td class="small">{e.key.unwrap_or_default()}</td>
                    <td class="small mono">{e.model.or(e.provider).unwrap_or_default()}</td>
                    <td class="small">{e.detail}</td>
                </tr>
            }
        })
        .collect_view();
    view! {
        <table class="grid">
            <thead>
                <tr><th>"Time"</th><th>"Kind"</th><th>"Request"</th><th>"Key"</th><th>"Model / Provider"</th><th>"Detail"</th></tr>
            </thead>
            <tbody>{body}</tbody>
        </table>
    }
}

// ---------------------------------------------------------------------------------------------
// Login (no server functions — plain form POST to the axum auth handlers)
// ---------------------------------------------------------------------------------------------

#[component]
pub fn LoginPage() -> impl IntoView {
    let query = use_query_map();
    let has_error = move || query.read().get("error").is_some();
    let redirect_to = move || query.read().get("redirect_to").unwrap_or_default();

    view! {
        <div class="login-wrap">
            <form class="login card" method="post" action="/auth/login">
                <h1><span class="leaf">"🍃"</span>" llmleaf control"</h1>
                <p class="muted">"Sign in to manage keys, verdicts, and usage."</p>
                {move || has_error().then(|| view! { <div class="error-box">"Sign-in failed. Try again."</div> })}
                <input type="hidden" name="redirect_to" prop:value=redirect_to/>
                <label>"Master password"</label>
                <input type="password" name="password" autocomplete="current-password" autofocus=true/>
                <button class="btn" type="submit">"Sign in"</button>
                <a class="btn-ghost" href=move || format!("/auth/oidc/login?redirect_to={}", redirect_to())>
                    "Sign in with SSO"
                </a>
            </form>
        </div>
    }
}
