//! The Leptos application: the document shell, the routed layout with its nav, and shared presentation
//! helpers. Page components live in [`pages`]; the typed RPC they call lives in [`server`].

use leptos::prelude::*;
use leptos_meta::{provide_meta_context, MetaTags, Stylesheet, Title};
use leptos_router::components::{Route, Router, Routes, A};
use leptos_router::hooks::use_location;
use leptos_router::StaticSegment;

pub mod pages;
pub mod server;

use crate::dto::Session;

/// The HTML document shell rendered by the server (and the file/error fallback).
pub fn shell(options: LeptosOptions) -> impl IntoView {
    view! {
        <!DOCTYPE html>
        <html lang="en">
            <head>
                <meta charset="utf-8"/>
                <meta name="viewport" content="width=device-width, initial-scale=1"/>
                <AutoReload options=options.clone() />
                <HydrationScripts options/>
                <MetaTags/>
            </head>
            <body>
                <App/>
            </body>
        </html>
    }
}

#[component]
pub fn App() -> impl IntoView {
    provide_meta_context();

    view! {
        <Stylesheet id="leptos" href="/pkg/llmleaf-web.css"/>
        <Title text="llmleaf · control"/>
        <Router>
            <Nav/>
            <main class="container">
                <Routes fallback=|| view! { <p class="muted">"Page not found."</p> }>
                    <Route path=StaticSegment("") view=pages::Dashboard/>
                    <Route path=StaticSegment("keys") view=pages::KeysPage/>
                    <Route path=StaticSegment("accounting") view=pages::AccountingPage/>
                    <Route path=StaticSegment("routes") view=pages::RoutesPage/>
                    <Route path=StaticSegment("events") view=pages::EventsPage/>
                    <Route path=StaticSegment("login") view=pages::LoginPage/>
                </Routes>
            </main>
        </Router>
    }
}

/// The top navigation. Links + logout appear only when an operator session resolves; on the login page
/// it stays minimal.
#[component]
fn Nav() -> impl IntoView {
    let location = use_location();
    let who = Resource::new(
        move || location.pathname.get(),
        |_| async move { server::whoami().await.ok().flatten() },
    );

    view! {
        <header class="nav">
            <A href="/" attr:class="brand">
                <span class="leaf">"🍃"</span>
                <span>"llmleaf"</span>
                <span class="muted">"control"</span>
            </A>
            <Suspense fallback=|| ()>
                {move || who.get().flatten().map(|session: Session| view! {
                    <nav class="links">
                        <A href="/">"Dashboard"</A>
                        <A href="/keys">"Keys"</A>
                        <A href="/accounting">"Accounting"</A>
                        <A href="/routes">"Core"</A>
                        <A href="/events">"Events"</A>
                    </nav>
                    <div class="who">
                        <span class="muted">{session.subject}</span>
                        <a class="btn-ghost" href="/auth/logout">"Sign out"</a>
                    </div>
                })}
            </Suspense>
        </header>
    }
}

// ---------------------------------------------------------------------------------------------
// Presentation helpers (shared by the page components; pure, compile both sides)
// ---------------------------------------------------------------------------------------------

/// Group digits with thin separators: `1234567` → `1,234,567`.
pub fn fmt_int(n: u64) -> String {
    let s = n.to_string();
    let bytes = s.as_bytes();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    for (i, b) in bytes.iter().enumerate() {
        if i > 0 && (bytes.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(*b as char);
    }
    out
}

/// USD with four decimals (provider costs are small): `$0.0009`.
pub fn fmt_cost(c: f64) -> String {
    format!("${c:.4}")
}

/// Compact token count: `1.2k`, `3.4M`, else the integer.
pub fn fmt_tokens(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}k", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

/// Time-of-day `HH:MM:SS` (UTC) from unix-ms — enough for an event log without a date library.
pub fn fmt_hms(ms: u64) -> String {
    let s = (ms / 1000) % 86_400;
    format!("{:02}:{:02}:{:02}", s / 3600, (s % 3600) / 60, s % 60)
}

/// A suspension target rendered relative to now (seconds), e.g. `for 3600s` or `(expired)`.
pub fn fmt_suspend(until: u64, now_secs: u64) -> String {
    if until > now_secs {
        format!("suspended {}s", until - now_secs)
    } else {
        "(expired)".to_string()
    }
}
