//! `llmleaf-bench` — print the time + allocation table for the core hot-path loops and components.
//!
//! Usage:
//!   llmleaf-bench [FILTER] [--filter S] [--csv]
//!
//!   FILTER / --filter S   only run cases whose name contains S (e.g. `keys`, `hotpath`, `openai`).
//!   --csv                 emit CSV instead of the aligned table (for tracking numbers over time).
//!
//! Examples:
//!   cargo run -p llmleaf-bench --release
//!   cargo run -p llmleaf-bench --release -- hotpath
//!   cargo run -p llmleaf-bench --release -- --csv > bench.csv
//!
//! Build in `--release`: an unoptimised hot path is not the hot path SOUL.md is talking about.

use std::process::ExitCode;

use llmleaf_bench::alloc::{self, CountingAllocator};

/// Counting must be live for the allocation columns to mean anything (see [`alloc`]).
#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

fn main() -> ExitCode {
    let opts = match Options::parse(std::env::args().skip(1)) {
        Ok(Some(opts)) => opts,
        Ok(None) => return ExitCode::SUCCESS, // --help
        Err(e) => {
            eprintln!("llmleaf-bench: {e}\n\nrun `llmleaf-bench --help` for usage");
            return ExitCode::FAILURE;
        }
    };

    if !alloc::is_active() {
        eprintln!(
            "llmleaf-bench: the counting allocator is not installed — allocation figures would be \
             meaningless. This is a build bug (the `#[global_allocator]` was dropped)."
        );
        return ExitCode::FAILURE;
    }

    if !cfg!(debug_assertions) {
        // release: nothing to say.
    } else {
        eprintln!(
            "note: running a debug build — timings are not representative. Re-run with `--release`."
        );
    }

    let suite = llmleaf_bench::run(opts.filter);
    if suite.rows().is_empty() {
        eprintln!("llmleaf-bench: no cases matched the filter");
        return ExitCode::FAILURE;
    }

    if opts.csv {
        print!("{}", suite.csv());
    } else {
        print!("{}", suite.table());
    }
    ExitCode::SUCCESS
}

struct Options {
    filter: Option<String>,
    csv: bool,
}

impl Options {
    fn parse(args: impl Iterator<Item = String>) -> Result<Option<Self>, String> {
        let mut opts = Options {
            filter: None,
            csv: false,
        };
        let mut args = args.peekable();
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "-h" | "--help" => {
                    print_help();
                    return Ok(None);
                }
                "--csv" => opts.csv = true,
                "-f" | "--filter" => {
                    opts.filter = Some(args.next().ok_or("--filter expects a value")?);
                }
                other if other.starts_with('-') => {
                    return Err(format!("unknown option {other:?}"));
                }
                // A bare positional is taken as the filter.
                other => opts.filter = Some(other.to_string()),
            }
        }
        Ok(Some(opts))
    }
}

fn print_help() {
    println!(
        "llmleaf-bench — time + allocation benchmarks for the core hot path\n\n\
         USAGE:\n  \
           llmleaf-bench [FILTER] [OPTIONS]\n\n\
         OPTIONS:\n  \
           FILTER                only run cases whose name contains this substring\n  \
           -f, --filter S        same as the positional FILTER\n  \
           --csv                 emit CSV instead of the aligned table\n  \
           -h, --help            print this help\n\n\
         EXAMPLES:\n  \
           cargo run -p llmleaf-bench --release\n  \
           cargo run -p llmleaf-bench --release -- hotpath\n  \
           cargo run -p llmleaf-bench --release -- --csv > bench.csv"
    );
}
