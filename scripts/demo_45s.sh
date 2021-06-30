#!/usr/bin/env bash
# Recording script for a ~45 s screen capture: benchmark, chart, then Python usage.
# Assumes `make setup build` has already been run.
set -euo pipefail
cd "$(dirname "$0")/.."
say() { printf '\n\033[1m# %s\033[0m\n' "$*"; sleep 1; }
say "1/3  make bench  (C++ engine vs pure Python vs pandas)"
MF_BENCH_EVENTS="${MF_BENCH_EVENTS:-200000}" MF_BENCH_LOOKUPS="${MF_BENCH_LOOKUPS:-10000}" make -s bench
make -s chart
say "2/3  chart written to docs/benchmark.png"
say "3/3  Python usage"
MF_MODULE_DIR="$PWD/build/python" .venv/bin/python examples/usage.py
