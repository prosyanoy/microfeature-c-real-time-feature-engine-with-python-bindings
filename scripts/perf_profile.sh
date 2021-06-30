#!/usr/bin/env bash
# Profile the native benchmark with perf and (if available) render a flame graph.
# Usage: scripts/perf_profile.sh [benchmark_filter]   (default: BM_ReplayFeatures)
# Flame graphs need Brendan Gregg's FlameGraph scripts: set FLAMEGRAPH_DIR to a checkout.
set -euo pipefail
cd "$(dirname "$0")/.."
FILTER="${1:-BM_ReplayFeatures}"
OUT=build/perf
mkdir -p "$OUT"
command -v perf >/dev/null || { echo "perf not installed (apt install linux-tools-generic)"; exit 1; }
perf record -e "${PERF_EVENT:-cpu-clock:u}" -F 999 --call-graph dwarf -o "$OUT/perf.data" -- \
  ./build/microfeature_bench --benchmark_filter="$FILTER" --benchmark_min_time=2s
perf report -i "$OUT/perf.data" --no-children --stdio --percent-limit 1.5 > "$OUT/report.txt" 2>/dev/null
echo "text report: $OUT/report.txt"
if [[ -n "${FLAMEGRAPH_DIR:-}" ]]; then
  perf script -i "$OUT/perf.data" | "$FLAMEGRAPH_DIR/stackcollapse-perf.pl" | \
    "$FLAMEGRAPH_DIR/flamegraph.pl" > "$OUT/flame.svg"
  echo "flame graph: $OUT/flame.svg"
fi
