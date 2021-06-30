# Performance notes

All numbers below were measured on a 2-vCPU 2.4 GHz Linux VM (Python 3.14, GCC 15, `-O3`
Release build). Reproduce with `make bench` (Python harness) and `make bench-cpp`
(Google Benchmark). Absolute numbers will differ on your machine and vary ~20% run to run; ratios are what matter. The committed chart (`docs/benchmark.png`) is from a separate full-size run, so it differs slightly from the tables here.

## Python harness (`make bench`, 300k events, 2k keys, 60 s window)

Features for every event in the stream (count, sum, mean, rate, EWMA):

| implementation | events/s | vs pandas |
|---|---:|---:|
| C++ engine (`replay_features`) | ~3.1 M | ~14x |
| pure Python (dict of deques) | ~0.30 M | ~1.4x |
| pandas (`groupby.rolling` + per-key `ewm`) | ~0.22 M | 1x |

Ingest only: C++ 6.5 M events/s on one thread, ~8-9 M/s with 2-4 threads; pure Python ~0.6 M/s.

Feature-lookup latency as seen from Python (per call, `perf_counter_ns`):

| implementation | p50 | p99 |
|---|---:|---:|
| `Engine.lookup(key, now)` | 0.7 us | 1.3 us |
| `Engine.features(one_key_array, now)` | 1.2 us | 2.0 us |
| pure Python dict/deque | 1.0 us | 2.7 us |
| pandas boolean-mask lookup | 648 us | 1094 us |
| `Engine.features(1024 keys, now)`, per key | 0.13 us | 0.20 us |

Honest reading: a *single* lookup from Python is dominated by interpreter and pybind11
call overhead (~0.5 us), so the C++ engine only ties a well-written pure-Python dict. The
engine wins when work is batched (stream replay, batched lookups) or when the caller is C++.

## Native (Google Benchmark, `make bench-cpp`)

| benchmark | result |
|---|---|
| ingest, 1 thread, 1M events / 10k keys | ~6.3 M events/s |
| `ingest_parallel`, 2 / 4 / 8 threads | ~8.2 / 8.1 / 7.9 M events/s |
| `replay_features` (ingest + query per event) | ~3.4 M events/s |
| single `query` (10k keys) | ~132 ns |
| query + 12% writes, 1 thread / 4 threads (per op) | 164 ns / 214 ns CPU |

Parallel scaling is limited by this VM having only 2 vCPUs, not by the design; more cores
are needed to measure the shard scaling curve properly.

## Profiling

`scripts/perf_profile.sh [filter]` records the native benchmark with `perf` (user-space
`cpu-clock`, DWARF call graphs), writes `build/perf/report.txt`, and renders
`build/perf/flame.svg` if `FLAMEGRAPH_DIR` points at a FlameGraph checkout.

**It could not be run for the numbers above**: the benchmark VM has
`kernel.perf_event_paranoid=4`, which forbids unprivileged `perf record`. No flame graph is
therefore checked in. Run it with `sudo sysctl kernel.perf_event_paranoid=1` on a machine you control.

What the code structure predicts you will see (unverified by profile, derived from the
benchmarks): ingestion time is dominated by the shard mutex + `unordered_map` lookup (cache
misses on 10k keys), the `exp()` in the EWMA update, and the binary searches in `query`;
the ring-buffer append itself is O(1) for in-order events.
