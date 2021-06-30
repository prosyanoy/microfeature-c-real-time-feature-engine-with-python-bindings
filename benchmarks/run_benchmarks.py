"""Throughput and latency benchmark: C++ engine vs pure Python vs pandas.

Configuration comes from environment variables (see .env.example). Prints a Markdown
report and writes benchmarks/results/latest.json.
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.environ.get("MF_MODULE_DIR", str(ROOT / "build" / "python")))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import microfeature as mf  # noqa: E402
from baselines import PandasLookup, PurePythonEngine, pandas_features  # noqa: E402

N = int(os.environ.get("MF_BENCH_EVENTS", 300_000))
KEYS = int(os.environ.get("MF_BENCH_KEYS", 2000))
LOOKUPS = int(os.environ.get("MF_BENCH_LOOKUPS", 20_000))
THREADS = int(os.environ.get("MF_BENCH_THREADS", 4))
W = int(os.environ.get("MF_WINDOW_MS", 60_000))
H = int(os.environ.get("MF_HALFLIFE_MS", 30_000))
SHARDS = int(os.environ.get("MF_SHARDS", 64))


def make_stream(seed=1):
    rng = np.random.default_rng(seed)
    keys = rng.integers(0, KEYS, N).astype(np.uint64)
    ts = np.cumsum(rng.integers(0, 4, N)).astype(np.int64)  # time-ordered, ms
    vals = np.round(rng.gamma(2.0, 3.0, N), 3)
    return keys, ts, vals


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return time.perf_counter() - t0, out


def latencies(fn, args_list):
    samples = np.empty(len(args_list))
    for i, a in enumerate(args_list):
        t0 = time.perf_counter_ns()
        fn(*a)
        samples[i] = time.perf_counter_ns() - t0
    us = samples / 1000.0
    return {"p50_us": float(np.percentile(us, 50)), "p99_us": float(np.percentile(us, 99)),
            "mean_us": float(us.mean()), "n": len(us)}


def main():
    keys, ts, vals = make_stream()
    new_engine = lambda: mf.Engine(window_ms=W, halflife_ms=H, num_shards=SHARDS)  # noqa: E731
    res: dict = {"config": dict(events=N, keys=KEYS, window_ms=W, halflife_ms=H, shards=SHARDS,
                                lookups=LOOKUPS, cpu_count=os.cpu_count(),
                                machine=platform.platform(), python=platform.python_version())}

    # --- A. compute features for every event of the stream -------------------------
    t_cpp, got = timed(lambda: new_engine().replay_features(keys, ts, vals, inclusive=True))
    t_py, ref = timed(lambda: PurePythonEngine(W, H).replay_features(keys, ts, vals, inclusive=True))
    t_pd, pdref = timed(lambda: pandas_features(keys, ts, vals, W, H))
    res["stream_features"] = {name: {"seconds": t, "events_per_sec": N / t}
                              for name, t in [("cpp", t_cpp), ("pure_python", t_py), ("pandas", t_pd)]}
    # Sanity: the contenders agree (ties in ts make pandas positional, so compare the others).
    np.testing.assert_allclose(got[:, [0, 1, 4]], ref[:, [0, 1, 4]], rtol=1e-6, atol=1e-6)

    # --- B. ingest-only throughput -------------------------------------------------
    ing = {}
    t, _ = timed(lambda: new_engine().ingest(keys, ts, vals))
    ing["cpp_1_thread"] = N / t
    for th in sorted({2, THREADS}):
        t, _ = timed(lambda: new_engine().ingest_parallel(keys, ts, vals, th))
        ing[f"cpp_{th}_threads"] = N / t

    def py_ingest():
        e = PurePythonEngine(W, H)
        for k, t_, v in zip(keys.tolist(), ts.tolist(), vals.tolist()):
            e.ingest(k, t_, v)
        return e
    t, py_engine = timed(py_ingest)
    ing["pure_python"] = N / t
    res["ingest_events_per_sec"] = ing

    # --- C. lookup latency against a warmed store ----------------------------------
    eng = new_engine()
    eng.ingest(keys, ts, vals)
    now = int(ts[-1])
    rng = np.random.default_rng(2)
    probe = rng.integers(0, KEYS, LOOKUPS).astype(np.uint64)
    probe_arrays = [(probe[i:i + 1], now) for i in range(LOOKUPS)]  # pre-built: no alloc in timing
    probe_ints = [(int(k), now) for k in probe]
    lat = {
        "cpp_features_1_key": latencies(eng.features, probe_arrays),
        "cpp_lookup": latencies(eng.lookup, probe_ints),
        "pure_python": latencies(py_engine.features, probe_ints),
    }
    pdl = PandasLookup(keys, ts, vals, W)
    lat["pandas_boolean_mask"] = latencies(pdl.features, probe_ints[:min(LOOKUPS, 2000)])
    batch = rng.integers(0, KEYS, 1024).astype(np.uint64)
    reps = [(batch, now)] * 2000
    b = latencies(eng.features, reps)
    lat["cpp_features_batch_1024_per_key"] = {k: (v / 1024 if k.endswith("_us") else v) for k, v in b.items()}
    lat["cpp_features_batch_1024_per_call"] = b
    res["lookup_latency"] = lat

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "latest.json").write_text(json.dumps(res, indent=2))
    print(report(res))


def report(res):
    c = res["config"]
    lines = [f"### Setup\n{c['events']:,} events, {c['keys']:,} keys, window {c['window_ms']} ms, "
             f"{c['shards']} shards, {c['cpu_count']} CPUs, Python {c['python']}\n",
             "### Features for every event in the stream (events/s, higher is better)",
             "| implementation | seconds | events/s | speed-up vs pandas |", "|---|---:|---:|---:|"]
    sf = res["stream_features"]
    for name, r in sf.items():
        lines.append(f"| {name} | {r['seconds']:.3f} | {r['events_per_sec']:,.0f} | "
                     f"{r['events_per_sec'] / sf['pandas']['events_per_sec']:.1f}x |")
    lines += ["", "### Ingest only (events/s)", "| implementation | events/s |", "|---|---:|"]
    lines += [f"| {k} | {v:,.0f} |" for k, v in res["ingest_events_per_sec"].items()]
    lines += ["", "### Feature lookup latency from Python (microseconds)",
              "| implementation | p50 | p99 | mean | samples |", "|---|---:|---:|---:|---:|"]
    lines += [f"| {k} | {v['p50_us']:.2f} | {v['p99_us']:.2f} | {v['mean_us']:.2f} | {v['n']} |"
              for k, v in res["lookup_latency"].items()]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
