"""Minimal Python usage of the engine (shown at the end of the 45-second demo)."""
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.environ.get("MF_MODULE_DIR", str(Path(__file__).resolve().parents[1] / "build" / "python")))
import microfeature as mf  # noqa: E402

rng = np.random.default_rng(0)
n = 1_000_000
keys = rng.integers(0, 1000, n).astype(np.uint64)      # e.g. store ids
ts = np.cumsum(rng.integers(0, 3, n)).astype(np.int64)  # event time, ms
vals = rng.gamma(2.0, 3.0, n)                           # e.g. items per order

eng = mf.Engine(window_ms=60_000, halflife_ms=30_000)
eng.ingest_parallel(keys, ts, vals, threads=4)          # GIL released; threads interleave, so some events arrive "late" and are counted
now = int(ts[-1])
feats = eng.features(np.array([3, 7, 999_999], dtype=np.uint64), now)  # (3, 5) float64
for k, row in zip((3, 7, 999_999), feats):
    print(f"store {k:>6}:", {n_: round(float(v), 3) for n_, v in zip(mf.FEATURE_NAMES, row)})
print("stats:", eng.stats())
