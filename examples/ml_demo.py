"""End-to-end ML loop: streaming features from the C++ engine -> scikit-learn model.

Simulates a grocery-delivery fleet. Customers place orders at stores (store "busy-ness" =
recent order arrivals) and shoppers pick them (shopper "pick rate" = recent picking speed).
Each order's fulfilment time depends on both. We walk the merged event timeline in time order,
look up point-in-time features from two engines *before* each order is ingested (no label or
future leakage), then train a model on the early orders and score the later ones.
"""
from __future__ import annotations

import heapq
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.environ.get("MF_MODULE_DIR", str(ROOT / "build" / "python")))

import microfeature as mf  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.metrics import mean_absolute_error, r2_score  # noqa: E402

DAYS = int(os.environ.get("MF_DEMO_DAYS", 4))
SEED = int(os.environ.get("MF_DEMO_SEED", 7))
WINDOW_MS = int(os.environ.get("MF_WINDOW_MS", 600_000))   # 10 minute windows for this demo
HALFLIFE_MS = int(os.environ.get("MF_HALFLIFE_MS", 300_000))
N_STORES, SHOPPERS_PER_STORE = 12, 8
DAY_MS = 24 * 3600 * 1000
SHOPPER_KEY_BASE = 1_000_000  # shoppers and stores share one key space in this demo


def simulate(rng):
    """Return orders as arrays sorted by time, with latent ground truth for the label."""
    base_rate = rng.uniform(0.4, 1.5, N_STORES)           # orders / minute at the daily mean
    phase = rng.uniform(0, 2 * np.pi, N_STORES)
    sec_per_item = rng.uniform(15, 45, N_STORES * SHOPPERS_PER_STORE)  # shopper skill (latent)
    rows = []
    for s in range(N_STORES):
        peak = base_rate[s] * 2.0 / 60_000.0               # max orders per ms (thinning bound)
        n = rng.poisson(peak * DAYS * DAY_MS)
        t = np.sort(rng.uniform(0, DAYS * DAY_MS, n))
        lam = base_rate[s] * (1 + 0.9 * np.sin(2 * np.pi * t / DAY_MS + phase[s])) / 60_000.0
        t = t[rng.uniform(0, peak, n) < lam]
        # Occasional demand bursts (e.g. weather, promotions).
        for _ in range(DAYS * 2):
            c = rng.uniform(0, DAYS * DAY_MS)
            extra = c + rng.exponential(10 * 60_000, rng.poisson(25))
            t = np.concatenate([t, extra[extra < DAYS * DAY_MS]])
        for ts in t:
            shopper = s * SHOPPERS_PER_STORE + int(rng.integers(SHOPPERS_PER_STORE))
            rows.append((int(ts), s, shopper, int(1 + rng.poisson(9))))
    rows.sort()
    ts = np.array([r[0] for r in rows], dtype=np.int64)
    store = np.array([r[1] for r in rows], dtype=np.uint64)
    shopper = np.array([r[2] for r in rows], dtype=np.uint64)
    items = np.array([r[3] for r in rows], dtype=np.float64)
    # Fulfilment time (minutes): picking time grows with the store's real-time load.
    busy = np.zeros(len(ts))
    for s in range(N_STORES):  # true load = orders at this store in the previous 10 minutes
        idx = np.flatnonzero(store == s)
        busy[idx] = np.arange(len(idx)) - np.searchsorted(ts[idx], ts[idx] - 600_000)
    minutes = (3 + items * sec_per_item[shopper] / 60 * (1 + 0.12 * busy)
               + rng.normal(0, 0.8, len(ts)))
    return ts, store, shopper, items, np.maximum(minutes, 1.0)


def build_features(ts, store, shopper, items, minutes):
    """Single time-ordered pass: features first, then ingest; shoppers learn from completions."""
    stores = mf.Engine(window_ms=WINDOW_MS, halflife_ms=HALFLIFE_MS)
    shoppers = mf.Engine(window_ms=WINDOW_MS, halflife_ms=HALFLIFE_MS)
    done = []  # heap of (completion_ts, order index)
    X = np.empty((len(ts), 2 + 2 + 3))
    lookup_ns = 0
    for i in range(len(ts)):
        now = int(ts[i])
        while done and done[0][0] <= now:  # completed picks become visible to the shopper engine
            cts, j = heapq.heappop(done)
            shoppers.ingest_one(int(shopper[j]), cts, items[j] / minutes[j])  # items per minute
        t0 = time.perf_counter_ns()
        s_feat = stores.lookup(int(store[i]), now)
        p_feat = shoppers.lookup(int(shopper[i]), now)
        lookup_ns += time.perf_counter_ns() - t0
        ewma = p_feat[4]
        X[i] = [s_feat[0], s_feat[1], p_feat[0], np.nan_to_num(ewma, nan=1.0),
                np.nan_to_num(p_feat[2], nan=1.0), items[i], now % DAY_MS / 3.6e6]
        stores.ingest_one(int(store[i]), now, items[i])
        heapq.heappush(done, (now + int(minutes[i] * 60_000), i))
    return X, lookup_ns / len(ts) / 1000.0


def main():
    rng = np.random.default_rng(SEED)
    ts, store, shopper, items, y = simulate(rng)
    t0 = time.perf_counter()
    X, lookup_us = build_features(ts, store, shopper, items, y)
    print(f"{len(ts):,} orders over {DAYS} days; features built in {time.perf_counter() - t0:.2f}s "
          f"(mean 2-lookup latency {lookup_us:.2f} us)")

    cols = ["store_orders_10m", "store_items_10m", "shopper_picks_10m", "shopper_ewma_pick_rate",
            "shopper_mean_pick_rate", "items", "hour_of_day"]
    split = int(len(y) * 0.7)  # chronological split: train on the past, score the future
    sets = {"baseline (items + hour only)": [5, 6], "with streaming features": list(range(len(cols)))}
    print(f"train {split:,} / test {len(y) - split:,} orders (chronological split)\n")
    print(f"{'model':32s} {'MAE (min)':>10s} {'R^2':>7s}")
    results = {}
    for name, idx in sets.items():
        model = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.08, random_state=SEED)
        model.fit(X[:split][:, idx], y[:split])
        pred = model.predict(X[split:][:, idx])
        results[name] = (mean_absolute_error(y[split:], pred), r2_score(y[split:], pred))
        print(f"{name:32s} {results[name][0]:10.2f} {results[name][1]:7.3f}")
    base, feat = results.values()
    print(f"\nMAE improvement from streaming features: {100 * (1 - feat[0] / base[0]):.1f}%")
    return results


if __name__ == "__main__":
    main()
