"""Reference implementations the C++ engine is compared against.

Both compute the same five features as `microfeature.Engine`
(count, sum, mean, rate_per_sec, ewma) over the window (now - W, now].
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np
import pandas as pd

LN2 = math.log(2.0)


class PurePythonEngine:
    """Idiomatic pure-Python store: dict of deques with running sums (time-ordered input)."""

    def __init__(self, window_ms: int, halflife_ms: int):
        self.window_ms = window_ms
        self.tau = halflife_ms / LN2
        self.win: dict[int, deque] = {}
        self.sums: dict[int, float] = {}
        self.ew: dict[int, list] = {}  # key -> [num, den, last_ts]

    def ingest(self, key: int, ts: int, value: float) -> None:
        dq = self.win.get(key)
        if dq is None:
            dq = self.win[key] = deque()
            self.sums[key] = 0.0
        dq.append((ts, value))
        self.sums[key] += value
        lo = ts - self.window_ms
        while dq[0][0] <= lo:
            self.sums[key] -= dq.popleft()[1]
        e = self.ew.get(key)
        if e is None:
            self.ew[key] = [value, 1.0, ts]
        else:
            d = math.exp(-(ts - e[2]) / self.tau)
            e[0] = e[0] * d + value
            e[1] = e[1] * d + 1.0
            e[2] = ts

    def features(self, key: int, now: int) -> tuple:
        dq = self.win.get(key)
        if dq is None:
            return (0.0, 0.0, math.nan, 0.0, math.nan)
        lo = now - self.window_ms
        # Sum lazily over expired head entries, as a hand-written store would.
        count = len(dq)
        total = self.sums[key]
        for ts, v in dq:
            if ts > lo:
                break
            count -= 1
            total -= v
        mean = total / count if count else math.nan
        e = self.ew[key]
        return (float(count), total if count else 0.0, mean,
                count / (self.window_ms / 1000.0), e[0] / e[1])

    def replay_features(self, keys, ts, values, inclusive=True) -> np.ndarray:
        """Process a time-ordered stream, returning one feature row per event."""
        out = np.empty((len(keys), 5))
        ingest, features = self.ingest, self.features
        for i, (k, t, v) in enumerate(zip(keys.tolist(), ts.tolist(), values.tolist())):
            if inclusive:
                ingest(k, t, v)
                out[i] = features(k, t)
            else:
                out[i] = features(k, t)
                ingest(k, t, v)
        return out


def pandas_features(keys, ts, values, window_ms: int, halflife_ms: int) -> np.ndarray:
    """Inclusive per-event features via pandas rolling/ewm; rows returned in input order.

    Timestamps must be unique per key (rolling windows are positional on ties).
    """
    df = pd.DataFrame({"key": keys, "ts": ts, "v": values, "pos": np.arange(len(keys))})
    df["dt"] = pd.to_datetime(df["ts"], unit="ms")
    df = df.sort_values(["key", "ts"], kind="stable").reset_index(drop=True)
    w = f"{window_ms}ms"
    roll = df.groupby("key").rolling(w, on="dt")["v"]
    count = roll.count().to_numpy()
    total = roll.sum().to_numpy()
    mean = roll.mean().to_numpy()
    ewma = np.empty(len(df))
    # groupby().rolling() output is ordered by key then time, same as the sorted frame.
    halflife = pd.Timedelta(milliseconds=halflife_ms)
    starts = np.flatnonzero(np.r_[True, df["key"].to_numpy()[1:] != df["key"].to_numpy()[:-1]])
    ends = np.r_[starts[1:], len(df)]
    v = df["v"]
    dt = df["dt"]
    for s, e in zip(starts, ends):
        ewma[s:e] = v.iloc[s:e].ewm(halflife=halflife, times=dt.iloc[s:e]).mean().to_numpy()
    res = np.column_stack([count, total, mean, count / (window_ms / 1000.0), ewma])
    out = np.empty_like(res)
    out[df["pos"].to_numpy()] = res
    return out


class PandasLookup:
    """Naive 'feature store' on pandas: per-key frames filtered with boolean masks per lookup."""

    def __init__(self, keys, ts, values, window_ms: int):
        df = pd.DataFrame({"key": keys, "ts": ts, "v": values}).sort_values("ts")
        self.groups = {k: g[["ts", "v"]].reset_index(drop=True) for k, g in df.groupby("key")}
        self.window_ms = window_ms

    def features(self, key: int, now: int) -> tuple:
        g = self.groups.get(key)
        if g is None:
            return (0.0, 0.0, math.nan, 0.0, math.nan)
        w = g[(g["ts"] > now - self.window_ms) & (g["ts"] <= now)]["v"]
        c = float(w.count())
        return (c, float(w.sum()), float(w.mean()) if c else math.nan,
                c / (self.window_ms / 1000.0), math.nan)
