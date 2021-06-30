"""Parity of the C++ engine with pandas rolling/ewm and the pure-Python baseline."""
import numpy as np
import pytest

import microfeature as mf
from baselines import PurePythonEngine, pandas_features

W, H = 30_000, 10_000


def make_stream(n=6000, keys=25, span=300_000, seed=0):
    rng = np.random.default_rng(seed)
    k = rng.integers(0, keys, n).astype(np.uint64)
    # Unique timestamps: pandas rolling windows are positional on ties.
    t = rng.choice(span, size=n, replace=False).astype(np.int64)
    v = np.round(rng.gamma(2.0, 3.0, n), 3)
    return k, t, v


def assert_close(a, b):
    assert a.shape == b.shape
    np.testing.assert_allclose(a[:, [0, 3]], b[:, [0, 3]], rtol=0, atol=1e-9)  # count, rate
    np.testing.assert_allclose(a[:, [1, 2, 4]], b[:, [1, 2, 4]], rtol=1e-9, atol=1e-8)


def test_matches_pandas_rolling_and_ewm():
    k, t, v = make_stream()
    eng = mf.Engine(window_ms=W, halflife_ms=H)
    got = eng.replay_features(k, t, v, inclusive=True)
    want = pandas_features(k, t, v, W, H)
    assert_close(got, want)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_matches_pandas_with_shuffled_input(seed):
    k, t, v = make_stream(seed=seed)
    perm = np.random.default_rng(seed).permutation(len(k))
    eng = mf.Engine(window_ms=W, halflife_ms=H)
    got = eng.replay_features(k[perm], t[perm], v[perm], inclusive=True)
    assert_close(got, pandas_features(k[perm], t[perm], v[perm], W, H))


def test_matches_pure_python_baseline_including_pointwise_lookups():
    k, t, v = make_stream(seed=4)
    order = np.argsort(t, kind="stable")
    k, t, v = k[order], t[order], v[order]
    base = PurePythonEngine(W, H)
    want = base.replay_features(k, t, v, inclusive=False)
    got = mf.Engine(window_ms=W, halflife_ms=H).replay_features(k, t, v, inclusive=False)
    assert_close(got, want)

    eng = mf.Engine(window_ms=W, halflife_ms=H)
    eng.ingest(k, t, v)
    now = int(t[-1])
    keys = np.arange(30, dtype=np.uint64)  # includes never-seen keys 25..29
    lookup = eng.features(keys, now)
    ref = np.array([base.features(int(x), now) for x in keys])
    assert_close(lookup, ref)


def test_parallel_ingest_matches_replay():
    k, t, v = make_stream(n=20000, seed=5)
    a, b = mf.Engine(window_ms=W, halflife_ms=H), mf.Engine(window_ms=W, halflife_ms=H)
    a.replay(k, t, v)
    b.ingest_parallel(k, t, v, threads=4)
    keys = np.arange(25, dtype=np.uint64)
    now = int(t.max())
    assert_close(a.features(keys, now), b.features(keys, now))
    for key in range(25):
        np.testing.assert_array_equal(a.window_samples(key), b.window_samples(key))
