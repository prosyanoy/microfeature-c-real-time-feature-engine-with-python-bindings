import numpy as np
import pytest

import microfeature as mf


def test_feature_names_and_shape():
    assert mf.FEATURE_NAMES == ("count", "sum", "mean", "rate_per_sec", "ewma")
    e = mf.Engine(window_ms=1000, halflife_ms=500)
    e.ingest_one(7, 100, 2.0)
    out = e.features(np.array([7, 8], dtype=np.uint64), 100)
    assert out.shape == (2, 5) and out.dtype == np.float64
    assert out[0, 0] == 1 and out[0, 1] == 2.0 and out[0, 2] == 2.0
    assert out[1, 0] == 0 and np.isnan(out[1, 2]) and np.isnan(out[1, 4])


def test_lookup_equals_batch_row():
    e = mf.Engine(window_ms=1000, halflife_ms=500)
    e.ingest(np.array([1, 1, 2]), np.array([10, 20, 30]), np.array([1.0, 3.0, 5.0]))
    row = e.lookup(1, 20)
    assert row.shape == (5,)
    np.testing.assert_array_equal(row, e.features(np.array([1], dtype=np.uint64), 20)[0])
    assert row[2] == 2.0


def test_features_at_uses_per_row_time():
    e = mf.Engine(window_ms=100, halflife_ms=50)
    e.ingest(np.array([1, 1]), np.array([0, 90]), np.array([1.0, 1.0]))
    out = e.features_at(np.array([1, 1], dtype=np.uint64), np.array([90, 190]))
    assert list(out[:, 0]) == [2.0, 0.0]


def test_input_validation():
    e = mf.Engine()
    with pytest.raises(ValueError):
        e.ingest(np.zeros(3), np.zeros(2), np.zeros(3))
    with pytest.raises(ValueError):
        mf.Engine(window_ms=0)


def test_stats_and_clear():
    e = mf.Engine(window_ms=1000, halflife_ms=500)
    n = e.ingest(np.array([1, 1, 1]), np.array([5000, 100, 5001]), np.array([1.0, 2.0, np.nan]))
    assert n == 1
    s = e.stats()
    assert s["accepted"] == 1 and s["late_dropped"] == 1 and s["rejected"] == 1 and s["keys"] == 1
    e.clear()
    assert e.stats()["keys"] == 0


def test_replay_features_point_in_time_excludes_self():
    e = mf.Engine(window_ms=1000, halflife_ms=500)
    out = e.replay_features(np.array([1, 1]), np.array([0, 10]), np.array([4.0, 6.0]))
    assert out[0, 0] == 0 and out[1, 0] == 1 and out[1, 1] == 4.0
