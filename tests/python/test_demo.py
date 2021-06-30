import os


def test_ml_demo_streaming_features_beat_baseline(monkeypatch, capsys):
    monkeypatch.setenv("MF_DEMO_DAYS", "2")
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "examples" / "ml_demo.py"
    spec = importlib.util.spec_from_file_location("ml_demo", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    results = mod.main()
    base, feat = results.values()
    assert feat[0] < 0.7 * base[0]  # MAE at least 30% better with streaming features
    assert feat[1] > base[1]
