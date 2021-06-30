"""Render docs/benchmark.png from benchmarks/results/latest.json (run `make bench` first)."""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
res = json.loads((HERE / "results" / "latest.json").read_text())
ACCENT, MUTED, INK = "#2a6fdb", "#9aa3ad", "#1f2933"


def panel(ax, title, rows, fmt, note):
    names = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    ax.barh(names, vals, color=[ACCENT if n.startswith("C++") else MUTED for n in names], height=0.55)
    ax.set_xscale("log")
    ax.invert_yaxis()
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")
    ax.set_xlabel(note, color="#52606d", fontsize=9)
    for y, v in enumerate(vals):
        ax.text(v * 1.15, y, fmt(v), va="center", fontsize=9, color=INK)
    ax.set_xlim(min(vals) / 1.5, max(vals) * 12)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(left=False, colors=INK)
    ax.grid(axis="x", color="#e4e7eb", linewidth=0.8)
    ax.set_axisbelow(True)


lat, sf = res["lookup_latency"], res["stream_features"]
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.2), dpi=150)
panel(a, "p99 feature-lookup latency (lower is better)",
      [("C++ engine", lat["cpp_features_1_key"]["p99_us"]),
       ("pure Python", lat["pure_python"]["p99_us"]),
       ("pandas", lat["pandas_boolean_mask"]["p99_us"])],
      lambda v: f"{v:,.1f} µs", "microseconds, log scale (single-key lookup, called from Python)")
panel(b, "Throughput: features per event (higher is better)",
      [("C++ engine", sf["cpp"]["events_per_sec"]),
       ("pure Python", sf["pure_python"]["events_per_sec"]),
       ("pandas", sf["pandas"]["events_per_sec"])],
      lambda v: f"{v / 1e6:.2f} M/s", "events per second, log scale")
c = res["config"]
fig.text(0.01, 0.01, f"{c['events']:,} events, {c['keys']:,} keys, {c['window_ms'] // 1000}s window, "
         f"{c['cpu_count']} vCPU. Reproduce: make bench && make chart", fontsize=8, color="#52606d")
fig.tight_layout(rect=(0, 0.04, 1, 1))
out = HERE.parent / "docs" / "benchmark.png"
fig.savefig(out)
print("wrote", out, file=sys.stderr)
