#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M2M Chart Generator - Formato NotebookLM / Científico

Genera gráficos con estilo Nature/IEEE usando SOLO datos reales
de benchmark_results.json o benchmarks/results/benchmark_latest.json.

NUNCA fabrica datos. Si no hay datos, genera gráficos vacíos con mensaje.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

# ─── Paths ─────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
ASSETS_DIR = PROJECT_DIR / "assets"
BENCHMARK_JSON = PROJECT_DIR / "benchmark_results.json"
BENCHMARKS_DIR = PROJECT_DIR / "benchmarks" / "results"
BENCHMARK_LATEST = BENCHMARKS_DIR / "benchmark_latest.json"

ASSETS_DIR.mkdir(exist_ok=True)

# ─── Scientific Style Configuration ─────────────────────────────────────────
# Nature/IEEE inspired style
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "axes.grid.which": "major",
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
    "legend.fontsize": 9,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "0.8",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "text.color": "#1a1a1a",
    "axes.labelcolor": "#1a1a1a",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
})

# Color palette (colorblind-friendly)
COLORS = {
    "cpu": "#0072B2",      # Blue
    "vulkan": "#D55E00",    # Vermillion
    "transformed": "#009E73",  # Teal
    "linear": "#CC79A7",    # Pink
    "cuda": "#F0E442",      # Yellow
}


def _load_benchmark() -> Optional[Dict[str, Any]]:
    """Carga datos de benchmark desde cualquier ubicación disponible."""
    for path in [BENCHMARK_JSON, BENCHMARK_LATEST]:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)

    # Buscar el más reciente en benchmarks/results/
    if BENCHMARKS_DIR.exists():
        files = sorted(BENCHMARKS_DIR.glob("benchmark_*.json"), reverse=True)
        for f in files:
            if f.name != "benchmark_latest.json":
                with open(f, "r", encoding="utf-8") as fh:
                    return json.load(fh)

    return None


def _save_fig(fig: plt.Figure, filename: str) -> None:
    """Guarda gráfico en PNG (300dpi) y PDF vectorial."""
    png_path = ASSETS_DIR / filename
    pdf_path = png_path.with_suffix(".pdf")

    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", facecolor="white")
    print(f"  [OK] {png_path.name} (+ PDF)")
    plt.close(fig)


def _extract_latency_data(data: Dict) -> Optional[Dict]:
    """Extrae datos de latencia del benchmark."""
    results = {}

    # New format (backends dict)
    backends = data.get("backends", {})
    for name, backend in backends.items():
        if "error" in backend:
            continue
        retrieval = backend.get("retrieval", {})
        results[name] = {
            "avg_ms": retrieval.get("avg_latency_ms", 0),
            "p50_ms": retrieval.get("p50_latency_ms", 0),
            "p95_ms": retrieval.get("p95_latency_ms", 0),
            "p99_ms": retrieval.get("p99_latency_ms", 0),
            "min_ms": retrieval.get("min_latency_ms", 0),
            "max_ms": retrieval.get("max_latency_ms", 0),
            "qps": retrieval.get("throughput_qps", 0),
            "n_queries": retrieval.get("n_queries", 0),
        }

    # Old format
    old_results = data.get("results", {})
    for name, r in old_results.items():
        short_name = "linear" if "linear" in name.lower() else "m2m"
        if short_name not in results:
            results[short_name] = {
                "avg_ms": r.get("avg_latency_ms", 0),
                "p95_ms": r.get("p95_latency_ms", 0),
                "p99_ms": r.get("p99_latency_ms", 0),
                "qps": r.get("throughput_qps", 0),
                "n_queries": 1,
            }

    # Linear baseline
    baseline = data.get("linear_baseline", {})
    if baseline and "linear" not in results:
        results["linear"] = {
            "avg_ms": baseline.get("avg_latency_ms", 0),
            "p50_ms": baseline.get("p50_latency_ms", 0),
            "p95_ms": baseline.get("p95_latency_ms", 0),
            "p99_ms": baseline.get("p99_latency_ms", 0),
            "qps": baseline.get("throughput_qps", 0),
            "n_queries": 1,
        }

    return results if results else None


# ═══════════════════════════════════════════════════════════════════════════════
# Charts
# ═══════════════════════════════════════════════════════════════════════════════


def chart_latency_comparison(data: Dict, latency_data: Dict) -> None:
    """Gráfico de barras: latencia promedio por backend con barras de error P95-P50."""
    print("\n[1/4] Latency comparison...")

    # Filter backends with data
    backends_to_plot = {k: v for k, v in latency_data.items() if v["n_queries"] > 1}
    if not backends_to_plot:
        # Fallback: use any available
        backends_to_plot = latency_data

    if not backends_to_plot:
        print("  [SKIP] No latency data available")
        return

    names = list(backends_to_plot.keys())
    avgs = [backends_to_plot[n]["avg_ms"] for n in names]
    p50s = [backends_to_plot[n].get("p50_ms", 0) for n in names]
    p95s = [backends_to_plot[n].get("p95_ms", 0) for n in names]

    # Error bars: asymmetric (p95 - avg for upper, avg - p50 for lower)
    yerr_lower = [max(0, a - p) for a, p in zip(avgs, p50s)]
    yerr_upper = [u - a for a, u in zip(avgs, p95s)]

    colors = [COLORS.get(n, "#999999") for n in names]
    display_names = [n.capitalize() for n in names]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(display_names, avgs, color=colors, edgecolor="#333333",
                  linewidth=0.8, width=0.6, yerr=[yerr_lower, yerr_upper],
                  capsize=4, error_kw={"linewidth": 1.0, "capthick": 1.0})

    ax.set_ylabel("Latency (ms)")
    ax.set_title("Query Latency by Backend")

    # Value labels
    for bar, val in zip(bars, avgs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Config info
    config = data.get("config", {})
    n_splats = config.get("n_splats", "?")
    k_val = config.get("k", "?")
    ax.set_xlabel(f"N = {n_splats:,} vectors  |  K = {k_val}  |  Error bars: P50–P95")

    # Add n= annotation
    for i, n in enumerate(names):
        nq = backends_to_plot[n].get("n_queries", "")
        if nq:
            ax.text(bars[i].get_x() + bars[i].get_width() / 2, 0.5,
                    f"n={nq}", ha="center", va="bottom", fontsize=7,
                    color="#666666")

    fig.tight_layout()
    _save_fig(fig, "chart_latency.png")


def chart_throughput(data: Dict, latency_data: Dict) -> None:
    """Gráfico de throughput (QPS) por backend."""
    print("\n[2/4] Throughput...")

    backends_to_plot = {k: v for k, v in latency_data.items() if v.get("qps", 0) > 0}
    if not backends_to_plot:
        print("  [SKIP] No throughput data available")
        return

    names = list(backends_to_plot.keys())
    qps_vals = [backends_to_plot[n]["qps"] for n in names]
    colors = [COLORS.get(n, "#999999") for n in names]
    display_names = [n.capitalize() for n in names]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(display_names, qps_vals, color=colors, edgecolor="#333333",
                  linewidth=0.8, width=0.6)

    ax.set_ylabel("Throughput (queries/sec)")
    ax.set_title("Search Throughput by Backend")

    for bar, val in zip(bars, qps_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Speedup annotation
    if "linear" in backends_to_plot and len(backends_to_plot) > 1:
        linear_qps = backends_to_plot["linear"]["qps"]
        for n, qps in zip(names, qps_vals):
            if n != "linear" and linear_qps > 0:
                speedup = qps / linear_qps
                idx = names.index(n)
                ax.annotate(
                    f"{speedup:.1f}x",
                    xy=(bars[idx].get_x() + bars[idx].get_width() / 2, qps),
                    xytext=(bars[idx].get_x() + bars[idx].get_width() / 2, qps * 1.1),
                    fontsize=10, fontweight="bold", color=COLORS.get(n, "#333"),
                    ha="center",
                    arrowprops=dict(arrowstyle="->", color=COLORS.get(n, "#333"), lw=1),
                )

    fig.tight_layout()
    _save_fig(fig, "chart_throughput.png")


def chart_percentile_breakdown(data: Dict, latency_data: Dict) -> None:
    """Gráfico de barras apiladas: percentiles P50, P95, P99 por backend."""
    print("\n[3/4] Percentile breakdown...")

    backends_with_percentiles = {}
    for k, v in latency_data.items():
        if v.get("p50_ms", 0) > 0 and v.get("p95_ms", 0) > 0:
            backends_with_percentiles[k] = v

    if not backends_with_percentiles:
        print("  [SKIP] No percentile data available")
        return

    names = list(backends_with_percentiles.keys())
    display_names = [n.capitalize() for n in names]
    p50 = [backends_with_percentiles[n]["p50_ms"] for n in names]
    p95 = [backends_with_percentiles[n]["p95_ms"] for n in names]
    p99 = [backends_with_percentiles[n].get("p99_ms", 0) for n in names]

    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width, p50, width, label="P50", color="#4C72B0", edgecolor="#333", linewidth=0.5)
    ax.bar(x, p95, width, label="P95", color="#DD8452", edgecolor="#333", linewidth=0.5)
    ax.bar(x + width, p99, width, label="P99", color="#C44E52", edgecolor="#333", linewidth=0.5)

    ax.set_ylabel("Latency (ms)")
    ax.set_title("Latency Distribution by Percentile")
    ax.set_xticks(x)
    ax.set_xticklabels(display_names)
    ax.legend(loc="upper left")

    fig.tight_layout()
    _save_fig(fig, "chart_percentiles.png")


def chart_speedup_summary(data: Dict, latency_data: Dict) -> None:
    """Gráfico de speedup relativo al linear baseline."""
    print("\n[4/4] Speedup summary...")

    if "linear" not in latency_data or latency_data["linear"]["avg_ms"] == 0:
        print("  [SKIP] No linear baseline for speedup calculation")
        return

    linear_avg = latency_data["linear"]["avg_ms"]
    speedups = {}
    for name, v in latency_data.items():
        if name != "linear" and v["avg_ms"] > 0:
            speedups[name] = linear_avg / v["avg_ms"]

    if not speedups:
        print("  [SKIP] No backends to compare")
        return

    names = list(speedups.keys())
    vals = [speedups[n] for n in names]
    colors = [COLORS.get(n, "#999999") for n in names]
    display_names = [n.capitalize() for n in names]

    fig, ax = plt.subplots(figsize=(8, 5))

    # Horizontal bar chart
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, vals, color=colors, edgecolor="#333333", linewidth=0.8, height=0.5)

    ax.set_xlabel("Speedup vs Linear Scan")
    ax.set_title("Relative Speedup by Backend")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(display_names)
    ax.axvline(x=1.0, color="#999999", linestyle="--", linewidth=0.8, label="Baseline (1x)")

    for bar, val in zip(bars, vals):
        label = f"{val:.1f}x"
        x_pos = max(val + 0.3, bar.get_width() + 0.3)
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                label, va="center", fontsize=9, fontweight="bold")

    ax.legend(loc="lower right")
    fig.tight_layout()
    _save_fig(fig, "chart_speedup.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    print("=" * 60)
    print("M2M Chart Generator — Scientific Format (Nature/IEEE)")
    print("Data source: benchmark_results.json (REAL DATA ONLY)")
    print("=" * 60)

    data = _load_benchmark()

    if data is None:
        print("\n[WARNING] No benchmark data found!")
        print("  Searched:")
        print(f"    - {BENCHMARK_JSON}")
        print(f"    - {BENCHMARK_LATEST}")
        print(f"    - {BENCHMARKS_DIR}/benchmark_*.json")
        print("\n  Run 'python benchmarks/run_benchmark.py' first.")
        sys.exit(1)

    print(f"\n[DATA] Loaded benchmark from: {data.get('timestamp', 'unknown')}")
    print(f"  Config: N={data.get('config', {}).get('n_splats', '?')}, "
          f"K={data.get('config', {}).get('k', '?')}, "
          f"dim={data.get('config', {}).get('latent_dim', '?')}")
    print(f"  System: {data.get('system_specs', {}).get('platform', 'unknown')}")

    latency_data = _extract_latency_data(data)

    if latency_data is None:
        print("\n[ERROR] Could not extract latency data from benchmark.")
        sys.exit(1)

    print(f"  Backends found: {list(latency_data.keys())}")

    chart_latency_comparison(data, latency_data)
    chart_throughput(data, latency_data)
    chart_percentile_breakdown(data, latency_data)
    chart_speedup_summary(data, latency_data)

    print("\n" + "=" * 60)
    print("All charts generated successfully")
    print(f"Output: {ASSETS_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
