#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M2M Benchmark Chart Generator
==============================
Generates publication-ready charts from **live benchmark data only**.
Every number on every chart comes from running the benchmark right now on
your machine — nothing is hardcoded.

Workflow
--------
  1. Run `benchmarks/run_benchmark.py --device both`
  2. Read the fresh JSON from `benchmarks/results/benchmark_latest.json`
  3. Optionally read `benchmarks/results/scalability_latest.json` for Chart 5
  4. Generate 5 charts into `assets/`

Usage
-----
  # Run benchmark first, then generate charts:
  python benchmarks/run_benchmark.py --device both
  python scripts/generate_charts.py

  # Or let this script run the benchmark automatically:
  python scripts/generate_charts.py --run-benchmark
  python scripts/generate_charts.py --run-benchmark --device cpu
"""

import sys
import json
import subprocess
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
RESULTS_DIR   = PROJECT_ROOT / "benchmarks" / "results"
BENCHMARK_JSON = RESULTS_DIR / "benchmark_latest.json"
SCALABILITY_JSON = RESULTS_DIR / "scalability_latest.json"
GPU_BATCH_JSON = PROJECT_ROOT / "gpu_batch_validation_results.json"
ASSETS_DIR    = PROJECT_ROOT / "assets"
ASSETS_DIR.mkdir(exist_ok=True)


# ── Theme ─────────────────────────────────────────────────────────────────────
plt.style.use('dark_background')
plt.rcParams.update({
    'figure.facecolor':  '#0d1117',
    'axes.facecolor':    '#161b22',
    'axes.edgecolor':    '#30363d',
    'axes.labelcolor':   '#c9d1d9',
    'text.color':        '#c9d1d9',
    'xtick.color':       '#8b949e',
    'ytick.color':       '#8b949e',
    'grid.color':        '#21262d',
    'font.family':       'sans-serif',
    'font.size':          11,
})

C = {
    'linear': '#8b949e',
    'cpu':    '#58a6ff',
    'vulkan': '#f78166',
    'accent': '#7ee787',
    'purple': '#d2a8ff',
    'edge':   '#30363d',
    'title':  '#e6edf3',
}


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def save_chart(fig, filename: str, dpi: int = 200):
    p = ASSETS_DIR / filename
    fig.savefig(p, dpi=dpi, bbox_inches='tight',
                facecolor='#0d1117', edgecolor='none', pad_inches=0.3)
    print(f"  [OK] {p.name}")
    plt.close(fig)


def bar_labels(ax, bars, fmt='{:.1f}', offset=0, fs=10, color='#c9d1d9'):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + offset,
                fmt.format(h), ha='center', va='bottom',
                fontsize=fs, fontweight='bold', color=color)


def load_benchmark_json() -> dict:
    """Load benchmark_latest.json — fail loudly if missing."""
    if not BENCHMARK_JSON.exists():
        print(f"\n[ERROR] {BENCHMARK_JSON} not found.")
        print("  Run first:  python benchmarks/run_benchmark.py --device both")
        sys.exit(1)
    with open(BENCHMARK_JSON, encoding='utf-8') as f:
        return json.load(f)


def run_benchmark_subprocess(device: str = "both"):
    """Invoke run_benchmark.py as a subprocess."""
    script = PROJECT_ROOT / "benchmarks" / "run_benchmark.py"
    print(f"\n[RUN] python {script.name} --device {device}\n")
    result = subprocess.run(
        [sys.executable, str(script), "--device", device],
        cwd=str(PROJECT_ROOT),
    )
    if result.returncode != 0:
        print("[ERROR] Benchmark failed — aborting chart generation.")
        sys.exit(result.returncode)


def _r(d: dict, *keys, default="—"):
    """Safe nested dict access, returns formatted string or default."""
    val = d
    for k in keys:
        if not isinstance(val, dict) or k not in val:
            return default
        val = val[k]
    return val


# ═════════════════════════════════════════════════════════════════════════════
# Chart 1 — Main: Linear vs CPU vs Vulkan (latency + throughput)
# ═════════════════════════════════════════════════════════════════════════════

def chart_main_benchmark(data: dict):
    """3-way comparison: Linear vs M2M CPU vs M2M Vulkan GPU."""
    bl    = data.get("linear_baseline", {})
    cpu_r = data.get("backends", {}).get("cpu", {}).get("retrieval", {})
    vk_r  = data.get("backends", {}).get("vulkan", {}).get("retrieval", {})
    cfg   = data.get("config", {})
    dmeta = data.get("data_meta", {})

    # Build series — only include backends that ran successfully
    series = [("Linear\n(Brute Force)", C['linear'],
               bl.get("avg_latency_ms", 0), bl.get("throughput_qps", 0))]
    if cpu_r and "avg_latency_ms" in cpu_r:
        series.append(("M2M\n(CPU)", C['cpu'],
                       cpu_r["avg_latency_ms"], cpu_r["throughput_qps"]))
    if vk_r and "avg_latency_ms" in vk_r:
        series.append(("M2M\n(Vulkan GPU)", C['vulkan'],
                       vk_r["avg_latency_ms"], vk_r["throughput_qps"]))

    labels = [s[0] for s in series]
    colors = [s[1] for s in series]
    lats   = [s[2] for s in series]
    qpss   = [s[3] for s in series]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    b1 = ax1.bar(labels, lats, color=colors, alpha=0.9, edgecolor=C['edge'], width=0.55)
    ax1.set_ylabel('Avg Latency (ms)', fontsize=12, fontweight='bold')
    ax1.set_title('Search Latency  (lower is better)', fontsize=13, fontweight='bold', pad=12)
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    ax1.set_ylim(0, max(lats) * 1.38)
    bar_labels(ax1, b1, fmt='{:.2f} ms', offset=max(lats)*0.01)

    # Speedup annotation for each M2M bar
    bl_lat = lats[0]
    for i, (lbl, col, lat, _) in enumerate(series[1:], start=1):
        if lat > 0:
            sp = bl_lat / lat
            ax1.annotate(f'{sp:.1f}×', xy=(i, lat), xytext=(i, lat + max(lats)*0.15),
                         fontsize=12, fontweight='bold', color=col, ha='center')

    b2 = ax2.bar(labels, qpss, color=colors, alpha=0.9, edgecolor=C['edge'], width=0.55)
    ax2.set_ylabel('Queries per Second', fontsize=12, fontweight='bold')
    ax2.set_title('Retrieval Throughput  (higher is better)', fontsize=13, fontweight='bold', pad=12)
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    ax2.set_ylim(0, max(qpss) * 1.35)
    bar_labels(ax2, b2, fmt='{:.1f} QPS', offset=max(qpss)*0.01)

    n  = cfg.get("n_splats", "?")
    nq = cfg.get("n_queries", "?")
    k  = cfg.get("k", "?")
    src = dmeta.get("source", "real data")
    fig.suptitle(
        f'M2M vs Linear Search · {src}\n'
        f'{n:,} embeddings · {nq} queries · K={k}',
        fontsize=13, fontweight='bold', y=1.03, color=C['title']
    )
    plt.tight_layout()
    save_chart(fig, 'benchmark_results.png')


# ═════════════════════════════════════════════════════════════════════════════
# Chart 2 — Latency percentiles: CPU vs Vulkan
# ═════════════════════════════════════════════════════════════════════════════

def chart_latency_distribution(data: dict):
    cpu_r = data.get("backends", {}).get("cpu", {}).get("retrieval", {})
    vk_r  = data.get("backends", {}).get("vulkan", {}).get("retrieval", {})

    series = []
    if cpu_r and "avg_latency_ms" in cpu_r:
        series.append(("CPU", C['cpu'], [
            cpu_r["avg_latency_ms"], cpu_r["p95_latency_ms"], cpu_r["p99_latency_ms"]]))
    if vk_r and "avg_latency_ms" in vk_r:
        series.append(("Vulkan GPU", C['vulkan'], [
            vk_r["avg_latency_ms"], vk_r["p95_latency_ms"], vk_r["p99_latency_ms"]]))

    if not series:
        print("  [SKIP] chart_latency_distribution — no backend data")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(3)
    w = 0.3 if len(series) == 2 else 0.5
    offsets = [-w/2, w/2] if len(series) == 2 else [0]

    all_bars = []
    for (label, col, vals), off in zip(series, offsets):
        bars = ax.bar(x + off, vals, w, label=label, color=col, alpha=0.9, edgecolor=C['edge'])
        bar_labels(ax, bars, fmt='{:.1f}', offset=0.3, fs=10, color=col)
        all_bars.extend(vals)

    ax.set_ylabel('Latency (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Latency Percentiles · CPU vs Vulkan GPU', fontsize=14, fontweight='bold', pad=12)
    ax.set_xticks(x); ax.set_xticklabels(['Avg', 'P95', 'P99'], fontsize=12)
    ax.legend(loc='upper left', framealpha=0.8, edgecolor=C['edge'])
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.set_ylim(0, max(all_bars) * 1.3)
    plt.tight_layout()
    save_chart(fig, 'benchmark_latency.png')


# ═════════════════════════════════════════════════════════════════════════════
# Chart 3 — Data Lake: Ingest / Std Training / Gen Training
# ═════════════════════════════════════════════════════════════════════════════

def chart_data_lake_metrics(data: dict):
    backends_data = data.get("backends", {})
    rows = []
    for dev, col in [("cpu", C['cpu']), ("vulkan", C['vulkan'])]:
        bk = backends_data.get(dev)
        if bk and "error" not in bk:
            rows.append({
                "label": dev.upper(),
                "color": col,
                "ingest": bk.get("ingest_throughput_qps", 0),
                "std_tp": bk.get("standard_training", {}).get("throughput_splats_per_sec", 0),
                "std_loss": bk.get("standard_training", {}).get("avg_loss", 0),
                "gen_tp": bk.get("generative_training", {}).get("throughput_splats_per_sec", 0),
                "gen_loss": bk.get("generative_training", {}).get("avg_loss", 0),
            })

    if not rows:
        print("  [SKIP] chart_data_lake_metrics — no backend data")
        return

    labels = [r["label"] for r in rows]
    colors = [r["color"] for r in rows]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    # Ingest
    vals = [r["ingest"] for r in rows]
    b1 = ax1.bar(labels, vals, color=colors, alpha=0.9, edgecolor=C['edge'], width=0.45)
    ax1.set_ylabel('Splats / second', fontsize=12, fontweight='bold')
    ax1.set_title('Ingest Throughput', fontsize=14, fontweight='bold', pad=12)
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    ax1.set_ylim(0, max(vals) * 1.3)
    bar_labels(ax1, b1, fmt='{:.0f}', offset=max(vals)*0.01)
    if len(vals) == 2 and vals[0] > 0:
        pct = (vals[1] / vals[0] - 1) * 100
        sign = "+" if pct >= 0 else ""
        ax1.text(1, vals[1]*0.5, f'{sign}{pct:.0f}%\nVulkan',
                 ha='center', fontsize=13, fontweight='bold', color=C['accent'],
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='#0d1117',
                           edgecolor=C['accent'], alpha=0.9))

    # Standard training
    vals2 = [r["std_tp"] for r in rows]
    b2 = ax2.bar(labels, vals2, color=colors, alpha=0.9, edgecolor=C['edge'], width=0.45)
    ax2.set_ylabel('Splats / second', fontsize=12, fontweight='bold')
    ax2.set_title('Standard Training\n(SOC Importance Sampling)', fontsize=14, fontweight='bold', pad=12)
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    ax2.set_ylim(0, max(vals2) * 1.3)
    bar_labels(ax2, b2, fmt='{:,.0f}', offset=max(vals2)*0.01)
    for bar, r in zip(b2, rows):
        ax2.text(bar.get_x() + bar.get_width()/2, max(vals2)*0.12,
                 f'loss: {r["std_loss"]:.4f}', ha='center', fontsize=9,
                 color='#8b949e', style='italic')

    # Generative training
    vals3 = [r["gen_tp"] for r in rows]
    b3 = ax3.bar(labels, vals3, color=colors, alpha=0.9, edgecolor=C['edge'], width=0.45)
    ax3.set_ylabel('Splats / second', fontsize=12, fontweight='bold')
    ax3.set_title('Generative Training\n(Langevin Augmentation)', fontsize=14, fontweight='bold', pad=12)
    ax3.grid(axis='y', linestyle='--', alpha=0.3)
    ax3.set_ylim(0, max(vals3) * 1.3)
    bar_labels(ax3, b3, fmt='{:,.0f}', offset=max(vals3)*0.01)
    for bar, r in zip(b3, rows):
        ax3.text(bar.get_x() + bar.get_width()/2, max(vals3)*0.12,
                 f'loss: {r["gen_loss"]:.4f}', ha='center', fontsize=9,
                 color='#8b949e', style='italic')

    n  = data.get("config", {}).get("n_splats", "?")
    src = data.get("data_meta", {}).get("source", "real data")
    fig.suptitle(f'M2M Data Lake Metrics · {src} ({n:,} samples)',
                 fontsize=15, fontweight='bold', y=1.02, color=C['title'])
    plt.tight_layout()
    save_chart(fig, 'benchmark_data_lake.png')


# ═════════════════════════════════════════════════════════════════════════════
# Chart 4 — Summary table
# ═════════════════════════════════════════════════════════════════════════════

def chart_summary_table(data: dict):
    bl    = data.get("linear_baseline", {})
    cpu_b = data.get("backends", {}).get("cpu", {})
    vk_b  = data.get("backends", {}).get("vulkan", {})
    cfg   = data.get("config", {})
    dmeta = data.get("data_meta", {})

    cpu_r = cpu_b.get("retrieval", {}) if "error" not in cpu_b else {}
    vk_r  = vk_b.get("retrieval", {})  if "error" not in vk_b  else {}

    def fmt_ms(d, key): return f'{d[key]:.2f} ms' if d.get(key) is not None else '—'
    def fmt_qps(d, key): return f'{d[key]:.2f}' if d.get(key) is not None else '—'

    bl_lat = bl.get("avg_latency_ms", 0)
    cpu_sp = f'{bl_lat/cpu_r["avg_latency_ms"]:.1f}×' if cpu_r.get("avg_latency_ms") else '—'
    vk_sp  = f'{bl_lat/vk_r["avg_latency_ms"]:.1f}×'  if vk_r.get("avg_latency_ms")  else '—'

    headers = ['Metric', 'Linear Scan', 'M2M CPU', 'M2M Vulkan GPU']
    rows_ = [
        ['Avg Latency',       fmt_ms(bl,'avg_latency_ms'),  fmt_ms(cpu_r,'avg_latency_ms'),  fmt_ms(vk_r,'avg_latency_ms')],
        ['P95 Latency',       '—',                          fmt_ms(cpu_r,'p95_latency_ms'),  fmt_ms(vk_r,'p95_latency_ms')],
        ['P99 Latency',       '—',                          fmt_ms(cpu_r,'p99_latency_ms'),  fmt_ms(vk_r,'p99_latency_ms')],
        ['Throughput (QPS)',  fmt_qps(bl,'throughput_qps'), fmt_qps(cpu_r,'throughput_qps'), fmt_qps(vk_r,'throughput_qps')],
        ['Speedup vs Linear', '1.0×',                       cpu_sp,                          vk_sp],
        ['Ingest Rate',       '—',
         f'{cpu_b.get("ingest_throughput_qps",0):.0f} sp/s' if "error" not in cpu_b else '—',
         f'{vk_b.get("ingest_throughput_qps",0):.0f} sp/s'  if "error" not in vk_b  else '—'],
        ['Std Training',      '—',
         f'{cpu_b.get("standard_training",{}).get("throughput_splats_per_sec",0):,.0f} sp/s' if "error" not in cpu_b else '—',
         f'{vk_b.get("standard_training",{}).get("throughput_splats_per_sec",0):,.0f} sp/s'  if "error" not in vk_b  else '—'],
        ['Gen Training',      '—',
         f'{cpu_b.get("generative_training",{}).get("throughput_splats_per_sec",0):,.0f} sp/s' if "error" not in cpu_b else '—',
         f'{vk_b.get("generative_training",{}).get("throughput_splats_per_sec",0):,.0f} sp/s'  if "error" not in vk_b  else '—'],
    ]

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.axis('off')
    table = ax.table(cellText=rows_, colLabels=headers, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)

    col_colors = [None, C['linear'], C['cpu'], C['vulkan']]
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#30363d')
        if row == 0:
            cell.set_facecolor('#21262d')
            cell.set_text_props(fontweight='bold', color='#e6edf3')
        else:
            cell.set_facecolor('#0d1117')
            if col > 0:
                cell.set_text_props(color=col_colors[col])
            else:
                cell.set_text_props(color='#c9d1d9')

    n   = cfg.get("n_splats", "?")
    dim = cfg.get("latent_dim", "?")
    src = dmeta.get("source", "real data")
    ax.set_title(f'M2M Full Benchmark Summary · {src}  ({n:,} samples, {dim}D)',
                 fontsize=13, fontweight='bold', pad=20, color=C['title'])
    save_chart(fig, 'benchmark_summary.png')


# ═════════════════════════════════════════════════════════════════════════════
# Chart 5 — GPU Batch Search scalability (from validate_gpu_batch output)
# ═════════════════════════════════════════════════════════════════════════════

def chart_gpu_batch_scalability():
    """
    Reads gpu_batch_validation_results.json produced by
    benchmarks/validate_gpu_batch.py and plots single-query vs batch-dispatch
    speedup across 10k / 50k / 100k.
    """
    if not GPU_BATCH_JSON.exists():
        print(f"  [SKIP] chart_gpu_batch_scalability — {GPU_BATCH_JSON.name} not found")
        print("          Run:  python benchmarks/validate_gpu_batch.py")
        return

    with open(GPU_BATCH_JSON, encoding='utf-8') as f:
        raw = json.load(f)

    scales_data = raw.get("gpu_vector_index", {})
    if not scales_data:
        print("  [SKIP] chart_gpu_batch_scalability — no gpu_vector_index data")
        return

    scales, single_ms, batch_ms, speedups = [], [], [], []
    for s_str, info in scales_data.items():
        # Expect keys: single_loop_ms_total, batch_dispatch_ms_avg, batch_speedup
        sl = info.get("single_loop_ms_total")
        bd = info.get("batch_dispatch_ms_avg")
        sp = info.get("batch_speedup")
        if sl is not None and bd is not None:
            scales.append(int(s_str))
            single_ms.append(sl)
            batch_ms.append(bd)
            speedups.append(sp or (sl / bd if bd > 0 else 0))

    if not scales:
        print("  [SKIP] chart_gpu_batch_scalability — insufficient data keys")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    x   = np.arange(len(scales))
    w   = 0.35
    lbl = [f'{s//1000}k' for s in scales]

    b1 = ax1.bar(x - w/2, single_ms, w, label='Single-query loop', color=C['cpu'],    alpha=0.9, edgecolor=C['edge'])
    b2 = ax1.bar(x + w/2, batch_ms,  w, label='Batch dispatch',   color=C['vulkan'], alpha=0.9, edgecolor=C['edge'])
    ax1.set_ylabel('Total time (ms)  for 20 queries', fontsize=11, fontweight='bold')
    ax1.set_title('Single-Query Loop vs Batch Dispatch', fontsize=13, fontweight='bold', pad=12)
    ax1.set_xticks(x); ax1.set_xticklabels(lbl, fontsize=12)
    ax1.legend(framealpha=0.8, edgecolor=C['edge'])
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    bar_labels(ax1, b1, fmt='{:.0f}', offset=1, color=C['cpu'])
    bar_labels(ax1, b2, fmt='{:.0f}', offset=1, color=C['vulkan'])

    sp_bars = ax2.bar(lbl, speedups, color=C['accent'], alpha=0.9, edgecolor=C['edge'], width=0.4)
    ax2.axhline(y=1.0, color=C['edge'], linestyle='--', linewidth=1.5, label='1× baseline')
    ax2.set_ylabel('Batch speedup  vs single-query', fontsize=11, fontweight='bold')
    ax2.set_title('GPUVectorIndex Batch Speedup', fontsize=13, fontweight='bold', pad=12)
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    ax2.set_ylim(0, max(speedups) * 1.4)
    bar_labels(ax2, sp_bars, fmt='{:.2f}×', offset=0.03, color=C['accent'])
    ax2.legend(framealpha=0.8, edgecolor=C['edge'])

    fig.suptitle('GPUVectorIndex · Persistent Buffer Batch Search',
                 fontsize=14, fontweight='bold', y=1.02, color=C['title'])
    plt.tight_layout()
    save_chart(fig, 'benchmark_gpu_batch.png')


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--run-benchmark", action="store_true",
                        help="Run benchmarks/run_benchmark.py before generating charts")
    parser.add_argument("--device", default="both", choices=["cpu", "vulkan", "both"],
                        help="Device to pass to run_benchmark.py (only with --run-benchmark)")
    args = parser.parse_args()

    print("=" * 70)
    print("  M2M Benchmark Chart Generator  (live data only)")
    print("=" * 70)

    # Step 1 — optionally run benchmark
    if args.run_benchmark:
        run_benchmark_subprocess(args.device)

    # Step 2 — load fresh results (fail loudly if missing)
    print(f"\n[LOAD] {BENCHMARK_JSON.name}...")
    data = load_benchmark_json()
    ts   = data.get("timestamp", "unknown")
    src  = data.get("data_meta", {}).get("source", "real data")
    n    = data.get("config", {}).get("n_splats", "?")
    print(f"  Timestamp: {ts}")
    print(f"  Dataset:   {src}  ({n:,} samples)")

    # Step 3 — generate charts
    print()
    print("[1/5] Main benchmark (Linear vs CPU vs Vulkan)...")
    chart_main_benchmark(data)

    print("[2/5] Latency percentiles (Avg / P95 / P99)...")
    chart_latency_distribution(data)

    print("[3/5] Data Lake metrics (Ingest / Std Training / Gen Training)...")
    chart_data_lake_metrics(data)

    print("[4/5] Summary table...")
    chart_summary_table(data)

    print("[5/5] GPU Batch Search scalability...")
    chart_gpu_batch_scalability()

    print()
    print("=" * 70)
    print(f"[OK] Charts saved → {ASSETS_DIR}")
    print("     benchmark_results.png")
    print("     benchmark_latency.png")
    print("     benchmark_data_lake.png")
    print("     benchmark_summary.png")
    print("     benchmark_gpu_batch.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
