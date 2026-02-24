#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M2M Benchmark Chart Generator

Generates charts from REAL benchmark data (sklearn Handwritten Digits dataset).
Includes:
  - Linear search baseline vs M2M CPU vs M2M Vulkan GPU
  - Data lake metrics: ingest, standard training, generative training
  - Latency distribution (Avg / P95 / P99)

Data source: examples/validate_data_lake.py → data_lake_real_metrics.json
Charts match the performance section in README.md.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
import time
from pathlib import Path

# Paths
script_dir = Path(__file__).parent
project_dir = script_dir.parent
assets_dir = project_dir / 'assets'
assets_dir.mkdir(exist_ok=True)

# === REAL BENCHMARK DATA (from README.md) ===
# Source: examples/validate_data_lake.py with sklearn digits dataset
# 10,000 real embeddings (640D), 1,000 queries, K=10

DATA = {
    'dataset': 'sklearn Handwritten Digits',
    'dataset_detail': '1,797 real images → 640D latent space, upsampled to 10K',
    'n_splats': 10_000,
    'n_queries': 1_000,
    'k': 10,
    'dim': 640,

    # Linear search baseline (brute-force O(N) torch.cdist)
    # Measured on same hardware, same dataset
    'linear': {
        'avg_latency_ms': 93.53,
        'throughput_qps': 10.70,
    },

    # M2M on CPU only
    'cpu': {
        'retrieval_qps': 48.97,
        'avg_latency_ms': 18.92,
        'p95_latency_ms': 24.10,
        'p99_latency_ms': 30.44,
        'ingest_throughput': 890,
        'standard_training_throughput': 49_651,
        'generative_training_throughput': 38_667,
        'standard_training_loss': 2.3029,
        'generative_training_loss': 2.3035,
    },

    # M2M with Vulkan GPU compute shaders
    'vulkan': {
        'retrieval_qps': 43.07,
        'avg_latency_ms': 23.22,
        'p95_latency_ms': 27.93,
        'p99_latency_ms': 34.03,
        'ingest_throughput': 1_046,
        'standard_training_throughput': 48_784,
        'generative_training_throughput': 38_534,
        'standard_training_loss': 2.3037,
        'generative_training_loss': 2.3027,
    },
}

# GitHub dark theme
plt.style.use('dark_background')
plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor': '#161b22',
    'axes.edgecolor': '#30363d',
    'axes.labelcolor': '#c9d1d9',
    'text.color': '#c9d1d9',
    'xtick.color': '#8b949e',
    'ytick.color': '#8b949e',
    'grid.color': '#21262d',
    'font.family': 'sans-serif',
    'font.size': 11,
})

C = {
    'linear': '#8b949e',    # Gray (baseline)
    'cpu': '#58a6ff',       # Blue
    'vulkan': '#f78166',    # Orange
    'accent': '#7ee787',    # Green
    'purple': '#d2a8ff',    # Purple
    'edge': '#30363d',
}


def save_chart(fig, filename, dpi=200):
    filepath = assets_dir / filename
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='#0d1117',
                edgecolor='none', pad_inches=0.3)
    print(f"  [OK] {filepath.name}")
    plt.close(fig)


def try_load_live_data():
    """Load live benchmark JSON if available and overlay onto DATA."""
    live_file = project_dir / 'data_lake_real_metrics.json'
    if not live_file.exists():
        return False

    with open(live_file, 'r') as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        return False

    for entry in raw:
        hw = entry.get('hardware', '')
        rr = entry.get('retrieval_routing', {})
        st = entry.get('standard_training', {})
        gt = entry.get('generative_training', {})
        ingest = entry.get('ingest_throughput_qps', 0)

        # Determine which backend
        if 'Vulkan: True' in hw or 'vulkan' in hw.lower():
            target = DATA['vulkan']
        else:
            target = DATA['cpu']

        if rr:
            target['retrieval_qps'] = rr.get('qps', target['retrieval_qps'])
            target['avg_latency_ms'] = rr.get('avg_latency_ms', target['avg_latency_ms'])
            target['p95_latency_ms'] = rr.get('p95_latency_ms', target['p95_latency_ms'])
            target['p99_latency_ms'] = rr.get('p99_latency_ms', target['p99_latency_ms'])
        if ingest:
            target['ingest_throughput'] = ingest
        if st:
            target['standard_training_throughput'] = st.get('throughput_splats_per_sec',
                                                            target['standard_training_throughput'])
            target['standard_training_loss'] = st.get('avg_loss', target['standard_training_loss'])
        if gt:
            target['generative_training_throughput'] = gt.get('throughput_splats_per_sec',
                                                              target['generative_training_throughput'])
            target['generative_training_loss'] = gt.get('avg_loss', target['generative_training_loss'])

    print("  [OK] Loaded live data from data_lake_real_metrics.json")
    return True


def add_bar_labels(ax, bars, fmt='{:.1f}', offset=0, fontsize=10, color='#c9d1d9'):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + offset,
                fmt.format(h), ha='center', va='bottom',
                fontsize=fontsize, fontweight='bold', color=color)


# ──────────────────────────────────────────────────────────────
# Chart 1: Main benchmark — Linear vs CPU vs Vulkan
# ──────────────────────────────────────────────────────────────
def chart_main_benchmark():
    """3-way comparison: Linear Search vs M2M CPU vs M2M Vulkan GPU."""
    d = DATA
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    labels = ['Linear Search\n(Brute Force)', 'M2M\n(CPU)', 'M2M\n(Vulkan GPU)']
    colors = [C['linear'], C['cpu'], C['vulkan']]

    # Latency
    lats = [d['linear']['avg_latency_ms'], d['cpu']['avg_latency_ms'], d['vulkan']['avg_latency_ms']]
    bars1 = ax1.bar(labels, lats, color=colors, alpha=0.9, edgecolor=C['edge'], width=0.55)
    ax1.set_ylabel('Avg Latency (ms)', fontsize=12, fontweight='bold')
    ax1.set_title('Search Latency (lower is better)', fontsize=13, fontweight='bold', pad=12)
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    ax1.set_ylim(0, max(lats) * 1.35)
    add_bar_labels(ax1, bars1, fmt='{:.2f} ms', offset=1)

    # Speedup annotations
    cpu_speedup = d['linear']['avg_latency_ms'] / d['cpu']['avg_latency_ms']
    ax1.annotate(f'{cpu_speedup:.1f}x faster', xy=(1, lats[1]),
                 xytext=(1.4, lats[0] * 0.7),
                 fontsize=11, fontweight='bold', color=C['cpu'],
                 arrowprops=dict(arrowstyle='->', color=C['cpu'], lw=1.5),
                 ha='center')

    # Throughput
    qps = [d['linear']['throughput_qps'], d['cpu']['retrieval_qps'], d['vulkan']['retrieval_qps']]
    bars2 = ax2.bar(labels, qps, color=colors, alpha=0.9, edgecolor=C['edge'], width=0.55)
    ax2.set_ylabel('Queries per Second', fontsize=12, fontweight='bold')
    ax2.set_title('Retrieval Throughput (higher is better)', fontsize=13, fontweight='bold', pad=12)
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    ax2.set_ylim(0, max(qps) * 1.35)
    add_bar_labels(ax2, bars2, fmt='{:.1f} QPS', offset=0.5)

    fig.suptitle(
        f'M2M vs Linear Search · Real Data ({d["n_splats"]:,} embeddings, '
        f'{d["n_queries"]:,} queries, K={d["k"]})',
        fontsize=14, fontweight='bold', y=1.02, color='#e6edf3'
    )
    plt.tight_layout()
    save_chart(fig, 'benchmark_results.png')


# ──────────────────────────────────────────────────────────────
# Chart 2: Latency distribution — Avg / P95 / P99
# ──────────────────────────────────────────────────────────────
def chart_latency_distribution():
    """CPU vs Vulkan latency percentiles."""
    d = DATA
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(3)
    w = 0.3
    cpu_lats = [d['cpu']['avg_latency_ms'], d['cpu']['p95_latency_ms'], d['cpu']['p99_latency_ms']]
    vk_lats = [d['vulkan']['avg_latency_ms'], d['vulkan']['p95_latency_ms'], d['vulkan']['p99_latency_ms']]

    b1 = ax.bar(x - w/2, cpu_lats, w, label='CPU', color=C['cpu'], alpha=0.9, edgecolor=C['edge'])
    b2 = ax.bar(x + w/2, vk_lats, w, label='Vulkan GPU', color=C['vulkan'], alpha=0.9, edgecolor=C['edge'])

    ax.set_ylabel('Latency (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Latency Distribution · CPU vs Vulkan GPU', fontsize=14, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(['Avg', 'P95', 'P99'], fontsize=12)
    ax.legend(loc='upper left', framealpha=0.8, edgecolor=C['edge'])
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.set_ylim(0, max(max(cpu_lats), max(vk_lats)) * 1.3)

    add_bar_labels(ax, b1, fmt='{:.1f}', offset=0.3, fontsize=10, color=C['cpu'])
    add_bar_labels(ax, b2, fmt='{:.1f}', offset=0.3, fontsize=10, color=C['vulkan'])

    plt.tight_layout()
    save_chart(fig, 'benchmark_latency.png')


# ──────────────────────────────────────────────────────────────
# Chart 3: Data Lake Metrics — ingest, training, generative
# ──────────────────────────────────────────────────────────────
def chart_data_lake_metrics():
    """Data lake operational metrics: ingest, standard training, generative training."""
    d = DATA
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    backends = ['CPU', 'Vulkan GPU']
    colors = [C['cpu'], C['vulkan']]

    # --- Ingest Throughput ---
    ingest = [d['cpu']['ingest_throughput'], d['vulkan']['ingest_throughput']]
    bars1 = ax1.bar(backends, ingest, color=colors, alpha=0.9, edgecolor=C['edge'], width=0.45)
    ax1.set_ylabel('Splats / sec', fontsize=12, fontweight='bold')
    ax1.set_title('Ingest Throughput', fontsize=14, fontweight='bold', pad=12)
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    ax1.set_ylim(0, max(ingest) * 1.3)
    add_bar_labels(ax1, bars1, fmt='{:.0f}', offset=10)
    if ingest[1] > ingest[0]:
        pct = ((ingest[1] / ingest[0]) - 1) * 100
        ax1.text(1, ingest[1] * 0.55, f'+{pct:.0f}%\nVulkan',
                 ha='center', fontsize=13, fontweight='bold', color=C['accent'],
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='#0d1117',
                           edgecolor=C['accent'], alpha=0.9))

    # --- Standard Training ---
    std_tp = [d['cpu']['standard_training_throughput'], d['vulkan']['standard_training_throughput']]
    bars2 = ax2.bar(backends, std_tp, color=colors, alpha=0.9, edgecolor=C['edge'], width=0.45)
    ax2.set_ylabel('Splats / sec', fontsize=12, fontweight='bold')
    ax2.set_title('Standard Training\n(SOC Importance Sampling)', fontsize=14, fontweight='bold', pad=12)
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    ax2.set_ylim(0, max(std_tp) * 1.3)
    add_bar_labels(ax2, bars2, fmt='{:,.0f}', offset=500)

    # Add loss values below bars
    for i, (bar, loss) in enumerate(zip(bars2, [d['cpu']['standard_training_loss'],
                                                 d['vulkan']['standard_training_loss']])):
        ax2.text(bar.get_x() + bar.get_width() / 2, max(std_tp) * 0.15,
                 f'loss: {loss:.4f}', ha='center', fontsize=9, color='#8b949e',
                 style='italic')

    # --- Generative Training ---
    gen_tp = [d['cpu']['generative_training_throughput'], d['vulkan']['generative_training_throughput']]
    bars3 = ax3.bar(backends, gen_tp, color=colors, alpha=0.9, edgecolor=C['edge'], width=0.45)
    ax3.set_ylabel('Splats / sec', fontsize=12, fontweight='bold')
    ax3.set_title('Generative Training\n(Langevin Augmentation)', fontsize=14, fontweight='bold', pad=12)
    ax3.grid(axis='y', linestyle='--', alpha=0.3)
    ax3.set_ylim(0, max(gen_tp) * 1.3)
    add_bar_labels(ax3, bars3, fmt='{:,.0f}', offset=400)

    for i, (bar, loss) in enumerate(zip(bars3, [d['cpu']['generative_training_loss'],
                                                 d['vulkan']['generative_training_loss']])):
        ax3.text(bar.get_x() + bar.get_width() / 2, max(gen_tp) * 0.15,
                 f'loss: {loss:.4f}', ha='center', fontsize=9, color='#8b949e',
                 style='italic')

    fig.suptitle(
        f'M2M Data Lake Metrics · Real Data ({d["dataset"]}, {d["n_splats"]:,} samples)',
        fontsize=15, fontweight='bold', y=1.02, color='#e6edf3'
    )
    plt.tight_layout()
    save_chart(fig, 'benchmark_data_lake.png')


# ──────────────────────────────────────────────────────────────
# Chart 4: Summary table
# ──────────────────────────────────────────────────────────────
def chart_summary_table():
    """Visual summary table for README."""
    d = DATA
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')

    cpu_speedup = d['linear']['avg_latency_ms'] / d['cpu']['avg_latency_ms']
    vk_speedup = d['linear']['avg_latency_ms'] / d['vulkan']['avg_latency_ms']

    headers = ['Metric', 'Linear Scan', 'M2M CPU', 'M2M Vulkan GPU']
    rows = [
        ['Avg Latency',
         f'{d["linear"]["avg_latency_ms"]:.2f} ms',
         f'{d["cpu"]["avg_latency_ms"]:.2f} ms',
         f'{d["vulkan"]["avg_latency_ms"]:.2f} ms'],
        ['Throughput (QPS)',
         f'{d["linear"]["throughput_qps"]:.2f}',
         f'{d["cpu"]["retrieval_qps"]:.2f}',
         f'{d["vulkan"]["retrieval_qps"]:.2f}'],
        ['Speedup vs Linear',
         '1.0x',
         f'{cpu_speedup:.1f}x',
         f'{vk_speedup:.1f}x'],
        ['P95 Latency', '—',
         f'{d["cpu"]["p95_latency_ms"]:.2f} ms',
         f'{d["vulkan"]["p95_latency_ms"]:.2f} ms'],
        ['P99 Latency', '—',
         f'{d["cpu"]["p99_latency_ms"]:.2f} ms',
         f'{d["vulkan"]["p99_latency_ms"]:.2f} ms'],
        ['Ingest Rate', '—',
         f'{d["cpu"]["ingest_throughput"]:.0f} splats/s',
         f'{d["vulkan"]["ingest_throughput"]:.0f} splats/s'],
        ['Std Training', '—',
         f'{d["cpu"]["standard_training_throughput"]:,.0f} splats/s',
         f'{d["vulkan"]["standard_training_throughput"]:,.0f} splats/s'],
        ['Gen Training', '—',
         f'{d["cpu"]["generative_training_throughput"]:,.0f} splats/s',
         f'{d["vulkan"]["generative_training_throughput"]:,.0f} splats/s'],
    ]

    table = ax.table(cellText=rows, colLabels=headers, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.7)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#30363d')
        if row == 0:
            cell.set_facecolor('#21262d')
            cell.set_text_props(fontweight='bold', color='#e6edf3')
        else:
            cell.set_facecolor('#0d1117')
            cell.set_text_props(color='#c9d1d9')
            if col == 1:
                cell.set_text_props(color=C['linear'])
            elif col == 2:
                cell.set_text_props(color=C['cpu'])
            elif col == 3:
                cell.set_text_props(color=C['vulkan'])

    ax.set_title(
        f'M2M Full Benchmark Summary · {d["dataset"]} ({d["n_splats"]:,} samples, {d["dim"]}D)',
        fontsize=14, fontweight='bold', pad=20, color='#e6edf3'
    )
    save_chart(fig, 'benchmark_summary.png')


def main():
    """Generate all charts from real benchmark data."""
    print("=" * 70)
    print("M2M Benchmark Chart Generator (Real Data)")
    print("=" * 70)
    print()
    print(f"  Dataset: {DATA['dataset']} ({DATA['dataset_detail']})")
    print(f"  Config:  {DATA['n_splats']:,} splats, "
          f"{DATA['n_queries']:,} queries, K={DATA['k']}, {DATA['dim']}D")
    print()

    try_load_live_data()
    print()

    print("[1/4] Main benchmark (Linear vs CPU vs Vulkan)...")
    chart_main_benchmark()

    print("[2/4] Latency distribution (Avg / P95 / P99)...")
    chart_latency_distribution()

    print("[3/4] Data lake metrics (Ingest / Training / Generative)...")
    chart_data_lake_metrics()

    print("[4/4] Summary table...")
    chart_summary_table()

    print()
    print("=" * 70)
    print(f"[OK] All 4 charts generated → {assets_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
