#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M2M Benchmark Chart Generator

Generates charts from ACTUAL benchmark data only.
No simulated or invented data.
"""

import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

# Paths
script_dir = Path(__file__).parent
project_dir = script_dir.parent
assets_dir = project_dir / 'assets'
assets_dir.mkdir(exist_ok=True)
benchmark_file = project_dir / 'benchmark_results.json'

# Professional colors
COLORS = {
    'primary': '#00C4B6',
    'secondary': '#FF4B4B',
    'accent': '#9D4EDD',
}

# Dark theme
plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#0d1117'
plt.rcParams['axes.facecolor'] = '#0d1117'
plt.rcParams['axes.edgecolor'] = '#30363d'
plt.rcParams['axes.labelcolor'] = '#c9d1d9'
plt.rcParams['text.color'] = '#c9d1d9'

def save_chart(fig, filename, dpi=300):
    """Save chart."""
    filepath = assets_dir / filename
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='#0d1117')
    print(f"[OK] Generated: {filepath}")
    plt.close(fig)

def load_benchmark_data():
    """Load actual benchmark data."""
    if not benchmark_file.exists():
        print(f"[ERROR] Benchmark file not found: {benchmark_file}")
        return None
    
    with open(benchmark_file, 'r') as f:
        return json.load(f)

def chart_real_benchmark():
    """Generate chart from real benchmark data only."""
    data = load_benchmark_data()
    if not data:
        return
    
    config = data['configuration']
    results = data['results']
    
    # Extract data
    systems = ['Linear Search\n(O(N))', 'M2M\n(HRM2 + KNN)']
    latencies = [
        results['Linear Search (O(N))']['avg_latency_ms'],
        results['M2M (HRM2 + KNN)']['avg_latency_ms']
    ]
    throughputs = [
        results['Linear Search (O(N))']['throughput_qps'],
        results['M2M (HRM2 + KNN)']['throughput_qps']
    ]
    speedup = results['M2M (HRM2 + KNN)']['speedup_vs_linear']
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Chart 1: Latency
    colors = [COLORS['secondary'], COLORS['primary']]
    bars1 = ax1.bar(systems, latencies, color=colors, alpha=0.85, edgecolor='#30363d')
    ax1.set_ylabel('Avg Latency (ms)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Query Latency ({config["n_splats"]:,} splats, k={config["k"]})', 
                  fontsize=13, fontweight='bold', pad=15)
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Add labels
    for bar, lat in zip(bars1, latencies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height, 
                f'{lat:.2f}ms', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Chart 2: Throughput
    bars2 = ax2.bar(systems, throughputs, color=colors, alpha=0.85, edgecolor='#30363d')
    ax2.set_ylabel('Throughput (QPS)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Throughput ({config["n_splats"]:,} splats, {config["queries"]} queries)', 
                  fontsize=13, fontweight='bold', pad=15)
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Add labels
    for bar, qps in zip(bars2, throughputs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, height, 
                f'{qps:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add speedup annotation
    ax2.text(1.5, max(throughputs) * 0.7, f'{speedup:.1f}x\nspeedup', 
             ha='center', fontsize=14, fontweight='bold', color=COLORS['primary'],
             bbox=dict(boxstyle='round', facecolor='#0d1117', edgecolor=COLORS['primary']))
    
    plt.tight_layout()
    save_chart(fig, 'benchmark_results.png')
    
    print(f"\n[BENCHMARK DATA]")
    print(f"  Splats: {config['n_splats']:,}")
    print(f"  Queries: {config['queries']:,}")
    print(f"  K: {config['k']}")
    print(f"  Device: {config['device']}")
    print(f"  Linear: {latencies[0]:.2f}ms, {throughputs[0]:.1f} QPS")
    print(f"  M2M: {latencies[1]:.2f}ms, {throughputs[1]:.1f} QPS")
    print(f"  Speedup: {speedup:.1f}x")

def main():
    """Generate charts from real data only."""
    print("=" * 70)
    print("M2M Benchmark Chart Generator (Real Data Only)")
    print("=" * 70)
    print()
    
    print("[INFO] Generating charts from benchmark_results.json...")
    chart_real_benchmark()
    
    print()
    print("=" * 70)
    print("[OK] Charts generated from validated benchmark data")
    print("=" * 70)

if __name__ == "__main__":
    main()
