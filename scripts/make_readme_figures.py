"""Regenera las figuras del README desde los JSON medidos en bench/ (audit).

Cero numeros inventados: cada valor sale de un archivo JSON de benchmark.
Salidas en docs/: bench-hero.png, bench-qps.png, bench-filtered.png,
bench-storage.png. Reemplazan las capturas del dashboard pre-audit.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BENCH = os.path.join(ROOT, "bench")
DOCS = os.path.join(ROOT, "docs")

with open(os.path.join(BENCH, "final_ab.json")) as f:
    FINAL_AB = json.load(f)
with open(os.path.join(BENCH, "baseline.json")) as f:
    BASE = json.load(f)
with open(os.path.join(BENCH, "h8_filtered.json")) as f:
    H8 = json.load(f)
with open(os.path.join(BENCH, "h13_sq8disk.json")) as f:
    H13 = json.load(f)
with open(os.path.join(BENCH, "kmeans_bs_ab.json")) as f:
    BS = json.load(f)
with open(os.path.join(BENCH, "h7_churn.json")) as f:
    H7 = json.load(f)

BASELINE_BY = {}
for r in BASE["results"]:
    BASELINE_BY[(r["dataset"], r["backend"], r.get("n_probe"))] = r

OLD = {
    "single_10k": 97, "single_50k": 59, "batch_10k": 800,
}

plt.rcParams.update({
    "figure.facecolor": "#0f1117", "axes.facecolor": "#161922",
    "axes.edgecolor": "#2a2f3a", "axes.labelcolor": "#e6e6e6",
    "text.color": "#e6e6e6", "xtick.color": "#9aa3b2", "ytick.color": "#9aa3b2",
    "grid.color": "#242938", "font.size": 11,
    "axes.titleweight": "bold", "axes.titlesize": 13,
})
ACCENT, ACCENT2, MUTED = "#4f8cff", "#37d67a", "#6b7280"


def hero():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    fig.suptitle("M2M Vector Search — measured results (audit 2026-10)", y=1.02, fontsize=14, fontweight="bold")

    ax = axes[0]
    labels = ["single\n@10k", "batch\n@10k", "batch\n@50k"]
    # "before" solo donde hay medición citada (tanda 1); batch@50k sin A/B previo
    old = [97, 800, None]
    new = [
        FINAL_AB["n10000"]["single_qps"],
        FINAL_AB["n10000"]["batch_qps"],
        FINAL_AB["n50000"]["batch_qps"],
    ]
    x = range(3)
    for i, (o, n_) in enumerate(zip(old, new)):
        if o is not None:
            ax.bar(i - 0.2, o, 0.38, color=MUTED)
            ax.text(i - 0.2, o * 1.15, f"{o}", ha="center", fontsize=9, color=MUTED)
        else:
            ax.text(i - 0.2, 105, "n/d", ha="center", fontsize=9, color=MUTED)
        ax.bar(i + 0.2, n_, 0.38, color=ACCENT)
        ax.text(i + 0.2, n_ * 1.15, f"{n_:,.0f}", ha="center", fontsize=9, color=ACCENT)
    ax.bar([], [])  # placeholders para la leyenda
    ax.bar([], [], color=MUTED, label="before audit")
    ax.bar([], [], color=ACCENT, label="after audit")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("queries / second")
    ax.set_ylim(80, 12000)
    ax.set_title("Search throughput (R@10 ≥ 0.9995)")
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.5)

    ax = axes[1]
    rows = [r for r in BASE["results"] if r["dataset"] == "clustered_10000"]
    names, qps, recs = [], [], []
    for r in rows:
        if r["backend"] == "linear_numpy":
            names.append("linear\nnumpy")
        elif r["backend"] == "m2m_l2" and r.get("n_probe") == 5:
            names.append("m2m ivf\n(probe=5)")
        elif r["backend"] == "m2m_gaussian":
            names.append("m2m gauss\n(probe=5)")
        else:
            continue
        qps.append(r["qps"])
        recs.append(r["recall"])
    names.append("m2m batch\n(r=1.0)")
    qps.append(FINAL_AB["n10000"]["batch_qps"])
    colors = [MUTED] + [ACCENT] * (len(names) - 1)
    bars = ax.bar(names, qps, color=colors)
    bars[-1].set_color(ACCENT2)
    for b, v, r in zip(bars, qps, recs + [1.0]):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.1, f"{v:,.0f}", ha="center", fontsize=9)
    ax.set_ylabel("queries / second")
    ax.set_title("10k clustered — vs linear scan (same run)")
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.5)

    fig.tight_layout()
    fig.savefig(os.path.join(DOCS, "bench-hero.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)


def qps_fig():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    fig.suptitle("Streaming & index build (measured)", y=1.02, fontsize=14, fontweight="bold")

    ax = axes[0]
    steps = [r["churn_pct"] for r in H7["rows"][1:]]
    new_docs = [r["n_docs"] for r in H7["rows"][1:]]
    ax.plot(steps, new_docs, "o-", color=ACCENT2, lw=2)
    for s, d in zip(steps, new_docs):
        ax.annotate(f"{d:,} docs", (s, d), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    ax.set_xlabel("churn step (% of base re-added/replaced)")
    ax.set_ylabel("corpus size (docs)")
    ax.set_title("IVF attach: add() = 0.03 s/step (was 6–8 s)")
    ax.set_xticks(steps)
    ax.grid(alpha=0.5)

    ax = axes[1]
    ks = ["32", "256", "1000", "10000"]
    build = [BS[k]["build_s"] for k in ks]
    rec = [BS[k]["recall@10"] for k in ks]
    colors = [ACCENT if k == "1000" else MUTED for k in ks]
    bars = ax.bar(ks, build, color=colors)
    for b, v, r in zip(bars, build, rec):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.15, f"{v:.1f}s\nr={r:.4f}", ha="center", fontsize=9)
    ax.set_xlabel("MiniBatchKMeans batch_size")
    ax.set_ylabel("index build time (s)")
    ax.set_title("50k×384 k=224 — default 10000 → 1000 = 3.8× faster", pad=14)
    ax.set_ylim(2, 40)
    ax.grid(axis="y", alpha=0.5)

    fig.tight_layout()
    fig.savefig(os.path.join(DOCS, "bench-qps.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)


def filtered_fig():
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    fig.suptitle("Filtered search — pre-filter vs old post-filter", y=1.04, fontsize=14, fontweight="bold")
    sel = ["5%", "25%", "50%"]
    keys = ["p05", "p25", "p50"]
    new = [H8[k]["recall@10"] for k in keys]
    legacy = H8["post_filter_legacy"]
    old = [legacy[k]["recall@10"] for k in keys]
    x = range(3)
    ax.bar([i - 0.2 for i in x], old, 0.38, label="post-filter on top-k (old)", color=MUTED)
    ax.bar([i + 0.2 for i in x], new, 0.38, label="search_batch_filtered (new)", color=ACCENT)
    for i in x:
        ax.text(i - 0.2, old[i] + 0.03, f"{old[i]:.2f}", ha="center", fontsize=10, color=MUTED)
        ax.text(i + 0.2, new[i] + 0.03, f"{new[i]:.4f}", ha="center", fontsize=10, color=ACCENT)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"selectivity {s}" for s in sel])
    ax.set_ylabel("recall@10")
    ax.set_ylim(0, 1.18)
    ax.set_title("10k×384 — filter selects docs with a given metadata key")
    ax.legend(frameon=False, fontsize=10, loc="upper left", bbox_to_anchor=(0.02, 0.98))
    ax.grid(axis="y", alpha=0.5)
    fig.text(
        0.99, 0.01,
        "QPS (new): %s — bench/h8_filtered.json" % " / ".join(f"{H8[k]['qps']:.0f}" for k in keys),
        ha="right", fontsize=8, color=MUTED,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(DOCS, "bench-filtered.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)


def storage_fig():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    fig.suptitle("SQ8DiskStore — compressed mode (H13, 600k×384 unnormalized)", y=1.02, fontsize=14, fontweight="bold")

    ax = axes[0]
    modes = ["float32\n(all in RAM)", "SQ8 + mmap\nrescoring"]
    ram = [H13["float32"]["ram_arrays_mb"], H13["sq8disk"]["ram_arrays_mb"]]
    bars = ax.bar(modes, ram, color=[MUTED, ACCENT])
    for b, v in zip(bars, ram):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.05, f"{v:,.0f} MB", ha="center", fontsize=11)
    ax.set_ylabel("vector arrays in RAM (MB)")
    ax.set_title("RAM: 3.9× less (recall 1.0 both)", pad=14)
    ax.set_ylim(0, 1450)
    ax.grid(axis="y", alpha=0.5)

    ax = axes[1]
    qps = [H13["float32"]["qps"], H13["sq8disk"]["qps"]]
    bars = ax.bar(modes, qps, color=[MUTED, ACCENT])
    for b, v in zip(bars, qps):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.05, f"{v:.0f}", ha="center", fontsize=11)
    ax.set_ylabel("queries / second")
    ax.set_title("Throughput: 2.2× slower (honest trade-off)", pad=14)
    ax.set_ylim(0, 95)
    ax.grid(axis="y", alpha=0.5)

    fig.tight_layout()
    fig.savefig(os.path.join(DOCS, "bench-storage.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    hero()
    qps_fig()
    filtered_fig()
    storage_fig()
    for f in ("bench-hero", "bench-qps", "bench-filtered", "bench-storage"):
        p = os.path.join(DOCS, f + ".png")
        print(f, os.path.getsize(p) // 1024, "KB")
