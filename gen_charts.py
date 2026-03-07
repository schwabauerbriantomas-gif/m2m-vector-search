# Generar graficos
import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt

assets = Path(__file__).parent / "assets"
assets.mkdir(exist_ok=True, parents=True)

print("Generando graficos...")

# Grafico 1: Benchmark
fig, ax = plt.subplots(figsize=(10, 5))
methods = ["Linear Scan", "M2M CPU", "M2M Vulkan"]
times = [30.06, 89.24, 51.88]
colors = ["#FF6B6B", "#3498DB", "#00C4B6"]
bars = ax.bar(methods, times, color=colors)
ax.set_ylabel("Latencia (ms)")
ax.set_title("Comparacion de Latencia (10K vectores, 640D)")
for bar, val in zip(bars, times):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{val}ms",
        ha="center",
        va="bottom",
    )
plt.tight_layout()
fig.savefig(assets / "chart_benchmark_comparison.png", dpi=150)
print("  chart_benchmark_comparison.png OK")
plt.close()


# Grafico 3: Tipos de datos
fig, ax = plt.subplots(figsize=(10, 5))
types = ["Imagenes", "Geo", "Audio", "Texto"]
compat = [0.9, 0.8, 0.6, 0.3]
colors = ["#00C4B6" if c > 0.5 else "#FF6B6B" for c in compat]
ax.barh(types, compat, color=colors)
ax.set_xlabel("Compatibilidad M2M")
ax.set_title("When to use M2M")
ax.axvline(x=0.5, color="black", linestyle="--")
plt.tight_layout()
fig.savefig(assets / "chart_data_types.png", dpi=150)
print("  chart_data_types.png OK")
plt.close()

# Grafico 4: Workflow
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")

ax.text(
    5,
    9,
    "INICIO",
    ha="center",
    fontsize=14,
    fontweight="bold",
    bbox=dict(boxstyle="round", facecolor="#00C4B6"),
)
ax.text(
    5,
    7,
    "Analizar\nDatos",
    ha="center",
    fontsize=12,
    bbox=dict(boxstyle="round", facecolor="#2C3E50", edgecolor="white"),
    color="white",
)
ax.text(
    5,
    5,
    "Score\n> 0.2?",
    ha="center",
    fontsize=12,
    bbox=dict(boxstyle="round", facecolor="#FFE66D"),
)
ax.text(
    2,
    3,
    "Linear\nScan",
    ha="center",
    fontsize=12,
    bbox=dict(boxstyle="round", facecolor="#FF6B6B"),
)
ax.text(
    8,
    3,
    "Probar\nM2M",
    ha="center",
    fontsize=12,
    bbox=dict(boxstyle="round", facecolor="#00C4B6"),
)

ax.annotate("", xy=(5, 7.8), xytext=(5, 8.2), arrowprops=dict(arrowstyle="->", lw=2))
ax.annotate("", xy=(5, 5.8), xytext=(5, 6.2), arrowprops=dict(arrowstyle="->", lw=2))
ax.annotate("", xy=(2.5, 4), xytext=(4, 5), arrowprops=dict(arrowstyle="->", lw=2))
ax.annotate("", xy=(7.5, 4), xytext=(6, 5), arrowprops=dict(arrowstyle="->", lw=2))

ax.text(3, 4.5, "NO", fontsize=10, fontweight="bold")
ax.text(7, 4.5, "SI", fontsize=10, fontweight="bold")

plt.tight_layout()
fig.savefig(assets / "chart_workflow.png", dpi=150)
print("  chart_workflow.png OK")
plt.close()

print("\nTodos los graficos generados!")
