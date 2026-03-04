# Generar graficos
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

assets = Path(r"C:\Users\Brian\.openclaw\workspace\projects\m2m\assets")
assets.mkdir(exist_ok=True)

print("Generando graficos...")

# Grafico 1: Benchmark
fig, ax = plt.subplots(figsize=(10, 5))
methods = ['Linear Scan', 'M2M']
times = [94.79, 0.99]
colors = ['#FF6B6B', '#00C4B6']
bars = ax.bar(methods, times, color=colors)
ax.set_ylabel('Latencia (ms)')
ax.set_title('Comparacion de Latencia (100K vectores)')
for bar, val in zip(bars, times):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val}ms', ha='center')
plt.tight_layout()
fig.savefig(assets / 'chart_benchmark_comparison.png', dpi=150)
print("  chart_benchmark_comparison.png OK")
plt.close()

# Grafico 2: Arquitectura
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Cajas
ax.add_patch(plt.Rectangle((1, 7), 8, 2, fill=True, color='#3498DB'))
ax.text(5, 8, 'CAPA APLICACION', ha='center', va='center', color='white', fontsize=12, fontweight='bold')

ax.add_patch(plt.Rectangle((1, 4), 8, 2.5, fill=True, color='#2ECC71'))
ax.text(5, 5.25, 'M2M CORE ENGINE', ha='center', va='center', color='white', fontsize=12, fontweight='bold')

ax.add_patch(plt.Rectangle((1, 1), 8, 2.5, fill=True, color='#E74C3C'))
ax.text(5, 2.25, 'MEMORIA (VRAM/RAM/SSD)', ha='center', va='center', color='white', fontsize=12, fontweight='bold')

plt.tight_layout()
fig.savefig(assets / 'chart_architecture.png', dpi=150)
print("  chart_architecture.png OK")
plt.close()

# Grafico 3: Tipos de datos
fig, ax = plt.subplots(figsize=(10, 5))
types = ['Imagenes', 'Geo', 'Audio', 'Texto']
compat = [0.9, 0.8, 0.6, 0.3]
colors = ['#00C4B6' if c > 0.5 else '#FF6B6B' for c in compat]
ax.barh(types, compat, color=colors)
ax.set_xlabel('Compatibilidad M2M')
ax.set_title('Cuando usar M2M')
ax.axvline(x=0.5, color='black', linestyle='--')
plt.tight_layout()
fig.savefig(assets / 'chart_data_types.png', dpi=150)
print("  chart_data_types.png OK")
plt.close()

# Grafico 4: Workflow
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

ax.text(5, 9, 'INICIO', ha='center', fontsize=14, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='#00C4B6'))
ax.text(5, 7, 'Analizar\nDatos', ha='center', fontsize=12,
        bbox=dict(boxstyle='round', facecolor='#2C3E50', edgecolor='white'), color='white')
ax.text(5, 5, 'Score\n> 0.2?', ha='center', fontsize=12,
        bbox=dict(boxstyle='round', facecolor='#FFE66D'))
ax.text(2, 3, 'Linear\nScan', ha='center', fontsize=12,
        bbox=dict(boxstyle='round', facecolor='#FF6B6B'))
ax.text(8, 3, 'Probar\nM2M', ha='center', fontsize=12,
        bbox=dict(boxstyle='round', facecolor='#00C4B6'))

ax.annotate('', xy=(5, 7.8), xytext=(5, 8.2), arrowprops=dict(arrowstyle='->', lw=2))
ax.annotate('', xy=(5, 5.8), xytext=(5, 6.2), arrowprops=dict(arrowstyle='->', lw=2))
ax.annotate('', xy=(2.5, 4), xytext=(4, 5), arrowprops=dict(arrowstyle='->', lw=2))
ax.annotate('', xy=(7.5, 4), xytext=(6, 5), arrowprops=dict(arrowstyle='->', lw=2))

ax.text(3, 4.5, 'NO', fontsize=10, fontweight='bold')
ax.text(7, 4.5, 'SI', fontsize=10, fontweight='bold')

plt.tight_layout()
fig.savefig(assets / 'chart_workflow.png', dpi=150)
print("  chart_workflow.png OK")
plt.close()

print("\nTodos los graficos generados!")
