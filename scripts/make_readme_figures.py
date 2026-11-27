#!/usr/bin/env python
"""Regenera las cuatro figuras de benchmark del README leyendo SOLO los JSON medidos por
``bench/*.py`` con el protocolo ``bench/_protocol.py`` (audit r4-10, r5-01..r5-15).

* Ninguna cifra vive en este archivo: QPS, recall, tiempos, tamaños, commits y ratios salen de
  ``benchmarks/results/*.json`` (rama) y ``benchmarks/results/before_a74d9e4/*.json`` (commit
  base, medido con los mismos scripts). Lo que no está medido se rotula "n/d" (audit r4-17).
* Barras lineales desde 0 (sin ejes log truncados), dot-plot QPS x recall con TODOS los datasets
  (uniforme y 50k incluidos), ratios calculados, títulos que describen su panel, leyendas con
  handles reales, paleta sólida (sin alfa) y PNG sin canal alfa.
* Salida determinista: rcParams fijos, sin marcas de tiempo; ``docs/bench-manifest.json`` guarda
  figura -> JSON fuente + sha256 (y el sha256 del PNG). ``--check`` sólo corre el sanity check
  de que el script no contiene literales de cifras.

Uso:  python scripts/make_readme_figures.py [--check]
"""
import ast
import hashlib
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTADOS = os.path.join(ROOT, "benchmarks", "results")
ANTES = os.path.join(RESULTADOS, "before_a74d9e4")
DOCS = os.path.join(ROOT, "docs")
DPI = 130
ND = "n/d"
LITERALES_DE_LAYOUT = frozenset({DPI})  # únicos enteros >= 100 permitidos en este archivo

FONDO, EJES, REJILLA, TEXTO, TEXTO2 = "#0f1117", "#161922", "#242938", "#e6e6e6", "#9aa3b2"
DESPUES, ANTES_C, LINEAL, OTRA = "#4f8cff", "#8b8f98", "#f59e0b", "#37d67a"
AZULES = ["#1e3a8a", "#2563eb", "#60a5fa", "#93c5fd"]

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from PIL import Image  # noqa: E402


# --------------------------------------------------------------------------- sanity
def sin_cifras_hardcodeadas():
    """El script no puede contener literales numéricos 'de cifra': enteros >= 100 (salvo el DPI)
    (las conversiones de unidad se escriben como potencias: 10**2, 10**3)
    o floats con 3+ decimales. Las cifras de las figuras vienen de los JSON, no de acá."""
    with open(os.path.abspath(__file__), encoding="utf-8") as f:
        arbol = ast.parse(f.read())
    sospechosos = []
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Constant) and isinstance(nodo.value, (int, float)) \
                and not isinstance(nodo.value, bool):
            v = nodo.value
            if isinstance(v, int) and abs(v) >= 10 ** 2 and v not in LITERALES_DE_LAYOUT:
                sospechosos.append((nodo.lineno, v))
            if isinstance(v, float) and len(repr(v).split(".")[-1]) >= 3:
                sospechosos.append((nodo.lineno, v))
    if sospechosos:
        raise SystemExit(f"make_readme_figures: literales de cifras hardcodeados {sospechosos}; "
                         "toda cifra debe salir de benchmarks/results/*.json")


def sha256(ruta):
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


# ------------------------------------------------------------------------------ datos
def cargar(nombre, antes=False):
    ruta = os.path.join(ANTES if antes else RESULTADOS, nombre + ".json")
    if not os.path.exists(ruta):
        return None, ruta
    with open(ruta, encoding="utf-8") as f:
        return json.load(f), ruta


def _match(param, sub):
    return param == sub or param.startswith(sub + " ") or param.startswith(sub + " (")


def fila(doc, param=(), **ds):
    """Primera fila del dataset (kind, n, d) cuyo parametro contiene todas las subcadenas."""
    if not doc:
        return None
    for f in doc["resultados"]:
        d = f["dataset"]
        if all(d.get(k) == v for k, v in ds.items()) and all(_match(f["parametro"], s) for s in param):
            return f
    return None


def num(f, clave):
    """Valor medido o None (fila ausente o n/d)."""
    if f is None or f.get("estado") == ND or f.get(clave) is None:
        return None
    return float(f[clave])


def commit_de(doc):
    return doc["meta"]["commit"][:9] if doc else ND


def etiqueta_ds(kind, n, d):
    return f"{kind} {n // 10**3}k · d={d}"


def configs_de(doc):
    """Datasets (kind, n, d) en el orden en que se midieron (de la primera tabla)."""
    vistos = []
    for f in doc["resultados"]:
        d = f["dataset"]
        c = (d.get("kind"), d.get("n"), d.get("d"))
        if c not in vistos:
            vistos.append(c)
    return vistos


# ------------------------------------------------------------------------------ estilo
def estilo():
    plt.rcParams.update({
        "figure.facecolor": FONDO, "axes.facecolor": EJES, "axes.edgecolor": REJILLA,
        "axes.labelcolor": TEXTO, "text.color": TEXTO, "xtick.color": TEXTO2, "ytick.color": TEXTO2,
        "grid.color": REJILLA, "font.size": 10, "font.family": "DejaVu Sans",
        "axes.titleweight": "bold", "axes.titlesize": 11, "legend.frameon": False,
        "savefig.facecolor": FONDO, "axes.grid": True, "grid.linewidth": 0.6,
        "axes.axisbelow": True,
    })


def guardar(fig, nombre, titulo, fuentes, manifest):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")  # sin canal alfa
    ruta = os.path.join(DOCS, nombre + ".png")
    img.save(ruta, format="PNG")
    manifest[nombre] = {
        "titulo": titulo,
        "fuentes": [{"ruta": os.path.relpath(p, ROOT), "sha256": sha256(p)} for p in fuentes],
        "sha256_png": sha256(ruta),
        "pixeles": list(img.size),
    }
    print(f"{nombre}.png  {img.size[0]}x{img.size[1]} px  fuentes={len(fuentes)}")


def barras_agrupadas(ax, grupos, series, valor, titulo, ylabel, fmt="{:,.0f}", nota=None,
                     escala_nota=0.03):
    """Barras lineales desde 0: por cada grupo una barra por serie; ``valor(g, s)`` -> número o
    None (n/d, rotulado como tal). Devuelve el máximo graficado."""
    ancho = 0.8 / len(series)
    top = 0.0
    for gi, g in enumerate(grupos):
        for si, s in enumerate(series):
            v = valor(g, s)
            x = gi + (si - (len(series) - 1) / 2) * ancho
            if v is None:
                ax.text(x, 0, ND, ha="center", va="bottom", fontsize=7, color=TEXTO2, rotation=90)
                continue
            ax.bar(x, v, ancho * 0.92, color=s["color"], zorder=3)
            extra = ("\n" + nota(g, s)) if nota else ""
            ax.text(x, v, fmt.format(v) + extra, ha="center", va="bottom", fontsize=7, color=TEXTO)
            top = max(top, v)
    ax.set_xticks(range(len(grupos)))
    ax.set_xticklabels([g["etiqueta"] for g in grupos])
    ax.set_ylim(0, top * (1 + escala_nota) if top > 0 else 1)
    ax.set_title(titulo)
    ax.set_ylabel(ylabel)
    ax.grid(axis="x", visible=False)
    return top


def leyenda(ax, series, **kw):
    ax.legend(handles=[Patch(color=s["color"], label=s["etiqueta"]) for s in series], **kw)


# ------------------------------------------------------------------------------ figuras
def hero(bus, bus_antes, manifest):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.3, 1]})
    ca, cb = commit_de(bus), commit_de(bus_antes)
    proto = bus["protocolo"]
    k, nq = proto["k"], proto["n_queries"]
    fig.suptitle(f"Search throughput and recall — measured (after {ca}, before {cb})", y=1.02,
                 fontsize=13, fontweight="bold")

    # --- panel izquierdo: dot-plot QPS x recall con TODOS los datasets ------------------
    metodos = [
        {"etiqueta": "linear single (X @ q)", "param": ("linear_single",), "color": LINEAL},
        {"etiqueta": "linear batched B=200", "param": ("linear_batched B=200",), "color": "#b45309"},
        {"etiqueta": "search(q, k) single", "param": ("search(q,k) legacy",), "color": DESPUES},
        {"etiqueta": "search_batch B=200", "param": ("search_batch B=200",), "color": OTRA},
    ]
    configs = configs_de(bus)
    formas = "os^Dv*p"
    for i, (kind, n, d) in enumerate(configs):
        for m in metodos:
            f = fila(bus, m["param"], kind=kind, n=n, d=d)
            qps, rec = num(f, "qps"), num(f, "recall@10")
            if qps and rec is not None:
                ax1.scatter(qps, rec, marker=formas[i % len(formas)], color=m["color"], s=55,
                            edgecolors=FONDO, linewidths=0.8, zorder=3)
    ax1.set_xscale("log")
    ax1.set_ylim(-0.03, 1.05)
    ax1.set_xlabel("queries / second (log scale)")
    ax1.set_ylabel("recall@10 vs exact float64")
    ax1.set_title(f"QPS x recall@10, all datasets (k={k}, {nq} queries, mediana de repeticiones)")
    h1 = [Line2D([], [], marker="o", linestyle="", color=m["color"], markersize=7, label=m["etiqueta"])
          for m in metodos]
    h2 = [Line2D([], [], marker=formas[i % len(formas)], linestyle="", color=TEXTO2, markersize=7,
                 label=etiqueta_ds(*c)) for i, c in enumerate(configs)]
    ax1.legend(handles=h1 + h2, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    ax1.grid(axis="x", visible=False)

    # --- panel derecho: before vs after, barras lineales desde 0 --------------------------
    d_ref = max(c[2] for c in configs)
    grupos = [{"etiqueta": f"{etiqueta_ds(*c)}\n{m['etiqueta']}", "cfg": c, "m": m}
              for c in configs if c[0] == "clustered" and c[2] == d_ref for m in metodos[2:]]
    series = [{"etiqueta": f"before {cb}", "doc": bus_antes, "color": ANTES_C},
              {"etiqueta": f"after {ca}", "doc": bus, "color": DESPUES}]

    def valor(g, s):
        kind, n, d = g["cfg"]
        return num(fila(s["doc"], g["m"]["param"], kind=kind, n=n, d=d), "qps")

    def nota(g, s):
        b = valor(g, series[0])
        a = valor(g, series[1])
        return "" if s is not series[1] or not a or not b else f"{a / b:.2f}x"

    top = barras_agrupadas(ax2, grupos, series, valor,
                           f"Before ({cb}) vs after ({ca}) — clustered, d={d_ref}",
                           "queries / second", nota=nota, escala_nota=0.22)
    for gi in range(1, len(grupos)):
        ax2.axvline(gi - 0.5, color=REJILLA, linewidth=0.8)
    leyenda(ax2, series, loc="upper right")
    fig.tight_layout()
    guardar(fig, "bench-hero", "Search throughput and recall (dot-plot + before/after)",
            [os.path.join(RESULTADOS, "bench_search.json"), os.path.join(ANTES, "bench_search.json")],
            manifest)


def qps_fig(bus, manifest):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.6, 1]})
    ca = commit_de(bus)
    k = bus["protocolo"]["k"]
    fig.suptitle(f"Batch throughput and index build time — measured ({ca})", y=1.02, fontsize=13,
                 fontweight="bold")
    configs = configs_de(bus)
    grupos = [{"etiqueta": etiqueta_ds(*c), "cfg": c} for c in configs]
    bs = sorted({int(f["parametro"].split("B=")[1].split()[0]) for f in bus["resultados"]
                 if f["parametro"].startswith("search_batch B=")})
    series = [{"etiqueta": "linear single", "param": ("linear_single",), "color": LINEAL},
              {"etiqueta": "search(q, k) single", "param": ("search(q,k) legacy",), "color": DESPUES}]
    series += [{"etiqueta": f"search_batch B={B}", "param": (f"search_batch B={B}",),
                "color": AZULES[i % len(AZULES)]} for i, B in enumerate(bs)]

    def valor(g, s):
        return num(fila(bus, s["param"], kind=g["cfg"][0], n=g["cfg"][1], d=g["cfg"][2]), "qps")

    barras_agrupadas(ax1, grupos, series, valor,
                     f"Queries / second by batch size (k={k}, mediana de repeticiones)",
                     "queries / second", escala_nota=0.35)
    leyenda(ax1, series, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=8)

    datasets = bus.get("datasets", [])
    if datasets:
        g2 = [{"etiqueta": etiqueta_ds(x["kind"], x["n"], x["d"]), "x": x} for x in datasets]
        s2 = [{"etiqueta": "add(n) build", "color": DESPUES}]
        barras_agrupadas(ax2, g2, s2, lambda g, s: g["x"].get("build_s"),
                         "Index build time: one add() of n docs", "seconds", fmt="{:.2f} s",
                         nota=lambda g, s: f"{g['x']['n'] / g['x']['build_s']:,.0f} docs/s", escala_nota=0.3)
    else:
        ax2.set_title("Index build time: n/d (sin datasets en el JSON)")
    fig.tight_layout()
    guardar(fig, "bench-qps", "Batch throughput by batch size and index build time",
            [os.path.join(RESULTADOS, "bench_search.json")], manifest)


def filtered_fig(fil, fil_antes, manifest):
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ca, cb = commit_de(fil), commit_de(fil_antes)
    d0 = fil["resultados"][0]["dataset"]
    fig.suptitle(f"Filtered search recall@10 — {d0['kind']} {d0['n'] // 10**3}k · d={d0['d']} "
                 f"(after {ca}, before {cb})", y=1.02, fontsize=13, fontweight="bold")
    nombres = []
    for f in fil["resultados"]:
        if f.get("filtro_nombre") and f["filtro_nombre"] not in nombres:
            nombres.append(f["filtro_nombre"])
    sel = {}
    for f in fil["resultados"]:
        if f.get("filtro_nombre") and f["filtro_nombre"] not in sel:
            sel[f["filtro_nombre"]] = f.get("selectividad")
    grupos = [{"etiqueta": f"{n}\n(sel. {10 ** 2 * sel[n]:.0f}%)", "n": n} for n in nombres]
    series = [
        {"etiqueta": f"search(filter) single — before {cb}", "doc": fil_antes, "param": ("repetida",), "color": ANTES_C},
        {"etiqueta": f"search(filter) single — after {ca}", "doc": fil, "param": ("repetida",), "color": DESPUES},
        {"etiqueta": f"search_batch B=200 + post-filter — before {cb}", "doc": fil_antes, "param": ("post-filtro",), "color": "#b45309"},
        {"etiqueta": f"search_batch_filtered B=200 filter=dict — after {ca}", "doc": fil, "param": ("search_batch_filtered", "filter=dict"), "color": OTRA},
        {"etiqueta": f"search_batch B=200 + post-filter — after {ca}", "doc": fil, "param": ("post-filtro",), "color": LINEAL},
    ]

    def fila_de(s, nombre):
        if not s["doc"]:
            return None
        for f in s["doc"]["resultados"]:
            if f.get("filtro_nombre") == nombre and all(_match(f["parametro"], p) for p in s["param"]):
                return f
        return None

    def valor(g, s):
        return num(fila_de(s, g["n"]), "recall@10")

    def nota(g, s):
        q = num(fila_de(s, g["n"]), "qps")
        return "" if q is None else f"{q:,.0f} QPS"

    barras_agrupadas(ax, grupos, series, valor,
                     "Recall@10 vs exact top-10 among docs passing the filter (GT float64)",
                     "recall@10", fmt="{:.3f}", nota=nota, escala_nota=0.02)
    ax.set_ylim(0, 1.3)
    leyenda(ax, series, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=8)
    fig.tight_layout()
    guardar(fig, "bench-filtered", "Filtered search recall by filter and path",
            [p for p in (os.path.join(RESULTADOS, "bench_filtered.json"),
                         os.path.join(ANTES, "bench_filtered.json")) if os.path.exists(p)], manifest)


def storage_fig(sto, sto_antes, manifest):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    ca = commit_de(sto)
    fig.suptitle(f"Storage: splat arrays vs RSS, snapshot on disk, reopen time — measured ({ca})",
                 y=1.02, fontsize=13, fontweight="bold")
    builds = [f for f in sto["resultados"] if f.get("camino") == "build" and f.get("estado") != ND]
    grupos = [{"etiqueta": f"{f['dataset']['n'] // 10**3}k docs", "f": f} for f in builds]
    mb = 2 ** 20

    def arr(g, clave):
        return g["f"].get(clave)

    s1 = [{"etiqueta": "arrays μ,α,κ float32 (analytic n·(d+2)·4)", "clave": "arrays_analitico_mb", "color": DESPUES},
          {"etiqueta": "splat arrays reserved by the store", "clave": "arrays_reservados_mb", "color": OTRA},
          {"etiqueta": "RSS delta after build (process)", "clave": "rss_delta_mb", "color": LINEAL},
          {"etiqueta": "RSS peak during build (process)", "clave": "rss_pico_mb", "color": "#b45309"}]
    barras_agrupadas(axes[0], grupos, s1, lambda g, s: g["f"].get(s["clave"]),
                     "Arrays vs measured RSS (build)", "MB", fmt="{:,.0f}", escala_nota=0.2)
    leyenda(axes[0], s1, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=7)

    snaps = [f for f in sto["resultados"] if f.get("camino") == "snapshot" and f.get("estado") != ND]
    if snaps:
        g2 = [{"etiqueta": f"{f['dataset']['n'] // 10**3}k docs", "f": f} for f in snaps]
        s2 = [{"etiqueta": "snapshot dir", "color": DESPUES}]
        barras_agrupadas(axes[1], g2, s2, lambda g, s: g["f"].get("disco_mb"),
                         "Snapshot on disk: save(path)", "MB", fmt="{:,.0f}",
                         nota=lambda g, s: f"{g['f'].get('bytes_por_doc', 0):,.0f} B/doc", escala_nota=0.25)
    else:
        axes[1].set_title("Snapshot on disk: n/d")

    reopens = [{"etiqueta": f"after {ca}", "doc": sto, "color": DESPUES},
               {"etiqueta": f"before {commit_de(sto_antes)}", "doc": sto_antes, "color": ANTES_C}]

    def rr(g, s):
        if not s["doc"]:
            return None
        for f in s["doc"]["resultados"]:
            if f.get("camino") == "snapshot" and f.get("estado") != ND and \
                    f["dataset"]["n"] == g["f"]["dataset"]["n"]:
                return f["mediana"] / 10 ** 3
        return None

    if snaps:
        barras_agrupadas(axes[2], grupos[:len(snaps)], reopens, rr,
                         "Reopen: SimpleVectorDB(storage_path=path)", "seconds", fmt="{:.2f} s",
                         escala_nota=0.25)
        leyenda(axes[2], reopens, loc="upper left", fontsize=8)
    else:
        axes[2].set_title("Reopen time: n/d")
    esc = [f for f in sto["resultados"] if f.get("camino") == "escritura" and f.get("estado") != ND]
    if esc:
        partes = [f"{f['parametro'].replace('add(', '').replace(' docs)', '').replace(' ', '')}: "
                  f"{f['qps']:,.0f} docs/s" for f in esc]
        fig.text(0.99, 0.01, "writes (1 txn per add(), no save()): " + "  |  ".join(partes),
                 ha="right", fontsize=7, color=TEXTO2)
    fig.tight_layout()
    guardar(fig, "bench-storage", "Storage: arrays vs RSS, snapshot size, reopen time",
            [p for p in (os.path.join(RESULTADOS, "bench_storage.json"),
                         os.path.join(ANTES, "bench_storage.json")) if os.path.exists(p)], manifest)


def main():
    if "--check" in sys.argv[1:]:
        sin_cifras_hardcodeadas()
        print("make_readme_figures: sin literales de cifras")
        return 0
    sin_cifras_hardcodeadas()
    estilo()
    bus, p1 = cargar("bench_search")
    if not bus:
        raise SystemExit(f"falta {p1}: corré bench/bench_search.py (y la columna before)")
    manifest = {}
    hero(bus, cargar("bench_search", antes=True)[0], manifest)
    qps_fig(bus, manifest)
    fil, p3 = cargar("bench_filtered")
    if fil:
        filtered_fig(fil, cargar("bench_filtered", antes=True)[0], manifest)
    sto, p4 = cargar("bench_storage")
    if sto:
        storage_fig(sto, cargar("bench_storage", antes=True)[0], manifest)
    doc = {
        "generador": "scripts/make_readme_figures.py",
        "regenerar": "python scripts/make_readme_figures.py",
        "protocolo": "bench/_protocol.py",
        "matplotlib": matplotlib.__version__,
        "commit_after": commit_de(bus),
        "commit_before": commit_de(cargar("bench_search", antes=True)[0]),
        "figuras": {
            f"docs/{k}.png": v for k, v in sorted(manifest.items())
        },
    }
    with open(os.path.join(DOCS, "bench-manifest.json"), "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1, sort_keys=True, ensure_ascii=False)
        f.write("\n")
    print("-> docs/bench-manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
