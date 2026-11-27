#!/usr/bin/env python
"""Regenerate the benchmark figures, ``docs/bench-manifest.json`` and ``docs/BENCHMARKS.md``
reading ONLY the measured JSON files (``benchmarks/results/*.json`` = after,
``benchmarks/results/before_a74d9e4/*.json`` = before commit).

No figure lives in this file: QPS, recall, timings, sizes, commits, ratios, the batch
sizes and even the protocol line under each figure come from the JSON.  The script
refuses to produce output when any of these fails:

* a numeric literal that could be a hand-typed figure is found in this source
  (``--check`` runs only this test);
* two texts, or a text and a legend, overlap after the figure is drawn;
* a text falls outside the canvas;
* a value label sits on top of a bar;
* two consecutive runs do not produce byte-identical PNGs.

Design rules: English text, comma thousands / dot decimals, horizontal bars with the
label on the left and the value at the end of the bar, ``layout="constrained"``, one
palette shared by all figures (amber = NumPy linear scan, gray = before commit, blue =
single-query path after, greens = batch paths, violet/pink/cyan = datasets).

Usage:  python scripts/make_readme_figures.py [--check] [--solo bench-hero,bench-qps]
"""
import ast
import hashlib
import io
import json
import math
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from matplotlib.text import Text  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullFormatter, NullLocator  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402
import PIL  # noqa: E402
from PIL import Image  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTADOS = os.path.join(ROOT, "benchmarks", "results")
ANTES = os.path.join(RESULTADOS, "before_a74d9e4")
DOCS = os.path.join(ROOT, "docs")
DPI = 150
ANCHO = 13.0
PCT = 10 ** 2
MIL = 10 ** 3
ND = "n/d"
TOL = 0.5  # px: un solapamiento menor que esto es redondeo, no colision
MIN_MARGEN_PX = 10  # px minimos entre el rotulo mas largo y el borde derecho del axes
UMBRAL_RECALL = 1.0  # el recall se anota solo donde es menor que esto
T_VALOR = 8.5

# Texto mandado por el dueno del pass que SI lleva cifras (no salen de los JSON).
NOTA_RUIDO = ("Noise, stated honestly: between two runs of the same code most rows vary by about "
              "±10%, but the large batches (B=1000, and B=200 on the 10k clustered dataset) varied "
              "by up to 40%. Treat differences inside that band as noise.")
LITERALES_DE_LAYOUT = frozenset({DPI})  # unicos enteros >= 100 permitidos en este archivo
TEXTO_CON_CIFRAS_PERMITIDAS = {"NOTA_RUIDO", "_RE_SHA"}
_RE_COLOR = re.compile(r"^#[0-9a-fA-F]{6}$")
_RE_CIFRAS = re.compile(r"\d{3,}")
_RE_SHA = re.compile(r"(?i)sha-?256")  # nombre de algoritmo, no una cifra

FONDO, EJES, REJILLA, TEXTO, TEXTO2 = "#0f1117", "#161922", "#242938", "#e6e6e6", "#9aa3b2"
LINEAL = "#f59e0b"        # escaneo lineal NumPy
ANTES_C = "#8b8f98"       # commit before
SINGLE = "#4f8cff"        # camino de consulta individual, after
SINGLE_VAR = "#93c5fd"    # variante del camino individual (metadata / 1ra evaluacion en frio)
VERDES = ["#86efac", "#4ade80", "#22c55e", "#16a34a", "#15803d", "#166534"]  # lotes, por B
FN_C = "#2dd4bf"          # search_batch_filtered(filter_fn=)
POSTFILTRO_C = "#f87171"  # post-filtro legado
DS_PALETTE = ["#a78bfa", "#f472b6", "#22d3ee", "#fafafa", "#facc15"]  # por (n, d)
DS_NOMBRES = ["violet", "pink", "cyan", "white", "yellow"]
# nombres de modo del JSON (en espanol) -> etiqueta en ingles para la figura
MODO_EN = {"completo": "full vectors", "espectral_auto": "spectral (auto)"}
HNSW_C, LSH_C = "#e879f9", "#fb923c"
MEMORIA_C, PERSISTENTE_C = SINGLE, "#22c55e"


# --------------------------------------------------------------------------- sanity
def sin_cifras_hardcodeadas():
    """Este archivo no puede contener cifras: enteros >= 100 (salvo el DPI), floats con 3+
    decimales, ni rachas de 3+ digitos dentro de strings (salvo colores hex y el texto
    mandado con cifras). Toda cifra dibujada tiene que salir de benchmarks/results/*.json."""
    with open(os.path.abspath(__file__), encoding="utf-8") as f:
        arbol = ast.parse(f.read())
    docstrings, permitidos = set(), set()
    for nodo in ast.walk(arbol):
        if isinstance(nodo, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            primero = nodo.body[0] if nodo.body else None
            if isinstance(primero, ast.Expr) and isinstance(primero.value, ast.Constant) \
                    and isinstance(primero.value.value, str):
                docstrings.add(id(primero.value))
        if isinstance(nodo, ast.Assign):
            nombres = [t.id for t in nodo.targets if isinstance(t, ast.Name)]
            if any(n in TEXTO_CON_CIFRAS_PERMITIDAS for n in nombres):
                permitidos.update(id(x) for x in ast.walk(nodo.value))
    sospechosos = []
    for nodo in ast.walk(arbol):
        if not isinstance(nodo, ast.Constant) or id(nodo) in docstrings or id(nodo) in permitidos:
            continue
        v = nodo.value
        if isinstance(v, bool):
            continue
        if isinstance(v, int) and abs(v) >= PCT and v not in LITERALES_DE_LAYOUT:
            sospechosos.append((nodo.lineno, v))
        if isinstance(v, float) and len(repr(v).split(".")[-1]) >= 3:
            sospechosos.append((nodo.lineno, v))
        if isinstance(v, str) and not _RE_COLOR.match(v) and _RE_CIFRAS.search(_RE_SHA.sub("", v)):
            sospechosos.append((nodo.lineno, v[:60]))
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


def cargar_todo():
    D = {}
    for clave, nombre, antes in (("bus", "bench_search", False), ("bus_antes", "bench_search", True),
                                 ("fil", "bench_filtered", False), ("idx", "bench_index_types", False),
                                 ("sto", "bench_storage", False)):
        doc, ruta = cargar(nombre, antes)
        if doc is None and not antes:
            raise SystemExit(f"falta {ruta}")
        D[clave] = (doc, ruta)
    return D


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
    """Valor medido o None (fila ausente, n/d o null)."""
    if f is None or f.get("estado") == ND or f.get(clave) is None:
        return None
    return float(f[clave])


def commit_de(doc, n=7):
    return doc["meta"]["commit"][:n] if doc else ND


def commit_corto_de(doc):
    if not doc:
        return ND
    return doc["meta"].get("commit_corto") or commit_de(doc, 9)


def etiqueta_ds(kind, n, d):
    return f"{kind} {n // MIL}k · d={d}"


def configs_de(doc):
    """Datasets (kind, n, d) en el orden en que se midieron."""
    vistos = []
    for f in doc["resultados"]:
        d = f["dataset"]
        c = (d.get("kind"), d.get("n"), d.get("d"))
        if c not in vistos:
            vistos.append(c)
    return vistos


def colores_de(configs):
    """Mismo (n, d) => mismo color en todas las figuras."""
    vistos = []
    for (_, n, d) in configs:
        if (n, d) not in vistos:
            vistos.append((n, d))
    return {nd: DS_PALETTE[i % len(DS_PALETTE)] for i, nd in enumerate(vistos)}, vistos


def _parse_b(p):
    m = re.search(r"B=(\d+)", str(p))
    return int(m.group(1)) if m else None


def lotes_de(doc):
    """Tamanos de lote B medidos por search_batch, en orden."""
    bs = []
    for f in doc["resultados"]:
        if str(f["parametro"]).startswith("search_batch B="):
            b = _parse_b(f["parametro"])
            if b is not None and b not in bs:
                bs.append(b)
    return sorted(bs)


def color_lote(bus, b):
    return VERDES[lotes_de(bus).index(b) % len(VERDES)]


def ref_lineal(bus, b, cfg):
    """Fila del escaneo lineal para el lote B (consulta individual si B es el minimo medido)."""
    f = fila(bus, (f"linear_batched B={b}",), kind=cfg[0], n=cfg[1], d=cfg[2])
    if f is None and b == min(lotes_de(bus)):
        f = fila(bus, ("linear_single",), kind=cfg[0], n=cfg[1], d=cfg[2])
    return f


def cpu_corto(cpu):
    return cpu.split(" with ")[0].removeprefix("AMD ")


def fmt_qps(v):
    s = f"{v:,.0f}" if v >= MIL else f"{v:,.1f}"
    return s[:-2] if s.endswith(".0") else s


def fmt_rec(v):
    return f"{v:.3f}"


def fmt_num(v):
    """Cifra exacta del JSON con coma de miles y punto decimal (sin redondear)."""
    if isinstance(v, int) and not isinstance(v, bool):
        return f"{v:,}"
    s = f"{v:,.6f}".rstrip("0").rstrip(".")
    return s if s else "0"


def rango(vals, fmt):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "n/a"
    lo, hi = min(vals), max(vals)
    return fmt(lo) if fmt(lo) == fmt(hi) else f"{fmt(lo)}–{fmt(hi)}"


# ------------------------------------------------------------------------------ estilo
def estilo():
    plt.rcParams.update({
        "figure.facecolor": FONDO, "axes.facecolor": EJES, "axes.edgecolor": REJILLA,
        "axes.labelcolor": TEXTO, "text.color": TEXTO, "xtick.color": TEXTO2, "ytick.color": TEXTO2,
        "grid.color": REJILLA, "font.size": 10, "font.family": "DejaVu Sans",
        "axes.titleweight": "bold", "axes.titlesize": 11.5, "axes.titlelocation": "left",
        "axes.titlepad": 10, "legend.frameon": False, "legend.fontsize": 9,
        "xtick.labelsize": 9.5, "ytick.labelsize": 10, "axes.labelsize": 10,
        "savefig.facecolor": FONDO, "axes.grid": True, "grid.linewidth": 0.6,
        "axes.axisbelow": True,
    })


def pie_de(*docs, antes=None):
    """Linea de protocolo + commits, toda armada desde los JSON."""
    d = next(x for x in docs if x)
    m = d["meta"]
    hilos = m["hilos"]
    partes = [cpu_corto(m["cpu"]),
              f"BLAS threads = {hilos.get('OPENBLAS_NUM_THREADS') or hilos.get('OMP_NUM_THREADS')}"]
    p = d.get("protocolo")
    if p:
        partes += [f"{p['n_queries']} queries", f"k = {p['k']}"]
    partes.append("median of repetitions")
    commits = "after = " + commit_de(d)
    if antes:
        commits = "before = " + commit_de(antes) + ", " + commits
    return " · ".join(partes + [commits])


def nueva_figura(alto, **gridspec):
    fig = plt.figure(figsize=(ANCHO, alto), dpi=DPI, layout="constrained")
    return fig, (fig.add_gridspec(**gridspec) if gridspec else None)


def cerrar_figura(fig, titulo, pie):
    fig.suptitle(titulo, fontsize=13.5, fontweight="bold", color=TEXTO)
    # pie de protocolo como supxlabel: constrained layout le reserva lugar (fig.text no)
    fig.supxlabel(pie, fontsize=9.5, color="#b4bcc8")


def color_de(s, g):
    return s["color_de"](g) if "color_de" in s else s["color"]


def ajustar_xlim(ax, textos, margen=0.04):
    """Ensancha el eje x (lineal desde 0) hasta que el rotulo mas largo entre con holgura."""
    for _ in range(4):
        ax.figure.canvas.draw()
        renderer = ax.figure.canvas.get_renderer()
        x1 = max((t.get_window_extent(renderer).x1 for t in textos), default=None)
        if x1 is None:
            return
        x_izq = ax.transAxes.transform((0, 0))[0]
        ancho = ax.transAxes.transform((1, 0))[0] - x_izq
        if x1 <= x_izq + ancho - max(ancho * margen, MIN_MARGEN_PX):
            return
        xd = ax.transData.inverted().transform((x1, 0))[0]
        x0, _ = ax.get_xlim()
        ax.set_xlim(x0, x0 + (xd - x0) * (1 + 2 * margen))
    ax.figure.canvas.draw()


def ajustar_todos(fig):
    """Pase final: el ancho definitivo de cada axes se conoce recien cuando existen todas
    las etiquetas, titulos y leyendas de la figura; ahi se reajusta cada eje x lineal."""
    for ax in fig.axes:
        if ax.get_xscale() != "linear":
            continue
        textos = [t for t in ax.texts if t.get_text().strip() and t.get_ha() != "right"]
        if textos:
            ajustar_xlim(ax, textos)


def barras_h(ax, grupos, series, valor, fmt, sufijo=None, nombres="nunca", rotulo=T_VALOR):
    """Barras horizontales agrupadas: un grupo por etiqueta de y, una barra por serie.
    ``valor(gi, si)`` -> numero o None (n/a). El nombre de la serie va en la leyenda de la
    fila dedicada (leyenda_de_barras) o, con nombres="siempre", delante de cada valor."""
    n_s = len(series)
    paso = 0.8 / n_s
    textos, top = [], 0.0
    for gi, g in enumerate(grupos):
        for si, s in enumerate(series):
            v = valor(gi, si)
            y = gi + (si - (n_s - 1) / 2) * paso
            if v is None:
                textos.append(ax.annotate("n/a", (0, y), xytext=(4, 0), textcoords="offset points",
                                          ha="left", va="center", fontsize=rotulo, color=TEXTO2))
                continue
            ax.barh(y, v, paso * 0.86, color=color_de(s, g), zorder=3)
            nombre = s["etiqueta"] + ": " if nombres == "siempre" else ""
            extra = sufijo(gi, si, v) if sufijo else ""
            extra = ("  (" + extra + ")") if extra else ""
            textos.append(ax.annotate(nombre + fmt(v) + extra, (v, y), xytext=(4, 0),
                                      textcoords="offset points", ha="left", va="center",
                                      fontsize=rotulo, color=TEXTO))
            top = max(top, v)
    ax.set_yticks(range(len(grupos)))
    ax.set_yticklabels([g["etiqueta"] for g in grupos])
    for t, g in zip(ax.get_yticklabels(), grupos):
        if g.get("color"):
            t.set_color(g["color"])
    ax.set_ylim(len(grupos) - 0.5 + 0.12, -0.5 - 0.12)
    ax.set_xlim(0, top * 1.08 if top > 0 else 1)
    ticks_amables(ax)
    ax.grid(axis="y", visible=False)
    return textos, top


def leyenda_de_barras(fig, celda, series, ncol=None):
    """Fila dedicada bajo un panel con la leyenda de sus series: los handles son parches
    del mismo color que las barras efectivamente dibujadas."""
    ax_leg = fig.add_subplot(celda)
    ax_leg.axis("off")
    ax_leg.legend(handles=[Patch(facecolor=s["color"], edgecolor="none", label=s["etiqueta"]) for s in series],
                  loc="center", ncol=ncol or len(series), handlelength=1.4, columnspacing=1.8,
                  borderaxespad=0)
    return ax_leg


def ticks_amables(ax):
    """Maximo 4 divisiones, sin tick pegado al borde derecho (su etiqueta se saldria del
    lienzo) y con coma de miles / punto decimal en las etiquetas."""
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, prune="upper"))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.10g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())


def separadores(ax, grupos):
    for gi in range(1, len(grupos)):
        ax.axhline(gi - 0.5, color=REJILLA, linewidth=0.9, zorder=2)


def decadas_en_vista(ax, eje="x"):
    """Ticks solo en las decadas DENTRO de la vista: los ticks fuera de vista dejan
    etiquetas 'visibles' fuera del lienzo (visto con LogLocator en matplotlib 3.11)."""
    axis = {"x": ax.xaxis, "y": ax.yaxis}[eje]
    lo, hi = sorted(axis.get_view_interval())
    ticks = [10 ** e for e in range(math.floor(math.log10(lo)), math.ceil(math.log10(hi)) + 1)
             if lo <= 10 ** e <= hi]
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_minor_locator(NullLocator())
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    axis.set_minor_formatter(NullFormatter())


def eje_log_comas(ax):
    ax.set_xscale("log")
    decadas_en_vista(ax)


# ------------------------------------------------------------------------------ checker
def _nombre_texto(t):
    s = " ".join(t.get_text().split())
    return (s[:48] + "…") if len(s) > 48 else s


def verificar_layout(fig, nombre):
    """(a) cero colisiones entre textos y leyendas, (b) cero textos fuera del lienzo y,
    de yapa, ningun rotulo ni leyenda encima de una barra. Falla (SystemExit) si no se cumple."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ancho, alto = fig.bbox.width, fig.bbox.height
    cajas = []
    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        if t.axes is not None and not t.axes.get_visible():
            continue
        bb = t.get_window_extent(renderer)
        if bb.width < TOL or bb.height < TOL:
            continue
        cajas.append([_nombre_texto(t), bb, None, t])
    leyendas = list(fig.legends) + [lg for ax in fig.axes for lg in (ax.get_legend(),) if lg is not None]
    for lg in leyendas:
        if not lg.get_visible():
            continue
        propios = {id(x) for x in lg.findobj(Text)}
        for c in cajas:
            if c[3] is not None and id(c[3]) in propios:
                c[2] = id(lg)
        cajas.append(["legend", lg.get_window_extent(renderer), id(lg), None])

    def choca(a, b):
        inter = Bbox.intersection(a, b)
        return inter is not None and inter.width > TOL and inter.height > TOL

    problemas = []
    for i, a in enumerate(cajas):
        if a[1].x0 < -TOL or a[1].y0 < -TOL or a[1].x1 > ancho + TOL or a[1].y1 > alto + TOL:
            problemas.append(f"{nombre}: texto fuera del lienzo: '{a[0]}'")
        for b in cajas[i + 1:]:
            if a[2] is not None and a[2] == b[2]:
                continue
            if choca(a[1], b[1]):
                problemas.append(f"{nombre}: colision '{a[0]}' x '{b[0]}'")
    for ax in fig.axes:
        barras = [p.get_window_extent(renderer) for p in ax.patches if isinstance(p, Rectangle)
                  and p.get_visible() and p.get_width() > 0 and p.get_height() > 0]
        if not barras:
            continue
        for c in cajas:
            if c[3] is None or c[3].axes is not ax or c[2] is not None:
                continue
            if any(choca(c[1], pb) for pb in barras):
                problemas.append(f"{nombre}: rotulo encima de una barra: '{c[0]}'")
        for lg in leyendas:
            if lg.axes is ax and any(choca(lg.get_window_extent(renderer), pb) for pb in barras):
                problemas.append(f"{nombre}: leyenda encima de una barra")
    if problemas:
        raise SystemExit("make_readme_figures: verificacion de layout FALLA\n  " + "\n  ".join(problemas))
    return {"textos": len(cajas), "solapamientos": 0, "fuera_de_lienzo": 0, "rotulo_sobre_barra": 0}


def render(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, facecolor=FONDO)
    pulgadas = [round(fig.get_figwidth(), 1), round(fig.get_figheight(), 1)]
    plt.close(fig)
    buf.seek(0)
    with Image.open(buf) as img:
        pixeles = list(img.size)
        rgb = img.convert("RGB")  # sin canal alfa
        out = io.BytesIO()
        rgb.save(out, format="PNG")
    return out.getvalue(), pixeles, pulgadas


# ------------------------------------------------------------------------------ figuras
def fig_hero(D):
    bus, bus_antes = D["bus"][0], D["bus_antes"][0]
    configs = configs_de(bus)
    colores, _ = colores_de(configs)
    cl = [c for c in configs if c[0] == "clustered"]
    ca, cb = commit_de(bus), commit_de(bus_antes)
    legacy, con_meta, lin1 = ("search(q,k) legacy",), ("search(q,k) include_metadata=True",), ("linear_single",)

    def qps(doc, param, cfg):
        return num(fila(doc, param, kind=cfg[0], n=cfg[1], d=cfg[2]), "qps")

    series_a = [
        {"etiqueta": "Linear scan (NumPy)", "doc": bus, "param": lin1, "color": LINEAL},
        {"etiqueta": f"before {cb}, tuple", "doc": bus_antes, "param": legacy, "color": ANTES_C},
        {"etiqueta": f"after {ca}, tuple", "doc": bus, "param": legacy, "color": SINGLE},
        {"etiqueta": f"after {ca}, default (+metadata)", "doc": bus, "param": con_meta, "color": SINGLE_VAR},
    ]
    grupos_a = [{"etiqueta": etiqueta_ds(*c), "cfg": c, "color": colores[(c[1], c[2])]} for c in cl]

    def valor_a(gi, si):
        return qps(series_a[si]["doc"], series_a[si]["param"], grupos_a[gi]["cfg"])

    def sufijo_a(gi, si, v):
        if si < 2:
            return ""
        lin = valor_a(gi, 0)
        return f"{v / lin:.2f}× linear" if lin else ""

    bs = lotes_de(bus)
    series_b = [{"etiqueta": "search(q, k), single", "param": legacy, "color": SINGLE}]
    series_b += [{"etiqueta": f"search_batch B={b:,}", "param": (f"search_batch B={b}",),
                  "color": color_lote(bus, b)} for b in bs]
    grupos_b = [{"etiqueta": etiqueta_ds(*c), "cfg": c, "color": colores[(c[1], c[2])]} for c in configs]

    def valor_b(gi, si):
        cfg = grupos_b[gi]["cfg"]
        return num(fila(bus, series_b[si]["param"], kind=cfg[0], n=cfg[1], d=cfg[2]), "recall@10")

    fig, gs = nueva_figura(8.4, nrows=2, ncols=2, width_ratios=[1.35, 1], height_ratios=[1, 0.14])
    ax_a, ax_b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    textos, _ = barras_h(ax_a, grupos_a, series_a, valor_a, fmt_qps, sufijo_a)
    separadores(ax_a, grupos_a)
    ajustar_xlim(ax_a, textos)
    ax_a.set_xlabel("queries per second, one query per call (linear scale from 0)")
    ax_a.set_title("A · Single-query throughput on clustered data")
    leyenda_de_barras(fig, gs[1, 0], series_a, ncol=2)
    textos, _ = barras_h(ax_b, grupos_b, series_b, valor_b, fmt_rec)
    separadores(ax_b, grupos_b)
    ajustar_xlim(ax_b, textos, margen=0.03)
    ax_b.set_xlabel("recall@10 vs exact float64 top-10 (1.0 = exact)")
    ax_b.set_title("B · Recall@10 by dataset")
    leyenda_de_barras(fig, gs[1, 1], series_b, ncol=2)
    cerrar_figura(fig, "Search: single-query throughput before vs after, and recall@10 by dataset",
                  pie_de(bus, antes=bus_antes))
    return fig, ("Single-query throughput (linear scan vs before vs after) and recall@10 by dataset",
                 [D["bus"][1], D["bus_antes"][1]])


def fig_qps(D):
    bus = D["bus"][0]
    configs = configs_de(bus)
    colores, _ = colores_de(configs)
    bs = lotes_de(bus)
    fig, gs = nueva_figura(7.2, nrows=2, ncols=2, height_ratios=[0.15, 1], width_ratios=[1.4, 1])
    ax_leg = fig.add_subplot(gs[0, :])
    ax_leg.axis("off")
    axes = [fig.add_subplot(gs[1, 0])]
    axes.append(fig.add_subplot(gs[1, 1], sharey=axes[0]))
    axes[1].tick_params(labelleft=False)
    handles, y_min, y_max = [], [], []
    for ax, kind, letra in ((axes[0], "clustered", "A"), (axes[1], "uniform", "B")):
        dels = [c for c in configs if c[0] == kind]
        for c in dels:
            color = colores[(c[1], c[2])]
            mk, cara = ("o", color) if kind == "clustered" else ("s", FONDO)
            pts = []
            for b in bs:
                f = fila(bus, (f"search_batch B={b}",), kind=c[0], n=c[1], d=c[2])
                q, r = num(f, "qps"), num(f, "recall@10")
                if q is not None:
                    pts.append((b, q, r))
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, marker=mk, markersize=7,
                    markerfacecolor=cara, markeredgecolor=color, linewidth=2, zorder=4)
            y_min += [p[1] for p in pts]
            y_max += [p[1] for p in pts]
            if kind == "uniform":
                for (b, q, r) in pts:
                    if r is None or r >= UMBRAL_RECALL:
                        continue
                    ref = num(ref_lineal(bus, b, c), "qps")
                    # debajo del punto, salvo que la referencia de escaneo lineal quede
                    # justo abajo (su marcador tacharia el texto): entonces a la derecha
                    if ref is not None and 1 <= q / ref < 1.6:
                        ax.annotate(f"recall {r:.2f}", (b, q), xytext=(11, 0), textcoords="offset points",
                                    ha="left", va="center", fontsize=8, color=color)
                    else:
                        ax.annotate(f"recall {r:.2f}", (b, q), xytext=(0, -9), textcoords="offset points",
                                    ha="center", va="top", fontsize=8, color=color)
            xl, yl = [], []
            for b in bs:
                f = ref_lineal(bus, b, c)
                if f is not None and num(f, "qps") is not None:
                    xl.append(b)
                    yl.append(num(f, "qps"))
            if xl:
                ax.plot(xl, yl, color=LINEAL, linestyle="--", marker="x", markersize=9,
                        markeredgewidth=2, zorder=3, linewidth=1.6)
                # nombre del camino pegado a su primer punto (abajo a la derecha: la linea sube)
                corto = ("uniform " if c[0] == "uniform" else "") + f"{c[1] // MIL}k · d={c[2]}"
                ax.annotate(f"linear scan, {corto}", (xl[0], yl[0]), xytext=(7, -6), textcoords="offset points",
                            ha="left", va="top", fontsize=8, color=LINEAL)
                y_min += yl
                y_max += yl
            handles.append(Line2D([], [], color=color, marker=mk, markersize=7, markerfacecolor=cara,
                                  markeredgecolor=color, linewidth=2, label=etiqueta_ds(*c)))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.xaxis.set_major_locator(FixedLocator(bs))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlim(bs[0] / 1.9, bs[-1] * 1.9)
        ax.set_title(f"{letra} · {'Clustered' if kind == 'clustered' else 'Uniform'} data — search_batch")
        ax.set_xlabel("batch size B (queries per call, log scale)")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("queries per second (log scale)")
    axes[0].set_ylim(min(y_min) / 1.6, max(y_max) * 1.7)
    for ax in axes:
        decadas_en_vista(ax, "y")
    lin_bs = sorted({b for c in configs for b in bs if ref_lineal(bus, b, c) is not None})
    faltan = [b for b in bs if b not in lin_bs]
    etiqueta_lin = "NumPy linear scan (batched): measured at B = " + ", ".join(f"{b:,}" for b in lin_bs)
    if faltan:
        etiqueta_lin += ", not at B = " + ", ".join(f"{b:,}" for b in faltan)
    handles.append(Line2D([], [], color=LINEAL, marker="x", markersize=9, linestyle="--", linewidth=1.6,
                          label=etiqueta_lin))
    cerrar_figura(fig, "Batch throughput by batch size: search_batch vs NumPy linear scan", pie_de(bus))
    ax_leg.legend(handles=handles, loc="center", ncol=3)
    return fig, ("search_batch throughput by batch size against the NumPy linear scan", [D["bus"][1]])


def fig_filtered(D):
    fil = D["fil"][0]
    bus = D["bus"][0]
    filas = fil["resultados"]
    nombres, selectividad = [], {}
    for f in filas:
        n = f.get("filtro_nombre")
        if n and n not in nombres:
            nombres.append(n)
            selectividad[n] = f.get("selectividad")
    grupos = [{"etiqueta": f"{n} — {(selectividad[n] or 0) * PCT:.0f}% of docs", "n": n} for n in nombres]

    def valor_de(clave, series):
        def v(gi, si):
            f = next((x for x in filas if x.get("camino") == series[si]["camino"]
                      and x.get("filtro_nombre") == grupos[gi]["n"]), None)
            return num(f, clave)
        return v

    b_batch = next((b for b in (_parse_b(f["parametro"]) for f in filas
                               if f.get("camino") == "batch_filtro_dict") if b), None)
    series_a = [
        {"etiqueta": "search(filter=dict)", "camino": "single_filtro", "color": SINGLE},
        {"etiqueta": "search_batch_filtered, filter=dict", "camino": "batch_filtro_dict",
         "color": color_lote(bus, b_batch) if b_batch else SINGLE},
        {"etiqueta": "search_batch_filtered, filter_fn=callable", "camino": "batch_filtro_fn", "color": FN_C},
        {"etiqueta": f"search_batch B={b_batch:,} + post-filter top-10 (legacy)", "camino": "batch_postfiltro",
         "color": POSTFILTRO_C},
    ]
    series_b = [
        {"etiqueta": "repeated query", "camino": "single_filtro", "color": SINGLE},
        {"etiqueta": "first evaluation after a write", "camino": "single_filtro_frio", "color": SINGLE_VAR},
    ]
    ds0 = filas[0]["dataset"]
    fig, gs = nueva_figura(10.8, nrows=4, ncols=2, height_ratios=[1.3, 0.1, 1, 0.1])
    ax_a, ax_b, ax_c = fig.add_subplot(gs[0, :]), fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])
    textos, _ = barras_h(ax_a, grupos, series_a, valor_de("recall@10", series_a), fmt_rec)
    separadores(ax_a, grupos)
    ajustar_xlim(ax_a, textos, margen=0.03)
    ax_a.set_xlabel("recall@10 vs exact filtered top-10 (1.0 = exact)")
    ax_a.set_title(f"A · Recall@10 by filter and path — {etiqueta_ds(ds0['kind'], ds0['n'], ds0['d'])}")
    leyenda_de_barras(fig, gs[1, :], series_a, ncol=2)
    textos, _ = barras_h(ax_b, grupos, series_b, valor_de("qps", series_b), fmt_qps)
    separadores(ax_b, grupos)
    ajustar_xlim(ax_b, textos)
    ax_b.set_xlabel("queries per second, one query per call")
    ax_b.set_title("B · search(filter=dict): single query")
    leyenda_de_barras(fig, gs[3, 0], series_b, ncol=2)
    textos, _ = barras_h(ax_c, grupos, series_a[1:], valor_de("qps", series_a[1:]), fmt_qps)
    separadores(ax_c, grupos)
    ajustar_xlim(ax_c, textos)
    ax_c.set_xlabel(f"queries per second, batch B={b_batch:,}")
    ax_c.set_title(f"C · Batch B={b_batch:,} with a filter")
    leyenda_de_barras(fig, gs[3, 1], series_a[1:], ncol=1)
    cerrar_figura(fig, "Filtered search: recall@10 by filter and path, and throughput", pie_de(fil))
    return fig, ("Filtered search: recall@10 by filter and path, and throughput of each path", [D["fil"][1]])


def fig_index(D):
    idx, bus = D["idx"][0], D["bus"][0]
    filas = idx["resultados"]
    colores, _ = colores_de(configs_de(bus))
    lin1 = ("linear_single",)

    def cfg_de(f):
        d = f["dataset"]
        return (d["kind"], d["n"], d["d"])

    hrm = [f for f in filas if f.get("camino") == "hrm2_lod" and num(f, "qps") is not None]
    ds_hrm = []
    for f in hrm:
        if cfg_de(f) not in ds_hrm:
            ds_hrm.append(cfg_de(f))
    lods = sorted({f["lod"] for f in hrm if f.get("lod") is not None})
    grupos_a = [{"etiqueta": etiqueta_ds(*c), "cfg": c, "color": colores[(c[1], c[2])]} for c in ds_hrm]
    series_a = [{"etiqueta": f"lod {lod}", "lod": lod, "color_de": lambda g: g["color"]} for lod in lods]
    series_a.append({"etiqueta": "Linear scan (NumPy)", "color": LINEAL})

    def valor_a(gi, si):
        s, cfg = series_a[si], grupos_a[gi]["cfg"]
        if "lod" in s:
            f = next((x for x in hrm if x["lod"] == s["lod"] and cfg_de(x) == cfg), None)
        else:
            f = fila(bus, lin1, kind=cfg[0], n=cfg[1], d=cfg[2])
        return num(f, "qps")

    def por_camino(camino):
        return next((f for f in filas if f.get("camino") == camino and num(f, "qps") is not None), None)

    ref, hnsw, lsh = por_camino("baseline"), por_camino("hnsw"), por_camino("lsh")
    items = []
    if ref:
        items.append((f"Linear scan (NumPy), batched B={_parse_b(ref['parametro']):,}", LINEAL, ref))
    if hnsw:
        items.append(("HNSW (pure Python) — experimental", HNSW_C, hnsw))
    if lsh:
        items.append(("LSH cross-polytope — experimental", LSH_C, lsh))
    sq8 = [f for f in filas if f.get("camino") == "sq8" and num(f, "qps") is not None]

    fig, gs = nueva_figura(10.6, nrows=2, ncols=2, width_ratios=[1.35, 1], height_ratios=[1, 1.25])
    ax_a, ax_b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1], sharey=ax_c)
    textos, _ = barras_h(ax_a, grupos_a, series_a, valor_a, fmt_qps, nombres="siempre")
    separadores(ax_a, grupos_a)
    ajustar_xlim(ax_a, textos)
    ax_a.set_xlabel("queries per second, one query per call")
    ax_a.set_title(f"A · HRM2 index by level of detail (lod {', '.join(str(x) for x in lods)})\nvs the linear scan")

    vals = [num(f, "qps") for (_, _, f) in items]
    x_izq = min(vals) / 2 if vals else 1
    ax_b.set_xscale("log")
    for i, (etq, color, f) in enumerate(items):
        v = num(f, "qps")
        ax_b.hlines(i, x_izq, v, color=color, linewidth=2, zorder=3)
        ax_b.plot([v], [i], marker="o", markersize=9, color=color, zorder=4)
        rec, build = num(f, "recall@10"), num(f, "build_s")
        txt = f"{fmt_qps(v)} QPS · recall {fmt_rec(rec)}" if rec is not None else f"{fmt_qps(v)} QPS"
        txt += f" · build {build:,.2f} s" if build is not None else " · no build step"
        if i == 0:
            ax_b.annotate(txt, (v, i), xytext=(-6, 7), textcoords="offset points", ha="right",
                          va="bottom", fontsize=8, color=TEXTO)
        else:
            ax_b.annotate(txt, (v, i), xytext=(8, 0), textcoords="offset points", ha="left",
                          va="center", fontsize=8, color=TEXTO)
    ax_b.set_xlim(x_izq, min(vals) * 10 ** 3.3 if vals else 1)
    ax_b.set_yticks(range(len(items)))
    ax_b.set_yticklabels([x[0] for x in items])
    for t, x in zip(ax_b.get_yticklabels(), items):
        t.set_color(x[1])
    ax_b.set_ylim(len(items) - 0.45, -0.95)
    ax_b.grid(axis="y", visible=False)
    eje_log_comas(ax_b)
    ax_b.set_xlabel("queries per second (log scale)")
    n_hnsw = idx.get("n_hnsw_lsh") or (hnsw["dataset"]["n"] if hnsw else None)
    ax_b.set_title(f"B · HNSW and LSH vs linear scan\nN = {n_hnsw:,} docs, log scale")

    etiquetas_c, colores_c = [], []
    for f in sq8:
        d = f["dataset"]
        m = re.search(r"oversample=([^)]+)\)", f["parametro"])
        etiquetas_c.append(f"{d['n'] // MIL}k · {MODO_EN.get(f.get('modo'), f.get('modo'))}"
                          f" · oversample {m.group(1) if m else '?'}")
        colores_c.append(colores[(d["n"], d["d"])])
    for i, f in enumerate(sq8):
        q, rec, disco = num(f, "qps"), num(f, "recall@10"), num(f, "disco_mb")
        ax_c.barh(i, q, 0.62, color=colores_c[i], zorder=3)
        textos.append(ax_c.annotate(f"{fmt_qps(q)} QPS · recall {fmt_rec(rec)}" if rec is not None else fmt_qps(q),
                                    (q, i), xytext=(4, 0), textcoords="offset points", ha="left",
                                    va="center", fontsize=T_VALOR, color=TEXTO))
        if disco is not None:
            ax_d.barh(i, disco, 0.62, color=colores_c[i], zorder=3)
            textos.append(ax_d.annotate(f"{disco:,.1f} MB", (disco, i), xytext=(4, 0),
                                        textcoords="offset points", ha="left", va="center",
                                        fontsize=T_VALOR, color=TEXTO))
    ax_c.set_yticks(range(len(sq8)))
    ax_c.set_yticklabels(etiquetas_c)
    for t, c in zip(ax_c.get_yticklabels(), colores_c):
        t.set_color(c)
    ax_d.tick_params(labelleft=False)
    ax_c.set_ylim(len(sq8) - 0.5 + 0.1, -0.5 - 0.1)
    ax_c.set_xlim(0, max(num(f, "qps") for f in sq8) * 1.08)
    ax_d.set_xlim(0, max(num(f, "disco_mb") for f in sq8 if num(f, "disco_mb")) * 1.08)
    ajustar_xlim(ax_c, [t for t in textos if t.axes is ax_c])
    ajustar_xlim(ax_d, [t for t in textos if t.axes is ax_d])
    ticks_amables(ax_c)
    ticks_amables(ax_d)
    ax_c.set_xlabel("queries per second (search_batch)")
    ax_d.set_xlabel("MB on disk")
    ax_c.set_title("C · SQ8 on disk: search_batch throughput")
    ax_d.set_title("D · SQ8 on disk: disk footprint")
    for ax in (ax_c, ax_d):
        ax.grid(axis="y", visible=False)
    cerrar_figura(fig, "Index types: HRM2 by level of detail, SQ8 on disk, and HNSW / LSH vs linear scan",
                  pie_de(idx, bus))
    return fig, ("Index types: HRM2 by level of detail, SQ8 on disk (QPS and MB), HNSW and LSH vs linear scan",
                 [D["idx"][1], D["bus"][1]])


def fig_storage(D):
    sto = D["sto"][0]
    filas = sto["resultados"]
    builds = sorted([f for f in filas if f.get("camino") == "build" and num(f, "mediana") is not None],
                    key=lambda f: f["dataset"]["n"])
    snaps = {f["dataset"]["n"]: f for f in filas if f.get("camino") == "snapshot" and num(f, "mediana") is not None}
    grupos_a = [{"etiqueta": f"clustered {f['dataset']['n'] // MIL}k docs", "n": f["dataset"]["n"]} for f in builds]
    series_a = [{"etiqueta": "in memory: build with add(n)", "clave": "build", "color": MEMORIA_C},
                {"etiqueta": "on disk: reopen from snapshot", "clave": "snapshot", "color": PERSISTENTE_C}]

    def valor_a(gi, si):
        f = next((x for x in builds if x["dataset"]["n"] == grupos_a[gi]["n"]), None) \
            if series_a[si]["clave"] == "build" else snaps.get(grupos_a[gi]["n"])
        v = num(f, "mediana")
        return v / MIL if v is not None else None

    esc = [f for f in filas if f.get("camino") == "escritura" and num(f, "qps") is not None]
    docs_vals = sorted({f["docs"] for f in esc})
    grupos_b = [{"etiqueta": f"add({x:,} docs)", "docs": x} for x in docs_vals]
    series_b = [{"etiqueta": "in memory: without storage_path", "con": False, "color": MEMORIA_C},
                {"etiqueta": "on disk: with storage_path", "con": True, "color": PERSISTENTE_C}]

    def valor_b(gi, si):
        f = next((x for x in esc if x["docs"] == grupos_b[gi]["docs"]
                  and bool(x.get("con_storage_path")) == series_b[si]["con"]), None)
        return num(f, "qps")

    unit = sorted([f for f in filas if f.get("camino") == "add_unitario" and num(f, "mediana") is not None],
                  key=lambda f: bool(f.get("con_storage_path")))
    fig, gs = nueva_figura(6.6, nrows=2, ncols=3, height_ratios=[1, 0.16], width_ratios=[1.15, 1.0, 1.55])
    ax_a, ax_b, ax_c = (fig.add_subplot(gs[0, k]) for k in range(3))
    textos, _ = barras_h(ax_a, grupos_a, series_a, valor_a, lambda v: fmt_num(round(v, 2)) + " s")
    ajustar_xlim(ax_a, textos)
    ax_a.set_xlabel("seconds (median)")
    ax_a.set_title("A · Build in memory vs reopen from disk")
    textos, _ = barras_h(ax_b, grupos_b, series_b, valor_b, fmt_qps)
    ajustar_xlim(ax_b, textos)
    ax_b.set_xlabel("documents per second (median)")
    ax_b.set_title("B · add() throughput")
    etiquetas_c = ["with storage_path" if f.get("con_storage_path") else "without storage_path" for f in unit]
    textos = []
    for i, f in enumerate(unit):
        med, p5, p95 = num(f, "mediana"), num(f, "p5"), num(f, "p95")
        color = PERSISTENTE_C if f.get("con_storage_path") else MEMORIA_C
        ax_c.barh(i, med, 0.5, color=color, zorder=3)
        ax_c.plot([p5, p95], [i, i], color=TEXTO, linewidth=1.8, solid_capstyle="round", zorder=4)
        textos.append(ax_c.annotate(f"median {med:,.2f} ms\np5 {p5:,.2f} · p95 {p95:,.2f}",
                                    (max(med, p95), i), xytext=(5, 0), textcoords="offset points",
                                    ha="left", va="center", fontsize=T_VALOR, color=TEXTO))
    ax_c.set_yticks(range(len(unit)))
    ax_c.set_yticklabels(etiquetas_c)
    ax_c.set_ylim(len(unit) - 0.5 + 0.35, -0.5 - 0.35)
    ax_c.set_xlim(0, max(num(f, "p95") for f in unit) * 2.2)
    ajustar_xlim(ax_c, textos)
    ticks_amables(ax_c)
    ax_c.set_xlabel("milliseconds per add(1 doc)\nbar = median, line = p5–p95")
    ax_c.set_title("C · add(1 doc) latency")
    # una sola leyenda a lo ancho (azul = en memoria, verde = en disco): dos leyendas por
    # columna le quitaban ancho al panel C en el constrained layout
    leyenda_de_barras(fig, gs[1, :], series_a + series_b, ncol=4)
    cerrar_figura(fig, "Storage: build vs reopen, add() throughput and single-doc add latency", pie_de(sto))
    return fig, ("Storage: build vs reopen, add() throughput with and without storage_path, add(1 doc) latency",
                 [D["sto"][1]])


FIGURAS = (("bench-hero", fig_hero), ("bench-qps", fig_qps), ("bench-filtered", fig_filtered),
           ("bench-index", fig_index), ("bench-storage", fig_storage))


# ------------------------------------------------------------------------------ hechos
def hechos(D):
    """Todo lo que las leyendas de BENCHMARKS.md afirman, calculado desde los JSON."""
    bus, bus_antes, fil, idx, sto = (D[k][0] for k in ("bus", "bus_antes", "fil", "idx", "sto"))
    cfgs = configs_de(bus)
    colores, nds = colores_de(cfgs)
    legacy, con_meta, lin1 = ("search(q,k) legacy",), ("search(q,k) include_metadata=True",), ("linear_single",)

    def q(doc, param, cfg):
        return num(fila(doc, param, kind=cfg[0], n=cfg[1], d=cfg[2]), "qps")

    H = {"cfgs": cfgs, "colores": colores,
         "paleta": ", ".join(f"{DS_NOMBRES[i % len(DS_NOMBRES)]} = {nd[0] // MIL}k · d={nd[1]}"
                             for i, nd in enumerate(nds))}
    cl = [c for c in cfgs if c[0] == "clustered"]
    H["cl"] = cl
    H["d_ref"] = max(c[2] for c in cl)
    H["ratio"] = {}
    for nombre, param, doc in (("tuple", legacy, bus), ("default", con_meta, bus), ("before", legacy, bus_antes)):
        H["ratio"][nombre] = {etiqueta_ds(*c): q(doc, param, c) / q(bus, lin1, c) for c in cl
                              if q(doc, param, c) and q(bus, lin1, c)}
    m2m = [f for f in bus["resultados"] if f.get("camino") == "m2m" and num(f, "recall@10") is not None]
    H["rec_clustered"] = [num(f, "recall@10") for f in m2m if f["dataset"]["kind"] == "clustered"]
    H["rec_uniform"] = [num(f, "recall@10") for f in m2m if f["dataset"]["kind"] == "uniform"]
    bs = lotes_de(bus)
    H["bs"] = bs
    H["b1"] = [(etiqueta_ds(*c), q(bus, (f"search_batch B={bs[0]}",), c), q(bus, lin1, c)) for c in cfgs]
    H["celdas"] = []
    for b in bs[1:]:
        for c in cfgs:
            lin, lib = num(ref_lineal(bus, b, c), "qps"), q(bus, (f"search_batch B={b}",), c)
            if lin is not None and lib is not None:
                H["celdas"].append((b, etiqueta_ds(*c), lin, lib))
    H["lin_bs"] = sorted({b for c in cfgs for b in bs if ref_lineal(bus, b, c) is not None})
    ff = fil["resultados"]
    H["ds_fil"] = etiqueta_ds(ff[0]["dataset"]["kind"], ff[0]["dataset"]["n"], ff[0]["dataset"]["d"]) \
        if ff else "n/a"
    H["post"] = [num(f, "recall@10") for f in ff if f.get("camino") == "batch_postfiltro"
                 and num(f, "recall@10") is not None]
    H["exactas"] = [num(f, "recall@10") for f in ff if f.get("camino") in
                    ("single_filtro", "batch_filtro_dict", "batch_filtro_fn") and num(f, "recall@10") is not None]
    H["frio"] = []
    for f in ff:
        if f.get("camino") == "single_filtro_frio" and num(f, "qps"):
            cal = next((x for x in ff if x.get("camino") == "single_filtro"
                        and x.get("filtro_nombre") == f.get("filtro_nombre")), None)
            if cal and num(cal, "qps"):
                H["frio"].append(num(f, "qps") and num(cal, "qps") / num(f, "qps"))
    H["invalidacion"] = next((f["invalidacion"] for f in ff if f.get("invalidacion")), "")
    H["filtros"] = []
    for f in ff:
        n = f.get("filtro_nombre")
        if n and n not in [x[0] for x in H["filtros"]]:
            H["filtros"].append((n, f.get("selectividad")))
    ifilas = idx["resultados"]
    hrm = [f for f in ifilas if f.get("camino") == "hrm2_lod" and num(f, "qps") is not None]
    H["lods"] = sorted({f["lod"] for f in hrm})
    H["hrm_lin"] = {}
    for f in hrm:
        d = f["dataset"]
        lin = num(fila(bus, lin1, kind=d["kind"], n=d["n"], d=d["d"]), "qps")
        if lin:
            H["hrm_lin"].setdefault(etiqueta_ds(d["kind"], d["n"], d["d"]), []).append((f["lod"], num(f, "qps") / lin))
    sq8 = [f for f in ifilas if f.get("camino") == "sq8" and num(f, "qps") is not None]
    H["sq8_todo"] = len(sq8)
    H["sq8_baja"] = []
    for f in sq8:
        m = re.search(r"oversample=([^)]+)\)", f["parametro"])
        if num(f, "recall@10") is not None and num(f, "recall@10") < UMBRAL_RECALL:
            H["sq8_baja"].append((f["dataset"]["n"] // MIL, MODO_EN.get(f.get("modo"), f.get("modo")),
                                  m.group(1) if m else "?", num(f, "recall@10")))
    H["sq8_modos"] = sorted({MODO_EN.get(f.get("modo"), f.get("modo")) for f in sq8})

    def por_camino(camino):
        return next((f for f in ifilas if f.get("camino") == camino and num(f, "qps") is not None), None)

    ref, hnsw, lsh = por_camino("baseline"), por_camino("hnsw"), por_camino("lsh")
    H["n_hnsw"] = idx.get("n_hnsw_lsh")
    H["ref"], H["hnsw"], H["lsh"] = ref, hnsw, lsh
    H["lin_10k"] = q(bus, lin1, next(c for c in cfgs if c[1] == min(x[1] for x in cfgs) and c[2] == H["d_ref"]))
    sfilas = sto["resultados"]
    H["reopen_build"] = []
    for f in sfilas:
        if f.get("camino") == "snapshot" and num(f, "mediana"):
            b = next((x for x in sfilas if x.get("camino") == "build" and x["dataset"]["n"] == f["dataset"]["n"]), None)
            if b and num(b, "mediana"):
                H["reopen_build"].append((f["dataset"]["n"], num(f, "mediana") / num(b, "mediana")))
    H["add1"] = {bool(f.get("con_storage_path")): (num(f, "mediana"), num(f, "p5"), num(f, "p95"))
                 for f in sfilas if f.get("camino") == "add_unitario" and num(f, "mediana") is not None}
    H["reps_sto"] = {}
    for f in sfilas:
        H["reps_sto"].setdefault(f.get("camino"), set()).add(f.get("n_repeticiones"))
    return H


# ------------------------------------------------------------------------------ BENCHMARKS.md
CLAVES_BASE = ["dataset.kind", "dataset.n", "dataset.d", "parametro", "camino", "B", "n_repeticiones",
               "mediana", "p5", "p95", "qps", "recall@10", "rss_pico_mb", "build_s"]
SALTAR = {"tabla", "commit", "meta", "dataset.seed"}


def resumen_veredicto(v):
    plano = {k: x for k, x in v.items() if not isinstance(x, (dict, list))}
    ver = v.get("verdict")
    if isinstance(ver, dict):
        for k, x in ver.items():
            if not isinstance(x, (dict, list)):
                plano["verdict." + k] = x
    return "scalars only (full object in the JSON): " + json.dumps(plano, ensure_ascii=False, separators=(",", ":"))


def aplanar(f):
    out = {}
    for k, v in f.items():
        if k in SALTAR:
            continue
        if k == "dataset" and isinstance(v, dict):
            for kk, vv in v.items():
                if kk != "seed":
                    out["dataset." + kk] = vv
        elif k == "veredicto" and isinstance(v, dict):
            out[k] = resumen_veredicto(v)
        elif isinstance(v, (dict, list)):
            out[k] = json.dumps(v, ensure_ascii=False, separators=(",", ":"))
        else:
            out[k] = v
    return out


def celda(v):
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return fmt_num(v)
    return str(v).replace("|", "\\|").replace("\n", " ")


def tabla_md(doc):
    filas = [aplanar(f) for f in doc["resultados"]]
    cols = [c for c in CLAVES_BASE if any(c in f for f in filas)]
    for f in filas:
        for k in f:
            if k not in cols:
                cols.append(k)
    lineas = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for f in filas:
        lineas.append("| " + " | ".join(celda(f.get(c)) for c in cols) + " |")
    return "\n".join(lineas)


def leyenda_hero(H):
    r = H["ratio"]
    t = ("Left: single-query throughput on clustered data — the NumPy linear scan, `search(q, k)` at the before "
         "and after commits (legacy tuple path), and the default call that also returns metadata; the "
         "parenthesis is after ÷ linear scan. ")
    ref = [c for c in H["cl"] if c[2] == H["d_ref"]]
    rt = [r["tuple"].get(etiqueta_ds(*c)) for c in ref]
    rd = [r["default"].get(etiqueta_ds(*c)) for c in ref]
    t += (f"At d = {H['d_ref']} the tuple path is {rango(rt, lambda x: f'{x:.2f}×')} the linear scan "
          f"(default path {rango(rd, lambda x: f'{x:.2f}×')}). ")
    for c in H["cl"]:
        if c[2] == H["d_ref"]:
            continue
        rt2, rd2 = r["tuple"].get(etiqueta_ds(*c)), r["default"].get(etiqueta_ds(*c))
        if rt2 is not None and rt2 < 1 and (rd2 is None or rd2 < 1):
            t += f"At d = {c[2]} the linear scan is faster than every library path ({rt2:.2f}× tuple, {rd2:.2f}× default). "
        elif rt2 is not None:
            t += f"At d = {c[2]} the tuple path is {rt2:.2f}× the linear scan. "
    t += (f"Right: recall@10 against the exact float64 brute-force top-10 — {rango(H['rec_clustered'], lambda x: f'{x:.3f}')} "
          f"on every clustered row and {rango(H['rec_uniform'], lambda x: f'{x:.2f}')} on uniform data, where the "
          "index misses most of the true top-10. ")
    t += ("What it does not show: peak throughput (next figure), other hardware or BLAS thread counts, "
          "or the cause of the recall collapse.")
    return t


def leyenda_qps(H):
    gana = [c for c in H["celdas"] if c[2] > c[3]]
    pierde = sorted({f"{c[1]} at B={c[0]:,}" for c in H["celdas"] if c[2] <= c[3]})
    bs_txt = ", ".join(f"{b:,}" for b in sorted({c[0] for c in H["celdas"]}))
    t = (f"`search_batch` throughput by batch size B, one line per dataset ({H['paleta']}; hollow squares = "
         "uniform data), against the NumPy linear scan (amber, dashed); log-log axes. Recall@10 is annotated "
         "only where it is below 1.0. ")
    if H["celdas"]:
        t += (f"For B = {bs_txt} the batched linear scan is faster than `search_batch` in {len(gana)} of "
              f"{len(H['celdas'])} dataset × batch cells")
        t += ("; the library is ahead only on " + ", ".join(pierde) + ". ") if pierde else ". "
    b1 = [x for x in H["b1"] if x[1] and x[2]]
    if b1:
        t += f"At B = {H['bs'][0]} the library is faster on {sum(1 for x in b1 if x[1] > x[2])} of {len(b1)} datasets. "
    faltan = [b for b in H["bs"] if b not in H["lin_bs"]]
    if faltan:
        t += (f"The linear scan was measured at B = {', '.join(f'{b:,}' for b in H['lin_bs'])} only; the dashed "
              f"segment crossing B = {', '.join(f'{b:,}' for b in faltan)} is not a measurement. ")
    t += "What it does not show: per-query latency (Table 1), multi-threaded BLAS, or the filtered and on-disk paths."
    return t


def leyenda_filtered(H):
    t = (f"Filtered search on {H['ds_fil']}: A — recall@10 per filter and path; B — single-query throughput; "
         f"C — batch throughput. Filters: " + ", ".join(f"`{n}` ({(s or 0) * PCT:.1f}% of the documents)" for n, s in H["filtros"])
         + ". ")
    if H["post"] and H["exactas"]:
        t += (f"`search(filter=dict)` and both `search_batch_filtered` variants return the exact filtered top-10 "
              f"({rango(H['exactas'], lambda x: f'{x:.3f}')} in every row); the legacy post-filter — fetch an "
              f"unfiltered top-10 and drop the rows that fail the filter — returns {rango(H['post'], lambda x: f'{x:.3f}')}. ")
    if H["frio"]:
        t += (f"The first evaluation after a public write is {rango(H['frio'], lambda x: f'{x:.1f}×')} slower than a "
              "repeated one (the benchmark forces a cold cache before each cold query — `invalidacion` in the "
              f"JSON: \"{H['invalidacion']}\"). ")
    t += "What it does not show: other selectivities, 50k-docs corpora, or the cost of keeping the masks up to date."
    return t


def leyenda_index(H):
    t = (f"A — single-query QPS of the HRM2 index by level of detail (lod {', '.join(str(x) for x in H['lods'])}) "
         "against the linear scan measured in the same session (`bench_search.json`); B — on a "
         f"{(H['n_hnsw'] or 0):,}-doc corpus, pure-Python HNSW and cross-polytope LSH against the batched linear scan "
         "recorded in the same table (log scale); C and D — SQ8 on disk: `search_batch` throughput and disk footprint "
         "per mode (" + ", ".join(f"`{m}`" for m in H["sq8_modos"]) + ") and oversampling, recall at the end of each "
         "bar in C. ")
    mas_rapidos = sorted({lod for v in H["hrm_lin"].values() for (lod, r) in v if r > 1})
    detalle = "; ".join(k + ": " + ", ".join(f"lod {lod} = {r:.2f}×" for lod, r in sorted(v))
                        for k, v in H["hrm_lin"].items())
    if mas_rapidos:
        t += f"HRM2 is faster than the linear scan only at lod = {', '.join(str(x) for x in mas_rapidos)} ({detalle}). "
    else:
        t += f"HRM2 is never faster than the linear scan ({detalle}). "
    if H["sq8_baja"]:
        t += (f"SQ8 recall is below 1.0 in {len(H['sq8_baja'])} of {H['sq8_todo']} rows ("
              + ", ".join(f"{n}k · {m} · oversample {o}: {rec:.3f}" for (n, m, o, rec) in H["sq8_baja"]) + "). ")
    else:
        t += "SQ8 recall is 1.0 in every measured row. "
    if H["hnsw"] and H["lsh"] and H["ref"]:
        t += (f"HNSW ({fmt_qps(num(H['hnsw'], 'qps'))} QPS, build {num(H['hnsw'], 'build_s'):,.1f} s) and LSH "
              f"({fmt_qps(num(H['lsh'], 'qps'))} QPS) are far below the linear reference ({fmt_qps(num(H['ref'], 'qps'))} QPS). ")
    t += ("What it does not show: tuned or native HNSW/LSH, SQ8 above 50k docs, or a like-for-like HNSW/LSH "
          "comparison — the reference row is a batched scan and both index types answer one query per call.")
    return t


def leyenda_storage(H):
    t = ("A — building the index in memory (`add(n)`) versus reopening it from a snapshot (`save(path)` + "
         "`SimpleVectorDB(storage_path=path)`); B — `add()` throughput with and without `storage_path`; C — latency "
         "of `add(1 doc)` with and without `storage_path` (bar = median, line = p5–p95). ")
    if H["reopen_build"]:
        t += ("Reopening costs " + ", ".join(f"{r:.2f}× the in-memory build at {n // MIL}k docs" for n, r in H["reopen_build"])
              + ". ")
    if False in H["add1"] and True in H["add1"]:
        s, c = H["add1"][False], H["add1"][True]
        t += (f"A single-doc add takes a median of {s[0]:,.2f} ms without `storage_path` and {c[0]:,.2f} ms with it "
              f"(p95 {c[2]:,.2f} ms). ")
    nombres = {"build": "build", "snapshot": "reopen", "escritura": "`add()`", "add_unitario": "`add(1 doc)`"}
    reps = "; ".join(f"{nombres.get(k, k)}: " + "/".join(str(x) for x in sorted(v, key=lambda z: (z is None, z or 0)))
                     for k, v in H["reps_sto"].items())
    t += (f"Repetitions per row in this table are few ({reps}): read these as order-of-magnitude figures, not as "
          "precise medians. What it does not show: a before/after comparison (there is no before storage table), "
          "concurrent writers, or other storage backends.")
    return t


LEYENDAS = (leyenda_hero, leyenda_qps, leyenda_filtered, leyenda_index, leyenda_storage)


def benchmarks_md(D, manifest):
    bus, fil, idx, sto = D["bus"][0], D["fil"][0], D["idx"][0], D["sto"][0]
    bus_antes = D["bus_antes"][0]
    H = hechos(D)
    m = bus["meta"]
    hilos = m["hilos"]
    L = ["# Benchmarks (measured)", ""]
    L += ["Every number on this page — the tables, the captions and the five figures — is generated from the JSON "
          "files in `benchmarks/results/` by `scripts/make_readme_figures.py`; nothing is typed by hand. The same "
          "script checks every figure (no overlapping texts, nothing outside the canvas, no value label on top of a "
          "bar, byte-identical PNGs on two consecutive runs) and writes `bench-manifest.json` with the SHA-256 of "
          "every source file and figure.", ""]
    L += ["```",
          "PYTHONPATH=$PWD/src python scripts/make_readme_figures.py   # figures + manifest + this page",
          "python scripts/make_readme_figures.py --check                 # no hand-typed numbers in the script",
          "```", ""]
    n_after = len({commit_corto_de(D[k][0]) for k in ("bus", "fil", "idx", "sto")})
    L += [f"* **Code measured (\"after\"):** commit `{commit_corto_de(bus)}` (`meta.commit`; the four after JSON "
          f"files record {n_after} distinct commit{'s' if n_after != 1 else ''}).",
          f"* **Comparison base (\"before\"):** commit `{commit_corto_de(bus_antes)}`, files under "
          "`benchmarks/results/before_a74d9e4/` (search, filtered and index tables; there is no before storage table).",
          f"* **Machine:** {m['cpu']}, {m['cpu_count']} logical CPUs; {m['plataforma']}; Python {m['python']}, "
          f"NumPy {m['numpy']}; BLAS threads: OPENBLAS_NUM_THREADS = {hilos.get('OPENBLAS_NUM_THREADS')}, "
          f"OMP_NUM_THREADS = {hilos.get('OMP_NUM_THREADS')} (single thread on purpose: with the default BLAS "
          "threads the small operations spin-wait on this host).",
          ""]
    # ---------------------------------------------------------------- protocolo
    p = bus["protocolo"]
    L += ["## Protocol", "",
          "| Item | Value (from the JSON) |", "|---|---|",
          f"| Queries | {p['n_queries']} per dataset, drawn with the same generator as the corpus "
          f"(query seed {p['seed_queries']}, data seed {p['seed_datos']}), k = {p['k']} |",
          "| Ground truth | exact float64 brute-force dot product against every row (argpartition + sort); "
          "recall@10 = |top-10 ∩ GT| / |GT| |",
          f"| Repetitions | warm-up {p['warmup']}; each row repeats between {p['min_reps']} and {p['max_reps']} "
          f"times until at least {p['min_tiempo_s']:.1f} s of wall time; `qps` = median over repetitions of "
          "(queries per repetition ÷ wall time of that repetition); `mediana` / `p5` / `p95` = latency percentiles "
          "over all queries of all repetitions |",
          "| Memory | `rss_pico_mb` = peak process RSS sampled from `/proc/self/statm` |",
          f"| Filtered table | one corpus ({etiqueta_ds(fil['resultados'][0]['dataset']['kind'], fil['resultados'][0]['dataset']['n'], fil['resultados'][0]['dataset']['d'])}); "
          f"reps {fil['protocolo']['min_reps']}–{fil['protocolo']['max_reps']} |",
          f"| Index-type table | blocks `{idx.get('bloques')}`; HNSW/LSH/gate corpus N = {idx.get('n_hnsw_lsh'):,}; "
          f"gate timeout {idx.get('gate_timeout_s')} s |",
          f"| Storage table | reopen {sto.get('reopen_reps')} repetitions, `add()` {sto.get('escritura_reps')}, "
          f"`add(1 doc)` {sto.get('adds_unitarios')} single adds |",
          ""]
    L += ["Datasets (`bench_search.json` → `datasets`):", "",
          "| kind | n | d | n_docs | n_coarse | n_probe | build_s | gt_s | RSS peak during build (MB) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for x in bus.get("datasets", []):
        L.append(f"| {x['kind']} | {fmt_num(x['n'])} | {x['d']} | {fmt_num(x['n_docs'])} | {x['n_coarse']} | "
                 f"{x['n_probe']} | {fmt_num(x['build_s'])} | {fmt_num(x['gt_s'])} | {fmt_num(x['rss_pico_build_mb'])} |")
    L += ["", "Corpus generators, as recorded in `protocolo`: clustered = C = round(sqrt(n)) centers ~ N(0, I_d)·2, "
          "point = center + N(0, I_d), L2-normalized float32; uniform = N(0, I_d), L2-normalized float32. "
          "The verbatim `protocolo` text of each JSON is reproduced in the appendix.", ""]
    L += ["Load average (1/5/15 min) before → after each table:", "",
          "| Table | Commit | Load before | Load after |", "|---|---|---|---|"]
    for etiqueta, (doc, ruta) in (("bench_search (after)", D["bus"]), ("bench_search (before)", D["bus_antes"]),
                                  ("bench_filtered", D["fil"]), ("bench_index_types", D["idx"]),
                                  ("bench_storage", D["sto"])):
        if doc:
            mm = doc["meta"]
            L.append(f"| `{os.path.relpath(ruta, ROOT)}` | `{commit_corto_de(doc)}` | `{mm.get('loadavg_antes', mm.get('uptime_antes', 'n/a'))}` | "
                     f"`{mm.get('loadavg_despues', mm.get('uptime_despues', 'n/a'))}` |")
    L.append("")
    # ---------------------------------------------------------------- figuras
    L += ["## Figures", "", "All figures: horizontal bars from zero with the value at the end of the bar (line "
          "charts only where a batch-size or a log-scale comparison needs them), English text, comma thousands "
          "and dot decimals. Colour key shared by every figure: amber = NumPy linear scan, gray = before commit, "
          f"blue = single-query path (after), greens = batch paths, {H['paleta']}.", ""]
    for i, (ruta_png, info) in enumerate(manifest.items(), 1):
        nombre = os.path.basename(ruta_png)
        L += [f"### Figure {i} — `{nombre}`", "", f"![{info['titulo']}]({nombre})", "",
              LEYENDAS[i - 1](H), "",
              "Sources: " + "; ".join(f"`{f['ruta']}` (sha256 `{f['sha256']}`)" for f in info["fuentes"]) + ".", ""]
    # ---------------------------------------------------------------- tablas
    L += ["## Tables", "",
          "Column names are the JSON keys and values are printed exactly as stored (no rounding). `n/a` = null "
          "in the JSON; a row recorded as `n/d` (not measured) keeps its `estado` / `motivo` columns.", ""]
    for i, (titulo_tabla, doc, ruta) in enumerate((
            ("bench_search.json — after", bus, D["bus"][1]),
            ("before_a74d9e4/bench_search.json — before, used by Figure 1", bus_antes, D["bus_antes"][1]),
            ("bench_filtered.json", fil, D["fil"][1]),
            ("bench_index_types.json", idx, D["idx"][1]),
            ("bench_storage.json", sto, D["sto"][1])), 1):
        if doc is None:
            continue
        L += [f"### Table {i} — `{os.path.relpath(ruta, ROOT)}` ({titulo_tabla}; commit `{commit_corto_de(doc)}`, "
              f"sha256 `{sha256(ruta)}`)", "", tabla_md(doc), ""]
    L += ["## Noise", "", NOTA_RUIDO, ""]
    L += ["## Appendix — verbatim `protocolo` recorded in each JSON", ""]
    for etiqueta, (doc, ruta) in (("bench_search (after)", D["bus"]), ("bench_search (before)", D["bus_antes"]),
                                  ("bench_filtered", D["fil"]), ("bench_index_types", D["idx"])):
        if doc and doc.get("protocolo"):
            L += [f"<details><summary>{etiqueta} — <code>{os.path.relpath(ruta, ROOT)}</code></summary>", "",
                  "```json", json.dumps(doc["protocolo"], ensure_ascii=False, indent=1), "```", "", "</details>", ""]
    return "\n".join(L)


# ------------------------------------------------------------------------------ main
def generar(D, solo=None):
    pngs, manifest = {}, {}
    for nombre, fn in FIGURAS:
        if solo and nombre not in solo:
            continue
        fig, (titulo, fuentes) = fn(D)
        ajustar_todos(fig)
        verif = verificar_layout(fig, nombre)
        png, pixeles, pulgadas = render(fig)
        pngs[nombre] = png
        manifest[f"docs/{nombre}.png"] = {
            "titulo": titulo,
            "fuentes": [{"ruta": os.path.relpath(p, ROOT), "sha256": sha256(p)} for p in fuentes],
            "sha256_png": hashlib.sha256(png).hexdigest(),
            "pixeles": pixeles,
            "pulgadas": pulgadas,
            "verificacion": verif,
        }
    md = benchmarks_md(D, manifest) if not solo else ""
    return pngs, manifest, md


def main():
    sin_cifras_hardcodeadas()
    argv = sys.argv[1:]
    if "--check" in argv:
        print("make_readme_figures: sin literales de cifras (AST del script, incluidos strings)")
        return 0
    estilo()
    solo = set(argv[argv.index("--solo") + 1].split(",")) if "--solo" in argv else None
    D = cargar_todo()
    pngs1, manifest, md = generar(D, solo)
    pngs2, manifest2, md2 = generar(D, solo)
    for nombre, png in pngs1.items():
        h1, h2 = hashlib.sha256(png).hexdigest(), hashlib.sha256(pngs2[nombre]).hexdigest()
        info = manifest[f"docs/{nombre}.png"]
        print(f"{nombre}.png  {info['pixeles'][0]}x{info['pixeles'][1]} px  sha256 {h1[:16]}…  "
              f"textos={info['verificacion']['textos']}  colisiones=0  fuera=0  sobre_barra=0  "
              f"determinista={'si' if h1 == h2 else 'NO'}")
        if h1 != h2:
            raise SystemExit(f"make_readme_figures: {nombre}.png NO es determinista ({h1} vs {h2})")
    if md != md2 or json.dumps(manifest, sort_keys=True) != json.dumps(manifest2, sort_keys=True):
        raise SystemExit("make_readme_figures: BENCHMARKS.md / manifiesto no son deterministas")
    if solo:
        print("modo --solo: nada escrito")
        return 0
    for nombre, png in pngs1.items():
        with open(os.path.join(DOCS, nombre + ".png"), "wb") as f:
            f.write(png)
    doc = {
        "generador": "scripts/make_readme_figures.py",
        "regenerar": "PYTHONPATH=$PWD/src python scripts/make_readme_figures.py",
        "protocolo": "bench/_protocol.py",
        "matplotlib": matplotlib.__version__,
        "pil": getattr(PIL, "__version__", "?"),
        "numpy": D["bus"][0]["meta"].get("numpy"),
        "dpi": DPI,
        "ancho_pulgadas": ANCHO,
        "commit_after": commit_corto_de(D["bus"][0]),
        "commit_before": commit_corto_de(D["bus_antes"][0]),
        "determinista": True,
        "verificacion": "cero colisiones texto/leyenda, cero textos fuera del lienzo, cero rotulos sobre barras, "
                        "PNG identicos en dos corridas seguidas (sha256)",
        "figuras": manifest,
    }
    with open(os.path.join(DOCS, "bench-manifest.json"), "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1, sort_keys=True, ensure_ascii=False)
        f.write("\n")
    with open(os.path.join(DOCS, "BENCHMARKS.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print("-> docs/bench-manifest.json, docs/BENCHMARKS.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
