"""H13: SQ8DiskStore end-to-end — protocolo honesto.

Dataset pre-generado UNA vez en disco (600k x 384 float32 = 920MB). Cada
escenario corre en su propio python (RSS aislado) y SOLO mide search:
  A. float32: carga npy completo en RAM (control)
  B. sq8disk: build_from_float_npy + search (SQ8 en RAM, float32 via mmap)
GT precomputado (indices exactos) por el generador.
"""
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

N, DIM, NQ, K = 600_000, 384, 100, 10

GEN = r'''
import os, sys
import numpy as np
tmp = sys.argv[1]
N, DIM = 600_000, 384
rng = np.random.RandomState(42)
centers = rng.randn(300, DIM).astype(np.float32)
data_path = os.path.join(tmp, "data.npy")
mm = np.lib.format.open_memmap(data_path, mode="w+", dtype=np.float32, shape=(N, DIM))
chunk = 100_000
for i in range(0, N, chunk):
    mm[i:i+chunk] = centers[rng.randint(0, 300, size=chunk)] + 0.35*rng.randn(chunk, DIM).astype(np.float32)
mm.flush(); del mm
data = np.load(data_path, mmap_mode="r")
queries = centers[rng.randint(0, 300, size=100)] + 0.35*rng.randn(100, DIM).astype(np.float32)
queries /= np.linalg.norm(queries, axis=1, keepdims=True)
# GT exacto: d2 completo preasignado (100 x 600k float32 = 240MB), un solo
# argpartition al final — sin merge incremental que pueda corromper indices.
qq = np.einsum("ij,ij->i", queries, queries)
d2full = np.empty((100, N), dtype=np.float32)
for i in range(0, N, chunk):
    blk = np.asarray(data[i:i+chunk])
    d2full[:, i:i+chunk] = qq[:, None] - 2.0*(queries @ blk.T) + np.einsum("ij,ij->i", blk, blk)[None, :]
gt = np.argpartition(d2full, 9, axis=1)[:, :10]
np.save(os.path.join(tmp, "queries.npy"), queries)
np.save(os.path.join(tmp, "gt.npy"), gt)
print("dataset + GT listos")
'''

CHILD = r'''
import resource, sys, time, json, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
mode, tmp = sys.argv[1], sys.argv[2]
K = 10
queries = np.load(os.path.join(tmp, "queries.npy"))
gt = np.load(os.path.join(tmp, "gt.npy"))
data_path = os.path.join(tmp, "data.npy")

t0 = time.perf_counter()
if mode == "float32":
    data = np.load(data_path)  # 920MB en RAM
    t_load = time.perf_counter() - t0
    t1 = time.perf_counter()
    qq = np.einsum("ij,ij->i", queries, queries)
    d2 = np.einsum("ij,ij->i", data, data)[None, :] - 2.0*(queries @ data.T) + qq[:, None]
    top = np.argpartition(d2, K-1, axis=1)[:, :K]
    ram_arr = data.nbytes
else:
    from m2m.sq8_disk import SQ8DiskStore
    t_b = time.perf_counter()
    st = SQ8DiskStore.build_from_float_npy(data_path, os.path.join(tmp, "dump"))
    t_load = time.perf_counter() - t_b
    t1 = time.perf_counter()
    top = st.search_batch(queries, k=K, oversample=2)
    ram_arr = st.ram_bytes()
dt = time.perf_counter() - t1

rec = np.mean([len(set(top[i].tolist()) & set(gt[i].tolist()))/K for i in range(len(queries))])
peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
print(json.dumps({"mode": mode, "recall": round(float(rec), 4), "qps": round(len(queries)/dt, 1),
                  "load_or_build_s": round(t_load, 1), "ram_arrays_mb": round(ram_arr/1e6, 1),
                  "peak_rss_mb": round(peak_mb, 1)}))
'''


def main():
    results = {}
    tmp = tempfile.mkdtemp(prefix="h13_")
    try:
        gen = os.path.join(tmp, "gen.py")
        with open(gen, "w") as f:
            f.write(GEN)
        out = subprocess.run([sys.executable, gen, tmp], capture_output=True, text=True, timeout=600)
        assert out.returncode == 0, out.stderr[-300:]
        print(out.stdout.strip())

        child = os.path.join(tmp, "child.py")
        with open(child, "w") as f:
            f.write(CHILD)
        for mode in ("float32", "sq8disk"):
            out = subprocess.run(
                [sys.executable, child, mode, tmp],
                capture_output=True, text=True, timeout=900,
            )
            assert out.returncode == 0, f"{mode}: {out.stderr[-400:]}"
            r = json.loads(out.stdout.strip().splitlines()[-1])
            results[mode] = r
            print(f"{mode}: {r}")
        results["disk_files_mb"] = {
            "data.npy": os.path.getsize(os.path.join(tmp, "data.npy")) / 1e6,
            "dump_sq8_mb": os.path.getsize(os.path.join(tmp, "dump", "sq8.npy")) / 1e6,
        }
        print(f"disco: {results['disk_files_mb']}")
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
    with open(os.path.join(HERE, "h13_sq8disk.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
