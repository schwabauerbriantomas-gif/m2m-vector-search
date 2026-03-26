import numpy as np
import sys
sys.path.insert(0, r'C:\Users\Brian\Desktop\m2m-vector-search-main')

cache_path = r'C:\Users\Brian\Desktop\m2m-vector-search-main\benchmarks\results\hf_embeddings_cache_10000_640.npy'
splats_np = np.load(cache_path)
print(f"Loaded embeddings: {splats_np.shape}")

from m2m.hrm2_engine import HRM2Engine, GaussianSplat

dim = splats_np.shape[1]
engine = HRM2Engine(embedding_dim=dim)

gsplats = []
for i, pos in enumerate(splats_np):
    gs = GaussianSplat(
        id=i, position=pos.astype(np.float32),
        color=np.random.rand(3).astype(np.float32), opacity=1.0,
        scale=np.ones(3, dtype=np.float32),
        rotation=np.zeros(4, dtype=np.float32),
    )
    gsplats.append(gs)

engine.add_splats(gsplats)
engine.index()

stats = engine.get_stats()
print(f"Stats: {stats}")

# Get fine cluster assignments
from sklearn.metrics import silhouette_score as sk_sil, calinski_harabasz_score as sk_ch, davies_bouldin_score as sk_db

for name, labels in [("coarse", engine.coarse_assignments), ("fine", engine.fine_assignments)]:
    if labels is not None:
        labels = np.array(labels)
        print(f"\n--- {name} clusters ({len(np.unique(labels))}) ---")
        sil = sk_sil(splats_np, labels)
        ch = sk_ch(splats_np, labels)
        db = sk_db(splats_np, labels)
        print(f"Silhouette Score: {sil:.4f}")
        print(f"Calinski-Harabasz Index: {ch:.2f}")
        print(f"Davies-Bouldin Index: {db:.4f}")
        print(f"Number of clusters: {len(np.unique(labels))}")
    else:
        print(f"{name} assignments: None")
