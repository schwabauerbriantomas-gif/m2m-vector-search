import numpy as np
import sys
sys.path.insert(0, r'C:\Users\Brian\Desktop\m2m-vector-search-main')

cache_path = r'C:\Users\Brian\Desktop\m2m-vector-search-main\benchmarks\results\hf_embeddings_cache_10000_640.npy'
try:
    splats = np.load(cache_path)
except:
    splats = np.random.randn(1000, 640).astype(np.float32)

print(f"Splats shape: {splats.shape}, dtype: {splats.dtype}")

try:
    from m2m.energy import EnergyFunction
    from m2m.config import M2MConfig
    print("Using EnergyFunction")
    
    config = M2MConfig()
    energy_fn = EnergyFunction(config)
    print(f"EnergyFunction initialized: {type(energy_fn)}")
    
    # Check what E_comp, E_geom, E_splats are
    print(f"E_comp type: {type(energy_fn.E_comp)}")
    print(f"E_geom type: {type(energy_fn.E_geom)}")
    print(f"E_splats type: {type(energy_fn.E_splats)}")
    
    queries = np.random.randn(1000, 640).astype(np.float32)
    
    # Try compute or __call__
    try:
        energies = energy_fn.compute(queries)
    except:
        energies = energy_fn(queries)
    
    energies = np.array(energies, dtype=np.float64)
    print(f"Energy stats:")
    print(f"  mean: {np.mean(energies):.6f}")
    print(f"  std: {np.std(energies):.6f}")
    print(f"  min: {np.min(energies):.6f}")
    print(f"  max: {np.max(energies):.6f}")
    print(f"  NaN count: {np.isnan(energies).sum()}")
    print(f"  Inf count: {np.isinf(energies).sum()}")
    
    from scipy.spatial.distance import cdist
    dists = cdist(queries[:100], splats, metric='euclidean')
    nearest_dist = np.min(dists, axis=1)
    nearest_energy = np.array(energies[:100], dtype=np.float64)
    
    corr = np.corrcoef(nearest_dist, nearest_energy)[0,1]
    print(f"  Distance-energy correlation: {corr:.4f}")
    print(f"  (Positive = energy increases with distance = correct)")
    
except Exception as e:
    print(f"Energy error: {type(e).__name__}: {e}")
    import traceback; traceback.print_exc()
