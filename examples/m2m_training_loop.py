import numpy as np
import time
import sys
import os

# Add parent dir to path to find m2m
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from m2m import M2MEngine, M2MConfig, create_m2m

def dummy_training_loop(engine: M2MEngine, steps: int = 100):
    """
    Simula un loop de entrenamiento que empuja gradients simulados asíncronamente
    a través del Data Lake (en este mock, simplemente generamos embeddings y actualizamos).
    """
    print(f"\n[Training] Starting simulated training for {steps} steps...")
    
    for step in range(steps):
        # 1. Forward pass simulado (ej. obtener queries)
        # Dummy batch of 32 queries
        queries = np.random.randn(32, engine.config.latent_dim).astype(np.float32)
        queries /= np.linalg.norm(queries, axis=-1, keepdims=True) + 1e-8
        
        # 2. Obtener mu's (keys) usando el VectorDB (forward)
        mu, a, k = engine.search(queries, k=5)
        
        # 3. Simulate backward pass and optimization
        loss = np.random.rand() * (100 - step) / 100 # Simulated decreasing loss
        
        if step % 20 == 0:
            print(f"  Step {step:03d} | Simulated Loss: {loss:.4f} | Processed batch size 32")
    
    print("\nSimulated training completed successfully using M2M Data Lake!")

def main():
    print("Initializing M2M Storage & Data Lake...")
    config = M2MConfig(
        device='cpu',
        n_splats_init=5000,
        max_splats=10000,
        enable_vulkan=False
    )
    m2m = create_m2m(config)
    
    # Simulate data ingestion
    print("Ingesting 5000 splats into M2M Data Lake...")
    dummy_data = np.random.randn(5000, config.latent_dim).astype(np.float32)
    m2m.add_splats(dummy_data)
    
    # Run simulated training loop
    start_time = time.time()
    dummy_training_loop(m2m, steps=100)
    print(f"\nTraining completed in {time.time() - start_time:.2f}s")
    print("\nTraining completed successfully using M2M Data Lake!")

if __name__ == '__main__':
    main()
