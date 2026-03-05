import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from m2m.gpu_vector_index import GPUVectorIndex

def validate_vulkan():
    print("Testing Vulkan Initialization and Compute Shader Execution...")
    try:
        # Create a small toy index
        index_vectors = np.random.randn(100, 640).astype(np.float32)
        # Normalize
        index_vectors /= np.linalg.norm(index_vectors, axis=1, keepdims=True)
        
        # Initialize the GPU Vector Index directly
        # This will compile the shader and load the C++ bindings
        gpu_index = GPUVectorIndex(index_vectors, max_batch_size=10)
        
        # Dispatch a query
        queries = np.random.randn(2, 640).astype(np.float32)
        queries /= np.linalg.norm(queries, axis=1, keepdims=True)
        
        indices, distances = gpu_index.batch_search(queries, k=5)
        
        print(f"\n[SUCCESS] Vulkan shader executed properly!")
        print(f"Indices returned shape: {indices.shape} (Expected: (2, 5))")
        print(f"Distances returned shape: {distances.shape} (Expected: (2, 5))")
        
    except Exception as e:
        print(f"\n[ERROR] Vulkan Validation Failed:")
        print(str(e))
        print("Please ensure your GPU supports Vulkan and you have the Vulkan SDK installed if compiling shaders.")

if __name__ == "__main__":
    validate_vulkan()
