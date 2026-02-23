import torch
import torch.nn as nn
import torch.optim as optim
import time
import sys
import os
import json
import argparse

# Add parent dir to path to find m2m
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from m2m import M2MConfig, create_m2m

class SimpleMLP(nn.Module):
    def __init__(self, input_dim=640, hidden_dim=256, output_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

def load_real_dataset(num_samples=10000, dim=640):
    try:
        print("[INFO] Loading real-world structured dataset (Handwritten Digits) via scikit-learn...")
        from sklearn.datasets import load_digits
        import numpy as np
        
        # Real-world data: 1797 images of 8x8 handwritten digits
        digits = load_digits()
        X = digits.data
        
        # Repeat or sample to get up to num_samples
        if X.shape[0] < num_samples:
            indices = np.random.choice(X.shape[0], num_samples, replace=True)
            X = X[indices]
        else:
            X = X[:num_samples]
            
        tensor_emb = torch.tensor(X, dtype=torch.float32)
        
        # Project real data to 640 dimensions to match the M2M spec
        if tensor_emb.shape[1] != dim:
            print(f"[INFO] Projecting {tensor_emb.shape[1]}D digit vectors to {dim}D M2M latent space...")
            proj = nn.Linear(tensor_emb.shape[1], dim, bias=False)
            with torch.no_grad():
                tensor_emb = proj(tensor_emb)
                
        print(f"[SUCCESS] Loaded {len(tensor_emb)} real-world embeddings (Clusters: {len(digits.target_names)})")
        return tensor_emb
    except Exception as e:
        print(f"[WARNING] Could not load scikit dataset, falling back: {e}")
        torch.manual_seed(42)
        clusters = torch.randn(50, dim) * 5
        assignments = torch.randint(0, 50, (num_samples,))
        data = clusters[assignments] + torch.randn(num_samples, dim) * 0.5
        return data

def validate_data_lake(device='cpu', enable_vulkan=False):
    print(f"\n{'='*50}\nInitializing M2M Validation on {device} (Vulkan: {enable_vulkan})\n{'='*50}")
    config = M2MConfig(
        device=device,
        n_splats_init=10000,
        max_splats=20000,
        enable_vulkan=enable_vulkan
    )
    m2m = create_m2m(config)
    
    dataset_size = 10000
    mock_embeddings = load_real_dataset(num_samples=dataset_size, dim=config.latent_dim).to(config.device)
    
    # Ingest Data
    start_ingest = time.time()
    m2m.add_splats(mock_embeddings)
    ingest_time = time.time() - start_ingest
    print(f"Ingested {dataset_size} splats in {ingest_time:.2f}s ({(dataset_size/ingest_time):.2f} splats/sec)")
    
    metrics = {
        "hardware": f"{device} (Vulkan: {enable_vulkan})",
        "dataset_size": dataset_size,
        "ingest_throughput_qps": dataset_size / ingest_time,
        "retrieval_routing": {},
        "standard_training": {},
        "generative_training": {}
    }
    
    # Training Loop - Standard
    print("\n--- Standard Training Loop (SOC Importance Sampling) ---")
    dataloader = m2m.export_to_dataloader(batch_size=256, importance_sampling=True, generate_samples=False)
    
    model_std = SimpleMLP(input_dim=config.latent_dim).to(config.device)
    optimizer_std = optim.Adam(model_std.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    start_time = time.time()
    total_loss_std = 0
    batches = 0
    for batch_idx, batch_mu in enumerate(dataloader):
        batch_mu = batch_mu.to(config.device)
        targets = torch.randint(0, 10, (batch_mu.shape[0],)).to(config.device)
        
        optimizer_std.zero_grad()
        outputs = model_std(batch_mu)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer_std.step()
        total_loss_std += loss.item()
        batches += 1
        
    epoch_time = time.time() - start_time
    avg_loss_std = total_loss_std / max(1, batches)
    print(f"Standard Epoch completed in {epoch_time:.2f}s | Avg Loss: {avg_loss_std:.4f}")
    
    metrics["standard_training"]["epoch_time_seconds"] = epoch_time
    metrics["standard_training"]["avg_loss"] = avg_loss_std
    metrics["standard_training"]["throughput_splats_per_sec"] = dataset_size / epoch_time

    # Training Loop - Generative
    print("\n--- Generative Training Loop (Langevin Augmentation) ---")
    gen_dataloader = m2m.export_to_dataloader(batch_size=256, generate_samples=True)
    
    model_gen = SimpleMLP(input_dim=config.latent_dim).to(config.device)
    optimizer_gen = optim.Adam(model_gen.parameters(), lr=1e-3)
    
    start_time = time.time()
    total_loss_gen = 0
    batches = 0
    for batch_idx, batch_mu in enumerate(gen_dataloader):
        batch_mu = batch_mu.to(config.device)
        targets = torch.randint(0, 10, (batch_mu.shape[0],)).to(config.device)
        
        optimizer_gen.zero_grad()
        outputs = model_gen(batch_mu)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer_gen.step()
        total_loss_gen += loss.item()
        batches += 1
        
    epoch_time_gen = time.time() - start_time
    avg_loss_gen = total_loss_gen / max(1, batches)
    print(f"Generative Epoch completed in {epoch_time_gen:.2f}s | Avg Loss: {avg_loss_gen:.4f}")
    
    metrics["generative_training"]["epoch_time_seconds"] = epoch_time_gen
    metrics["generative_training"]["avg_loss"] = avg_loss_gen
    metrics["generative_training"]["throughput_splats_per_sec"] = dataset_size / epoch_time_gen

    # Retrieval Benchmark
    print("\n--- Semantic Router Retrieval Benchmark (1000 queries) ---")
    import numpy as np
    n_queries = 1000
    queries = mock_embeddings[:n_queries]
    
    # Warmup
    m2m.search(queries[0].unsqueeze(0), k=10)
    
    latencies = []
    for i in range(n_queries):
        q = queries[i].unsqueeze(0)
        start_q = time.perf_counter()
        m2m.search(q, k=10)
        latencies.append((time.perf_counter() - start_q) * 1000) # ms
        
    latencies = np.array(latencies)
    qps = n_queries / (np.sum(latencies) / 1000.0)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    
    print(f"Retrieval QPS: {qps:.2f}")
    print(f"p95 Latency:   {p95:.2f} ms")
    print(f"p99 Latency:   {p99:.2f} ms")
    print(f"Avg Latency:   {np.mean(latencies):.2f} ms")
    
    metrics["retrieval_routing"] = {
        "qps": float(qps),
        "p95_latency_ms": float(p95),
        "p99_latency_ms": float(p99),
        "avg_latency_ms": float(np.mean(latencies))
    }

    print("\n--- VALIDATION METRICS ---")
    print(json.dumps(metrics, indent=4))
    
    return metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--vulkan', action='store_true')
    args = parser.parse_args()
    
    all_metrics = []
    if args.cpu:
        all_metrics.append(validate_data_lake(device='cpu', enable_vulkan=False))
    if args.vulkan:
        all_metrics.append(validate_data_lake(device='cuda' if torch.cuda.is_available() else 'cpu', enable_vulkan=True))
        
    if not args.cpu and not args.vulkan:
        # Run both by default
        all_metrics.append(validate_data_lake(device='cpu', enable_vulkan=False))
        all_metrics.append(validate_data_lake(device='cuda' if torch.cuda.is_available() else 'cpu', enable_vulkan=True))
        
    with open("data_lake_real_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=4)
