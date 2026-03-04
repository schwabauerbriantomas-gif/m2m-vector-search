import torch
import torch.nn as nn
import torch.optim as optim
import time
import sys
import os

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

def main():
    print("Initializing M2M Storage & Data Lake...")
    config = M2MConfig(
        device='cuda' if torch.cuda.is_available() else 'cpu',
        n_splats_init=5000,
        max_splats=10000,
        enable_vulkan=False
    )
    m2m = create_m2m(config)
    
    # Simulate data ingestion
    print("Ingesting 5000 splats into M2M Data Lake...")
    dummy_data = torch.randn(5000, config.latent_dim).to(config.device)
    m2m.add_splats(dummy_data)
    
    print("\n--- Standard Training Loop (SOC Importance Sampling) ---")
    dataloader = m2m.export_to_dataloader(batch_size=256, importance_sampling=True, generate_samples=False)
    
    model = SimpleMLP(input_dim=config.latent_dim).to(config.device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    start_time = time.time()
    for batch_idx, batch_mu in enumerate(dataloader):
        batch_mu = batch_mu.to(config.device)
        # Dummy targets
        targets = torch.randint(0, 10, (batch_mu.shape[0],)).to(config.device)
        
        optimizer.zero_grad()
        outputs = model(batch_mu)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
    print(f"Standard Epoch completed in {time.time() - start_time:.2f}s")
    
    print("\n--- Generative Training Loop (Langevin Augmentation) ---")
    gen_dataloader = m2m.export_to_dataloader(batch_size=256, generate_samples=True)
    
    start_time = time.time()
    for batch_idx, batch_mu in enumerate(gen_dataloader):
        batch_mu = batch_mu.to(config.device)
        targets = torch.randint(0, 10, (batch_mu.shape[0],)).to(config.device)
        
        optimizer.zero_grad()
        outputs = model(batch_mu)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
    print(f"Generative Epoch completed in {time.time() - start_time:.2f}s")
    print("\nTraining completed successfully using M2M Data Lake!")

if __name__ == '__main__':
    main()
