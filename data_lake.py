import torch
from torch.utils.data import IterableDataset
import numpy as np
import threading
import queue

class M2MDataLake(IterableDataset):
    """
    M2M Data Lake for PyTorch Training.
    Provides Tier-Aware Streaming, SOC Importance Sampling, and Langevin Generative Augmentation.
    """
    def __init__(
        self,
        m2m_engine,
        batch_size: int = 32,
        importance_sampling: bool = False,
        generate_samples: bool = False,
        langevin_steps: int = 5,
        prefetch_batches: int = 2
    ):
        super().__init__()
        # Handle both M2MEngine and M2MMemory
        self.m2m = m2m_engine.m2m if hasattr(m2m_engine, 'm2m') else m2m_engine
        self.batch_size = batch_size
        self.importance_sampling = importance_sampling
        self.generate_samples = generate_samples
        self.langevin_steps = langevin_steps
        self.prefetch_batches = prefetch_batches
        
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        total_splats = self.m2m.splats.n_active
        
        if total_splats == 0:
            return iter([])
            
        if worker_info is None:
            start_idx = 0
            end_idx = total_splats
        else:
            per_worker = int(np.ceil(total_splats / float(worker_info.num_workers)))
            start_idx = worker_info.id * per_worker
            end_idx = min(start_idx + per_worker, total_splats)
            
        indices = list(range(start_idx, end_idx))
        
        # SOC Importance Sampling
        if self.importance_sampling:
            # Sort indices by kappa (concentration) to prioritize important memories
            # High kappa = more concentrated/certain = more important
            kappas = self.m2m.splats.kappa[start_idx:end_idx].cpu().numpy()
            sorted_relative_indices = np.argsort(kappas)[::-1] 
            indices = [indices[int(i)] for i in sorted_relative_indices]
            
        # Tier-Aware Streaming via Prefetch Queue
        preload_queue = queue.Queue(maxsize=self.prefetch_batches)
        stop_event = threading.Event()
        
        def prefetch_worker():
            for i in range(0, len(indices), self.batch_size):
                if stop_event.is_set():
                    break
                batch_indices = indices[i:i + self.batch_size]
                
                # Fetch mu from Hot Tier (VRAM)
                # In a real integrated Memory Manager, we'd trigger `prefetch_to_warm` here
                # Here we directly clone the active tensor segment.
                batch_mu = self.m2m.splats.mu[batch_indices].clone()
                
                preload_queue.put(batch_mu)
            preload_queue.put(None) # Sentinel
            
        thread = threading.Thread(target=prefetch_worker, daemon=True)
        thread.start()
        
        try:
            while True:
                batch_mu = preload_queue.get()
                if batch_mu is None:
                    break
                    
                if self.generate_samples:
                    # Langevin Generative Augmentation
                    # Generate novel variations of the splat
                    try:
                        generated = self.m2m.sample(batch_mu, n_steps=self.langevin_steps)
                        if not isinstance(generated, torch.Tensor):
                            raise ValueError("DummyModule detected")
                        batch_mu = generated
                    except Exception:
                        # Fallback if sample is a dummy module
                        batch_mu = batch_mu + torch.randn_like(batch_mu) * 0.01
                    
                yield batch_mu
        finally:
            stop_event.set()
