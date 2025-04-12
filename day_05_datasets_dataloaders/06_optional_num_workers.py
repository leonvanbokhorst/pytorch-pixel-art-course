import torch
from torch.utils.data import Dataset, DataLoader
import time
import os  # To get cpu count


# --- Re-define Custom Dataset (or import) --- #
class SimpleTensorDataset(Dataset):
    def __init__(self, num_samples=500, feature_dim=100):
        self.num_samples = num_samples
        # Simulate some work during init
        print(
            f"Generating {num_samples} complex dataset samples (simulating load time)..."
        )
        time.sleep(0.1)
        self.features = torch.randn(num_samples, feature_dim)
        self.labels = torch.rand(num_samples, 1) > 0.5  # Binary labels
        print("Dataset generated.")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Simulate some work needed to load/process a single item
        time.sleep(0.001)  # Tiny delay per item
        return self.features[idx], self.labels[idx]


# --------------------------------------------- #

# --- Configuration --- #
NUM_SAMPLES = 2000  # Use more samples to potentially see effect
FEATURE_DIM = 256
BATCH_SIZE = 64

# --- Main Execution Guard --- #
if __name__ == "__main__":
    # --- Create Dataset Instance --- #
    dataset = SimpleTensorDataset(num_samples=NUM_SAMPLES, feature_dim=FEATURE_DIM)
    print(f"\nDataset created. Length: {len(dataset)}")

    # --- DataLoader with num_workers=0 (Default) --- #
    print(f"\n--- DataLoader 1: num_workers=0 (Main Process) ---")
    dataloader_nw0 = DataLoader(
        dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )

    start_time = time.time()
    print("Iterating through DataLoader (num_workers=0)...")
    for batch_idx, (features, labels) in enumerate(dataloader_nw0):
        # Simulate some minimal work with the batch
        _ = features.mean()
        if batch_idx == 0:
            print(f"  First batch feature shape: {features.shape}")
    end_time = time.time()
    duration_nw0 = end_time - start_time
    print(f"Iteration finished. Duration: {duration_nw0:.4f} seconds")

    # --- DataLoader with num_workers > 0 --- #
    # Use a reasonable number of workers, e.g., number of CPU cores
    # Be cautious on Windows or in interactive environments (like Jupyter)
    # where multiprocessing can have overhead or issues.
    num_cpu_cores = os.cpu_count() or 2  # Get CPU cores, default to 2 if unknown
    NUM_WORKERS = min(4, num_cpu_cores)  # Cap at 4 for this example

    print(f"\n--- DataLoader 2: num_workers={NUM_WORKERS} (Parallel Loading) --- ")

    # NOTE: Creating the DataLoader with num_workers > 0 might take a moment
    # as it needs to spawn the worker processes.
    print(f"Creating DataLoader with {NUM_WORKERS} workers...")
    dataloader_nwN = DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        # pin_memory=True # Often used with GPU for faster transfer
    )

    start_time = time.time()
    print(f"Iterating through DataLoader (num_workers={NUM_WORKERS})...")
    for batch_idx, (features, labels) in enumerate(dataloader_nwN):
        # Simulate some minimal work with the batch
        _ = features.mean()
        if batch_idx == 0:
            print(f"  First batch feature shape: {features.shape}")
    end_time = time.time()
    duration_nwN = end_time - start_time
    print(f"Iteration finished. Duration: {duration_nwN:.4f} seconds")

    print("\n--- Comparison --- ")
    print(f"Time with num_workers=0: {duration_nw0:.4f}s")
    print(f"Time with num_workers={NUM_WORKERS}: {duration_nwN:.4f}s")

    if duration_nwN < duration_nw0:
        print(
            "Using multiple workers resulted in faster data loading (as expected in many cases)."
        )
    elif duration_nwN > duration_nw0:
        print("Using multiple workers was SLOWER. This can happen due to:")
        print(
            "  - Small dataset/fast __getitem__: Overhead of multiprocessing outweighs benefits."
        )
        print(
            "  - System constraints or specific OS behavior (e.g., Windows limitations)."
        )
        print("  - High CPU usage from other processes.")
    else:
        print("No significant time difference observed.")

    print("\n`num_workers > 0` allows data loading to happen in parallel processes,")
    print("which can significantly speed up training if data loading is a bottleneck,")
    print("especially with complex preprocessing in `__getitem__` or slow disk I/O.")
    print(
        "However, it introduces overhead and might not always be faster for simple cases."
    )
