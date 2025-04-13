import time  # To see shuffle effect changes over runs

import torch
from torch.utils.data import DataLoader, Dataset


# --- Re-define Custom Dataset (or import) --- #
class SimpleTensorDataset(Dataset):
    def __init__(self, num_samples=12, feature_dim=4):
        self.num_samples = num_samples
        print(f"Generating {num_samples} dataset samples...")
        # Create predictable features based on index for easier shuffle tracking
        self.features = torch.arange(num_samples).float().unsqueeze(1) * 10
        self.features = self.features.repeat(
            1, feature_dim
        )  # [0,0,0,0], [10,10,10,10], ...
        self.labels = torch.arange(num_samples).float()  # Label is just the index

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Return index along with data to track shuffling
        return self.features[idx], self.labels[idx], idx


# --------------------------------------------- #

# --- Configuration --- #
NUM_SAMPLES = 12
FEATURE_DIM = 4

# --- Create Dataset Instance --- #
dataset = SimpleTensorDataset(num_samples=NUM_SAMPLES, feature_dim=FEATURE_DIM)
print(f"\nDataset created. Length: {len(dataset)}")

# --- Option 1: Default Batching (batch_size=1, shuffle=False) --- #
# This is technically possible but rarely used. Not shown here.

# --- Option 2: Basic Batching (batch_size=4, shuffle=False) --- #
print("\n--- DataLoader 1: batch_size=4, shuffle=False ---")
dataloader_bs4_noshuffle = DataLoader(dataset=dataset, batch_size=4, shuffle=False)

for batch_idx, (features, labels, indices) in enumerate(dataloader_bs4_noshuffle):
    print(f" Batch {batch_idx + 1}:")
    print(f"   Features shape: {features.shape}")  # [4, 4]
    print(f"   Labels shape: {labels.shape}")  # [4]
    print(
        f"   Original Indices: {indices}"
    )  # Should be sequential [0, 1, 2, 3], [4, 5, 6, 7], ...

# --- Option 3: Batching with Shuffling (batch_size=4, shuffle=True) --- #
print("\n--- DataLoader 2: batch_size=4, shuffle=True ---")
# Set a manual seed for reproducibility *of this specific shuffle order*
# Remove or change seed to see different shuffles across runs.
torch.manual_seed(int(time.time()))  # Use time for different shuffles each run
dataloader_bs4_shuffle = DataLoader(dataset=dataset, batch_size=4, shuffle=True)

for batch_idx, (features, labels, indices) in enumerate(dataloader_bs4_shuffle):
    print(f" Batch {batch_idx + 1}:")
    print(f"   Features shape: {features.shape}")  # [4, 4]
    print(f"   Labels shape: {labels.shape}")  # [4]
    print(f"   Original Indices: {indices}")  # Should be shuffled!

# --- Option 4: Different Batch Size (batch_size=3, shuffle=False) --- #
print("\n--- DataLoader 3: batch_size=3, shuffle=False ---")
dataloader_bs3_noshuffle = DataLoader(dataset=dataset, batch_size=3, shuffle=False)
num_batches_bs3 = (NUM_SAMPLES + 3 - 1) // 3
print(f" Expected number of batches: {num_batches_bs3}")

for batch_idx, (features, labels, indices) in enumerate(dataloader_bs3_noshuffle):
    print(f" Batch {batch_idx + 1}:")
    print(f"   Features shape: {features.shape}")  # [3, 4]
    print(f"   Labels shape: {labels.shape}")  # [3]
    print(f"   Original Indices: {indices}")  # Sequential [0, 1, 2], [3, 4, 5], ...

print("\nKey Takeaways:")
print("- `batch_size` controls how many samples are grouped together.")
print("- `shuffle=True` randomizes the order of samples *before* batching each epoch.")
