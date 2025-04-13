import torch
from torch.utils.data import DataLoader, Dataset


# --- Re-define Custom Dataset (or import from 01_creating_custom_dataset.py) --- #
# For simplicity in this example, we redefine it here.
class SimpleTensorDataset(Dataset):
    def __init__(self, num_samples=20, feature_dim=5, label_dim=1):
        self.num_samples = num_samples
        print(f"Generating {num_samples} dataset samples...")
        self.features = torch.randn(num_samples, feature_dim)
        self.labels = torch.randint(0, 2, (num_samples, label_dim)).float()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ---------------------------------------------------------------------------- #

# --- Configuration --- #
NUM_SAMPLES = 25  # Total samples in the dataset
BATCH_SIZE = 5  # How many samples per batch
FEATURE_DIM = 8
LABEL_DIM = 1

print(f"Total Samples: {NUM_SAMPLES}")
print(f"Batch Size: {BATCH_SIZE}")
num_batches = (NUM_SAMPLES + BATCH_SIZE - 1) // BATCH_SIZE  # Calculate expected batches
print(f"Expected Number of Batches: {num_batches}")

# --- Create Dataset Instance --- #
dataset = SimpleTensorDataset(
    num_samples=NUM_SAMPLES, feature_dim=FEATURE_DIM, label_dim=LABEL_DIM
)
print(f"\nDataset created. Length: {len(dataset)}")

# --- Create DataLoader Instance --- #
# Wrap the Dataset in a DataLoader
# Key arguments:
#   dataset: The dataset object to load from.
#   batch_size: How many samples per batch to load.
#   shuffle: Set to True to have the data reshuffled at every epoch (default: False).
#   num_workers: How many subprocesses to use for data loading (default: 0 -> main process).
data_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=False)
print(f"\nDataLoader created.")
print(f" - Batch size: {BATCH_SIZE}")
print(" - Shuffle: False")

# --- Iterate over the DataLoader --- #
print("\nIterating through the DataLoader...")
for batch_idx, (features_batch, labels_batch) in enumerate(data_loader):
    print(f"\nBatch {batch_idx + 1}/{num_batches}")
    print(
        f"  Features batch shape: {features_batch.shape}"
    )  # Should be [BATCH_SIZE, FEATURE_DIM]
    print(
        f"  Labels batch shape: {labels_batch.shape}"
    )  # Should be [BATCH_SIZE, LABEL_DIM]

    # In a training loop, you would typically move data to a device (CPU/GPU)
    # and feed it into your model here.
    # e.g., features_batch = features_batch.to(device)
    #        labels_batch = labels_batch.to(device)
    #        outputs = model(features_batch)
    #        loss = criterion(outputs, labels_batch)
    #        ...etc

print("\nFinished iterating through the DataLoader.")
print(
    f"Successfully retrieved data in batches of size {BATCH_SIZE} (last batch might be smaller)."
)
