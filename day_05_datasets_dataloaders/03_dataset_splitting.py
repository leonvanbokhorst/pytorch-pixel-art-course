import torch
from torch.utils.data import DataLoader, Dataset, random_split


# --- Re-define a simple Dataset (or import) --- #
class SimpleTensorDataset(Dataset):
    def __init__(self, num_samples=1000, feature_dim=10):
        self.num_samples = num_samples
        self.features = torch.randn(num_samples, feature_dim)
        self.labels = torch.randint(0, 2, (num_samples, 1)).float()
        print(f"Generated dataset with {num_samples} samples.")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# --------------------------------------------- #

# --- Configuration --- #
TOTAL_SAMPLES = 1000
FEATURE_DIM = 20
TRAIN_FRACTION = 0.8  # Use 80% for training, 20% for validation

# --- Create the Full Dataset --- #
full_dataset = SimpleTensorDataset(num_samples=TOTAL_SAMPLES, feature_dim=FEATURE_DIM)
print(f"\nFull dataset length: {len(full_dataset)}")

# --- Calculate Split Sizes --- #
train_size = int(TRAIN_FRACTION * len(full_dataset))
val_size = len(full_dataset) - train_size  # Ensure total adds up

print("Splitting dataset into:")
print(f" - Training size: {train_size}")
print(f" - Validation size: {val_size}")

# --- Perform the Split --- #
# torch.random_split takes the dataset and a list of lengths for the splits.
# It returns a list of Dataset objects (subsets).
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

print("\nDataset split complete.")
print(f"Length of train_dataset: {len(train_dataset)}")
print(f"Length of val_dataset: {len(val_dataset)}")

# --- Verify the Split (Optional) --- #
# Accessing elements still works like a regular dataset
if len(train_dataset) > 0:
    train_feature_0, train_label_0 = train_dataset[0]
    print(f"\nFirst training sample feature shape: {train_feature_0.shape}")
if len(val_dataset) > 0:
    val_feature_0, val_label_0 = val_dataset[0]
    print(f"First validation sample feature shape: {val_feature_0.shape}")

# Check if the splits are indeed subsets (different objects)
print(f"Are train and val datasets the same object? {train_dataset is val_dataset}")

# --- Using the Split Datasets with DataLoaders --- #
print("\nCreating DataLoaders for the split datasets...")
BATCH_SIZE = 64
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False
)  # No shuffle for validation

print(f"Train DataLoader created. Num batches: {len(train_loader)}")
print(f"Validation DataLoader created. Num batches: {len(val_loader)}")

print("\n`random_split` is the standard way to create train/validation/test splits.")
print("It ensures no data leakage between the sets.")
