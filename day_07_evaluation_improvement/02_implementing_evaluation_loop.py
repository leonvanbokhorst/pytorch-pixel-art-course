import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# --- Re-define/Import Components from Setup --- #
# Model Definition
class SimpleClassificationNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.layer_1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer_2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.relu(x)
        x = self.layer_2(x)
        return x


# Dataset Definition
class ClassificationDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# Configuration and Data/Loader Creation
INPUT_FEATURES = 10
HIDDEN_FEATURES = 15
NUM_CLASSES = 3
BATCH_SIZE = 32
NUM_VAL_SAMPLES = 500
X_val = torch.randn(NUM_VAL_SAMPLES, INPUT_FEATURES)
y_val = torch.randint(0, NUM_CLASSES, (NUM_VAL_SAMPLES,))
val_dataset = ClassificationDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Instantiate Model and Criterion
# IMPORTANT: Assume this model instance has trained weights loaded!
# For demo, we just use the initialized weights.
model = SimpleClassificationNet(INPUT_FEATURES, HIDDEN_FEATURES, NUM_CLASSES)
# model.load_state_dict(torch.load('trained_weights.pth')) # Load weights here in practice
criterion = nn.CrossEntropyLoss()
# -------------------------------------------- #

print("--- Starting Evaluation --- ")

# --- The Evaluation Loop --- #

# 1. Set Model to Evaluation Mode
# This disables layers like Dropout and uses running stats for BatchNorm.
model.eval()
print("Model set to evaluation mode: model.eval()")

# Initialize metrics
total_val_loss = 0.0
num_samples_processed = 0

# 2. Use torch.no_grad() Context Manager
# This disables gradient calculations, saving memory and computation during inference.
print("Entering torch.no_grad() context...")
with torch.no_grad():
    # Iterate over the validation data loader
    for batch_idx, (batch_X, batch_y) in enumerate(val_loader):
        # Move data to the same device as the model if necessary (e.g., GPU)
        # batch_X = batch_X.to(device)
        # batch_y = batch_y.to(device)

        # Perform forward pass
        outputs = model(batch_X)

        # Calculate loss
        loss = criterion(outputs, batch_y)

        # Accumulate loss
        # Multiply by batch size because criterion usually returns average loss per sample in batch
        total_val_loss += loss.item() * batch_X.size(0)
        num_samples_processed += batch_X.size(0)

        if batch_idx == 0:
            print(
                f"  Processed first batch. Output shape: {outputs.shape}, Labels shape: {batch_y.shape}"
            )

# --- Calculate Average Metrics --- #
if num_samples_processed != len(val_loader.dataset):
    print(
        f"Warning: Processed {num_samples_processed} samples, but dataset size is {len(val_loader.dataset)}"
    )

avg_val_loss = total_val_loss / num_samples_processed

print("\n--- Evaluation Complete --- ")
print(f"Number of validation samples processed: {num_samples_processed}")
print(f"Average Validation Loss: {avg_val_loss:.4f}")

print("\nKey differences from training loop:")
print("- `model.eval()` is called before the loop.")
print("- The loop is wrapped in `with torch.no_grad():`.")
print("- No `optimizer.zero_grad()`, `loss.backward()`, or `optimizer.step()` calls.")
print("- Focus is on calculating metrics (like loss) on unseen data.")
print("\nNext step: Calculating specific metrics like accuracy.")
