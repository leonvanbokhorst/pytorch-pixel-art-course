import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# --- 1. Define the Target Device --- #
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Selected device: CUDA (NVIDIA GPU)")
elif torch.backends.mps.is_available():
    # Avoid MPS if stability issues arise
    # device = torch.device("mps")
    # print("Selected device: MPS (Apple Silicon GPU)")
    device = torch.device("cpu")  # Fallback to CPU
    print("Selected device: CPU (MPS available but using CPU for stability)")
else:
    device = torch.device("cpu")
    print("Selected device: CPU")
print(f"Using device: {device}")


# --- Re-define Components (Classification Setup) --- #
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


# Config and Validation Data Generation
INPUT_FEATURES = 15
HIDDEN_FEATURES = 25
NUM_CLASSES = 5
BATCH_SIZE = 64
NUM_VAL_SAMPLES = 1000
# Generate data on CPU first
X_val = torch.randn(NUM_VAL_SAMPLES, INPUT_FEATURES, device="cpu")
y_val = torch.randint(0, NUM_CLASSES, (NUM_VAL_SAMPLES,), device="cpu")
val_dataset = ClassificationDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Instantiate Model and Criterion
# Assume model has been trained and weights loaded
model = SimpleClassificationNet(INPUT_FEATURES, HIDDEN_FEATURES, NUM_CLASSES)
# model.load_state_dict(torch.load('trained_weights.pth')) # Load weights in practice
criterion = nn.CrossEntropyLoss()
# ---------------------------------------------------- #

# --- Move Model to Device (BEFORE evaluation loop) --- #
print(f"\nMoving model to device: {device}...")
model = model.to(device)
print(f"Model parameters are on: {next(model.parameters()).device}")

# --- Modified Evaluation Loop --- #
print(f"\nStarting evaluation on {device}...")
start_eval_time = time.time()

model.eval()  # Set model to evaluation mode

# Initialize metrics
total_val_loss = 0.0
total_correct = 0
num_samples_processed = 0

with torch.no_grad():  # Disable gradient calculations
    for batch_idx, (batch_X, batch_y) in enumerate(val_loader):
        # --- Move Data Batch to Device (INSIDE batch loop) --- #
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        # ------------------------------------------------------- #

        # Forward pass (on the specified device)
        outputs = model(batch_X)

        # Calculate loss
        loss = criterion(outputs, batch_y)
        total_val_loss += loss.item() * batch_X.size(0)

        # Calculate Accuracy
        predicted_labels = torch.argmax(outputs, dim=1)
        total_correct += (predicted_labels == batch_y).sum().item()

        num_samples_processed += batch_X.size(0)

        # Verify tensors are on the correct device in the first batch
        if batch_idx == 0:
            print(
                f"  Inside loop: batch_X device: {batch_X.device}, batch_y device: {batch_y.device}"
            )
            print(f"  Inside loop: model output device: {outputs.device}")

# Calculate Average Metrics
avg_val_loss = total_val_loss / num_samples_processed
accuracy = total_correct / num_samples_processed

end_eval_time = time.time()
print("\nEvaluation finished.")
print(f"Total evaluation time: {end_eval_time - start_eval_time:.2f} seconds")
print(f"Average Validation Loss: {avg_val_loss:.4f}")
print(
    f"Validation Accuracy: {accuracy:.4f} ({total_correct}/{num_samples_processed} correct)"
)

print(
    "\nKey change: Added `.to(device)` for model (once) and data batches (inside loop).",
    " Used `model.eval()` and `torch.no_grad()` context.",
)
