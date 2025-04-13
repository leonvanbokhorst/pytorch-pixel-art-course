import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# --- 1. Define the Target Device --- #
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Selected device: CUDA (NVIDIA GPU)")
elif torch.backends.mps.is_available():
    # Avoid MPS for this specific simple regression example if having issues
    # device = torch.device("mps")
    # print("Selected device: MPS (Apple Silicon GPU)")
    device = torch.device("cpu")  # Fallback to CPU for simplicity here
    print("Selected device: CPU (MPS available but using CPU for stability)")
else:
    device = torch.device("cpu")
    print("Selected device: CPU")
print(f"Using device: {device}")


# --- Re-define Components (Model, Data, etc.) --- #
# Model Definition
class SimpleRegressionNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):
        super().__init__()
        self.layer_1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer_2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.relu(x)
        x = self.layer_2(x)
        return x


# Dataset Definition
class RegressionDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# Config and Data Generation
NUM_SAMPLES = 5000  # Slightly larger dataset
INPUT_FEATURES = 20
HIDDEN_FEATURES = 50
OUTPUT_FEATURES = 1
BATCH_SIZE = 128
W_true = torch.randn(INPUT_FEATURES, OUTPUT_FEATURES, device="cpu") * 3  # Start on CPU
b_true = torch.randn(OUTPUT_FEATURES, device="cpu") * 2
X_data = torch.randn(NUM_SAMPLES, INPUT_FEATURES, device="cpu")
y_data = (
    X_data @ W_true
    + b_true
    + torch.randn(NUM_SAMPLES, OUTPUT_FEATURES, device="cpu") * 0.5
)
train_dataset = RegressionDataset(X_data, y_data)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Instantiate Model, Loss, Optimizer
model = SimpleRegressionNet(INPUT_FEATURES, HIDDEN_FEATURES, OUTPUT_FEATURES)
criterion = nn.MSELoss()
learning_rate = 0.001  # Adam often needs smaller LR
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Using Adam
# ---------------------------------------------------- #

# --- Move Model to Device (BEFORE training loop) --- #
print(f"\nMoving model to device: {device}...")
model = model.to(device)
print(f"Model parameters are on: {next(model.parameters()).device}")
# Loss function is typically stateless, but if it had parameters, move it too:
# criterion = criterion.to(device)

# --- Modified Training Loop --- #
num_epochs = 5  # Fewer epochs for quick demo
print(f"\nStarting training for {num_epochs} epochs on {device}...")
start_loop_time = time.time()

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    num_batches_processed = 0

    for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
        # --- Move Data Batch to Device (INSIDE batch loop) --- #
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        # ------------------------------------------------------- #

        # Rest of the loop is the same, but operations happen on 'device'
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        num_batches_processed += 1

        # Verify tensors are on the correct device in the first batch
        if epoch == 0 and batch_idx == 0:
            print(
                f"  Inside loop: batch_X device: {batch_X.device}, batch_y device: {batch_y.device}"
            )
            print(f"  Inside loop: model output device: {outputs.device}")

    avg_epoch_loss = epoch_loss / num_batches_processed
    print(f"Epoch {epoch+1}/{num_epochs} completed. Average Loss: {avg_epoch_loss:.4f}")

end_loop_time = time.time()
print("\nTraining finished.")
print(f"Total training time: {end_loop_time - start_loop_time:.2f} seconds")

print(
    "\nKey change: Added `.to(device)` for model (once) and data batches (inside loop)."
)
