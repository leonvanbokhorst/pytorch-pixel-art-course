import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


# --- Re-define Components (or import) --- #
# Model Definition
class SimpleBinaryClassifier(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.layer_1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer_2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.relu(x)
        x = self.layer_2(x)
        return x


# Dataset Definition
class BinaryClassificationDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# Config and Data Generation
NUM_SAMPLES = 1000
INPUT_FEATURES = 10
HIDDEN_FEATURES = 8
BATCH_SIZE = 64
X_data = torch.randn(NUM_SAMPLES, INPUT_FEATURES)
true_boundary = X_data[:, : INPUT_FEATURES // 2].sum(dim=1) > 0
y_data_binary = true_boundary.float().unsqueeze(1)
train_dataset = BinaryClassificationDataset(X_data, y_data_binary)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Instantiate Model, Loss, Optimizer
model = SimpleBinaryClassifier(INPUT_FEATURES, HIDDEN_FEATURES)
criterion = nn.BCEWithLogitsLoss()  # Use BCEWithLogitsLoss
learning_rate = 0.05  # Might need adjustment
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
# ----------------------------------------- #

# --- Training Configuration --- #
num_epochs = 15
print(f"Starting binary classification training for {num_epochs} epochs...")

# --- The Training Loop --- #
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    epoch_correct = 0
    num_batches_processed = 0

    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()

        # Forward pass -> get raw logits
        outputs = model(batch_X)

        # Calculate loss using BCEWithLogitsLoss
        # Requires raw logits (outputs) and float labels (batch_y)
        loss = criterion(outputs, batch_y)

        loss.backward()
        optimizer.step()

        # --- Calculate Accuracy (within training loop for monitoring) --- #
        # Apply sigmoid to logits, then threshold at 0.5 to get predictions
        predicted_probs = torch.sigmoid(outputs)
        predicted_labels = (predicted_probs >= 0.5).float()
        correct_in_batch = (predicted_labels == batch_y).sum().item()
        epoch_correct += correct_in_batch
        # ---------------------------------------------------------------- #

        epoch_loss += loss.item()
        num_batches_processed += 1

    # Calculate average loss and accuracy for the epoch
    avg_epoch_loss = epoch_loss / num_batches_processed
    epoch_accuracy = epoch_correct / len(train_loader.dataset)
    print(
        f"Epoch {epoch+1}/{num_epochs} completed. Avg Loss: {avg_epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}"
    )

print("\nTraining finished.")

# Note: This loop includes accuracy calculation during training for demonstration.
# Often, accuracy is calculated separately on a validation set (Day 7).
