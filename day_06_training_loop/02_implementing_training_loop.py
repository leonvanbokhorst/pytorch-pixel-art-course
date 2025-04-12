import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# --- Re-define Components (or import) --- #
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


# Data Generation & Dataset Definition
class RegressionDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


NUM_SAMPLES = 1000
INPUT_FEATURES = 5
HIDDEN_FEATURES = 10
OUTPUT_FEATURES = 1
BATCH_SIZE = 64
W_true = torch.randn(INPUT_FEATURES, OUTPUT_FEATURES) * 3
b_true = torch.randn(OUTPUT_FEATURES) * 2
X_data = torch.randn(NUM_SAMPLES, INPUT_FEATURES)
y_data = X_data @ W_true + b_true + torch.randn(NUM_SAMPLES, OUTPUT_FEATURES) * 0.5
train_dataset = RegressionDataset(X_data, y_data)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Instantiate Model, Loss, Optimizer
model = SimpleRegressionNet(INPUT_FEATURES, HIDDEN_FEATURES, OUTPUT_FEATURES)
criterion = nn.MSELoss()
learning_rate = 0.01
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
# ----------------------------------------- #

# --- Training Configuration --- #
num_epochs = 15  # Number of times to iterate over the entire dataset
print(f"Starting training for {num_epochs} epochs...")

# --- The Training Loop --- #
for epoch in range(num_epochs):
    # Set the model to training mode
    # This is important for layers like Dropout and BatchNorm, which behave differently during training and evaluation
    model.train()

    epoch_loss = 0.0  # Track loss for this epoch
    num_batches_processed = 0

    # Iterate over batches of data provided by the DataLoader
    for batch_X, batch_y in train_loader:
        # --- Core Training Steps --- #

        # 1. Zero the gradients
        # Gradients accumulate by default, so we need to clear them before computing gradients for the current batch
        optimizer.zero_grad()

        # 2. Forward pass
        # Pass the batch of features through the model to get predictions
        outputs = model(batch_X)

        # 3. Calculate loss
        # Compare model outputs with the true labels using the loss function
        loss = criterion(outputs, batch_y)

        # 4. Backward pass
        # Compute gradients of the loss with respect to model parameters (requires_grad=True)
        loss.backward()

        # 5. Update weights (Optimizer step)
        # Adjust model parameters based on the computed gradients and the optimizer algorithm (e.g., SGD)
        optimizer.step()

        # --- End of Core Training Steps --- #

        # Accumulate loss for monitoring
        # loss.item() gets the scalar value of the loss tensor
        epoch_loss += loss.item()
        num_batches_processed += 1

    # Calculate average loss for the epoch
    avg_epoch_loss = epoch_loss / num_batches_processed
    print(f"Epoch {epoch+1}/{num_epochs} completed. Average Loss: {avg_epoch_loss:.4f}")

print("\nTraining finished.")
print(f"Final loss after {num_epochs} epochs: {avg_epoch_loss:.4f}")

# After training, you would typically evaluate the model on a separate test set (see Day 7).
