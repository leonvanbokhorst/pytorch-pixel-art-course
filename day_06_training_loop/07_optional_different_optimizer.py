import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


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

# Instantiate Model, Loss
base_model = SimpleRegressionNet(INPUT_FEATURES, HIDDEN_FEATURES, OUTPUT_FEATURES)
criterion = nn.MSELoss()
# ----------------------------------------- #


# --- Training Function (copied from previous example) --- #
def train_model(
    model_to_train, data_loader, loss_criterion, optimizer_instance, num_epochs=10
):
    print(
        f"Training with Optimizer: {optimizer_instance.__class__.__name__}, LR: {optimizer_instance.defaults['lr']}"
    )
    epoch_losses = []
    for epoch in range(num_epochs):
        model_to_train.train()
        running_loss = 0.0
        batches = 0
        for batch_X, batch_y in data_loader:
            optimizer_instance.zero_grad()
            outputs = model_to_train(batch_X)
            loss = loss_criterion(outputs, batch_y)
            loss.backward()
            optimizer_instance.step()
            running_loss += loss.item()
            batches += 1
        avg_loss = running_loss / batches
        epoch_losses.append(avg_loss)
        # Only print first and last epoch loss for brevity
        if epoch == 0 or epoch == num_epochs - 1:
            print(f"  Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.4f}")
        elif (epoch + 1) % 5 == 0:  # Print every 5 epochs
            print(
                f"  Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.4f} (intermediate)"
            )
    print("Training complete.")
    return epoch_losses


# --- Experiment with Different Optimizers --- #
num_epochs_compare = 20  # Use a few more epochs to see differences
lr_common = 0.01  # Use the same LR for a direct comparison (Adam often uses smaller LR)

# 1. SGD Optimizer
print("\n--- Training Run 1: SGD Optimizer --- ")
model_sgd = copy.deepcopy(base_model)  # Fresh model
optimizer_sgd = optim.SGD(model_sgd.parameters(), lr=lr_common)
losses_sgd = train_model(
    model_sgd, train_loader, criterion, optimizer_sgd, num_epochs_compare
)

# 2. Adam Optimizer
print("\n--- Training Run 2: Adam Optimizer --- ")
model_adam = copy.deepcopy(base_model)  # Fresh model
# Adam default LR is often 0.001, but we use 0.01 here for comparison
optimizer_adam = optim.Adam(model_adam.parameters(), lr=lr_common)
losses_adam = train_model(
    model_adam, train_loader, criterion, optimizer_adam, num_epochs_compare
)

# --- Observations --- #
print("\n--- Optimizer Comparison (LR = {lr_common}) --- ")
print(f"SGD Final Avg Loss:  {losses_sgd[-1]:.4f}")
print(f"Adam Final Avg Loss: {losses_adam[-1]:.4f}")

# You can plot these losses using matplotlib to visualize the convergence
# import matplotlib.pyplot as plt
# plt.plot(range(1, num_epochs_compare + 1), losses_sgd, label='SGD')
# plt.plot(range(1, num_epochs_compare + 1), losses_adam, label='Adam')
# plt.xlabel('Epoch')
# plt.ylabel('Average Loss')
# plt.title('SGD vs Adam Convergence')
# plt.legend()
# plt.grid(True)
# plt.show()

print("\nObservations:")
print("- Adam often converges faster (lower loss in fewer epochs) than basic SGD,")
print("  especially on more complex problems or with default learning rates.")
print(
    "- However, the best optimizer and its hyperparameters (like learning rate) can depend on the specific problem and dataset."
)
