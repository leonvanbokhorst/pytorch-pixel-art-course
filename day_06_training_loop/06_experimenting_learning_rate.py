import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import copy  # To easily reset model weights


# --- Re-define Components (or import) --- #
# Model Definition
class SimpleRegressionNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):
        super().__init__()
        # Use ParameterDict to easily re-initialize weights if needed later
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


# --- Training Function --- #
# Encapsulate the training loop in a function to avoid repetition
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
        print(f"  Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.4f}")
    print("Training complete.")
    return epoch_losses


# --- Experiment with Learning Rates --- #
num_epochs_short = 10  # Fewer epochs for quicker comparison

# 1. Standard Learning Rate
print("\n--- Training Run 1: Standard Learning Rate --- ")
lr_standard = 0.01
model_standard = copy.deepcopy(base_model)  # Use a fresh copy of the model
optimizer_standard = optim.SGD(model_standard.parameters(), lr=lr_standard)
losses_standard = train_model(
    model_standard, train_loader, criterion, optimizer_standard, num_epochs_short
)

# 2. High Learning Rate
print("\n--- Training Run 2: High Learning Rate --- ")
lr_high = 0.5  # Potentially too high
model_high = copy.deepcopy(base_model)  # Use a fresh copy
optimizer_high = optim.SGD(model_high.parameters(), lr=lr_high)
losses_high = train_model(
    model_high, train_loader, criterion, optimizer_high, num_epochs_short
)

# 3. Low Learning Rate
print("\n--- Training Run 3: Low Learning Rate --- ")
lr_low = 0.0001  # Potentially too low
model_low = copy.deepcopy(base_model)  # Use a fresh copy
optimizer_low = optim.SGD(model_low.parameters(), lr=lr_low)
losses_low = train_model(
    model_low, train_loader, criterion, optimizer_low, num_epochs_short
)

# --- Observations --- #
print("\n--- Learning Rate Comparison --- ")
print(f"Standard LR ({lr_standard}): Final Avg Loss = {losses_standard[-1]:.4f}")
print(f"High LR     ({lr_high}): Final Avg Loss = {losses_high[-1]:.4f}")
print(f"Low LR      ({lr_low}): Final Avg Loss = {losses_low[-1]:.4f}")

print("\nObservations:")
print("- Standard LR likely showed steady decrease in loss.")
print(
    "- High LR might show unstable loss (bouncing around) or even increasing loss (divergence)."
)
print("- Low LR likely showed very slow decrease in loss.")
print("Choosing an appropriate learning rate is crucial for effective training.")
