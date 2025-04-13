import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


# --- 1. Define Model --- #
# Let's use a simple multi-layer network for regression
class SimpleRegressionNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):
        super().__init__()
        self.layer_1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer_2 = nn.Linear(hidden_size, output_size)
        print("SimpleRegressionNet initialized.")

    def forward(self, x):
        x = self.layer_1(x)
        x = self.relu(x)
        x = self.layer_2(x)
        return x


# --- 2. Create Dummy Data, Dataset, DataLoader --- #
# Generate synthetic data for a simple regression problem
# y = X * W + b + noise

NUM_SAMPLES = 1000
INPUT_FEATURES = 5
HIDDEN_FEATURES = 10
OUTPUT_FEATURES = 1  # Regression -> 1 output value
BATCH_SIZE = 64

print(f"\nGenerating synthetic data...")
# True parameters (we want the model to learn these, approximately)
W_true = torch.randn(INPUT_FEATURES, OUTPUT_FEATURES) * 3
b_true = torch.randn(OUTPUT_FEATURES) * 2

X_data = torch.randn(NUM_SAMPLES, INPUT_FEATURES)
y_data = (
    X_data @ W_true + b_true + torch.randn(NUM_SAMPLES, OUTPUT_FEATURES) * 0.5
)  # Add noise

print(f" - X_data shape: {X_data.shape}")
print(f" - y_data shape: {y_data.shape}")


# Create a custom Dataset
class RegressionDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# Instantiate Dataset and DataLoader
train_dataset = RegressionDataset(X_data, y_data)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
print("Dataset and DataLoader created.")
print(f" - Num batches: {len(train_loader)}")

# --- 3. Instantiate the Model --- #
model = SimpleRegressionNet(
    input_size=INPUT_FEATURES, hidden_size=HIDDEN_FEATURES, output_size=OUTPUT_FEATURES
)
print(f"\nModel instantiated: {model}")

# --- 4. Define Loss Function --- #
# Mean Squared Error Loss for regression problems
criterion = nn.MSELoss()
print(f"\nLoss function defined: {criterion}")

# --- 5. Define Optimizer --- #
# Stochastic Gradient Descent (SGD)
learning_rate = 0.01
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
# Alternative: Adam optimizer
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
print(f"\nOptimizer defined: {optimizer}")
print(f" - Learning Rate: {learning_rate}")

print("\n--- Setup Complete --- ")
print("All components (Model, DataLoader, Loss, Optimizer) are ready.")
print("The next step is to implement the training loop using these components.")

# Note: This script only sets up the components.
# The actual training loop is implemented in the next example file.
