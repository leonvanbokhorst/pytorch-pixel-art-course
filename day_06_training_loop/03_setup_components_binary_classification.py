import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# --- 1. Define Model for Binary Classification --- #
# Output layer should have 1 output neuron (raw logit)
class SimpleBinaryClassifier(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.layer_1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer_2 = nn.Linear(hidden_size, 1)  # Single output logit
        print("SimpleBinaryClassifier initialized.")

    def forward(self, x):
        x = self.layer_1(x)
        x = self.relu(x)
        x = self.layer_2(x)
        return x  # Return raw logits


# --- 2. Create Dummy Data, Dataset, DataLoader --- #
# Generate synthetic data for binary classification

NUM_SAMPLES = 1000
INPUT_FEATURES = 10
HIDDEN_FEATURES = 8
BATCH_SIZE = 64

print(f"\nGenerating synthetic binary classification data...")
# Create features
X_data = torch.randn(NUM_SAMPLES, INPUT_FEATURES)

# Create true underlying relationship (e.g., based on sum of first few features)
# This is just to make the data learnable, not necessarily realistic
true_boundary = X_data[:, : INPUT_FEATURES // 2].sum(dim=1) > 0
y_data_binary = true_boundary.float().unsqueeze(
    1
)  # Convert boolean to float 0.0 or 1.0

print(f" - X_data shape: {X_data.shape}")
print(f" - y_data shape: {y_data_binary.shape}")
print(f" - Example y_data: {y_data_binary[:10].view(-1)}")  # Show some 0s and 1s
print(f" - Class balance: {y_data_binary.mean():.2f} (Fraction of class 1)")


# Create Dataset
class BinaryClassificationDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = (
            labels  # Labels should be float (0.0 or 1.0) for BCEWithLogitsLoss
        )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# Instantiate Dataset and DataLoader
train_dataset = BinaryClassificationDataset(X_data, y_data_binary)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
print("Dataset and DataLoader created.")

# --- 3. Instantiate the Model --- #
model = SimpleBinaryClassifier(input_size=INPUT_FEATURES, hidden_size=HIDDEN_FEATURES)
print(f"\nModel instantiated: {model}")

# --- 4. Define Loss Function --- #
# Binary Cross Entropy with Logits Loss
# Combines a Sigmoid layer and BCELoss in one single class.
# More numerically stable than using a plain Sigmoid followed by BCELoss.
# Expects RAW LOGITS from the model and FLOAT labels (0.0 or 1.0).
criterion = nn.BCEWithLogitsLoss()
print(f"\nLoss function defined: {criterion}")

# --- 5. Define Optimizer --- #
learning_rate = 0.01
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
print(f"\nOptimizer defined: {optimizer}")

print("\n--- Binary Classification Setup Complete --- ")
print("Components ready for binary classification training loop.")
