import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# --- Re-use Model Definition (e.g., from Day 4/6, adapted for classification) --- #
class SimpleClassificationNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.layer_1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer_2 = nn.Linear(hidden_size, num_classes)  # Output raw scores (logits)
        print("SimpleClassificationNet initialized.")
        # Note: No softmax here, CrossEntropyLoss prefers raw logits

    def forward(self, x):
        x = self.layer_1(x)
        x = self.relu(x)
        x = self.layer_2(x)
        return x


# --- Re-use Dataset Definition --- #
class ClassificationDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels  # Expecting integer labels for CrossEntropyLoss

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# --- Configuration & Assume Trained Model --- #
# Assume we have a model instance that was trained previously (like in Day 6)
# For this script, we'll just create a new instance, but imagine it holds trained weights.
INPUT_FEATURES = 10
HIDDEN_FEATURES = 15
NUM_CLASSES = 3  # Example: 3 classes
BATCH_SIZE = 32

model = SimpleClassificationNet(INPUT_FEATURES, HIDDEN_FEATURES, NUM_CLASSES)
print(f"\nModel instantiated: {model}")
# In a real scenario: model.load_state_dict(torch.load('path/to/trained_weights.pth'))

# --- Create Validation Data --- #
# IMPORTANT: This data should NOT have been used during training!
NUM_VAL_SAMPLES = 500
print(f"\nGenerating {NUM_VAL_SAMPLES} validation samples...")
X_val = torch.randn(NUM_VAL_SAMPLES, INPUT_FEATURES)
# Generate random integer labels (0, 1, or 2)
y_val = torch.randint(0, NUM_CLASSES, (NUM_VAL_SAMPLES,))

print(f" - X_val shape: {X_val.shape}")
print(f" - y_val shape: {y_val.shape}")
print(f" - Example y_val labels: {y_val[:10]}...")

# --- Create Validation Dataset and DataLoader --- #
val_dataset = ClassificationDataset(X_val, y_val)
# Shuffle is typically False for validation/testing
val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
print(f"Validation Dataset and DataLoader created.")
print(f" - Num validation batches: {len(val_loader)}")

# --- Define Loss Function (should match training) --- #
# CrossEntropyLoss is common for multi-class classification
# It combines LogSoftmax and NLLLoss
criterion = nn.CrossEntropyLoss()
print(f"\nLoss function defined: {criterion}")

print("\n--- Evaluation Setup Complete --- ")
print("Components ready for evaluation: Model, Validation DataLoader, Loss Criterion.")
print("The next step is to implement the evaluation loop.")

# Note: This script only sets up the components for evaluation.
# The actual evaluation loop is implemented in the next example file.
