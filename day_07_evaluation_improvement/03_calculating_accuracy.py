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
# Assume this model instance has trained weights loaded!
model = SimpleClassificationNet(INPUT_FEATURES, HIDDEN_FEATURES, NUM_CLASSES)
# model.load_state_dict(torch.load('trained_weights.pth')) # Load weights here in practice
criterion = nn.CrossEntropyLoss()
# -------------------------------------------- #

print("--- Starting Evaluation with Accuracy Calculation --- ")

# --- The Evaluation Loop with Accuracy --- #
model.eval()
print("Model set to evaluation mode.")

# Initialize metrics
total_val_loss = 0.0
total_correct = 0
num_samples_processed = 0

print("Entering torch.no_grad() context...")
with torch.no_grad():
    for batch_idx, (batch_X, batch_y) in enumerate(val_loader):
        # Forward pass
        outputs = model(batch_X)  # Shape: [batch_size, num_classes] (logits)

        # Calculate loss (optional, but often useful)
        loss = criterion(outputs, batch_y)
        total_val_loss += loss.item() * batch_X.size(0)

        # --- Calculate Accuracy --- #
        # Get predicted class index by finding the max logit score
        predicted_labels = torch.argmax(outputs, dim=1)  # Shape: [batch_size]

        # Compare predicted labels with true labels
        correct_in_batch = (predicted_labels == batch_y).sum().item()
        total_correct += correct_in_batch
        # -------------------------- #

        num_samples_processed += batch_X.size(0)

        if batch_idx == 0:
            print("  Processed first batch.")
            print(f"    Example Outputs (logits): {outputs[0]}")
            print(f"    Example Predicted Label: {predicted_labels[0]}")
            print(f"    Example True Label: {batch_y[0]}")
            print(f"    Correct in batch: {correct_in_batch}/{batch_X.size(0)}")

# --- Calculate Average Metrics --- #
avg_val_loss = total_val_loss / num_samples_processed
accuracy = total_correct / num_samples_processed

print("\n--- Evaluation Complete --- ")
print(f"Number of validation samples processed: {num_samples_processed}")
print(f"Average Validation Loss: {avg_val_loss:.4f}")
print(
    f"Validation Accuracy: {accuracy:.4f} ({total_correct}/{num_samples_processed} correct)"
)

# Note: Since the model is untrained (using initialized weights),
# the accuracy will likely be close to random chance (1 / NUM_CLASSES).
# For NUM_CLASSES=3, random chance is ~0.333.
