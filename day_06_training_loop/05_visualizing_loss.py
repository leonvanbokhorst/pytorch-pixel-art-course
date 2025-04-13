import matplotlib.pyplot as plt  # Import matplotlib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


# --- Re-define Components (Regression Setup) --- #
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
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

model = SimpleRegressionNet(INPUT_FEATURES, HIDDEN_FEATURES, OUTPUT_FEATURES)
criterion = nn.MSELoss()
learning_rate = 0.01
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
# --------------------------------------------- #

# --- Training Loop with Loss Recording --- #
num_epochs = 30  # Train a bit longer to see a curve
epoch_losses = []  # List to store average loss per epoch

print(f"Starting training for {num_epochs} epochs...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    num_batches = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        num_batches += 1

    avg_loss = running_loss / num_batches
    epoch_losses.append(avg_loss)  # Record the average loss for this epoch
    print(f"Epoch {epoch+1}/{num_epochs} completed. Average Loss: {avg_loss:.4f}")

print("\nTraining finished.")

# --- Plotting the Loss Curve --- #
print("\nPlotting the training loss...")

plt.figure(figsize=(10, 5))  # Create a figure
plt.plot(
    range(1, num_epochs + 1), epoch_losses, marker="o", linestyle="-"
)  # Plot losses vs epochs
plt.title("Training Loss Curve")  # Add title
plt.xlabel("Epoch")  # Add x-axis label
plt.ylabel("Average Loss")  # Add y-axis label
plt.grid(True)  # Add grid
plt.xticks(range(1, num_epochs + 1, max(1, num_epochs // 10)))  # Adjust x-axis ticks
# Save the plot to a file (optional)
# plt.savefig("training_loss_curve.png")
# print("Plot saved as training_loss_curve.png")
plt.show()  # Display the plot

print("\nVisualizing the loss curve helps understand training progress:")
print("- Steadily decreasing loss is good.")
print("- Flat loss might mean learning has stalled (LR too low? problem too hard?).")
print("- Jumpy/increasing loss might mean LR is too high.")
