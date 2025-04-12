import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter  # Import SummaryWriter
import os


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

# --- Setup TensorBoard --- #
# Create a SummaryWriter instance.
# Logs will be saved in a directory named 'runs/experiment_1' (or similar)
# You can specify a log_dir: SummaryWriter('logs/my_experiment')
log_dir = os.path.join("runs", "day6_tensorboard_demo")
writer = SummaryWriter(log_dir=log_dir)
print(f"TensorBoard logs will be saved in: {log_dir}")

# --- Training Loop with TensorBoard Logging --- #
num_epochs = 25
print(f"\nStarting training for {num_epochs} epochs with TensorBoard logging...")

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

    avg_epoch_loss = running_loss / num_batches
    print(f"Epoch {epoch+1}/{num_epochs} completed. Average Loss: {avg_epoch_loss:.4f}")

    # --- Log loss to TensorBoard --- #
    # Use writer.add_scalar(tag, scalar_value, global_step)
    # tag: Name of the scalar (e.g., 'Loss/train', 'Accuracy/validation')
    # scalar_value: The value to log
    # global_step: Typically the epoch number or batch number
    writer.add_scalar("Loss/train", avg_epoch_loss, epoch + 1)
    # You could also log learning rate, gradients, accuracy, etc.
    # writer.add_scalar('LearningRate', learning_rate, epoch + 1)
    # ------------------------------- #

print("\nTraining finished.")

# --- Close the TensorBoard Writer --- #
# Important to close the writer to flush all pending events to disk
writer.close()
print("TensorBoard writer closed.")

# --- How to View TensorBoard --- #
print("\nTo view TensorBoard:")
print("1. Open your terminal.")
print(f"2. Navigate to the parent directory of '{log_dir}' (likely your project root).")
print(f"3. Run the command: tensorboard --logdir={os.path.dirname(log_dir)}")
print("4. Open the URL provided in your browser (usually http://localhost:6006/).")
