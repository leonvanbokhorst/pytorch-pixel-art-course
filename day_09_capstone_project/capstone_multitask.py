import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import time
import os
from torch.utils.tensorboard import SummaryWriter

# --- Device Setup --- #
# Check for CUDA (NVIDIA GPU)
cuda_available = torch.cuda.is_available()

# Check for MPS (Apple Silicon GPU)
mps_available = torch.backends.mps.is_available() and torch.backends.mps.is_built()

if cuda_available:
    device = torch.device("cuda")
    print("Selected device: CUDA (NVIDIA GPU)")
elif mps_available:
    device = torch.device("mps")
    print("Selected device: MPS (Apple Silicon GPU)")
else:
    device = torch.device("cpu")
    print("Selected device: CPU")

print(f"Device object: {device}")

# --- Configuration (ADAPT THESE FOR YOUR DOMAIN) --- #
# Data parameters
NUM_SAMPLES = 10000
NUM_VAL_SAMPLES = 1000
INPUT_DIM = 10  # Adjust based on your features
# Model parameters
HIDDEN_DIM = 32  # Adjust complexity as needed
NUM_CLASSES = 1  # For binary classification head (adjust if multi-class)
# Training parameters
BATCH_SIZE = 128
LEARNING_RATE = 0.005
NUM_EPOCHS = 20
# Other parameters
RANDOM_SEED = 42  # For reproducible data generation
SAVE_MODEL_PATH = "day_09_capstone_project/multitask_model_final.pth"
TENSORBOARD_LOG_DIR = "runs/capstone_multitask_experiment"

# Set seed for reproducibility
torch.manual_seed(RANDOM_SEED)
if cuda_available:
    torch.cuda.manual_seed_all(RANDOM_SEED)
# Note: MPS reproducibility might have limitations


# --- 1. Define the MultiTaskDataset (ADAPT THIS FOR YOUR DOMAIN) --- #
class MultiTaskDataset(Dataset):
    """Generates or loads data for multi-task learning.

    *** ADAPT THIS CLASS FOR YOUR DOMAIN ***
    - Modify `__init__` to load your data (from files, etc.) or keep generation.
    - Ensure `self.features`, `self.target_regression`, `self.target_classification` are populated correctly.
    - Adjust target calculation logic based on your specific tasks.
    - If you only have one task, you can remove the unused target and corresponding model head.
    """

    def __init__(self, num_samples, input_dim, is_validation=False):
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.half_dim = input_dim // 2

        # --- Domain Specific Data Loading/Generation START --- #
        # Example: Generate synthetic data (Replace with your data loading)
        print(
            f"Generating {'validation' if is_validation else 'training'} dataset with {num_samples} samples..."
        )
        self.features = torch.randn(num_samples, input_dim)

        # Example: Calculate regression target (Replace with your logic)
        self.target_regression = self.features[:, : self.half_dim].sum(
            dim=1, keepdim=True
        )

        # Example: Calculate classification target (Replace with your logic)
        sum_second_half = self.features[:, self.half_dim :].sum(dim=1)
        self.target_classification = (
            (sum_second_half > 0).float().unsqueeze(1)
        )  # Needs float and shape [N, 1]
        # --- Domain Specific Data Loading/Generation END --- #

        print(f"  Features shape: {self.features.shape}")
        print(f"  Regression target shape: {self.target_regression.shape}")
        print(f"  Classification target shape: {self.target_classification.shape}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # This part usually stays the same, just returns the data points
        return (
            self.features[idx],
            self.target_regression[idx],
            self.target_classification[idx],
        )


# --- 2. Define the MultiTaskNet model (ADAPT THIS FOR YOUR DOMAIN) --- #
class MultiTaskNet(nn.Module):
    """A neural network with a shared body and two task-specific heads.

    *** ADAPT THIS CLASS FOR YOUR DOMAIN ***
    - Adjust `input_dim`, `hidden_dim`, `num_classes` based on your data and tasks.
    - Modify the architecture (layers, activations) of the body or heads if needed.
    - If you only have one task, remove the unused head.
    """

    def __init__(self, input_dim, hidden_dim, num_classes=1):
        super().__init__()
        # --- Shared Body Architecture --- #
        self.body_layer1 = nn.Linear(input_dim, hidden_dim)
        self.body_relu1 = nn.ReLU()
        self.body_layer2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.body_relu2 = nn.ReLU()
        # --- Add more body layers if needed --- #

        # --- Regression Head --- #
        # (Adjust output size if your regression target is multi-dimensional)
        self.regression_head = nn.Linear(hidden_dim // 2, 1)

        # --- Classification Head --- #
        # (Adjust output size for multi-class classification)
        self.classification_head = nn.Linear(hidden_dim // 2, num_classes)

        print("MultiTaskNet initialized:")
        print(f"  Input Dim: {input_dim}, Hidden Dim: {hidden_dim}")
        print(f"  Regression Head Output: {self.regression_head.out_features}")
        print(f"  Classification Head Output: {self.classification_head.out_features}")

    def forward(self, x):
        # Pass input through the shared body
        body_output = self.body_relu1(self.body_layer1(x))
        body_output = self.body_relu2(self.body_layer2(body_output))
        # --- Pass through additional body layers if added --- #

        # Get outputs from each head
        regression_output = self.regression_head(body_output)
        classification_output = self.classification_head(body_output)

        # Return both outputs (if only one task, return just that one)
        return regression_output, classification_output


# --- 3. Training Loop function (Usually less adaptation needed) --- #
def train_epoch(
    model,
    dataloader,
    regression_criterion,
    classification_criterion,
    optimizer,
    device,
    epoch_num,
    loss_weights=(0.5, 0.5),
):
    """Runs a single training epoch.

    Args:
        loss_weights (tuple): Weights for combining regression and classification loss.
        # ... other args ...

    Returns:
        # ...
    """
    model.train()  # Set model to training mode
    total_regression_loss = 0.0
    total_classification_loss = 0.0
    total_combined_loss = 0.0
    num_batches = len(dataloader)

    start_time = time.time()
    for batch_idx, (features, target_reg, target_cls) in enumerate(dataloader):
        features, target_reg, target_cls = (
            features.to(device),
            target_reg.to(device),
            target_cls.to(device),
        )

        optimizer.zero_grad()
        output_reg, output_cls = model(features)

        loss_reg = regression_criterion(output_reg, target_reg)
        loss_cls = classification_criterion(output_cls, target_cls)

        # --- Combine Losses (ADAPT WEIGHTING STRATEGY IF NEEDED) --- #
        reg_weight, cls_weight = loss_weights
        combined_loss = (reg_weight * loss_reg) + (cls_weight * loss_cls)

        combined_loss.backward()
        optimizer.step()

        total_regression_loss += loss_reg.item()
        total_classification_loss += loss_cls.item()
        total_combined_loss += combined_loss.item()

        # Optional: Print progress
        if batch_idx % (num_batches // 5) == 0 and batch_idx > 0:
            elapsed = time.time() - start_time
            print(
                f"  Epoch {epoch_num} | Batch {batch_idx}/{num_batches} | Reg Loss: {loss_reg.item():.4f} | Cls Loss: {loss_cls.item():.4f} | Combined: {combined_loss.item():.4f} | Time: {elapsed:.2f}s"
            )

    avg_reg_loss = total_regression_loss / num_batches
    avg_cls_loss = total_classification_loss / num_batches
    avg_combined_loss = total_combined_loss / num_batches
    epoch_duration = time.time() - start_time

    print(f"Epoch {epoch_num} Training Summary:")
    print(
        f"  Avg Reg Loss: {avg_reg_loss:.4f}, Avg Cls Loss: {avg_cls_loss:.4f}, Avg Combined: {avg_combined_loss:.4f}"
    )
    print(f"  Duration: {epoch_duration:.2f}s")

    return avg_reg_loss, avg_cls_loss, avg_combined_loss


# --- 4. Evaluation Loop function (ADAPT METRICS FOR YOUR DOMAIN) --- #
def evaluate_model(
    model, dataloader, regression_criterion, classification_criterion, device
):
    """Evaluates the model on the validation set.

    *** ADAPT EVALUATION METRICS FOR YOUR DOMAIN ***
    - Keep loss calculation as is if using the same loss functions.
    - Modify accuracy calculation for multi-class classification.
    - Add other relevant metrics (e.g., R-squared for regression, F1-score, Precision/Recall for classification).
    """
    model.eval()
    total_regression_loss = 0.0
    total_classification_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    num_batches = len(dataloader)

    print("\nStarting evaluation...")
    start_time = time.time()
    with torch.no_grad():
        for features, target_reg, target_cls in dataloader:
            features, target_reg, target_cls = (
                features.to(device),
                target_reg.to(device),
                target_cls.to(device),
            )

            output_reg, output_cls = model(features)

            # Calculate losses
            loss_reg = regression_criterion(output_reg, target_reg)
            loss_cls = classification_criterion(output_cls, target_cls)
            total_regression_loss += loss_reg.item()
            total_classification_loss += loss_cls.item()

            # --- Calculate Classification Accuracy (Example for Binary) --- #
            # ADAPT or ADD metrics here
            preds = (
                torch.sigmoid(output_cls) > 0.5
            )  # Apply sigmoid to logits, then threshold
            correct_predictions += (preds == target_cls).sum().item()
            total_samples += target_cls.size(0)
            # --- Add other metric calculations here (e.g., R2 score) --- #

    # --- Average Metrics --- #
    avg_reg_loss = total_regression_loss / num_batches
    avg_cls_loss = total_classification_loss / num_batches
    accuracy = (correct_predictions / total_samples) * 100 if total_samples > 0 else 0
    eval_duration = time.time() - start_time

    # --- Print Summary (ADAPT this too) --- #
    print("Evaluation Summary:")
    print(
        f"  Avg Reg Loss (MSE): {avg_reg_loss:.4f}"
    )  # Make sure label matches criterion
    print(
        f"  Avg Cls Loss (BCE): {avg_cls_loss:.4f}"
    )  # Make sure label matches criterion
    print(f"  Accuracy: {accuracy:.2f}%")
    # Print other metrics...
    print(f"  Duration: {eval_duration:.2f}s")

    # Return relevant metrics (ADAPT this)
    return avg_reg_loss, avg_cls_loss, accuracy


# --- 5. Main execution block (`if __name__ == '__main__':`) --- #
if __name__ == "__main__":
    print("\n--- Starting Capstone Project --- ")
    print(f"Using device: {device}")

    # --- Setup TensorBoard Writer --- #
    print(
        f"\nInitializing TensorBoard SummaryWriter (log dir: {TENSORBOARD_LOG_DIR})..."
    )
    writer = SummaryWriter(TENSORBOARD_LOG_DIR)
    # Optional: Add hyperparameters to TensorBoard for tracking
    writer.add_hparams(
        {
            "lr": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "epochs": NUM_EPOCHS,
            "hidden_dim": HIDDEN_DIM,
            "input_dim": INPUT_DIM,
        },
        {},  # We won't log final metrics here, but could if desired
    )

    # --- Create Datasets and DataLoaders (Uses the adaptable Dataset class) --- #
    print("\nCreating datasets...")
    train_dataset = MultiTaskDataset(
        num_samples=NUM_SAMPLES, input_dim=INPUT_DIM, is_validation=False
    )
    val_dataset = MultiTaskDataset(
        num_samples=NUM_VAL_SAMPLES, input_dim=INPUT_DIM, is_validation=True
    )

    print("\nCreating dataloaders...")
    # NOTE: num_workers > 0 can cause issues on Windows/macOS without careful handling.
    # Setting num_workers=0 is often safest for simple scripts like this.
    num_workers = 0
    print(f"Using {num_workers} workers for DataLoaders.")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers
    )

    # --- Initialize Model, Loss, Optimizer (ADAPT LOSS FUNCTIONS IF NEEDED) --- #
    print("\nInitializing model, loss functions, and optimizer...")
    # Uses the adaptable Model class
    model = MultiTaskNet(
        input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, num_classes=NUM_CLASSES
    ).to(device)

    # --- Define Loss Functions (Select based on your tasks) --- #
    regression_criterion = nn.MSELoss()  # Common choice for regression
    classification_criterion = (
        nn.BCEWithLogitsLoss()
    )  # Common choice for binary classification (takes logits)
    # For multi-class, use nn.CrossEntropyLoss() and adjust model output layer & target format

    # --- Define Loss Weights (Adjust how much each task contributes) --- #
    loss_weights = (
        0.5,
        0.5,
    )  # Example: Equal weight. Sum should ideally be 1, but not strictly necessary.

    # --- Define Optimizer (Adam is a good default) --- #
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- Training and Evaluation Loop --- #
    print(f"\n--- Starting Training for {NUM_EPOCHS} Epochs --- ")
    best_val_accuracy = -1.0  # Simple mechanism to track best model

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n===== Epoch {epoch}/{NUM_EPOCHS} =====")
        train_reg_loss, train_cls_loss, train_combined_loss = train_epoch(
            model,
            train_loader,
            regression_criterion,
            classification_criterion,
            optimizer,
            device,
            epoch,
            loss_weights,
        )

        # Log training losses to TensorBoard
        writer.add_scalar("Loss/Train_Regression", train_reg_loss, epoch)
        writer.add_scalar("Loss/Train_Classification", train_cls_loss, epoch)
        writer.add_scalar("Loss/Train_Combined", train_combined_loss, epoch)

        # Evaluate on validation set
        val_reg_loss, val_cls_loss, val_accuracy = evaluate_model(
            model, val_loader, regression_criterion, classification_criterion, device
        )

        # Log validation losses and accuracy to TensorBoard
        writer.add_scalar("Loss/Validation_Regression", val_reg_loss, epoch)
        writer.add_scalar("Loss/Validation_Classification", val_cls_loss, epoch)
        writer.add_scalar("Accuracy/Validation", val_accuracy, epoch)

        # --- Optional: Save model if it's the best so far (based on a chosen metric) --- #
        # ADAPT the metric used for saving (e.g., val_reg_loss if regression is primary)
        if (
            val_accuracy > best_val_accuracy
        ):  # Example: save based on validation accuracy
            best_val_accuracy = val_accuracy
            print(
                f"** New best validation accuracy: {best_val_accuracy:.2f}%. Saving model... **"
            )
            try:
                os.makedirs(os.path.dirname(SAVE_MODEL_PATH), exist_ok=True)
                torch.save(model.state_dict(), SAVE_MODEL_PATH)
                print(f"Model saved to {SAVE_MODEL_PATH}")
            except Exception as e:
                print(f"Error saving model: {e}")
        print("-----------------------------")

    print("\n--- Training Complete --- ")
    print(
        f"Final validation accuracy: {val_accuracy:.2f}% (Best: {best_val_accuracy:.2f}%)"
    )  # ADAPT final metric reporting

    # --- Close TensorBoard Writer --- #
    print("Closing TensorBoard writer...")
    writer.close()
    print("To view logs, run: tensorboard --logdir runs")

print("\nImports and device setup complete. Ready to define components.")
