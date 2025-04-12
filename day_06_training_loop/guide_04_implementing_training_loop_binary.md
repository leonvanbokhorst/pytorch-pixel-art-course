# Guide: 04 Implementing the Binary Classification Training Loop

This guide shows how to implement the training loop for a binary classification task, using the components (model, data, BCEWithLogitsLoss, optimizer) set up previously, as demonstrated in `04_implementing_training_loop_binary.py`.

**Core Concept:** The fundamental 5-step training loop (`zero_grad` -> `forward` -> `loss` -> `backward` -> `step`) remains the same regardless of the task (regression or classification). However, the specific loss function used and any additional monitoring metrics (like accuracy) will be task-specific.

## Training Loop Structure (Recap)

The loop iterates through epochs, and within each epoch, it iterates through batches from the `DataLoader`.

```python
# Conceptual Structure:
num_epochs = ...
model = ...
train_loader = ...
criterion = nn.BCEWithLogitsLoss() # Key difference
optimizer = ...

for epoch in range(num_epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        # 1. Zero Gradients
        optimizer.zero_grad()
        # 2. Forward Pass (Get Logits)
        outputs = model(batch_X)
        # 3. Calculate Loss (Using BCEWithLogitsLoss)
        loss = criterion(outputs, batch_y)
        # 4. Backward Pass
        loss.backward()
        # 5. Update Parameters
        optimizer.step()
        # (Optional: Calculate metrics like accuracy)
    # (Optional: Log epoch results)
```

## Key Aspects for Binary Classification

1. **Forward Pass Output:** The `outputs = model(batch_X)` call yields the raw logits from the model (shape `[batch_size, 1]`).
2. **Loss Calculation:** `loss = criterion(outputs, batch_y)` uses `nn.BCEWithLogitsLoss`, which correctly interprets the raw logits and the float `0.0`/`1.0` labels.
3. **Accuracy Calculation (Common Addition):** While loss tells us how wrong the model is on average, accuracy tells us the percentage of samples classified correctly. Since `BCEWithLogitsLoss` works on logits, we need a few extra steps to calculate accuracy within the loop:
    - **Convert Logits to Probabilities:** Apply the Sigmoid function to the raw logits: `predicted_probs = torch.sigmoid(outputs)`.
    - **Convert Probabilities to Labels:** Threshold the probabilities at 0.5 to get the predicted class (0 or 1): `predicted_labels = (predicted_probs >= 0.5).float()`.
    - **Compare and Count:** Compare the `predicted_labels` with the true `batch_y` and sum the number of correct predictions: `correct_in_batch = (predicted_labels == batch_y).sum().item()`.
    - **Accumulate:** Keep track of the total correct predictions across batches within an epoch.
    - **Calculate Epoch Accuracy:** After the epoch, divide the total correct count by the total number of samples in the dataset.

## Code Walkthrough

The script implements the loop with accuracy calculation:

```python
# Script Snippet (Inside Epoch Loop):
epoch_loss = 0.0
epoch_correct = 0
num_batches_processed = 0

for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
    optimizer.zero_grad()
    outputs = model(batch_X)        # Raw logits
    loss = criterion(outputs, batch_y) # Loss calculation
    loss.backward()
    optimizer.step()

    # --- Accuracy Calculation --- #
    predicted_probs = torch.sigmoid(outputs)          # Logits -> Probs
    predicted_labels = (predicted_probs >= 0.5).float() # Probs -> Labels (0/1)
    correct_in_batch = (predicted_labels == batch_y).sum().item()
    epoch_correct += correct_in_batch
    # -------------------------- #

    epoch_loss += loss.item()
    num_batches_processed += 1

# After inner loop (end of epoch)
avg_epoch_loss = epoch_loss / num_batches_processed
epoch_accuracy = epoch_correct / len(train_loader.dataset)
print(
    f"Epoch {epoch+1}/{num_epochs} completed. Avg Loss: {avg_epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}"
)
```

## Note on Training vs. Validation Accuracy

Calculating accuracy on the _training set_ within the loop is useful for monitoring progress. However, the primary measure of how well your model generalizes is its accuracy on a separate **validation set** (using `model.eval()` and `torch.no_grad()`), which we will cover in Day 7.

## Summary

The training loop for binary classification follows the standard 5 steps (`zero_grad`, `forward`, `loss`, `backward`, `step`). The key differences from regression lie in using a model that outputs a single logit and employing `nn.BCEWithLogitsLoss` as the criterion. Accuracy calculation is often added for monitoring, requiring conversion of logits to probabilities (via Sigmoid) and then thresholding to get predicted class labels.
