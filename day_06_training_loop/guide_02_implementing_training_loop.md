# Guide: 02 Implementing the Training Loop

This guide explains the structure and execution of a standard PyTorch training loop, bringing together the model, data, loss function, and optimizer to enable learning, as demonstrated in `02_implementing_training_loop.py`.

**Core Concept:** The training loop is the heart of the model learning process. It repeatedly feeds batches of data through the model, calculates the error (loss), computes gradients based on that error, and updates the model's parameters to minimize the error over time.

## Epochs and Batches

Training typically involves two nested loops:

- **Epoch:** One complete pass through the _entire_ training dataset. Models are usually trained for multiple epochs.
- **Batch (or Mini-Batch):** A smaller subset of the training data processed in one iteration of the inner loop. The `DataLoader` provides these batches.

## The Training Loop Structure

A typical PyTorch training loop looks like this:

```python
# Conceptual Structure:
num_epochs = ...
model = ...
data_loader = ...
criterion = ...
optimizer = ...

for epoch in range(num_epochs):
    # Set model to training mode
    model.train()

    # Loop over batches in the DataLoader
    for batch_data in data_loader:
        # --- Core 5 Steps --- #

        # 1. Zero Gradients
        optimizer.zero_grad()

        # 2. Forward Pass
        features, labels = batch_data # Unpack batch
        # features = features.to(device) # Move data to appropriate device
        # labels = labels.to(device)
        outputs = model(features)

        # 3. Calculate Loss
        loss = criterion(outputs, labels)

        # 4. Backward Pass (Compute Gradients)
        loss.backward()

        # 5. Update Parameters (Optimizer Step)
        optimizer.step()

        # --- End Core Steps --- #

        # (Optional: track/log loss, metrics, etc.)

    # (Optional: print epoch summary, perform validation)
```

Let's break down the key parts:

### Outer Loop (`for epoch...`)

Iterates over the entire dataset multiple times. The number of epochs is a hyperparameter.

### `model.train()`

This call sets the module and its submodules (like Dropout, BatchNorm) to training mode. This is important because some layers behave differently during training (e.g., Dropout randomly zeros elements) versus evaluation (e.g., Dropout is inactive). You should call `model.eval()` before running validation or testing.

### Inner Loop (`for batch_data in data_loader...`)

Iterates through the batches provided by the `DataLoader`. In each iteration, `batch_data` typically contains a batch of features and corresponding labels.

### The 5 Core Steps (per Batch)

These are executed for every single batch:

1. **`optimizer.zero_grad()`:**

    - **Why first?** Crucially resets the gradients stored in the `.grad` attribute of all parameters managed by the optimizer. If you don't do this, gradients from the _previous_ batch will be added to the gradients of the _current_ batch (as seen in Day 3 - Gradient Accumulation), leading to incorrect parameter updates.

2. **`outputs = model(features)` (Forward Pass):**

    - Passes the current batch of input features through the model's `forward` method to get predictions (`outputs`).

3. **`loss = criterion(outputs, labels)` (Loss Calculation):**

    - Compares the model's `outputs` with the true `labels` for the batch using the chosen loss function (`criterion`, e.g., `MSELoss`).
    - The result is a scalar tensor representing the average loss for the batch.

4. **`loss.backward()` (Backward Pass):**

    - This triggers `autograd` to compute the gradients of the `loss` with respect to all model parameters that have `requires_grad=True` and were involved in computing the `loss`.
    - The gradients are stored in the `.grad` attribute of each parameter.

5. **`optimizer.step()` (Parameter Update):**
    - The optimizer uses the gradients stored in the parameters' `.grad` attributes and its specific update rule (e.g., SGD, Adam) along with the learning rate (`lr`) to modify the parameter values (`param.data`). The goal is to adjust parameters in a direction that reduces the loss.

### Loss Monitoring

Inside the loop, it's common to track the loss to see if the model is learning. `loss.item()` extracts the Python scalar value from the loss tensor.

```python
# Script Snippet (Loss Tracking):
epoch_loss = 0.0
...
for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
    ...
    loss = criterion(outputs, batch_y)
    ...
    epoch_loss += loss.item() # Accumulate scalar loss value
...
avg_epoch_loss = epoch_loss / len(train_loader)
print(f"Epoch {epoch+1} Avg Loss: {avg_epoch_loss:.4f}")
```

A decreasing average loss per epoch generally indicates successful training.

## Summary

The PyTorch training loop iterates over epochs and batches. For each batch, it performs the critical sequence: zero gradients (`optimizer.zero_grad()`), compute predictions (`model(features)`), calculate loss (`criterion(outputs, labels)`), compute gradients (`loss.backward()`), and update parameters (`optimizer.step()`). Understanding and correctly implementing this loop is fundamental to training neural networks in PyTorch.
