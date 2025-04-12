# Guide: 04 Modifying Evaluation Loop for GPU/Device

This guide explains how to adapt the PyTorch evaluation loop to run on a specific compute device (like a GPU), ensuring consistency between the model and data placement, as shown in `04_modifying_evaluation_loop_gpu.py`.

**Core Concept:** Just like the training loop, the evaluation loop needs to be modified to leverage hardware accelerators. This involves moving the model to the target device and ensuring each batch of evaluation data is also moved to the same device before the forward pass.

## Prerequisites

- All evaluation components (Model instance with trained weights, Validation/Test DataLoader, Criterion) are instantiated.
- A `device` object (e.g., `torch.device("cuda")` or `torch.device("cpu")`) representing the target hardware has been created.

## Evaluation Loop Steps for Device Compatibility

Running the evaluation loop on a specific device involves these steps:

1.  **Move Model (Once, Before Loop):** Move the model instance (with loaded trained weights) to the target `device`.
    ```python
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleClassificationNet(...)
    # model.load_state_dict(torch.load(...)) # Load trained weights
    model = model.to(device)
    ```
2.  **Set Evaluation Mode:** Switch the model to evaluation mode.
    ```python
    model.eval()
    ```
3.  **Disable Gradients:** Wrap the main evaluation loop with the `torch.no_grad()` context manager.
    ```python
    with torch.no_grad():
        # ... evaluation loop ...
    ```
4.  **Iterate and Move Data Batches:** Loop through the validation/test `DataLoader`. For _each batch_:
    - Move the features and labels to the target `device`.
    ```python
    # Inside the no_grad loop:
    for batch_X, batch_y in val_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        # ... proceed with forward pass and metric calculation ...
    ```
5.  **Forward Pass & Metrics:** Perform the forward pass (`outputs = model(batch_X)`) and calculate loss/metrics (`loss = criterion(...)`, accuracy, etc.). These operations will automatically run on the `device` because both the model and the data are located there.

## Code Walkthrough

The script `04_modifying_evaluation_loop_gpu.py` combines these steps:

```python
# Script Snippet (Evaluation Loop):
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleClassificationNet(...).to(device) # Move model
# ... load weights, define criterion, val_loader ...

model.eval() # Step 1: Set eval mode
total_val_loss = 0.0
total_correct = 0
num_samples_processed = 0

with torch.no_grad(): # Step 2: Disable gradients
    for batch_X, batch_y in val_loader: # Step 3: Loop
        # Step 4: Move data
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        # Step 5: Forward pass & Metrics (on device)
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        total_val_loss += loss.item() * batch_X.size(0)

        predicted_labels = torch.argmax(outputs, dim=1)
        total_correct += (predicted_labels == batch_y).sum().item()

        num_samples_processed += batch_X.size(0)

# Calculate final metrics...
avg_val_loss = total_val_loss / num_samples_processed
accuracy = total_correct / num_samples_processed

print(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}")
```

## Summary

To run evaluation on a specific device (e.g., GPU), follow these steps: Move the model to the `device` once before the loop. Set the model to evaluation mode using `model.eval()`. Wrap the data iteration loop with `with torch.no_grad()`. Inside the loop, move each batch of validation/test data to the same `device` using `.to(device)` before performing the forward pass and calculating metrics. This ensures computations are performed efficiently on the target hardware without calculating unnecessary gradients.
