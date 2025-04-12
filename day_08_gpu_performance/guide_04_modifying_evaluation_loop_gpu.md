# Guide: 04 Modifying the Pixel Evaluation Loop for GPU Speed!

We made the training loop faster, now let's speed up the pop quiz! This guide shows how to modify the evaluation loop (from Day 7) to run on your chosen `device` (hopefully the GPU!), based on `04_modifying_evaluation_loop_gpu.py`.

**Core Concept:** Just like training, evaluation runs faster on a GPU. We need to make sure the trained model and the validation sprites are both on the target `device` before the model makes its predictions.

## Prerequisites

- You have your trained `model_to_evaluate` instance (ideally with loaded weights).
- You have your `val_loader` ready to serve validation sprites.
- You have your `criterion` (loss function).
- You have your target `device` object.

## Evaluation Loop Device Modifications

It's almost identical to the training loop modifications!

1.  **Teleport Model (Once, Before Loop):** Move the model to the `device` right after loading its trained weights.
    ```python
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_to_evaluate = YourPixelModel(...)
    # model_to_evaluate.load_state_dict(torch.load(...))
    model_to_evaluate = model_to_evaluate.to(device) # <<< Move model!
    ```
2.  **Set Evaluation Mode:** Call `model.eval()`.
    ```python
    model_to_evaluate.eval()
    ```
3.  **Enter Chill Zone:** Wrap the loop with `with torch.no_grad():`.
    ```python
    with torch.no_grad():
        # ... evaluation loop ...
    ```
4.  **Iterate and Teleport Batches:** Loop through the `val_loader`. For _each batch_ of validation sprites (and labels):

    - Move them to the target `device` using `.to(device)` (and reassign!).

    ```python
    # Inside the no_grad loop:
    for sprite_batch, label_batch in val_loader:
        # === Move CURRENT BATCH to Device === #
        sprite_batch = sprite_batch.to(device)
        label_batch = label_batch.to(device)
        # ================================== #

        # ... proceed with forward pass and metric calculation ...
    ```

5.  **Forward Pass & Metrics (on Device):** Now, when you call `outputs = model_to_evaluate(sprite_batch)` and calculate loss or accuracy, everything automatically runs on the `device` because both model and data are there.

## Code Example (Integrated)

```python
# Spell Snippet (Evaluation Loop):
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Setup --- #
model_to_evaluate = YourPixelModel(...)
# model_to_evaluate.load_state_dict(torch.load(...)) # Assume weights loaded
val_loader = ...
criterion = ...

# --- Move Model --- #
model_to_evaluate = model_to_evaluate.to(device)

# --- Evaluation --- #
model_to_evaluate.eval() # Step 1: Set eval mode
total_validation_loss = 0.0
# ... other metric accumulators ...

with torch.no_grad(): # Step 2: Disable gradients
    print("Starting evaluation on device:", device)
    for sprite_batch, label_batch in val_loader: # Step 3: Loop

        # Step 4: Move data batch to device
        sprite_batch = sprite_batch.to(device)
        label_batch = label_batch.to(device)

        # Step 5: Forward pass & Metrics (run on device)
        outputs = model_to_evaluate(sprite_batch)
        loss = criterion(outputs, label_batch)
        total_validation_loss += loss.item() * sprite_batch.size(0)

        # ... calculate accuracy or other metrics ...

# --- Calculate final average metrics --- #
average_validation_loss = total_validation_loss / len(val_loader.dataset)
# ... calculate average accuracy ...

print(f"Validation Loss: {average_validation_loss:.4f}")
# print(f"Validation Accuracy: {validation_accuracy:.4f}")
```

## Summary

Adapting the evaluation loop for the GPU (or other device) is simple:

1. Move the trained `model` to the `device` once.
2. Call `model.eval()`.
3. Wrap the loop in `with torch.no_grad()`.
4. Move each validation `batch` to the `device` inside the loop.
   The forward pass and metric calculations then automatically run on the target device, making your pixel model pop quiz much faster!
