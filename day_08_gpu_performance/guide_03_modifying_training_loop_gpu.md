# Guide: 03 Modifying Training Loop for GPU/Device

This guide explains the necessary modifications to run your PyTorch training loop on a specific compute device (like a GPU), leveraging the device selection logic covered previously, as shown in `03_modifying_training_loop_gpu.py`.

**Core Concept:** To utilize hardware acceleration (like a GPU), you must ensure that both the model and the input data reside on the target device _before_ performing operations that involve both (like the forward pass). The training loop needs minor adjustments to handle this data movement.

## Prerequisites

- All training components (Model, DataLoader, Criterion, Optimizer) are instantiated.
- A `device` object (e.g., `torch.device("cuda")` or `torch.device("cpu")`) representing the target hardware has been created.

## Modifications for Device Compatibility

Running the training loop on a specific device requires only two key changes:

1. **Move the Model (Once, Before the Loop):**

   - Before starting the `for epoch...` loop, move your entire model instance to the target device using `model = model.to(device)`.
   - This transfers all the model's parameters and buffers to the device's memory (e.g., GPU VRAM).
   - This is typically done only once.

   ```python
   # Before the epoch loop:
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   model = SimpleRegressionNet(...)
   # ... instantiate criterion, optimizer ...
   model = model.to(device) # Move model to device
   ```

2. **Move Data Batches (Repeatedly, Inside the Loop):**

   - _Inside_ the `for batch_X, batch_y in train_loader:` loop, for each batch retrieved from the `DataLoader`:
   - Move both the features (`batch_X`) and the labels (`batch_y`) to the _same_ target device using `.to(device)`.

   ```python
   # Inside the batch loop:
   for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
       # --- Move Data to Device --- #
       batch_X = batch_X.to(device)
       batch_y = batch_y.to(device)
       # --------------------------- #

       # Now proceed with training steps...
       optimizer.zero_grad()
       outputs = model(batch_X)
       loss = criterion(outputs, batch_y)
       loss.backward()
       optimizer.step()
   ```

## Why Move Data Inside the Loop?

- **DataLoader Output:** The `DataLoader` (especially with `num_workers > 0`) typically yields batches as CPU tensors.
- **Memory:** You usually don't load the entire dataset onto the GPU at once due to memory limitations. Moving batches individually keeps GPU memory usage manageable.

## Core Loop Remains Unchanged

Notice that the core 5 training steps (`optimizer.zero_grad()`, forward pass `model(batch_X)`, loss calculation `criterion(...)`, `loss.backward()`, `optimizer.step()`) **do not need to be changed**. PyTorch automatically performs these operations on the device where the involved tensors (model parameters and data batch) reside. Since we moved both `model` and the `batch_X`/`batch_y` to `device`, the computations will happen on that device (e.g., the GPU).

## Loss Function Device (Note)

Standard PyTorch loss functions like `nn.MSELoss` or `nn.CrossEntropyLoss` are generally stateless and don't have parameters, so they typically don't need to be explicitly moved to the device. However, if you were using a custom loss function that _was_ an `nn.Module` with its own parameters, you would need to move it to the device as well (`criterion = criterion.to(device)`).

## Summary

Adapting a PyTorch training loop for GPU (or another device) involves two main steps: 1. Move the entire model to the target `device` once before the loop using `model = model.to(device)`. 2. Move each data batch (features and labels) to the same `device` inside the batch processing loop using `batch_data = batch_data.to(device)`. The core training logic remains the same, as PyTorch operations automatically run on the device of their input tensors.
