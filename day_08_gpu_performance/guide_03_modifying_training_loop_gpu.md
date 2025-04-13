# Guide: 03 Modifying the Pixel Training Loop for GPU Speed!

We've got our `device` (GPU hopefully!) and we know how to teleport (`.to(device)`) the model and sprites. Now, let's modify our pixel training loop from Day 6 to actually _use_ the GPU for faster learning! Based on `03_modifying_training_loop_gpu.py`.

**Core Concept:** To make the training loop run on the GPU, we just need to ensure the model and the current batch of sprites are _both_ on the GPU right before the main calculations happen (the forward pass). It's surprisingly few changes!

## Prerequisites

- All your training ingredients are ready (Model instance, `train_loader`, Criterion, Optimizer).
- You have your target `device` object defined (e.g., `device = torch.device("cuda")`).

## The Two Key Teleportation Points

To make the loop device-aware, we add just two `.to(device)` calls:

1.  **Teleport the Model (ONCE, BEFORE the Epoch Loop):**

    - Right after creating your model instance and _before_ starting the main `for epoch...` loop, send the entire model to the target device.
    - Remember to reassign: `pixel_model = pixel_model.to(device)`.
    - This moves all the model's parameters to the GPU memory (VRAM), ready for fast computation.

    ```python
    # Before the epoch loop:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pixel_model = YourPixelModel(...)
    train_loader = ...
    criterion = ...
    optimizer = ...

    print(f"Moving model to device: {device}")
    pixel_model = pixel_model.to(device) # <<< Move model ONCE here!
    print("Model moved.")

    # Now start the epoch loop...
    for epoch in range(num_epochs):
       # ... rest of loop ...
    ```

2.  **Teleport the Sprite Batch (EVERY Iteration, INSIDE the Batch Loop):**

    - Inside the `for sprite_batch, label_batch in train_loader:` loop...
    - Right after getting the batch from the `DataLoader`, move the `sprite_batch` (and the `label_batch`, if you have one) to the _same_ `device` as the model.
    - Remember to reassign: `sprite_batch = sprite_batch.to(device)`.

    ```python
    # Inside the batch loop:
    for batch_idx, batch_data in enumerate(train_loader):
        # Assuming batch_data = (sprite_batch, label_batch)
        sprite_batch, label_batch = batch_data

        # === Move CURRENT BATCH to Device === #
        sprite_batch = sprite_batch.to(device)
        label_batch = label_batch.to(device)
        # ================================== #

        # Now the model and the data are on the same device!
        # Proceed with the standard 5 training steps:
        optimizer.zero_grad()
        outputs = pixel_model(sprite_batch) # Runs on device!
        loss = criterion(outputs, label_batch) # Runs on device!
        loss.backward() # Runs on device!
        optimizer.step() # Runs on device!
        # ... rest of batch processing ...
    ```

## Why Move Batches Inside the Loop?

- **`DataLoader` Output:** DataLoaders usually give you batches on the CPU.
- **GPU Memory Limits:** You typically can't fit your entire huge sprite collection onto the GPU's VRAM at once. Moving batch-by-batch is memory-efficient.

## The Core 5 Steps Don't Change!

Notice the beauty: The actual 5 core training steps (`zero_grad`, `model()`, `criterion()`, `loss.backward()`, `optimizer.step()`) **don't need modification**. PyTorch is smart; if the model parameters and the input batch (`sprite_batch`) are on the GPU, all those operations automatically run on the GPU!

## What About the Loss Function?

Standard loss functions like `nn.MSELoss`, `nn.BCEWithLogitsLoss`, `nn.CrossEntropyLoss` usually don't have internal parameters and don't need to be explicitly moved to the device. They just operate on the tensors they are given (which we already moved to the `device`).

## Summary

Making your pixel training loop GPU-ready is easy!

1. Move the `model` to the `device` **once** before the epoch loop (`model = model.to(device)`).
2. Move each `sprite_batch` (and `label_batch`) to the `device` **inside** the batch loop (`batch = batch.to(device)`).
   That's it! PyTorch handles the rest, running your computations on the GPU for potentially massive speedups.
