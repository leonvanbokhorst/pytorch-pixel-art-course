# Guide: 02 Teleporting Your Pixels: Moving Model & Data to the Device!

We know _where_ we want to work (`device` = CPU/GPU), but how do we get our pixel model and sprite data _there_? This guide explains the teleportation spell: the `.to()` method, as seen in `02_moving_model_to_device.py`.

**Core Concept:** For fast GPU pixel processing, the **model** doing the work and the **sprite data** being worked on **MUST be on the same device (the GPU!)**. By default, everything starts on the CPU. We need to explicitly use the `.to(device)` spell to move them.

## Requirements

1.  **A Pixel Model Instance:** Your instantiated `nn.Module` (e.g., `pixel_generator`).
2.  **A Device Object:** The `device` variable we created in Guide 1, holding `torch.device("cuda")`, `torch.device("mps")`, or `torch.device("cpu")`.

## The `.to(device)` Teleport Spell

Both models (`nn.Module`) and individual tensors (like our sprites) have this `.to()` method. You simply tell it the destination `device`.

## Moving Your Pixel Model (One Time Trip)

To move your entire model – all its layers, weights, biases, etc. – to the GPU (or CPU), call `.to(device)` on the model instance.

**🚨 CRITICAL CATCH! 🚨** Moving a model with `.to()` is **NOT** in-place! It doesn't modify the original model object; it _returns a new model object_ living on the target device. You **MUST capture this returned object** by reassigning it to your model variable.

```python
# Spell Snippet:

# 1. Define device (from Guide 1)
device = ... # Will be torch.device("cuda"), etc.

# 2. Instantiate model (starts on CPU by default)
# Assume YourPixelModel is defined
pixel_model = YourPixelModel(noise_dim=..., num_pixels=...)
# Check initial location (first parameter's device tells all)
initial_device = next(pixel_model.parameters()).device
print(f"Pixel Model initially on device: {initial_device}") # -> cpu

# 3. Teleport model to target device AND REASSIGN!
print(f"Teleporting model to {device}...")
pixel_model = pixel_model.to(device)

# 4. Verify the move
final_device = next(pixel_model.parameters()).device
print(f"Pixel Model now on device: {final_device}") # -> cuda (or mps/cpu)
```

- **When?** You usually move the model to the device **once** right after creating it and _before_ you start your training or evaluation loops.

## Moving Sprite Tensors (Every Batch!)

You move individual tensors (like a batch of sprites) the same way, using `tensor.to(device)`. This is also **NOT** in-place, so you must reassign!

```python
# Spell Snippet:
# Assume sprite_batch comes from DataLoader (initially on CPU)
sprite_batch_cpu = torch.randn(16, 1, 8, 8) # Batch of 16 8x8 sprites
print(f"Sprite batch initially on device: {sprite_batch_cpu.device}") # -> cpu

# Teleport sprite batch AND REASSIGN
sprite_batch_device = sprite_batch_cpu.to(device)
print(f"Sprite batch now on device: {sprite_batch_device.device}") # -> cuda (or mps/cpu)
```

- **When?** You need to move the sprite batches (and label batches, if you have them) to the target device **inside your training and evaluation loops**, right before you feed them into the model. This happens _for every single batch_.

## Consistency is King!

PyTorch will throw a fit (a `RuntimeError`) if you try an operation where the model is on one device (e.g., GPU) and the input data is on another (e.g., CPU).

```python
# Example Error Scenario:

# Model teleported to GPU
pixel_model_gpu = YourPixelModel(...).to(torch.device("cuda"))

# Input sprite batch left on CPU
sprite_batch_cpu = torch.randn(16, 1, 8, 8)

# TRYING TO RUN CPU DATA ON GPU MODEL = ERROR!
# output = pixel_model_gpu(sprite_batch_cpu) # <-- BOOM! RuntimeError!

# Correct Way:
sprite_batch_gpu = sprite_batch_cpu.to(torch.device("cuda"))
output = pixel_model_gpu(sprite_batch_gpu) # OK! Both on GPU.
```

## Summary

Use the magical `.to(device)` spell to teleport both your pixel model (`nn.Module`) and your sprite data tensors to your chosen workbench (CPU or GPU). Remember the golden rule: `.to()` is **NOT in-place**, so always reassign the result (`model = model.to(device)`, `sprite_batch = sprite_batch.to(device)`). Move the model once before the loops, and move the data batches inside the loops, ensuring everything involved in a calculation is on the same device!
