# Day 8: GPU Acceleration for Faster Pixel Art

**Topics:**

- Using GPUs (CUDA/MPS) for Pixel Speedup:
  - Checking availability (`torch.cuda.is_available()`, `torch.backends.mps.is_available()`)
  - Creating a device object (`torch.device`) to represent CPU or GPU.
  - Moving pixel models to GPU (`pixel_model.to(device)`).
  - Moving batches of sprite data to GPU (`sprite_batch.to(device)`).
- Device Consistency: Ensuring the model and the pixel data it's processing are on the _same_ device (CPU or GPU).
- Basic Pixel Performance Considerations:
  - Using `torch.no_grad()` during pixel generation/evaluation (already covered, but reinforces speed).
  - Relying on PyTorch's optimized tensor operations (vectorization) instead of slow Python loops for pixel manipulations.
  - Minimizing data transfer between CPU and GPU (e.g., prepare data on GPU if possible).
  - Mention of advanced techniques (like mixed precision for faster training of complex pixel models).

**Focus:** Leveraging GPU hardware (like NVIDIA CUDA or Apple Metal/MPS) to significantly accelerate pixel art model training and inference, and managing computations efficiently across devices.

## Key Resources

- **CUDA Semantics:** [https://pytorch.org/docs/stable/notes/cuda.html](https://pytorch.org/docs/stable/notes/cuda.html)
- **Apple Silicon (MPS) Backend:** [https://pytorch.org/docs/stable/notes/mps.html](https://pytorch.org/docs/stable/notes/mps.html)
- **`torch.device` Documentation:** [https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.device](https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.device)
- **`.to()` method (Tensor):** [https://pytorch.org/docs/stable/generated/torch.Tensor.to.html](https://pytorch.org/docs/stable/generated/torch.Tensor.to.html)
- **`.to()` method (Module):** [https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.to](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.to)
- **`torch.cuda` Utilities:** [https://pytorch.org/docs/stable/cuda.html](https://pytorch.org/docs/stable/cuda.html)

## Hands-On Pixel Acceleration Examples

- **Checking for GPU and Defining Device:** ([`01_checking_device.py`](./01_checking_device.py))
  - **Pixel Idea:** Write code to check `torch.cuda.is_available()` (and optionally `torch.backends.mps.is_available()`) and set the `device` variable accordingly (`cuda`, `mps`, or `cpu`).
  - **Purpose:** Show the standard way to automatically select the best available hardware for pixel processing.
- **Moving Pixel Model to Device:** ([`02_moving_model_to_device.py`](./02_moving_model_to_device.py))
  - **Pixel Idea:** Instantiate a pixel art model (generator/classifier). Move it to the selected `device`: `pixel_model = pixel_model.to(device)`. Check a parameter's device.
  - **Purpose:** Demonstrate transferring the entire pixel model (layers and parameters) to the GPU/CPU.
- **Modifying Pixel Training Loop for GPU:** ([`03_modifying_training_loop_gpu.py`](./03_modifying_training_loop_gpu.py))
  - **Pixel Idea:** Take the pixel training loop from Day 6. Move the `pixel_model` to the `device` _before_ the loop starts. Inside the batch loop, move the `sprite_batch` and `target_batch` tensors to the `device` just before feeding them to the model and calculating loss.
  - **Purpose:** Show the minimal changes required to run the pixel training loop on the selected device.
- **Modifying Pixel Evaluation Loop for GPU:** ([`04_modifying_evaluation_loop_gpu.py`](./04_modifying_evaluation_loop_gpu.py))
  - **Pixel Idea:** Take the pixel evaluation loop from Day 7. Ensure the `pixel_model` is on the `device`. Inside the `torch.no_grad()` loop, move the validation `sprite_batch` and `target_batch` to the `device`.
  - **Purpose:** Show the corresponding device changes needed for the pixel evaluation loop.
- **(Conceptual) Timing Pixel Operations:**
  - **Pixel Idea:** Suggest running the modified training/evaluation loops on a dataset of sprites and timing them with `device='cpu'` versus `device='cuda'` (if available) to observe the speed difference.
  - **Purpose:** Encourage experiencing the potential speedup GPUs offer for pixel art tasks firsthand.
