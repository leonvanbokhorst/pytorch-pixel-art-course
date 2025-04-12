# Day 8: GPU Acceleration and Performance Tips

**Topics:**

- Using GPUs (CUDA):
  - Checking availability (`torch.cuda.is_available()`)
  - Creating a device object (`torch.device`)
  - Moving models to GPU (`model.to(device)`)
  - Moving data batches to GPU (`data.to(device)`)
- Device Consistency: Ensuring all tensors in an operation are on the same device.
- Basic Performance Considerations:
  - Using `torch.no_grad()` for evaluation/inference.
  - Vectorized operations over Python loops.
  - Minimizing CPU-GPU data transfer.
  - Mention of advanced techniques (mixed precision, gradient checkpointing).

**Focus:** Leveraging GPU hardware to speed up training and inference, and understanding best practices for device management and efficient computation.

## Key Resources

- **CUDA Semantics:** [https://pytorch.org/docs/stable/notes/cuda.html](https://pytorch.org/docs/stable/notes/cuda.html) (Best practices for using CUDA with PyTorch, device selection, `to()`)
- **`torch.device` Documentation:** [https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.device](https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.device) (API for specifying CPU/GPU device)
- **`.to()` method (Tensor):** [https://pytorch.org/docs/stable/generated/torch.Tensor.to.html](https://pytorch.org/docs/stable/generated/torch.Tensor.to.html) (Moving tensors between devices/dtypes)
- **`.to()` method (Module):** [https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.to](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.to) (Moving entire models and their parameters)
- **`torch.cuda` Utilities:** [https://pytorch.org/docs/stable/cuda.html](https://pytorch.org/docs/stable/cuda.html) (Functions like `torch.cuda.is_available()`)

## Hands-On Examples

- **Checking for GPU and Defining Device:** ([`01_checking_device.py`](./01_checking_device.py))
  - **Code Idea:**
    ```python
    import torch
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU is available, using CUDA.")
    else:
        device = torch.device("cpu")
        print("GPU not available, using CPU.")
    ```
  - **Purpose:** Show the standard way to check for CUDA/MPS availability and define a `device` object.
- **Moving Model to Device:** ([`02_moving_model_to_device.py`](./02_moving_model_to_device.py))
  - **Code Idea:** Instantiate a model. Move it to the determined `device`: `model = model.to(device)`. Confirm parameter device.
  - **Purpose:** Demonstrate how to transfer the model's parameters to the target device.
- **Modifying Training Loop for GPU:** ([`03_modifying_training_loop_gpu.py`](./03_modifying_training_loop_gpu.py))
  - **Code Idea:** Take the training loop from Day 6. Move `model` to `device` before loop. Inside the batch loop, move `batch_X` and `batch_y` to `device`:
    ```python
    batch_X = batch_X.to(device)
    batch_y = batch_y.to(device)
    # ... rest of loop ...
    ```
  - **Purpose:** Show the minimal necessary changes to the training loop for device compatibility.
- **Modifying Evaluation Loop for GPU:** ([`04_modifying_evaluation_loop_gpu.py`](./04_modifying_evaluation_loop_gpu.py))
  - **Code Idea:** Take the evaluation loop from Day 7. Ensure `model` is on `device`. Inside the `torch.no_grad()` loop, move `batch_X` and `batch_y` to `device`.
    ```python
    batch_X = batch_X.to(device)
    batch_y = batch_y.to(device)
    # ... rest of loop ...
    ```
  - **Purpose:** Show the corresponding changes needed for the evaluation loop.
- **(Conceptual) Timing Comparison:**
  - **Code Idea:** No specific code, but suggest timing the loops from `03_...` and `04_...` on CPU vs GPU.
  - **Purpose:** Encourage experimentation to observe potential speedup.
