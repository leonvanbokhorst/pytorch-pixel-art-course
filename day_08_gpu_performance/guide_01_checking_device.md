# Guide: 01 Checking for GPU and Defining Device

This guide explains how to check for available hardware accelerators (GPUs) in PyTorch and define a `device` object to target computations, as demonstrated in `01_checking_device.py`.

**Core Concept:** Modern deep learning heavily relies on the parallel processing power of Graphics Processing Units (GPUs) to accelerate the demanding computations involved in training large models. PyTorch provides tools to detect available GPUs (primarily NVIDIA GPUs via CUDA, or Apple Silicon GPUs via Metal Performance Shaders - MPS) and direct computations to them.

## Why Use GPUs?

- **Parallelism:** GPUs contain thousands of cores optimized for performing the same operation on large amounts of data simultaneously (like the matrix multiplications and convolutions common in neural networks).
- **Speed:** This parallelism leads to significant speedups (often 10x, 100x, or more) for training and inference compared to running on a CPU.

## Checking for Available Devices

PyTorch provides functions to check for specific types of accelerators:

1. **CUDA (NVIDIA GPUs):**

    - `torch.cuda.is_available() -> bool`: Returns `True` if PyTorch detects a compatible NVIDIA GPU and the installed PyTorch version has CUDA support enabled.
    - If `True`, you can optionally get more info:
      - `torch.cuda.device_count() -> int`: Number of available GPUs.
      - `torch.cuda.current_device() -> int`: Index of the default GPU.
      - `torch.cuda.get_device_name(index) -> str`: Name of the GPU at a given index.

    ```python
    # Script Snippet (CUDA Check):
    is_cuda_available = torch.cuda.is_available()
    print(f"Is CUDA (GPU) available? {is_cuda_available}")
    if is_cuda_available:
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    ```

2. **MPS (Apple Silicon GPUs):**

    - `torch.backends.mps.is_available() -> bool`: Returns `True` if running on macOS 12.3+ with an Apple Silicon chip (M1, M2, etc.) and a PyTorch version (1.12+) that supports MPS.
    - `torch.backends.mps.is_built() -> bool`: Checks if the PyTorch installation was built with MPS support.

    ```python
    # Script Snippet (MPS Check):
    is_mps_available = torch.backends.mps.is_available()
    print(f"\nIs MPS (Apple Silicon GPU) available? {is_mps_available}")
    if is_mps_available:
        print(f"Was PyTorch built with MPS support? {torch.backends.mps.is_built()}")
    ```

## Defining the Target `device`

To make your code portable across different hardware setups, the standard practice is to check for available accelerators and define a `torch.device` object that represents the best available option.

- `torch.device("cuda")`: Represents the default NVIDIA GPU.
  You can specify a particular GPU using `torch.device("cuda:0")`, `torch.device("cuda:1")`, etc.
- `torch.device("mps")`: Represents the Apple Silicon GPU.
- `torch.device("cpu")`: Represents the CPU.

The common logic prioritizes CUDA, then MPS, then falls back to CPU:

```python
# Script Snippet (Device Selection):
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("\nSelected device: CUDA (NVIDIA GPU)")
elif torch.backends.mps.is_available():
    # Add check for is_built() if strict compatibility is needed
    device = torch.device("mps")
    print("\nSelected device: MPS (Apple Silicon GPU)")
else:
    device = torch.device("cpu")
    print("\nSelected device: CPU")

print(f"Device object: {device}")
```

## Using the `device` Object

This `device` object (which now holds `cuda`, `mps`, or `cpu`) is the key to controlling where tensors and models reside. In subsequent steps, you will use the `.to(device)` method to move objects to the selected hardware.

## Summary

Before leveraging GPU acceleration, use `torch.cuda.is_available()` and `torch.backends.mps.is_available()` to check for compatible hardware. Define a `torch.device` object based on the best available option (CUDA > MPS > CPU). This `device` object will be used later to explicitly move your model and data tensors onto the target hardware using the `.to()` method, enabling hardware acceleration.
