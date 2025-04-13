# Guide: 01 Finding Your Pixel Workbench: Checking for GPUs!

Want your pixel training to run way faster? We need to check if you have a powerful Graphics Processing Unit (GPU) installed and tell PyTorch to use it! This guide shows how to detect GPUs (NVIDIA via CUDA, or Apple Silicon via MPS) and choose the best workbench (`device`) for the job, as seen in `01_checking_device.py`.

**Core Concept:** Training pixel models involves tons of calculations. CPUs (your computer's main brain) are okay, but GPUs are like specialized art studios with thousands of tiny workers (cores) optimized for doing many simple math operations (like those in neural networks) all at the same time. Using a GPU can make training _massively_ faster.

## Why Use a GPU for Pixels?

- **MASSIVE Parallel Power:** GPUs are built for parallel tasks. They can process large batches of sprites or perform complex layer calculations much faster than a CPU, which typically handles tasks one after another.
- **Speed, Speed, Speed:** This parallelism translates directly into faster training epochs and quicker pixel generation/inference. What might take hours on a CPU could take minutes on a good GPU.

## Checking Your Hardware Arsenal

PyTorch needs specific drivers and compatible hardware. Here's how to check:

1.  **NVIDIA GPUs (CUDA):** The most common setup for deep learning.

    - **The Check:** `torch.cuda.is_available()` returns `True` if you have a compatible NVIDIA GPU with the right drivers and a CUDA-enabled PyTorch installed.
    - **Bonus Info:** If `True`, you can find out _how many_ GPUs (`torch.cuda.device_count()`) and their _names_ (`torch.cuda.get_device_name(0)` for the first one).

    ```python
    # Spell Snippet (CUDA Check):
    has_cuda = torch.cuda.is_available()
    print(f"Is CUDA (NVIDIA GPU) ready? {has_cuda}")
    if has_cuda:
        print(f"  Woohoo! Number of CUDA GPUs: {torch.cuda.device_count()}")
        print(f"  GPU Name (first one): {torch.cuda.get_device_name(0)}")
    ```

2.  **Apple Silicon GPUs (MPS - Metal Performance Shaders):** For newer Macs (M1, M2, etc.).

    - **The Check:** `torch.backends.mps.is_available()` returns `True` if you're on the right macOS version (12.3+) with Apple Silicon and a compatible PyTorch (1.12+).
    - **(Optional) Built Check:** `torch.backends.mps.is_built()` confirms PyTorch was compiled with MPS support.

    ```python
    # Spell Snippet (MPS Check):
    # Important: Check if the backend exists first!
    has_mps_support = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    print(f"\nIs MPS (Apple Silicon GPU) ready? {has_mps_support}")
    # Note: is_built() is often redundant if is_available() is True, but good for info.
    # if has_mps_support:
    #    print(f"  PyTorch build includes MPS support? {torch.backends.mps.is_built()}")
    ```

## Choosing Your Workbench (`torch.device`)

To make your code run on whatever hardware is best, we check availability and create a `torch.device` object. This object acts like a label telling PyTorch where to put things.

- `torch.device("cuda")`: Use the default NVIDIA GPU.
- `torch.device("mps")`: Use the Apple Silicon GPU.
- `torch.device("cpu")`: Stick to the main CPU.

The standard logic is: Prefer CUDA, then try MPS, otherwise fall back to CPU.

```python
# Spell Snippet (Device Selection):
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("\nWorkbench Selected: CUDA (NVIDIA GPU) - Fast Lane! 🔥")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
    print("\nWorkbench Selected: MPS (Apple Silicon GPU) - Pretty Fast! ✨")
else:
    device = torch.device("cpu")
    print("\nWorkbench Selected: CPU - Steady does it. 🐢")

print(f"Device object we'll use: {device}")
```

## Using the `device` Label

Now that we have our `device` object (holding "cuda", "mps", or "cpu"), we'll use it in the _next_ steps to actually move our pixel model and sprite data batches onto that chosen workbench using the `.to(device)` method.

## Summary

Check for GPU power using `torch.cuda.is_available()` (for NVIDIA) and `torch.backends.mps.is_available()` (for Apple Silicon). Create a `torch.device` object based on the best available option (CUDA > MPS > CPU). This `device` object is your target workbench, and you'll use it with `.to()` to move your model and data for accelerated pixel processing!
