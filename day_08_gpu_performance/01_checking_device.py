import torch

print(f"PyTorch version: {torch.__version__}")

# --- Check for CUDA (NVIDIA GPU) Availability --- #
is_cuda_available = torch.cuda.is_available()

print(f"\nIs CUDA (GPU) available? {is_cuda_available}")

if is_cuda_available:
    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")

    # Get the name of the current GPU (usually GPU 0)
    current_gpu_index = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(current_gpu_index)
    print(f"Current GPU Index: {current_gpu_index}")
    print(f"Current GPU Name: {gpu_name}")
else:
    print("No CUDA-compatible GPU detected.")

# --- Check for MPS (Apple Silicon GPU) Availability --- #
# Added in PyTorch 1.12
is_mps_available = torch.backends.mps.is_available()
print(f"\nIs MPS (Apple Silicon GPU) available? {is_mps_available}")

if is_mps_available:
    # Check if it was built with MPS support
    is_mps_built = torch.backends.mps.is_built()
    print(f"Was PyTorch built with MPS support? {is_mps_built}")
    if not is_mps_built:
        print("MPS available but not built. Full MPS support may not be enabled.")
else:
    print("No MPS-compatible GPU detected.")


# --- Define the Target Device --- #
# Standard practice: prioritize CUDA, then MPS, then CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("\nSelected device: CUDA (NVIDIA GPU)")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("\nSelected device: MPS (Apple Silicon GPU)")
else:
    device = torch.device("cpu")
    print("\nSelected device: CPU")

# The 'device' object can now be used to move tensors and models
print(f"Device object: {device}")

# You can create tensors directly on the device
# try:
#     tensor_on_device = torch.randn(3, 3, device=device)
#     print(f"\nSuccessfully created a tensor directly on {device}:\n{tensor_on_device}")
#     print(f"Tensor's device: {tensor_on_device.device}")
# except Exception as e:
#     print(f"\nCould not create tensor directly on {device}: {e}")

print("\nThis script demonstrates how to check for available accelerators")
print("and define a `device` object for use in subsequent steps.")
