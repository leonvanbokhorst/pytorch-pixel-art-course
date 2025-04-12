import torch

# Default integer tensor (usually int64)
default_int_tensor = torch.tensor([1, 2, 3])
print(f"Default Int Tensor: {default_int_tensor}, Dtype: {default_int_tensor.dtype}")

# Specify float32
float32_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
print(f"Float32 Tensor: {float32_tensor}, Dtype: {float32_tensor.dtype}")

# Specify float64 (double)
float64_tensor = torch.tensor([1, 2, 3], dtype=torch.float64)
print(f"Float64 Tensor: {float64_tensor}, Dtype: {float64_tensor.dtype}")

# Specify boolean
bool_tensor = torch.tensor([0, 1, 1, 0], dtype=torch.bool)
print(f"Boolean Tensor: {bool_tensor}, Dtype: {bool_tensor.dtype}")
