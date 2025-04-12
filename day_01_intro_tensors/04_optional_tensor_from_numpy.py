import torch
import numpy as np

# Create a NumPy array
numpy_array = np.array([1.0, 2.0, 3.0])
print(
    f"NumPy array: {numpy_array}, Type: {type(numpy_array)}, Dtype: {numpy_array.dtype}"
)

# Convert NumPy array to PyTorch tensor
tensor_from_numpy = torch.from_numpy(numpy_array)
print(
    f"Tensor from NumPy: {tensor_from_numpy}, Type: {type(tensor_from_numpy)}, Dtype: {tensor_from_numpy.dtype}"
)

# Important Note: The tensor and NumPy array share the same memory!
# Modifying one will modify the other (if they are on CPU)

numpy_array[0] = 99.0
print(f"Modified NumPy array: {numpy_array}")
print(f"Tensor after modifying NumPy array: {tensor_from_numpy}")

# Convert PyTorch tensor back to NumPy array
tensor = torch.tensor([4.0, 5.0, 6.0])
numpy_from_tensor = tensor.numpy()  # Only works for CPU tensors
print(f"Tensor: {tensor}")
print(
    f"NumPy array from tensor: {numpy_from_tensor}, Type: {type(numpy_from_tensor)}, Dtype: {numpy_from_tensor.dtype}"
)

# They also share memory here
tensor[0] = 111.0
print(f"Modified Tensor: {tensor}")
print(f"NumPy array after modifying tensor: {numpy_from_tensor}")
