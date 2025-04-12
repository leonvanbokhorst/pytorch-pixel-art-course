import torch

# Create a tensor
matrix = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(f"Original Tensor:\n{matrix}")

# Check and print shape
print(f"Shape: {matrix.shape}")

# Check and print number of dimensions (rank)
print(f"Number of dimensions: {matrix.ndim}")

# Check and print data type
print(f"Data type: {matrix.dtype}")
