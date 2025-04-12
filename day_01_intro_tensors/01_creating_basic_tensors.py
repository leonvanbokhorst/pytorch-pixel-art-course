import torch

# Create a 0D tensor (scalar)
scalar = torch.tensor(7)
print(f"Scalar: {scalar}, Shape: {scalar.shape}, Dtype: {scalar.dtype}")

# Create a 1D tensor (vector)
vector = torch.tensor([1, 2, 3])
print(f"Vector: {vector}, Shape: {vector.shape}, Dtype: {vector.dtype}")

# Create a 2D tensor (matrix)
matrix = torch.tensor([[1, 2], [3, 4]])
print(f"Matrix:\n{matrix}, Shape: {matrix.shape}, Dtype: {matrix.dtype}")
