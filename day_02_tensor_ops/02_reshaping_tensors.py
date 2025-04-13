import torch

# Create a 1D tensor
x = torch.arange(6)  # Creates tensor([0, 1, 2, 3, 4, 5])
print(f"Original tensor: {x}, Shape: {x.shape}")

# --- Using view --- #
# view requires the new shape to be compatible with the original number of elements
# It shares the underlying data with the original tensor (if possible)
view_reshape = x.view(2, 3)
print(f"\nReshaped with view(2, 3):\n{view_reshape}, Shape: {view_reshape.shape}")

# Changing the original tensor might affect the view
# x[0] = 99
# print(f"Original after change: {x}")
# print(f"View after change: {view_reshape}")

# --- Using reshape --- #
# reshape is more flexible and might return a copy or a view
# It's generally recommended over view unless you specifically need a view
reshape_reshape = x.reshape(3, 2)
print(
    f"\nReshaped with reshape(3, 2):\n{reshape_reshape}, Shape: {reshape_reshape.shape}"
)

# Try reshaping to an incompatible shape (will error)
# try:
#     x.reshape(2, 4)
# except RuntimeError as e:
#     print(f"\nError reshaping to (2, 4): {e}")

# Flattening a tensor
tensor_2d = torch.tensor([[1, 2], [3, 4]])
print(f"\nOriginal 2D tensor:\n{tensor_2d}")
flattened = tensor_2d.reshape(-1)  # -1 infers the dimension size
print(f"Flattened tensor: {flattened}, Shape: {flattened.shape}")

# --- Experimenting with odd-sized tensors --- #
print("\n=== Experimenting with odd-sized tensors ===")
odd_tensor = torch.arange(5)  # Creates tensor([0, 1, 2, 3, 4])
print(f"\nOriginal odd tensor: {odd_tensor}, Shape: {odd_tensor.shape}")

# Valid reshapes for 5 elements
print("\nValid reshapes:")
row_vector = odd_tensor.reshape(1, 5)
print(f"Row vector:\n{row_vector}, Shape: {row_vector.shape}")

col_vector = odd_tensor.reshape(5, 1)
print(f"Column vector:\n{col_vector}, Shape: {col_vector.shape}")

# Invalid reshapes (will error)
print("\nTrying invalid reshapes:")
try:
    odd_tensor.reshape(2, 2)  # Needs 4 elements
except RuntimeError as e:
    print(f"Error reshaping to (2, 2): {e}")

try:
    odd_tensor.reshape(2, 3)  # Needs 6 elements
except RuntimeError as e:
    print(f"Error reshaping to (2, 3): {e}")

# --- Experimenting with prime number tensor (7) --- #
print("\n=== Experimenting with prime number tensor (7) ===")
prime_tensor = torch.arange(7)  # Creates tensor([0, 1, 2, 3, 4, 5, 6])
print(f"\nOriginal prime tensor: {prime_tensor}, Shape: {prime_tensor.shape}")

# The only possible valid reshapes for a prime number
print("\nOnly possible valid reshapes for 7 elements:")
print(f"1D tensor: {prime_tensor}, Shape: {prime_tensor.shape}")

row_vector = prime_tensor.reshape(1, 7)
print(f"Row vector:\n{row_vector}, Shape: {row_vector.shape}")

col_vector = prime_tensor.reshape(7, 1)
print(f"Column vector:\n{col_vector}, Shape: {col_vector.shape}")

# Trying some invalid reshapes to demonstrate
print("\nTrying invalid reshapes (all will error):")
try:
    prime_tensor.reshape(2, 3)  # Needs 6 elements
except RuntimeError as e:
    print(f"Error reshaping to (2, 3): {e}")

try:
    prime_tensor.reshape(3, 2)  # Needs 6 elements
except RuntimeError as e:
    print(f"Error reshaping to (3, 2): {e}")

try:
    prime_tensor.reshape(2, 4)  # Needs 8 elements
except RuntimeError as e:
    print(f"Error reshaping to (2, 4): {e}")

# --- Experimenting with more factorable number tensor (12) --- #
print("\n=== Experimenting with more factorable number tensor (12) ===")
factorable_tensor = torch.arange(12)  # Creates tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
print(f"\nOriginal factorable tensor: {factorable_tensor}, Shape: {factorable_tensor.shape}")

# Valid reshapes for 12 elements
print("\nValid reshapes:")
row_vector = factorable_tensor.reshape(1, 12)
print(f"Row vector:\n{row_vector}, Shape: {row_vector.shape}")

col_vector = factorable_tensor.reshape(12, 1)   
print(f"Column vector:\n{col_vector}, Shape: {col_vector.shape}")

# 2d matrix
matrix = factorable_tensor.reshape(3, 4)
print(f"Matrix:\n{matrix}, Shape: {matrix.shape}")

# 3d tensor
tensor = factorable_tensor.reshape(2, 2, 3)
print(f"3D tensor:\n{tensor}, Shape: {tensor.shape}")   

# The -1 argument
flat_vector = factorable_tensor.reshape(-1)
print(f"Flat vector:\n{flat_vector}, Shape: {flat_vector.shape}")

# 2d matrix with -1
matrix_2d = factorable_tensor.reshape(2, -1)
print(f"Matrix 2D:\n{matrix_2d}, Shape: {matrix_2d.shape}")

# 3d tensor with -1
matrix_3d = factorable_tensor.reshape(2, 2, -1)
print(f"Matrix 3D:\n{matrix_3d}, Shape: {matrix_3d.shape}")

# The -1 argument can only be used once
try:
    factorable_tensor.reshape(2, -1, -1)
except RuntimeError as e:
    print(f"Error reshaping to (2, -1, -1): {e}")
