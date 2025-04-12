import torch

# --- Example 1: Scalar broadcasting --- #
# Adding a scalar to a tensor
a = torch.tensor([[1, 2], [3, 4]])
scalar = 10
result_scalar = a + scalar  # Scalar is broadcast to the shape of 'a'
print(f"Tensor a:\n{a}")
print(f"Scalar: {scalar}")
print(f"Result (a + scalar):\n{result_scalar}")  # [[11, 12], [13, 14]]

# --- Example 2: Row vector broadcasting --- #
# Adding a row vector to a 2D tensor
b = torch.tensor([[1, 2, 3], [4, 5, 6]])  # Shape (2, 3)
row_vector = torch.tensor([10, 20, 30])  # Shape (3,)
result_row = b + row_vector  # row_vector is broadcast to shape (2, 3)
# [[1+10, 2+20, 3+30], [4+10, 5+20, 6+30]]
print(f"\nTensor b (2x3):\n{b}")
print(f"Row vector (3,): {row_vector}")
print(f"Result (b + row_vector):\n{result_row}")

# --- Example 3: Column vector broadcasting --- #
# Adding a column vector to a 2D tensor
c = torch.tensor([[1, 2], [3, 4], [5, 6]])  # Shape (3, 2)
col_vector = torch.tensor([[10], [20], [30]])  # Shape (3, 1)
result_col = c + col_vector  # col_vector is broadcast to shape (3, 2)
# [[1+10, 2+10], [3+20, 4+20], [5+30, 6+30]]
print(f"\nTensor c (3x2):\n{c}")
print(f"Column vector (3x1):\n{col_vector}")
print(f"Result (c + col_vector):\n{result_col}")

# --- Example 4: Broadcasting with different dimensions --- #
d = torch.tensor([[1], [2], [3]])  # Shape (3, 1)
e = torch.tensor([10, 20])  # Shape (2,)
# How broadcasting works here:
# 1. 'e' is treated as shape (1, 2)
# 2. 'd' is expanded to (3, 2) by copying columns
# 3. 'e' is expanded to (3, 2) by copying rows
# 4. Element-wise addition is performed
result_diff_dims = d + e
# [[1+10, 1+20], [2+10, 2+20], [3+10, 3+20]]
print(f"\nTensor d (3x1):\n{d}")
print(f"Tensor e (2,): {e}")
print(f"Result (d + e) - Broadcasted to (3x2):\n{result_diff_dims}")

# --- Example 5: Incompatible shapes --- #
# These shapes cannot be broadcast together
f = torch.tensor([[1, 2, 3]])  # Shape (1, 3)
g = torch.tensor([[10], [20]])  # Shape (2, 1)
# try:
#     result_incompatible = f + g
# except RuntimeError as err:
#     print(f"\nCannot broadcast shapes {f.shape} and {g.shape}: {err}")
