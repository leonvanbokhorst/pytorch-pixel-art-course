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

# --- Example 6: Brightness Adjustment with Broadcasting --- #
print("\n=== Brightness Adjustment with Broadcasting ===")

# Create a simple 4x4 grayscale sprite (values 0-15)
sprite = torch.arange(16).reshape(4, 4)
print(f"\nOriginal Sprite (4x4):\n{sprite}")

# Add brightness using broadcasting
brightness_boost = 50
bright_sprite = sprite + brightness_boost
print(f"\nSprite + {brightness_boost} brightness:\n{bright_sprite}")

# Create a gradient effect using broadcasting
# First, create a row vector for horizontal gradient
horizontal_gradient = torch.arange(0, 40, 10)  # Creates [0, 10, 20, 30]
print(f"\nHorizontal Gradient Vector:\n{horizontal_gradient}")

# Add the gradient to each row of the sprite
gradient_sprite = sprite + horizontal_gradient
print(f"\nSprite + Horizontal Gradient:\n{gradient_sprite}")

# Create a vertical gradient
vertical_gradient = torch.arange(0, 40, 10).reshape(
    4, 1
)  # Creates [[0], [10], [20], [30]]
print(f"\nVertical Gradient Vector:\n{vertical_gradient}")

# Add the gradient to each column of the sprite
gradient_sprite_vertical = sprite + vertical_gradient
print(f"\nSprite + Vertical Gradient:\n{gradient_sprite_vertical}")

# --- Example 7: Different arange patterns --- #
print("\n=== Different arange patterns ===")

# Count by 2s (even numbers)
even_numbers = torch.arange(0, 16, 2).reshape(4, 2)
print(f"\nEven numbers (0 to 14 by 2):\n{even_numbers}")

# Count by 3s
threes = torch.arange(0, 16, 3).reshape(3, 2)
print(f"\nCount by 3s (0 to 15 by 3):\n{threes}")

# Start at 10, count by 5s
fives = torch.arange(10, 31, 5).reshape(5, 1)  # Changed to 5x1 to match 5 numbers
print(f"\nStart at 10, count by 5s (5x1):\n{fives}")

# Alternative: Get exactly 4 numbers
fives_4 = torch.arange(10, 30, 5).reshape(2, 2)  # Changed stop to 30 to get 4 numbers
print(f"\nStart at 10, count by 5s (2x2):\n{fives_4}")

# Count backwards
backwards = torch.arange(15, -1, -1).reshape(4, 4)
print(f"\nCount backwards (15 to 0):\n{backwards}")

# --- Example 8: Your Turn! --- #
print("\n=== Your Turn! ===")

sprite = torch.zeros((4, 4, 3))

# Option A
diagonal = torch.arange(0, 16, 1).reshape(4, 4)
anti_diagonal = torch.flip(diagonal, [1])
checker = torch.tensor([[0, 50] * 2, [50, 0] * 2] * 2)
sprite[:, :, 0] = diagonal
sprite[:, :, 1] = anti_diagonal
sprite[:, :, 2] = checker

print(f"\nSprite:\n{sprite}")

# Option B
sprite = torch.zeros((4, 4, 3))

diagonal = torch.arange(0, 16, 1).reshape(4, 4, 1)
anti_diagonal = torch.flip(diagonal, [1])
checker = torch.tensor([[[0], [50]] * 2, [[50], [0]] * 2] * 2)
sprite = diagonal + anti_diagonal + checker

print(f"\nSprite:\n{sprite}")

# Option C
base = torch.arange(0, 16, 1).reshape(4, 4)
diagonal = base.unsqueeze(-1)
anti_diagonal = torch.flip(base, [1]).unsqueeze(-1)
checker = torch.tensor([0, 50]).repeat(8).reshape(4, 4, 1)
sprite = torch.cat([diagonal, anti_diagonal, checker], dim=-1)

print(f"\nSprite:\n{sprite}")
