import torch

# Create a tensor
x = torch.tensor([1.0, 2.0, 3.0])
print(f"Original tensor x: {x}")
print(f"ID of x: {id(x)}")  # Memory address

# --- Standard operation (creates a new tensor) --- #
y = x + 5
print(f"\nResult of x + 5 (new tensor y): {y}")
print(f"Original tensor x is unchanged: {x}")
print(f"ID of y: {id(y)}")  # Different memory address

# --- In-place operation (modifies the original tensor) --- #
# In-place functions often end with an underscore '_'
x.add_(5)  # Modifies x directly
print(f"\nTensor x after x.add_(5): {x}")
print(f"ID of x is the same: {id(x)}")  # Same memory address

# Other examples:
x.mul_(2)  # Multiply x by 2 in-place
print(f"Tensor x after x.mul_(2): {x}")

x.sub_(1)  # Subtract 1 from x in-place
print(f"Tensor x after x.sub_(1): {x}")

# Why be cautious?
# 1. Breaks computation history for autograd (more on this later)
# 2. Can lead to unexpected behavior if other variables reference the same tensor
a = torch.tensor([10.0, 20.0])
b = a  # b now refers to the same underlying data as a
print(f"\nOriginal a: {a}, Original b: {b}")
b.add_(5)
print(f"b after b.add_(5): {b}")
print(f"a is also changed!: {a}")  # Because b is just another name for a's data
