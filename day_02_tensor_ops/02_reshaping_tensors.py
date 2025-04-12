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
