import torch

# Create a 2D tensor
tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"Original Tensor:\n{tensor}")

# --- Indexing --- #

# Get the first row (index 0)
first_row = tensor[0]
print(f"\nFirst row: {first_row}, Shape: {first_row.shape}")

# Get the element at row 1, column 2 (0-indexed)
element = tensor[1, 2]
print(
    f"Element at [1, 2]: {element}, Shape: {element.shape}"
)  # Note: it's a 0D tensor (scalar)

# --- Slicing --- #

# Get the second column (all rows, index 1 column)
second_column = tensor[:, 1]
print(f"\nSecond column: {second_column}, Shape: {second_column.shape}")

# Get the first two rows
first_two_rows = tensor[:2]
print(f"\nFirst two rows:\n{first_two_rows}, Shape: {first_two_rows.shape}")

# Get the sub-tensor with rows 1 and 2, columns 0 and 1
sub_tensor = tensor[1:3, 0:2]
print(f"\nSub-tensor (rows 1:3, cols 0:2):\n{sub_tensor}, Shape: {sub_tensor.shape}")
