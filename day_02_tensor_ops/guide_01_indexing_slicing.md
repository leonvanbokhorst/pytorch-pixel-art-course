# Guide: 01 Indexing and Slicing Tensors

This guide explains how to access specific elements or sections of PyTorch tensors using indexing and slicing, as demonstrated in `01_indexing_slicing.py`.

**Core Concept:** Often, you need to work with only a portion of your data represented in a tensor. Indexing and slicing provide powerful and flexible ways to select individual elements, rows, columns, or sub-regions of a tensor. The syntax is heavily inspired by Python lists and NumPy arrays.

## The Example Tensor

The script uses a 3x3 tensor for demonstration:

```python
# Script Snippet:
import torch

tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"Original Tensor:\n{tensor}")
# Output:
# Original Tensor:
# tensor([[1, 2, 3],
#         [4, 5, 6],
#         [7, 8, 9]])
```

## Indexing: Accessing Single Items

Indexing refers to accessing a single element or a slice along one dimension (like a whole row or column).

### 1. Accessing a Single Row

You can get an entire row by providing a single index.

```python
# Script Snippet:
first_row = tensor[0] # Get the row at index 0
print(f"\nFirst row: {first_row}, Shape: {first_row.shape}")
# Output:
# First row: tensor([1, 2, 3]), Shape: torch.Size([3])
```

- **Explanation:** `tensor[0]` selects the first row. Note that the result is a 1D tensor (a vector).

### 2. Accessing a Single Element

Provide indices for each dimension, separated by commas.

```python
# Script Snippet:
element = tensor[1, 2] # Get element at row index 1, column index 2
print(f"Element at [1, 2]: {element}, Shape: {element.shape}")
# Output:
# Element at [1, 2]: 6, Shape: torch.Size([])
```

- **Explanation:** `tensor[1, 2]` selects the element in the second row (index 1) and third column (index 2). The result is a 0D tensor (a scalar).

## Slicing: Accessing Sub-Tensors

Slicing uses the colon `:` notation (`start:stop:step`) to select ranges of elements along dimensions.

- `start`: The starting index (inclusive, defaults to 0).
- `stop`: The ending index (**exclusive**, defaults to the end).
- `step`: The increment (defaults to 1).
- A single colon `:` selects all elements along that dimension.

### 1. Accessing a Single Column

Use `:` for the row dimension and the specific index for the column dimension.

```python
# Script Snippet:
second_column = tensor[:, 1] # All rows (:), column index 1
print(f"\nSecond column: {second_column}, Shape: {second_column.shape}")
# Output:
# Second column: tensor([2, 5, 8]), Shape: torch.Size([3])
```

- **Explanation:** `tensor[:, 1]` selects all rows in the second column (index 1). The result is a 1D tensor.

### 2. Accessing a Range of Rows

Use slicing on the row dimension.

```python
# Script Snippet:
first_two_rows = tensor[:2] # Rows from index 0 up to (not including) index 2
print(f"\nFirst two rows:\n{first_two_rows}, Shape: {first_two_rows.shape}")
# Output:
# First two rows:
# tensor([[1, 2, 3],
#         [4, 5, 6]]), Shape: torch.Size([2, 3])
```

- **Explanation:** `tensor[:2]` selects rows starting from index 0 up to, but not including, index 2 (i.e., rows 0 and 1). The result is a 2D tensor.

### 3. Accessing a Rectangular Sub-Tensor (Block)

Apply slicing to multiple dimensions.

```python
# Script Snippet:
# Rows index 1 up to (not including) 3 => rows 1, 2
# Columns index 0 up to (not including) 2 => columns 0, 1
sub_tensor = tensor[1:3, 0:2]
print(f"\nSub-tensor (rows 1:3, cols 0:2):\n{sub_tensor}, Shape: {sub_tensor.shape}")
# Output:
# Sub-tensor (rows 1:3, cols 0:2):
# tensor([[4, 5],
#         [7, 8]]), Shape: torch.Size([2, 2])
```

- **Explanation:** `tensor[1:3, 0:2]` selects the block formed by rows 1 and 2, and columns 0 and 1. The result is a 2D tensor.

## Key Reminders

- **Zero-Based:** Indexing starts at 0.
- **Exclusive `stop`:** The `stop` index in slicing (`start:stop`) is _not_ included in the selection.
- **Shape:** Pay attention to the shape of the resulting tensor after indexing/slicing. Accessing single elements results in scalars, while accessing rows/columns results in vectors, and slicing ranges usually preserves the number of dimensions (but reduces their size).

## Summary

Indexing and slicing are fundamental tools for manipulating tensors in PyTorch. Mastering this NumPy-like syntax allows you to precisely extract the data you need for calculations, analysis, or feeding specific parts of your data into model layers.
