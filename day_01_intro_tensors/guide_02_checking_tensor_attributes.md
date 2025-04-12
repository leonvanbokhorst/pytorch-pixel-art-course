# Guide: 02 Checking Tensor Attributes

This guide explains the essential tensor attributes demonstrated in `02_checking_tensor_attributes.py`.

**Core Concept:** Every tensor in PyTorch has properties that describe its structure and the data it holds. Understanding these attributes is fundamental for debugging, ensuring tensor operations are valid, and designing neural network layers.

## The Example Tensor

The script starts by creating a 2-dimensional tensor (a matrix):

```python
# Script Snippet:
import torch

matrix = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(f"Original Tensor:\n{matrix}")
# Output:
# Original Tensor:
# tensor([[1, 2, 3],
#         [4, 5, 6]])
```

This tensor has 2 rows and 3 columns.

## Tensor Attributes Explored

The script then inspects and prints three key attributes of this tensor:

### 1. `.shape`

- **What it is:** An attribute that returns a `torch.Size` object (which behaves like a Python tuple) indicating the size of the tensor along each dimension.
- **Analogy:** The dimensions of a photograph (e.g., 1920 pixels wide by 1080 pixels high) or a shipping box (length x width x height).
- **Code:**

  ```python
  # Script Snippet:
  print(f"Shape: {matrix.shape}")
  # Output:
  # Shape: torch.Size([2, 3])
  ```

- **Explanation:** The output `torch.Size([2, 3])` tells us the tensor has 2 elements in its first dimension (rows) and 3 elements in its second dimension (columns).

### 2. `.ndim` (Number of Dimensions / Rank)

- **What it is:** An attribute that returns an integer representing the number of dimensions the tensor has. It's simply the length of the `.shape` tuple.
- **Analogy:** A point has 0 dimensions, a line has 1, a flat surface has 2, and a cube has 3. The `.ndim` tells you which kind of space the tensor lives in.
- **Code:**

  ```python
  # Script Snippet:
  print(f"Number of dimensions: {matrix.ndim}")
  # Output:
  # Number of dimensions: 2
  ```

- **Explanation:** The output `2` confirms that our `matrix` is a 2-dimensional tensor, corresponding to its shape `[2, 3]` having two numbers.
  - A scalar (like `torch.tensor(7)`) would have `ndim = 0`.
  - A vector (like `torch.tensor([1, 2, 3])`) would have `ndim = 1`.

### 3. `.dtype` (Data Type)

- **What it is:** An attribute that returns the data type of the elements stored within the tensor (e.g., floating-point number, integer, boolean).
- **Analogy:** The material something is made of (e.g., integer wood, float32 plastic, boolean glass).
- **Importance:** `dtype` affects memory usage, numerical precision, and compatibility with certain hardware (like GPUs) and operations.
- **Code:**

  ```python
  # Script Snippet:
  print(f"Data type: {matrix.dtype}")
  # Output:
  # Data type: torch.int64
  ```

- **Explanation:** The output `torch.int64` indicates that the tensor holds 64-bit integers. This was inferred by `torch.tensor()` because the input Python list `[[1, 2, 3], [4, 5, 6]]` contained only integers. If we had included a decimal (e.g., `[[1.0, 2, 3], [4, 5, 6]]`), the `dtype` would likely have defaulted to `torch.float32`.

## Summary

Checking `.shape`, `.ndim`, and `.dtype` is like asking a tensor "What do you look like?" and "What are you made of?". These are fundamental checks you'll perform constantly when working with PyTorch to ensure your data has the structure and type you expect before feeding it into models or performing calculations.
