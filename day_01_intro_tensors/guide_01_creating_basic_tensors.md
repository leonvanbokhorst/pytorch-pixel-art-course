# Guide: 01 Creating Basic Tensors

This guide dives deeper into the concepts demonstrated in `01_creating_basic_tensors.py`.

**Core Concept:** Tensors are the fundamental data structure in PyTorch. Think of them as multi-dimensional arrays, similar to NumPy's `ndarray`, but with superpowers like GPU acceleration and automatic differentiation capabilities (which we'll explore later!).

## The `torch.tensor()` Function

The primary way to create a tensor _directly from data_ (like Python lists or numbers) is using `torch.tensor()`. You pass the data you want to turn into a tensor into this function.

```python
# Script Snippet:
import torch
```

## Tensor Types Created

The script demonstrates creating the most common, basic tensor dimensions:

### 1. Scalar (0-Dimensional Tensor)

- **What it is:** A single numerical value. It has zero dimensions.
- **Analogy:** Think of a single temperature reading, a score, or just the number 7 floating in space.
- **Code:**

  ```python
  # Script Snippet:
  scalar = torch.tensor(7)
  print(f"Scalar: {scalar}, Shape: {scalar.shape}, Dtype: {scalar.dtype}")
  ```

- **Explanation:** We pass a single Python number (`7`) to `torch.tensor()`. The resulting tensor `scalar` holds this value.
- **Attributes Printed:**
  - `shape`: `torch.Size([])` - An empty shape indicates it's a scalar (0 dimensions).
  - `dtype`: `torch.int64` - PyTorch infers the data type. Since we provided a Python integer, it defaults to a 64-bit integer tensor.

### 2. Vector (1-Dimensional Tensor)

- **What it is:** A sequence of numbers, like a list. It has one dimension.
- **Analogy:** A shopping list, a row of seats in a cinema, the coefficients of a polynomial.
- **Code:**

  ```python
  # Script Snippet:
  vector = torch.tensor([1, 2, 3])
  print(f"Vector: {vector}, Shape: {vector.shape}, Dtype: {vector.dtype}")
  ```

- **Explanation:** We pass a Python list `[1, 2, 3]` to `torch.tensor()`. The resulting tensor `vector` holds these values in order.
- **Attributes Printed:**
  - `shape`: `torch.Size([3])` - Indicates one dimension with a length of 3.
  - `dtype`: `torch.int64` - Again, inferred from the Python integers in the list.

### 3. Matrix (2-Dimensional Tensor)

- **What it is:** A grid of numbers, arranged in rows and columns. It has two dimensions.
- **Analogy:** A spreadsheet, a chessboard, the pixels in a grayscale image.
- **Code:**

  ```python
  # Script Snippet:
  matrix = torch.tensor([[1, 2], [3, 4]])
  print(f"Matrix:\n{matrix}, Shape: {matrix.shape}, Dtype: {matrix.dtype}")
  ```

- **Explanation:** We pass a Python list _of lists_ `[[1, 2], [3, 4]]` to `torch.tensor()`. Each inner list becomes a row in the resulting 2D tensor `matrix`.
- **Attributes Printed:**
  - `shape`: `torch.Size([2, 2])` - Indicates two dimensions: 2 rows and 2 columns.
  - `dtype`: `torch.int64` - Inferred once more.

## Summary

This first script is all about getting comfortable with creating tensors using `torch.tensor()` and understanding how data maps to different tensor dimensions (scalars, vectors, matrices). You also saw how PyTorch automatically figures out the shape and data type, which are essential properties we'll use constantly.
