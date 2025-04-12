# Guide: 04 Broadcasting

This guide explains the concept of broadcasting in PyTorch, a powerful mechanism that allows element-wise operations on tensors with different shapes under certain compatibility rules, as demonstrated in `04_broadcasting.py`.

**Core Concept:** Broadcasting describes how PyTorch treats tensors with different shapes during arithmetic operations. If two tensors are "broadcastable," PyTorch implicitly expands the tensor with fewer dimensions or dimensions of size 1 to match the shape of the other tensor _without actually duplicating data in memory_. This allows element-wise operations to occur as if the tensors had the same shape.

## Why is Broadcasting Useful?

- **Conciseness:** Avoids manually tiling or expanding tensors (e.g., using `.repeat()` or `.expand()`) to make shapes match, leading to cleaner code.
- **Memory Efficiency:** The expansion is implicit; no actual memory is used to store the "copied" elements, saving significant memory, especially with large tensors.
- **Convenience:** Simplifies common operations like adding a bias vector to every row/column of a matrix or multiplying a matrix by a scalar.

## The Rules of Broadcasting

Two tensors are compatible for broadcasting if the following rules hold (comparing shapes from right to left, the "trailing dimensions"):

1. **Align Dimensions:** Start comparing shapes from the last dimension backward.
2. **Compare Sizes:** For each dimension pair:
    - If the dimension sizes are **equal**, move to the next dimension to the left.
    - If one of the dimension sizes is **1**, PyTorch "stretches" or "broadcasts" this dimension to match the size of the other tensor's dimension.
    - If the dimension sizes are **different** and **neither is 1**, the tensors are incompatible, and broadcasting will result in a `RuntimeError`.
3. **Prepend Dimensions:** If one tensor runs out of dimensions (comparing from right to left), PyTorch prepends dimensions of size 1 to its shape until the number of dimensions matches the other tensor, and then Rule 2 applies.

The resulting tensor's shape will have the maximum size along each dimension from the input tensors.

## Examples from the Script

### 1. Scalar Broadcasting

- **Operation:** `Tensor (2, 2) + Scalar (0D)`
- **Shapes:** `[2, 2]` and `[]`
- **Process:** The scalar is treated as having dimensions of size 1 prepended until it matches the number of dimensions (`[1, 1]`). Then, each dimension of size 1 is stretched to match the tensor (`[2, 2]`).
- **Result Shape:** `[2, 2]`

```python
# Script Snippet:
a = torch.tensor([[1, 2], [3, 4]])
scalar = 10
result_scalar = a + scalar
# Output:
# tensor([[11, 12],
#         [13, 14]])
```

### 2. Row Vector Broadcasting

- **Operation:** `Tensor (2, 3) + Row Vector (3,)`
- **Shapes:** `[2, 3]` and `[3]`
- **Process:**
  - Align right: `[2, 3]` vs `[3]`
  - Last dim: 3 == 3 (match)
  - Tensor `b` has more dims. Prepend 1 to row vector shape: `[2, 3]` vs `[1, 3]`.
  - Dim 0: Stretch size 1 to match size 2.
- **Result Shape:** `[2, 3]` (The row vector is added to each row of `b`)

```python
# Script Snippet:
b = torch.tensor([[1, 2, 3], [4, 5, 6]])  # Shape (2, 3)
row_vector = torch.tensor([10, 20, 30])  # Shape (3,)
result_row = b + row_vector
# Output:
# tensor([[11, 22, 33],
#         [14, 25, 36]])
```

### 3. Column Vector Broadcasting

- **Operation:** `Tensor (3, 2) + Column Vector (3, 1)`
- **Shapes:** `[3, 2]` and `[3, 1]`
- **Process:**
  - Align right: `[3, 2]` vs `[3, 1]`
  - Last dim: Stretch size 1 to match size 2.
  - Dim 0: 3 == 3 (match)
- **Result Shape:** `[3, 2]` (The column vector is added to each column of `c`)

```python
# Script Snippet:
c = torch.tensor([[1, 2], [3, 4], [5, 6]])    # Shape (3, 2)
col_vector = torch.tensor([[10], [20], [30]]) # Shape (3, 1)
result_col = c + col_vector
# Output:
# tensor([[11, 12],
#         [23, 24],
#         [35, 36]])
```

### 4. Broadcasting with Different Dimensions

- **Operation:** `Tensor (3, 1) + Tensor (2,)`
- **Shapes:** `[3, 1]` and `[2]`
- **Process:**
  - Tensor `e` (shape `[2]`) is treated as `[1, 2]` (Rule 3 - prepend 1).
  - Align right: `[3, 1]` vs `[1, 2]`
  - Last dim: Stretch size 1 (from `d`) to match size 2. Shapes become `[3, 2]` vs `[1, 2]`.
  - Dim 0: Stretch size 1 (from `e`) to match size 3. Shapes become `[3, 2]` vs `[3, 2]`.
- **Result Shape:** `[3, 2]`

```python
# Script Snippet:
d = torch.tensor([[1], [2], [3]])  # Shape (3, 1)
e = torch.tensor([10, 20])      # Shape (2,) -> treated as (1, 2)
result_diff_dims = d + e
# Output:
# tensor([[11, 21],
#         [12, 22],
#         [13, 23]])
```

### 5. Incompatible Shapes

- **Operation:** `Tensor (1, 3) + Tensor (2, 1)`
- **Shapes:** `[1, 3]` and `[2, 1]`
- **Process:**
  - Align right: `[1, 3]` vs `[2, 1]`
  - Last dim: Stretch size 1 (from `g`) to match size 3. Shapes conceptually `[1, 3]` vs `[2, 3]`.
  - Dim 0: Sizes are 1 and 2. They are different, and neither is 1. **Error!**
- **Result:** `RuntimeError`

```python
# Example (from script comments):
f = torch.tensor([[1, 2, 3]])  # Shape (1, 3)
g = torch.tensor([[10], [20]])  # Shape (2, 1)
# f + g -> RuntimeError: The size of tensor a (3) must match the size of tensor b (2) at non-singleton dimension 1
```

## Summary

Broadcasting is a fundamental concept in PyTorch and NumPy that simplifies operations on tensors with different shapes by defining rules for implicit dimension expansion. It avoids manual, memory-intensive tiling operations and makes code more concise. Understanding the broadcasting rules is key to writing efficient and correct tensor manipulations.
