# Guide: 05 Matrix Multiplication

This guide explains how to perform matrix multiplication in PyTorch, a fundamental operation in linear algebra and deep learning, as demonstrated in `05_matrix_multiplication.py`.

**Core Concept:** Matrix multiplication is a process of multiplying two matrices that results in a new matrix. Unlike element-wise multiplication (`*`), it involves calculating the dot product between the rows of the first matrix and the columns of the second matrix. This operation is the backbone of linear transformations and is used extensively in neural network layers.

## The Shape Compatibility Rule

This is the **most critical rule** for matrix multiplication:

To multiply matrix `A` by matrix `B` (resulting in `C = A @ B`), the number of **columns** in `A` must be **equal** to the number of **rows** in `B`.

- If `A` has shape `(m, k)`
- And `B` has shape `(k, n)`
- Then the result `C` will have shape `(m, n)`.

The common inner dimension `k` disappears, and the outer dimensions `m` and `n` define the shape of the result.

## Example Matrices

The script creates two compatible matrices using random data:

```python
# Script Snippet:
import torch

torch.manual_seed(42) # for reproducibility
A = torch.randn(2, 3) # Shape (m=2, k=3)
B = torch.randn(3, 2) # Shape (k=3, n=2)

print(f"Matrix A (2x3):\n{A}")
print(f"Matrix B (3x2):\n{B}")
```

- Matrix `A` has 3 columns.
- Matrix `B` has 3 rows.
- The inner dimensions match (`k=3`), so they can be multiplied.
- The resulting matrix `C` will have shape `(m=2, n=2)`.

## Performing Matrix Multiplication in PyTorch

PyTorch provides two main ways:

### 1. `torch.matmul(input, other)`

This is the primary PyTorch function for matrix multiplication.

```python
# Script Snippet:
C_matmul = torch.matmul(A, B)
print(f"\nResult using torch.matmul(A, B) (2x2):\n{C_matmul}")
```

### 2. The `@` Operator

Python 3.5+ introduced the `@` infix operator specifically for matrix multiplication, offering a more concise syntax.

```python
# Script Snippet:
C_operator = A @ B
print(f"\nResult using A @ B (2x2):\n{C_operator}")
```

Both methods yield the same result, as verified in the script:

```python
# Script Snippet:
print(f"\nAre results equal? {torch.allclose(C_matmul, C_operator)}")
# Output: True
```

## Incompatible Shapes

If the inner dimensions do not match, attempting matrix multiplication will raise a `RuntimeError`.

```python
# Script Snippet (Error Case):
A_wrong = torch.randn(2, 3) # Shape (2, k=3)
B_wrong = torch.randn(2, 2) # Shape (k=2, 2) -> Inner dimensions 3 != 2

try:
    C_wrong = A_wrong @ B_wrong
except RuntimeError as e:
    print(f"\nError multiplying shapes {A_wrong.shape} and {B_wrong.shape}: {e}")
# Output:
# Error multiplying shapes torch.Size([2, 3]) and torch.Size([2, 2]): size mismatch, m1: [2, 3], m2: [2, 2]...
```

## Summary

Matrix multiplication (`@` or `torch.matmul`) is distinct from element-wise multiplication (`*`). It combines rows and columns via dot products and is fundamental to linear transformations in neural networks. Always ensure the inner dimensions of the matrices are compatible (`(m, k) @ (k, n)`) before performing the operation. The result will have the shape defined by the outer dimensions (`(m, n)`).
