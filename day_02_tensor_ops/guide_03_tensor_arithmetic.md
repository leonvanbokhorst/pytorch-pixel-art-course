# Guide: 03 Tensor Arithmetic

This guide covers basic element-wise arithmetic operations between PyTorch tensors, as demonstrated in `03_tensor_arithmetic.py`.

**Core Concept:** PyTorch allows you to perform standard mathematical operations directly on tensors. For tensors of the same shape, these operations are typically performed _element-wise_. This means the operation is applied independently to each pair of corresponding elements from the input tensors.

## Shape Requirement

For the basic arithmetic operations shown here, the input tensors generally need to have the **exact same shape**. PyTorch applies the operation between the element at `[i, j]` in the first tensor and the element at `[i, j]` in the second tensor, producing a result at `[i, j]` in the output tensor.

_(There's a powerful exception called "broadcasting" where PyTorch _can_ handle operations on tensors with _different_ shapes under certain rules. We'll cover that in the next section!)_

## The Example Tensors

The script uses two 2x2 tensors:

```python
# Script Snippet:
import torch

a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[10, 20], [30, 40]])

print(f"Tensor a:\n{a}")
print(f"Tensor b:\n{b}")
# Output:
# Tensor a:
# tensor([[1, 2],
#         [3, 4]])
# Tensor b:
# tensor([[10, 20],
#         [30, 40]])
```

## Element-wise Operations

PyTorch supports standard Python arithmetic operators and corresponding `torch` functions.

### 1. Addition (`+` or `torch.add`)

Adds corresponding elements.

```python
# Script Snippet:
addition = a + b
# alternative: torch.add(a, b)
print(f"\nAddition (a + b):\n{addition}")
# Output:
# Addition (a + b):
# tensor([[11, 22],
#         [33, 44]])
```

- (1+10=11, 2+20=22, 3+30=33, 4+40=44)\*

### 2. Subtraction (`-` or `torch.sub`)

Subtracts corresponding elements.

```python
# Script Snippet:
subtraction = b - a
# alternative: torch.sub(b, a)
print(f"\nSubtraction (b - a):\n{subtraction}")
# Output:
# Subtraction (b - a):
# tensor([[ 9, 18],
#         [27, 36]])
```

- (10-1=9, 20-2=18, 30-3=27, 40-4=36)\*

### 3. Element-wise Multiplication (`*` or `torch.mul`)

Multiplies corresponding elements. **This is NOT matrix multiplication!** (That uses `@` or `torch.matmul`). This is sometimes called the Hadamard product.

```python
# Script Snippet:
multiplication = a * b
# alternative: torch.mul(a, b)
print(f"\nElement-wise Multiplication (a * b):\n{multiplication}")
# Output:
# Element-wise Multiplication (a * b):
# tensor([[ 10,  40],
#         [ 90, 160]])
```

- (1*10=10, 2*20=40, 3*30=90, 4*40=160)\*

### 4. Division (`/` or `torch.div`)

Divides corresponding elements.

```python
# Script Snippet:
division = b / a
# alternative: torch.div(b, a)
print(f"\nDivision (b / a):\n{division}")
# Output:
# Division (b / a):
# tensor([[10., 10.],
#         [10., 10.]])
```

- (10/1=10, 20/2=10, 30/3=10, 40/4=10)\*
- **Note:** Be mindful of division by zero if your tensors contain zeros. Also, division often results in float tensors (`torch.float32` or `torch.float64`), even if the inputs were integers, as seen here.

### 5. Exponentiation (`**` or `torch.pow`)

Raises elements of the first tensor to the power of the corresponding elements in the second tensor, or to a scalar power.

```python
# Script Snippet:
exponentiation = a**2 # Raise each element of 'a' to the power of 2
# alternative: torch.pow(a, 2)
print(f"\nExponentiation (a ** 2):\n{exponentiation}")
# Output:
# Exponentiation (a ** 2):
# tensor([[ 1,  4],
#         [ 9, 16]])
```

- (1^2=1, 2^2=4, 3^2=9, 4^2=16)\*

## Summary

Performing basic element-wise arithmetic in PyTorch is intuitive, using standard Python operators (`+`, `-`, `*`, `/`, `**`) or equivalent `torch` functions. The main requirement for these basic operations is that the tensors have compatible shapes (usually the same shape), and the operations are applied independently to corresponding elements. Remember the distinction between element-wise multiplication (`*`) and matrix multiplication (`@`).
