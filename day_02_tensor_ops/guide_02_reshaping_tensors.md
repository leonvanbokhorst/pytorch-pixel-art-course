# Guide: 02 Reshaping Tensors

This guide explores how to change the shape of a PyTorch tensor while preserving its elements, using methods demonstrated in `02_reshaping_tensors.py`.

**Core Concept:** Reshaping is a fundamental operation that changes the dimensions of a tensor without altering the total number of elements or the underlying data order (when read sequentially). It's like rearranging items on a shelf into different row/column configurations.

## Why Reshape Tensors?

- **Layer Input Requirements:** Different neural network layers expect inputs with specific shapes. For example, fully connected (dense) layers often require flattened 1D vectors, while convolutional layers expect multi-dimensional inputs (e.g., `[batch_size, channels, height, width]` for images).
- **Data Alignment:** Reshaping can align tensors for specific operations like broadcasting or matrix multiplication.
- **Feature Engineering:** You might reshape data to represent features in a different structure.

## The Goal: Same Elements, Different Dimensions

The key constraint is that the total number of elements must remain constant. If a tensor has 6 elements, you can reshape it into `(2, 3)`, `(3, 2)`, `(6, 1)`, `(1, 6)`, or `(6,)`, but not `(2, 4)` (which requires 8 elements).

## The Example Tensor

The script starts with a simple 1D tensor:

```python
# Script Snippet:
import torch

x = torch.arange(6) # Creates tensor([0, 1, 2, 3, 4, 5])
print(f"Original tensor: {x}, Shape: {x.shape}")
# Output:
# Original tensor: tensor([0, 1, 2, 3, 4, 5]), Shape: torch.Size([6])
```

## Reshaping Methods

PyTorch offers two primary methods:

### 1. `tensor.view(shape)`

- **How it works:** Attempts to create a new tensor with the specified `shape` that shares the _same underlying data_ as the original tensor. It doesn't copy memory, making it very fast.
- **Requirement:** The tensor's data must be _contiguous_ in memory in a way that allows the new shape. If not, `view` will raise an error.
- **Shared Data:** Because it often shares data, modifying the original tensor might change the view, and vice-versa (similar to the NumPy interaction in Day 1).

```python
# Script Snippet:
view_reshape = x.view(2, 3)
print(f"\nReshaped with view(2, 3):\n{view_reshape}, Shape: {view_reshape.shape}")
# Output:
# Reshaped with view(2, 3):
# tensor([[0, 1, 2],
#         [3, 4, 5]]), Shape: torch.Size([2, 3])
```

### 2. `tensor.reshape(shape)`

- **How it works:** Also changes the tensor's shape. It will try to return a _view_ (shared data) if possible (i.e., if the data is contiguous for the new shape). However, if a view isn't possible, `reshape` will return a _copy_ of the data with the desired shape.
- **Flexibility:** This makes `reshape` more robust than `view`. You get the shape you want, potentially at the cost of a memory copy if needed.
- **Recommendation:** Generally preferred over `view` unless you have a specific reason to guarantee a view or catch non-contiguous errors.

```python
# Script Snippet:
reshape_reshape = x.reshape(3, 2)
print(f"\nReshaped with reshape(3, 2):\n{reshape_reshape}, Shape: {reshape_reshape.shape}")
# Output:
# Reshaped with reshape(3, 2):
# tensor([[0, 1],
#         [2, 3],
#         [4, 5]]), Shape: torch.Size([3, 2])
```

## `view` vs. `reshape`: The Subtlety

- `x.view(shape)`: Guarantees to return a view or raise an error if not possible due to memory layout.
- `x.reshape(shape)`: Returns a view _if possible_, otherwise returns a copy. Safer bet in most cases.

## Flattening Tensors: The `-1` Trick

A common use case is flattening a multi-dimensional tensor into a 1D vector. You can achieve this by using `-1` for one of the dimensions in `reshape`. PyTorch will automatically infer the correct size for that dimension based on the total number of elements.

```python
# Script Snippet:
tensor_2d = torch.tensor([[1, 2], [3, 4]]) # Shape: [2, 2], 4 elements
print(f"\nOriginal 2D tensor:\n{tensor_2d}")

flattened = tensor_2d.reshape(-1) # Infer the size for one dimension
print(f"Flattened tensor: {flattened}, Shape: {flattened.shape}")
# Output:
# Original 2D tensor:
# tensor([[1, 2],
#         [3, 4]])
# Flattened tensor: tensor([1, 2, 3, 4]), Shape: torch.Size([4])
```

You can also use `-1` in combination with other dimensions, e.g., `tensor.reshape(batch_size, -1)` flattens all dimensions after the first one.

## Shape Compatibility

Remember, the total number of elements must match. Trying to reshape into an incompatible shape will result in an error.

```python
# Example (from script comments):
try:
    x.reshape(2, 4) # Original x has 6 elements, 2*4 = 8 elements - Incompatible!
except RuntimeError as e:
    print(f"\nError reshaping to (2, 4): {e}")
# Output:
# Error reshaping to (2, 4): shape '[2, 4]' is invalid for input of size 6
```

## Summary

Reshaping tensors using `view` or `reshape` is crucial for preparing data for different parts of a machine learning pipeline. `reshape` is generally more robust, while `view` can be slightly faster if you know the tensor is contiguous. The `-1` argument in `reshape` provides a convenient way to flatten tensors or infer dimensions. Always ensure the new shape is compatible with the total number of elements in the original tensor.
