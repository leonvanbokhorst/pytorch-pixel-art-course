# Guide: 04 (Optional) Tensor from NumPy

This guide covers the interaction between PyTorch Tensors and NumPy arrays, as demonstrated in `04_optional_tensor_from_numpy.py`.

**Core Concept:** PyTorch is designed to work seamlessly with NumPy, a fundamental package for scientific computing in Python. This allows you to leverage existing NumPy code and data within your PyTorch workflows.

## Why Bridge NumPy and PyTorch?

- **Leverage Existing Code:** Many data loading, preprocessing, and analysis pipelines are already written using NumPy.
- **Familiarity:** If you're coming from a NumPy background, understanding the connection helps ease the transition.
- **Data Handling:** Easily convert data between the two formats as needed for different stages of your project.

## NumPy Array -> PyTorch Tensor: `torch.from_numpy()`

The primary way to convert a NumPy `ndarray` into a PyTorch `Tensor` is using `torch.from_numpy()`.

```python
# Script Snippet:
import torch
import numpy as np

numpy_array = np.array([1.0, 2.0, 3.0])
print(f"NumPy array: {numpy_array}, ... Dtype: {numpy_array.dtype}")

# Conversion
tensor_from_numpy = torch.from_numpy(numpy_array)
print(f"Tensor from NumPy: {tensor_from_numpy}, ... Dtype: {tensor_from_numpy.dtype}")

# Output (Example):
# NumPy array: [1. 2. 3.], ... Dtype: float64
# Tensor from NumPy: tensor([1., 2., 3.], dtype=torch.float64), ... Dtype: torch.float64
```

- **Data Type:** Notice that PyTorch usually preserves the NumPy array's data type (e.g., `float64` remains `torch.float64`).

## PyTorch Tensor -> NumPy Array: `.numpy()`

To convert a PyTorch `Tensor` back into a NumPy `ndarray`, you can call the `.numpy()` method on the tensor.

**Important Constraint:** The `.numpy()` method **only works if the tensor resides on the CPU**. If the tensor is on a GPU, you must first move it to the CPU using `.cpu()` before calling `.numpy()` (e.g., `tensor.cpu().numpy()`). We'll cover CPU/GPU placement later.

```python
# Script Snippet:
tensor = torch.tensor([4.0, 5.0, 6.0]) # This tensor is on the CPU by default
numpy_from_tensor = tensor.numpy()
print(f"Tensor: {tensor}")
print(f"NumPy array from tensor: {numpy_from_tensor}, ... Dtype: {numpy_from_tensor.dtype}")

# Output (Example):
# Tensor: tensor([4., 5., 6.])
# NumPy array from tensor: [4. 5. 6.], ... Dtype: float64
```

## The Crucial Point: Shared Memory

**This is the most important concept in this section!**

When you use `torch.from_numpy()` or `tensor.numpy()`, the resulting object **shares the same underlying memory** as the original object (assuming the tensor is on the CPU).

- **What it means:** If you modify the data in the NumPy array, the corresponding tensor created with `torch.from_numpy()` will reflect those changes, and vice-versa.
- **Why?** Efficiency. No data needs to be copied, which saves time and memory, especially for large arrays.
- **Caveat:** This can lead to unexpected behavior if you're not aware of it!

```python
# Script Snippet Demonstrating Shared Memory:

# Modify NumPy array -> Tensor changes
numpy_array[0] = 99.0
print(f"Modified NumPy array: {numpy_array}")
# tensor_from_numpy will also show 99.0 at index 0
print(f"Tensor after modifying NumPy array: {tensor_from_numpy}")

# Modify Tensor -> NumPy array changes
tensor[0] = 111.0
print(f"Modified Tensor: {tensor}")
# numpy_from_tensor will also show 111.0 at index 0
print(f"NumPy array after modifying tensor: {numpy_from_tensor}")
```

## Want a Copy Instead?

If you _do not_ want the tensor and array to share memory, you should create the tensor using `torch.tensor()` directly with the NumPy array as input. This function _copies_ the data.

```python
independent_tensor = torch.tensor(numpy_array) # This creates a copy
```

Similarly, you can use `tensor.clone().numpy()` to get a NumPy copy that doesn't share memory with the original tensor.

## Summary

PyTorch provides efficient ways to convert between NumPy arrays and PyTorch tensors using `torch.from_numpy()` and `tensor.numpy()`. The key takeaway is that these methods typically result in shared memory (for CPU tensors), making modifications in one object visible in the other. Use `torch.tensor(numpy_array)` or `tensor.clone().numpy()` when you explicitly need an independent copy of the data.
