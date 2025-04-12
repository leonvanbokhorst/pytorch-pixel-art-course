# Guide: 06 Aggregation Functions

This guide covers tensor aggregation functions in PyTorch, which reduce tensors to summary values like sum, mean, min, or max, as shown in `06_aggregation_functions.py`.

**Core Concept:** Aggregation involves reducing the information in a tensor down to a smaller number of values (often a single scalar) that summarize the original data. Common examples include finding the total sum, the average value, or the minimum/maximum element.

## Why Aggregate?

- **Metrics:** Calculating performance metrics like total loss, average accuracy.
- **Analysis:** Finding minimum/maximum values, understanding data distribution (mean, standard deviation).
- **Feature Reduction:** Summarizing features along certain dimensions.

## Tensor-Wide Aggregation

The simplest form of aggregation applies the function across all elements of the tensor, returning a single scalar (0-dimensional tensor).

```python
# Script Snippet:
import torch

x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]) # Use floats for mean
print(f"Original tensor: {x}")

# Sum
total_sum = x.sum()
# Alternative: torch.sum(x)
print(f"\nSum of all elements: {total_sum}, Scalar? {total_sum.ndim == 0}") # Output: 15.0, True

# Mean
average = x.mean()
# Alternative: torch.mean(x)
print(f"Mean of all elements: {average}, Scalar? {average.ndim == 0}") # Output: 3.0, True

# Min / Max
minimum = x.min()
maximum = x.max()
# Alternatives: torch.min(x), torch.max(x)
print(f"Minimum element: {minimum}") # Output: 1.0
print(f"Maximum element: {maximum}") # Output: 5.0
```

- **Method vs. Function:** You can use either the tensor method (e.g., `x.sum()`) or the corresponding `torch` function (e.g., `torch.sum(x)`).

## Aggregation Along Specific Dimensions

Often, you want to aggregate only along a specific dimension of a multi-dimensional tensor. This is controlled using the `dim` argument.

**Key Idea:** The `dim` argument specifies the dimension **along which** the reduction occurs. This dimension will be removed (or reduced to size 1 if `keepdim=True`, not shown here) in the output tensor's shape.

Let's use a 2D matrix:

```python
# Script Snippet:
matrix = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) # Shape: (2, 3)
print(f"\nOriginal 2D Matrix:\n{matrix}")
# Output:
# Original 2D Matrix:
# tensor([[1., 2., 3.],
#         [4., 5., 6.]])
```

### Aggregating along `dim=0` (Collapsing Rows)

Specifying `dim=0` aggregates values _across the rows_. Think of collapsing the rows vertically.
The resulting tensor will have a shape reflecting the remaining dimensions (in this case, columns).

```python
# Script Snippet:
# Sum over columns (aggregate rows, dim=0)
sum_cols = matrix.sum(dim=0)
print(f"Sum over columns (dim=0): {sum_cols}, Shape: {sum_cols.shape}")
# Output:
# Sum over columns (dim=0): tensor([5., 7., 9.]), Shape: torch.Size([3])
# Explanation: [1+4, 2+5, 3+6] = [5, 7, 9]

# Mean over columns (dim=0)
mean_cols = matrix.mean(dim=0)
print(f"Mean over columns (dim=0): {mean_cols}, Shape: {mean_cols.shape}")
# Output:
# Mean over columns (dim=0): tensor([2.5000, 3.5000, 4.5000]), Shape: torch.Size([3])
# Explanation: [(1+4)/2, (2+5)/2, (3+6)/2] = [2.5, 3.5, 4.5]
```

- Input shape `(2, 3)`, `dim=0` specified -> Dimension 0 is removed -> Output shape `(3,)`

### Aggregating along `dim=1` (Collapsing Columns)

Specifying `dim=1` aggregates values _across the columns_. Think of collapsing the columns horizontally.
The resulting tensor will have a shape reflecting the remaining dimensions (in this case, rows).

```python
# Script Snippet:
# Sum over rows (aggregate columns, dim=1)
sum_rows = matrix.sum(dim=1)
print(f"Sum over rows (dim=1): {sum_rows}, Shape: {sum_rows.shape}")
# Output:
# Sum over rows (dim=1): tensor([ 6., 15.]), Shape: torch.Size([2])
# Explanation: [1+2+3, 4+5+6] = [6, 15]

# Mean over rows (dim=1)
mean_rows = matrix.mean(dim=1)
print(f"Mean over rows (dim=1): {mean_rows}, Shape: {mean_rows.shape}")
# Output:
# Mean over rows (dim=1): tensor([2., 5.]), Shape: torch.Size([2])
# Explanation: [(1+2+3)/3, (4+5+6)/3] = [2, 5]
```

- Input shape `(2, 3)`, `dim=1` specified -> Dimension 1 is removed -> Output shape `(2,)`

## Other Aggregation Functions

PyTorch includes many other useful aggregation functions, such as:

- `torch.std()` / `tensor.std()`: Standard deviation.
- `torch.prod()` / `tensor.prod()`: Product of elements.
- `torch.argmin()` / `tensor.argmin()`: Index of the minimum value.
- `torch.argmax()` / `tensor.argmax()`: Index of the maximum value.

## Summary

Aggregation functions (`sum`, `mean`, `min`, `max`, etc.) provide concise ways to summarize tensor data. They can operate over the entire tensor to produce a scalar result, or along specific dimensions using the `dim` argument to reduce the tensor's rank while summarizing across the specified dimension. Understanding how `dim` works is crucial for applying aggregations correctly in multi-dimensional scenarios.
