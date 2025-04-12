import torch

# Create a tensor
x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])  # Use floats for mean
print(f"Original tensor: {x}")

# --- Sum --- #
total_sum = x.sum()
# Alternative: torch.sum(x)
print(f"\nSum of all elements: {total_sum}, Scalar? {total_sum.ndim == 0}")

# --- Mean --- #
average = x.mean()
# Alternative: torch.mean(x)
print(f"Mean of all elements: {average}, Scalar? {average.ndim == 0}")

# --- Min / Max --- #
minimum = x.min()
maximum = x.max()
# Alternatives: torch.min(x), torch.max(x)
print(f"Minimum element: {minimum}")
print(f"Maximum element: {maximum}")

# --- Aggregation over dimensions --- #
# Create a 2D tensor
matrix = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print(f"\nOriginal 2D Matrix:\n{matrix}")

# Sum over columns (aggregate rows, dim=0)
sum_cols = matrix.sum(dim=0)
print(f"Sum over columns (dim=0): {sum_cols}, Shape: {sum_cols.shape}")

# Sum over rows (aggregate columns, dim=1)
sum_rows = matrix.sum(dim=1)
print(f"Sum over rows (dim=1): {sum_rows}, Shape: {sum_rows.shape}")

# Mean over columns (dim=0)
mean_cols = matrix.mean(dim=0)
print(f"Mean over columns (dim=0): {mean_cols}, Shape: {mean_cols.shape}")

# Mean over rows (dim=1)
mean_rows = matrix.mean(dim=1)
print(f"Mean over rows (dim=1): {mean_rows}, Shape: {mean_rows.shape}")
