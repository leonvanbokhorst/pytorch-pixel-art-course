# Guide: 05 DataLoader Options (Batching & Shuffling)

This guide revisits the core `DataLoader` arguments `batch_size` and `shuffle`, demonstrating their effects clearly, as shown in `05_dataloader_options.py`.

**Core Concept:** `DataLoader` provides several options to control how data is fetched, grouped, and ordered. The most fundamental are `batch_size` and `shuffle`.

## `batch_size` Revisited

- **Purpose:** Determines the number of samples grouped together in each batch yielded by the `DataLoader`.
- **Effect:** Controls the size of the first dimension of the tensors returned in each iteration.
- **Example:** If `batch_size=4`, the `features_batch` tensor will have shape `(4, *feature_shape)`.
- **Last Batch:** If the total dataset size is not perfectly divisible by `batch_size`, the last batch will contain the remaining samples (and thus might be smaller than `batch_size`), unless `drop_last=True` is set.

The script demonstrates creating loaders with different `batch_size` values (e.g., 4 vs. 3) to show how it affects the number of batches and the shape of the yielded tensors.

```python
# Script Snippet:
dataset = SimpleTensorDataset(num_samples=12)

# Batch size 4 -> 12 / 4 = 3 batches of size 4
dataloader_bs4 = DataLoader(dataset=dataset, batch_size=4, shuffle=False)
# for batch in dataloader_bs4: features_batch.shape -> torch.Size([4, ...])

# Batch size 3 -> 12 / 3 = 4 batches of size 3
dataloader_bs3 = DataLoader(dataset=dataset, batch_size=3, shuffle=False)
# for batch in dataloader_bs3: features_batch.shape -> torch.Size([3, ...])
```

## `shuffle` Revisited

- **Purpose:** Controls whether the data order is randomized before each epoch.
- **`shuffle=False` (Default):** The `DataLoader` fetches samples sequentially according to their index in the `Dataset` (0, 1, 2, ...).

  ```python
  # Script Snippet (shuffle=False):
  dataloader_noshuffle = DataLoader(dataset, batch_size=4, shuffle=False)
  # Batch 1 indices: tensor([0, 1, 2, 3]) 
  # Batch 2 indices: tensor([4, 5, 6, 7])
  # Batch 3 indices: tensor([8, 9, 10, 11])
    ```

- **`shuffle=True`:** Before each epoch (i.e., each full iteration through the `DataLoader`), the loader creates a randomly permuted list of all indices (`0` to `len(dataset)-1`). It then fetches samples and creates batches based on this _shuffled_ index order.

  ```python
  # Script Snippet (shuffle=True):
  # Set seed for potentially different shuffle each run
  torch.manual_seed(int(time.time()))
  dataloader_shuffle = DataLoader(dataset, batch_size=4, shuffle=True)
  # Example output (order will vary):
  # Batch 1 indices: tensor([ 7,  2,  9,  0])
  # Batch 2 indices: tensor([ 4,  6,  1, 11])
  # Batch 3 indices: tensor([ 3,  5, 10,  8])
  ```

- **Importance:** Shuffling is crucial during **training** to prevent the model from learning dependencies related to the data order and to ensure batches are representative of the overall dataset distribution.
- **Validation/Test:** You typically set `shuffle=False` for validation and testing loops to ensure consistent evaluation metrics across epochs or runs.

## Reproducibility vs. Randomness

- If you need the _exact same_ shuffling order every time you run the code (for debugging or strict reproducibility), set a fixed seed using `torch.manual_seed(...)` _before_ creating the `DataLoader` with `shuffle=True`.
- If you want different shuffling each time (which is normal for training), don't set a fixed seed, or use a changing seed like `int(time.time())` as shown in the script.

## Other Common `DataLoader` Options

While `batch_size` and `shuffle` are the most common, other useful arguments exist:

- **`num_workers`:** (Covered in the next section) Number of subprocesses to use for data loading. `0` means data is loaded in the main process.
- **`pin_memory` (bool):** If `True`, the loader will copy tensors into pinned memory before returning them. This can speed up CPU-to-GPU data transfers. Typically used when `num_workers > 0` and training on a GPU. Default is `False`.
- **`drop_last` (bool):** If `True`, drops the last incomplete batch if the dataset size is not evenly divisible by the batch size. If `False` (default), the last batch will be smaller.
- **`collate_fn` (callable):** A function that takes a list of samples (returned by `dataset.__getitem__`) and merges them into a batch. The default `collate_fn` handles tensors, numbers, dicts, and lists well, but you might provide a custom one for complex data structures (e.g., padding sequences of different lengths).

## Summary

`batch_size` controls the number of samples per batch, while `shuffle=True` randomizes the order of samples fetched from the `Dataset` before each epoch. These are the two primary arguments used to configure how `DataLoader` prepares data for model training and evaluation loops. Other options like `num_workers`, `pin_memory`, and `drop_last` offer further control over performance and batch handling.
