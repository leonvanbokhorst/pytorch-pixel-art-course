# Guide: 04 Using the DataLoader

This guide explains how to use `torch.utils.data.DataLoader` to efficiently load data in batches from your `Dataset`, as demonstrated in `04_using_dataloader.py`.

**Core Concept:** While `Dataset` defines _how_ to access individual data samples, `DataLoader` handles the process of retrieving these samples, grouping them into batches, optionally shuffling them, and potentially loading them in parallel using multiple processes. It's the standard tool for feeding data to your model during training and evaluation.

## Why Use `DataLoader`?

- **Batching:** Training on batches (mini-batches) of data instead of single samples is more computationally efficient and provides a more stable gradient estimate.
- **Shuffling:** Randomly shuffling the data before each epoch helps prevent the model from learning the order of samples and improves generalization.
- **Parallel Loading (Optional):** Can use multiple CPU worker processes (`num_workers`) to load data in the background while the GPU is busy with model computations, preventing data loading from becoming a bottleneck.
- **Automatic Collation:** Automatically combines individual samples fetched from the `Dataset` into batch tensors.

## Creating a `DataLoader`

You instantiate `DataLoader` by providing at least:

1. The `Dataset` object (or `Subset`) you want to load from.
2. The desired `batch_size` (number of samples per batch).
3. Optionally, `shuffle=True` to shuffle data before each epoch.

```python
# Script Snippet:
from torch.utils.data import Dataset, DataLoader

# Assume dataset is an instance of a Dataset class
dataset = SimpleTensorDataset(num_samples=25, feature_dim=8)
BATCH_SIZE = 5

data_loader = DataLoader(
    dataset=dataset,
    batch_size=BATCH_SIZE,
    shuffle=False # Typically True for training, False for validation/test
    # num_workers=0 # Default: load in main process
)
```

## Iterating Through Batches

The `DataLoader` object is an **iterable**. You can directly loop over it in a standard Python `for` loop. Each iteration yields one batch of data.

```python
# Script Snippet:
print("\nIterating through the DataLoader...")
# Calculate expected number of batches
num_batches = (len(dataset) + BATCH_SIZE - 1) // BATCH_SIZE

for batch_idx, batch_data in enumerate(data_loader):
    # Unpack the batch (assuming __getitem__ returned a tuple)
    features_batch, labels_batch = batch_data

    print(f"\nBatch {batch_idx + 1}/{num_batches}")
    print(f"  Features batch shape: {features_batch.shape}")
    print(f"  Labels batch shape: {labels_batch.shape}")

    # --- Training Loop Logic Would Go Here --- #
    # 1. Move batch to device (CPU/GPU)
    # features_batch = features_batch.to(device)
    # labels_batch = labels_batch.to(device)
    #
    # 2. Forward pass
    # outputs = model(features_batch)
    # ... etc ...
```

## What Happens Inside the Loop?

For each iteration, the `DataLoader` performs several steps automatically:

1. **Sampling:** Determines which indices to fetch for the current batch (respecting `shuffle` if enabled).
2. **Fetching:** Calls `dataset.__getitem__(idx)` for each sampled index (potentially using multiple worker processes if `num_workers > 0`).
3. **Collating:** Takes the individual samples returned by `__getitem__` (e.g., tuples of tensors `(feature, label)`) and combines them into batch tensors (e.g., `features_batch`, `labels_batch`). PyTorch handles stacking the tensors correctly along a new batch dimension (dimension 0).

## Batch Shapes

If your `dataset.__getitem__` returns `(feature, label)` where `feature` has shape `(F1, F2, ...)` and `label` has shape `(L1, L2, ...)`, the `DataLoader` will yield batches where:

- `features_batch` has shape `(batch_size, F1, F2, ...)`
- `labels_batch` has shape `(batch_size, L1, L2, ...)`
- Note: The last batch might be smaller than `batch_size` if the total dataset size is not evenly divisible by `batch_size`.

## Summary

`DataLoader` is the essential PyTorch utility for creating batches from a `Dataset`. By simply wrapping your `Dataset` instance and specifying `batch_size` and `shuffle`, you get an easy-to-use iterator that yields ready-to-use batches, significantly simplifying the data feeding process in training and evaluation loops.
