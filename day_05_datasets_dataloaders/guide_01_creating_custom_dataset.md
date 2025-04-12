# Guide: 01 Creating a Custom Dataset

This guide explains how to create a custom dataset class in PyTorch by inheriting from `torch.utils.data.Dataset`, as demonstrated in `01_creating_custom_dataset.py`.

**Core Concept:** To feed data into your PyTorch models efficiently, especially with utilities like `DataLoader`, you need a standardized way to access your data samples. The `torch.utils.data.Dataset` abstract class provides this standard. By subclassing it, you tell PyTorch how to get the total number of samples and how to retrieve any specific sample by its index.

## Why Use `Dataset`?

- **Abstraction:** Hides the details of data storage (files, memory, database) behind a consistent interface.
- **Integration:** Works seamlessly with `torch.utils.data.DataLoader` for batching, shuffling, and parallel loading.
- **Organization:** Keeps data loading and access logic separate from model training code.
- **Flexibility:** Allows complex loading procedures within `__getitem__` (e.g., reading images, parsing text, applying initial transformations).

## The `Dataset` Contract: Required Methods

When you create your custom dataset class, you _must_ implement three special methods:

1. **`__init__(self, ...)`:** The Constructor

    - **Purpose:** Initialize the dataset. This usually involves loading metadata, finding file paths, or loading the entire dataset into memory if it's small enough.
    - **Action:** Store necessary information (like data tensors, file paths, labels, length) as attributes of `self`.
    - **Example (`SimpleTensorDataset`):** In the script, this method generates random feature and label tensors and stores them along with the total number of samples.

      ```python
      # Script Snippet (__init__):
      def __init__(self, num_samples=100, feature_dim=10, label_dim=1):
          super().__init__()
          self.num_samples = num_samples
          # In real code: load file paths, read CSV, etc.
          self.features = torch.randn(num_samples, feature_dim)
          self.labels = torch.randint(0, 2, (num_samples, label_dim)).float()
      ```

2. **`__len__(self)`:** Get Dataset Size

    - **Purpose:** Return the total number of samples in the dataset.
    - **Action:** Must return an integer representing the dataset size.
    - **Why?** `DataLoader` needs this to know how many indices are valid and how many batches to create.
    - \*\*Example (`SimpleTensorDataset`):

      ```python
      # Script Snippet (__len__):
      def __len__(self):
          return self.num_samples
      ```

3. **`__getitem__(self, idx)`:** Get a Single Sample
    - **Purpose:** Retrieve the data sample corresponding to the given index `idx`.
    - **Action:** Implement the logic to load, process (if needed), and return the single data sample for index `idx`. The index `idx` will range from `0` to `len(self) - 1`.
    - **Return Value:** Typically returns a tuple, often `(features, label)` or `(image, caption)`, etc.
    - **Why?** This is the core method called by `DataLoader` (or manually via `dataset[idx]`) to fetch individual data points for batching.
    - \*\*Example (`SimpleTensorDataset`):

      ```python
      # Script Snippet (__getitem__):
      def __getitem__(self, idx):
          # Basic index checking
          if not 0 <= idx < self.num_samples:
              raise IndexError(f"Index {idx} is out of bounds...")
          # Retrieve pre-loaded data
          feature = self.features[idx]
          label = self.labels[idx]
          return feature, label # Return as a tuple
      ```

## Using Your Custom Dataset

Once defined, you can instantiate your dataset and access samples using standard Python indexing, which automatically calls your `__getitem__` method:

```python
# Script Snippet (Usage):
dataset = SimpleTensorDataset(num_samples=50, feature_dim=5, label_dim=1)

# Get total length
print(f"Dataset length: {len(dataset)}") # Calls __len__() -> 50

# Get the first sample
feature_0, label_0 = dataset[0] # Calls __getitem__(0)
print(f"First feature: {feature_0}")
print(f"First label: {label_0}")
```

## Summary

Creating a custom `Dataset` in PyTorch involves inheriting from `torch.utils.data.Dataset` and implementing the `__init__`, `__len__`, and `__getitem__` methods. This provides a standardized way for PyTorch utilities like `DataLoader` to interact with your data, regardless of its underlying storage or complexity.
