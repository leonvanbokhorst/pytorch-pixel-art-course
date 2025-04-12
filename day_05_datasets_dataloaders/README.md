# Day 5: Data Loading with Dataset and DataLoader

**Topics:**

- `torch.utils.data.Dataset`:
  - Purpose: Abstracting data access.
  - Required methods: `__init__`, `__len__`, `__getitem__`.
  - Creating a custom Dataset class.
- `torch.utils.data.DataLoader`:
  - Purpose: Batching, shuffling, parallel loading.
  - Key arguments: `dataset`, `batch_size`, `shuffle`, `num_workers`.
  - Iterating over the DataLoader to get batches.
- Efficiency: Understanding `num_workers` and batch size trade-offs.

**Focus:** Structuring data input pipelines efficiently and flexibly using PyTorch's standard utilities.

## Key Resources

- **PyTorch Official Tutorials - Datasets & DataLoaders:** [https://pytorch.org/tutorials/beginner/basics/data_tutorial.html](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html) (Covers creating custom Datasets, using DataLoader for iteration and batching)
- **`torch.utils.data.Dataset` Documentation:** [https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset) (API for the Dataset abstract class)
- **`torch.utils.data.DataLoader` Documentation:** [https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) (API for the DataLoader class, including arguments like `batch_size`, `shuffle`, `num_workers`)

## Hands-On Examples

- **Creating a Custom `Dataset`:** ([`01_creating_custom_dataset.py`](./01_creating_custom_dataset.py))
  - **Code Idea:** Define a `SimpleTensorDataset` class inheriting from `torch.utils.data.Dataset`. Implement `__init__` to store some sample data (e.g., pairs of random tensors), `__len__` to return the total number of samples, and `__getitem__` to retrieve a single data sample (feature/label pair) by index.
  - **Purpose:** Demonstrate the fundamental structure of a custom Dataset and how to implement its required methods.
- **Applying Data Transforms:** ([`02_applying_transforms.py`](./02_applying_transforms.py))
  - **Code Idea:** Create a slightly modified `Dataset` that accepts a `transform` argument in `__init__`. Inside `__getitem__`, apply the transform (if provided) to the feature data before returning it. Demonstrate with a simple lambda function or a `torchvision.transforms` example (like normalization, if applicable to the dummy data).
  - **Purpose:** Show how to incorporate data preprocessing or augmentation steps into the `Dataset` using transforms.
- **Dataset Splitting (Train/Validation):** ([`03_dataset_splitting.py`](./03_dataset_splitting.py))
  - **Code Idea:** Instantiate a full dataset (e.g., `SimpleTensorDataset`). Use `torch.utils.data.random_split` to divide it into training and validation subsets based on specified lengths or fractions. Print the lengths of the resulting datasets.
  - **Purpose:** Demonstrate the standard way to create training and validation sets from a single dataset for proper model evaluation.
- **Using the `DataLoader`:** ([`04_using_dataloader.py`](./04_using_dataloader.py))
  - **Code Idea:** Instantiate the `SimpleTensorDataset` (or use a split dataset from previous example). Create a `DataLoader` instance, passing the dataset to it. Iterate through the `DataLoader` using a `for` loop and print the shape of the batches obtained.
  - **Purpose:** Show how to wrap a `Dataset` in a `DataLoader` to easily iterate over data in batches.
- **`DataLoader` Options (Batching & Shuffling):** ([`05_dataloader_options.py`](./05_dataloader_options.py))
  - **Code Idea:** Create `DataLoader` instances from the same `Dataset` but with different `batch_size` values and with `shuffle=True` vs `shuffle=False`. Iterate to observe the effects.
  - **Purpose:** Illustrate the core functionalities of `DataLoader` for batching and shuffling data.
- **(Optional) `DataLoader` with `num_workers`:** ([`06_optional_num_workers.py`](./06_optional_num_workers.py))
  - **Code Idea:** Briefly demonstrate creating a `DataLoader` with `num_workers` > 0.
  - **Purpose:** Introduce the concept of parallel data loading.
