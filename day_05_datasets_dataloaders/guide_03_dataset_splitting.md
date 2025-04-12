# Guide: 03 Dataset Splitting (Train/Validation/Test)

This guide explains how to split a single PyTorch `Dataset` into multiple non-overlapping subsets (typically for training, validation, and testing) using `torch.utils.data.random_split`, as shown in `03_dataset_splitting.py`.

**Core Concept:** To train and evaluate machine learning models properly, you need separate datasets:

- **Training Set:** Used to train the model (adjust weights based on gradients).
- **Validation Set:** Used periodically during training to tune hyperparameters (like learning rate, model architecture) and check for overfitting. The model doesn't train _directly_ on this data.
- **Test Set:** Used _only once_ after all training and hyperparameter tuning is complete to get an unbiased estimate of the model's final performance on unseen data.

Splitting ensures that the model is evaluated on data it has never seen during the training process, providing a realistic measure of its generalization ability.

## `torch.utils.data.random_split`

PyTorch provides a convenient function for this purpose: `torch.utils.data.random_split`.

- **Input:** Takes the original, full `Dataset` object and a sequence (list or tuple) of integer lengths for the desired splits.
- **Important:** The sum of the lengths in the sequence _must_ equal the total length of the original dataset.
- **Output:** Returns a tuple of `torch.utils.data.Subset` objects, one for each specified length.

## How to Use `random_split`

The process typically involves:

1. **Instantiate Full Dataset:** Create an instance of your complete dataset.

    ```python
    # Script Snippet:
    from torch.utils.data import Dataset # Assuming SimpleTensorDataset is defined
    full_dataset = SimpleTensorDataset(num_samples=1000, ...)
    ```

2. **Calculate Split Lengths:** Determine the number of samples for each split (e.g., based on fractions).

    ```python
    # Script Snippet:
    TOTAL_SAMPLES = len(full_dataset)
    TRAIN_FRACTION = 0.8
    train_size = int(TRAIN_FRACTION * TOTAL_SAMPLES)
    val_size = TOTAL_SAMPLES - train_size # Remainder for validation
    ```

3. **Perform the Split:** Call `random_split` with the dataset and the calculated lengths.

    ```python
    # Script Snippet:
    from torch.utils.data import random_split
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    ```

    - If you needed a test set too: `train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_len, val_len, test_len])`

## `Subset` Objects

The objects returned by `random_split` (`train_dataset`, `val_dataset` in the example) are instances of `torch.utils.data.Subset`. A `Subset` wraps the original `full_dataset` but only exposes the specific indices assigned to it during the split. You can interact with a `Subset` just like a regular `Dataset`:

- Get its length: `len(train_dataset)`
- Access items by index: `train_dataset[i]` (This internally maps `i` to the correct index in the `full_dataset`).

```python
# Script Snippet:
print(f"Length of train_dataset: {len(train_dataset)}") # -> 800
print(f"Length of val_dataset: {len(val_dataset)}")   # -> 200
train_feature_0, train_label_0 = train_dataset[0] # Access first sample of train split
```

## Using Splits with `DataLoader`

Typically, you will create separate `DataLoader` instances for each split, often with different settings (e.g., shuffling for the training set but not for validation/test).

```python
# Script Snippet:
from torch.utils.data import DataLoader

BATCH_SIZE = 64
# Shuffle training data each epoch
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# No need to shuffle validation data
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
```

## Reproducibility Note

`random_split` performs a _random_ shuffling before assigning indices. If you need the _exact same_ split every time you run your code (for reproducibility), you should set the random seed _before_ calling `random_split`:

```python
# Example for reproducible split:
torch.manual_seed(42)
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
```

## Summary

`torch.utils.data.random_split` is the standard PyTorch utility for dividing a dataset into non-overlapping training, validation, and/or test subsets. Provide the full dataset and a list of desired integer lengths for each split. The resulting `Subset` objects can be used directly with `DataLoader`s to manage the different phases of model training and evaluation.
