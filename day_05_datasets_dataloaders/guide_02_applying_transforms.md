# Guide: 02 Applying Data Transforms

This guide explains how to integrate data preprocessing and augmentation into your custom PyTorch `Dataset` using transforms, as demonstrated in `02_applying_transforms.py`.

**Core Concept:** Raw data often needs transformation before being fed into a neural network. This can include preprocessing steps (like normalization, tensor conversion) or data augmentation techniques (like random rotations or flips for images) to improve model performance and robustness. PyTorch `Dataset` classes typically incorporate a `transform` argument to handle these operations efficiently.

## Why Use Transforms?

1. **Preprocessing:** Ensuring data is in the correct format, scale, or distribution expected by the model (e.g., converting PIL images to tensors, normalizing pixel values).
2. **Data Augmentation:** Artificially increasing the diversity of your training data by applying random transformations to samples each time they are accessed. This helps the model generalize better and reduces overfitting.

## Integrating Transforms into `Dataset`

The standard pattern involves two modifications to your custom `Dataset` class:

1. **Accept `transform` in `__init__`:**

    - Add an optional argument (e.g., `transform=None`) to your `__init__` method.
    - Store the passed transform callable (function or object) in an instance attribute (e.g., `self.transform`).

    ```python
    # Snippet: Modified __init__
    def __init__(self, ..., transform=None):
        # ... other initializations ...
        self.transform = transform
    ```

2. **Apply `transform` in `__getitem__`:**

    - Inside the `__getitem__` method, after loading or retrieving the raw data sample (specifically the features you want to transform), check if `self.transform` was provided.
    - If it exists, apply it to the feature data _before_ returning the sample.

    ```python
    # Snippet: Modified __getitem__
    def __getitem__(self, idx):
        # ... load feature, label ...
        feature = self.features[idx] # Example: retrieve feature
        label = self.labels[idx]     # Example: retrieve label

        # Apply transform *only* to the feature (usually)
        if self.transform:
            feature = self.transform(feature)

        return feature, label
    ```

## How it Works: On-the-Fly Transformation

By applying the transform within `__getitem__`, the transformation happens dynamically each time a sample is requested by the `DataLoader`. This is particularly beneficial for data augmentation, as a different random transformation can be applied to the same underlying sample each time it appears in a new epoch during training.

## Example: Simple Lambda Transform

The script uses a simple `lambda` function to demonstrate the concept:

```python
# Script Snippet (Usage):
add_10_transform = lambda x: x + 10 # Define transform

dataset_with_transform = SimpleTensorDatasetWithTransform(
    ..., transform=add_10_transform # Pass transform during instantiation
)

# When dataset_with_transform[0] is called:
# 1. __getitem__(0) retrieves original feature_0 and label_0
# 2. It applies add_10_transform to feature_0
# 3. It returns (feature_0 + 10, label_0)
feature_0_transformed, label_0_transformed = dataset_with_transform[0]
```

## `torchvision.transforms`

For more complex or standard operations, especially on images, the `torchvision.transforms` module is indispensable. It provides pre-built transforms for:

- Resizing, cropping, padding, flipping, rotating images.
- Converting between PIL Images and Tensors (`transforms.ToTensor`).
- Normalizing tensor data (`transforms.Normalize`).
- Adjusting brightness, contrast, saturation.

You can chain multiple transforms together using `transforms.Compose`:

```python
# Example (Conceptual - for image data)
# import torchvision.transforms as transforms
# image_transform = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
# dataset = YourImageDataset(..., transform=image_transform)
```

## Preprocessing vs. Augmentation

- **Preprocessing** transforms (like `ToTensor`, `Normalize`) are usually applied to both training and validation/test data.
- **Augmentation** transforms (like `RandomHorizontalFlip`) are typically applied _only_ to the training data to increase its variety without changing the validation/test distribution.
- You often create separate `Dataset` instances (or separate `transform` pipelines) for training and validation.

## Summary

Transforms are integrated into PyTorch datasets by passing a callable `transform` during `Dataset` initialization and applying it within the `__getitem__` method before returning the sample. This allows for flexible on-the-fly preprocessing and data augmentation, commonly using `torchvision.transforms` for standard operations.
