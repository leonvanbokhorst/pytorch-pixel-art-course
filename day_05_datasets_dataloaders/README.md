# Day 5: Surveying the Land & Gathering Resources - Datasets & DataLoaders

**Fueling the Magic**

Our Pixel Paladin can now build intricate magical blueprints (`nn.Module`)! But these powerful constructs need fuel – the raw pixel data from the world around us. Today, we learn how to find, organize, and efficiently transport these essential resources. We discover **Datasets**, which are like detailed maps or inventories pinpointing every pixel sprite, every tile, every piece of visual information we want to work with. Then, we meet the **DataLoaders**, our indispensable enchanted carts, which automatically and efficiently gather these resources from the locations specified by the Dataset map, delivering them in perfectly sized batches to our model workshop. Proper resource logistics are crucial for any grand magical undertaking!

---

## 🎯 Objectives

**Topics:**

- `torch.utils.data.Dataset` for Pixel Art:
  - Purpose: Creating a standard way to access individual pixel art images/sprites from a collection.
  - Required methods: `__init__` (e.g., load filenames or data), `__len__` (total number of sprites), `__getitem__` (load and return a single sprite tensor by index).
  - Creating a custom `PixelArtDataset` class.
- `torch.utils.data.DataLoader` for Pixel Batches:
  - Purpose: Efficiently loading batches of pixel art for training, handling shuffling, and using multiple CPU cores.
  - Key arguments: `dataset` (our `PixelArtDataset`), `batch_size` (how many sprites per batch), `shuffle` (randomize order each epoch?), `num_workers` (parallel loading).
  - Iterating over the `DataLoader` to get batches of sprites.
- Data Loading Efficiency: How `num_workers` and `batch_size` affect training speed.

**Focus:** Building efficient and organized pipelines for loading pixel art data into PyTorch models.

## Key Resources

- **PyTorch Official Tutorials - Datasets & DataLoaders:** [https://pytorch.org/tutorials/beginner/basics/data_tutorial.html](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)
- **`torch.utils.data.Dataset` Documentation:** [https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset)
- **`torch.utils.data.DataLoader` Documentation:** [https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)
- **`torchvision.datasets`:** [https://pytorch.org/vision/stable/datasets.html](https://pytorch.org/vision/stable/datasets.html) (Contains standard image datasets, good examples)
- **`torchvision.transforms`:** [https://pytorch.org/vision/stable/transforms.html](https://pytorch.org/vision/stable/transforms.html) (Common image transformations)

## Hands-On Pixel Loading Examples

- **Creating a Custom `PixelArtDataset`:** ([`01_creating_custom_dataset.py`](./01_creating_custom_dataset.py))
  - **Pixel Idea:** Define a `SimplePixelSpriteDataset` inheriting `Dataset`. `__init__` could take a list of small, predefined tensor sprites. `__len__` returns the list length. `__getitem__` returns the sprite tensor at the given index.
  - **Purpose:** Demonstrate the basic structure of a custom `Dataset` for simple pixel art data.
- **Applying Image Transforms:** ([`02_applying_transforms.py`](./02_applying_transforms.py))
  - **Pixel Idea:** Modify the `Dataset` to accept a `transform`. In `__getitem__`, apply a simple transform like converting a uint8 tensor (0-255) to a float tensor (0.0-1.0) using `sprite.float() / 255.0`. Maybe introduce `torchvision.transforms.ToTensor` if using PIL images conceptually.
  - **Purpose:** Show how to integrate preprocessing (like normalization) or augmentation into the pixel data loading process.
- **Splitting Pixel Datasets (Train/Validation):** ([`03_dataset_splitting.py`](./03_dataset_splitting.py))
  - **Pixel Idea:** Instantiate the `SimplePixelSpriteDataset`. Use `torch.utils.data.random_split` to divide it into training and validation sets of sprites.
  - **Purpose:** Show the standard practice for splitting pixel art data for training and evaluation.
- **Using `DataLoader` for Sprite Batches:** ([`04_using_dataloader.py`](./04_using_dataloader.py))
  - **Pixel Idea:** Wrap the `SimplePixelSpriteDataset` (or a split) in a `DataLoader`. Iterate through it and print the shape of the sprite batches (e.g., `[batch_size, channels, height, width]`).
  - **Purpose:** Demonstrate batching pixel art data using `DataLoader`.
- **`DataLoader` Options (Batch Size & Shuffling):** ([`05_dataloader_options.py`](./05_dataloader_options.py))
  - **Pixel Idea:** Create `DataLoader` instances with different `batch_size` values and `shuffle=True/False`. Iterate to show how batch sizes change and how shuffling affects the order of sprites seen.
  - **Purpose:** Illustrate controlling batch size and data order with `DataLoader`.
- **(Optional) Parallel Pixel Loading (`num_workers`):** ([`06_optional_num_workers.py`](./06_optional_num_workers.py))
  - **Pixel Idea:** Show creating a `DataLoader` with `num_workers > 0` for potentially faster loading of pixel art (especially if loading from disk and applying transforms).
  - **Purpose:** Introduce parallel data loading for pixel datasets.
