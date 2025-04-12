# Guide: 03 Slicing the Sprite Pie: Splitting Datasets!

Okay, you've got a whole folder (or list) of awesome pixel sprites in your `Dataset`. Now, how do you make sure your model actually _learns_ from them, instead of just memorizing? You split the pie! We need separate slices for training and evaluation. This guide shows how to use `torch.utils.data.random_split`, based on `03_dataset_splitting.py`.

**Core Concept:** To judge your model fairly, you can't test it on the exact same sprites it trained on (that's like giving it the answers before the test!). We need to split our collection:

- **Training Set:** The biggest slice. The model learns by looking at these sprites over and over, adjusting its internal knobs.
- **Validation Set:** A smaller slice, kept separate during training. We use this periodically to check how well the model is doing on sprites it _hasn't_ seen before. This helps us tune things like the learning rate and spot **overfitting** (when the model gets great at the training sprites but fails on new ones).
- **(Optional) Test Set:** Another slice, kept completely locked away until the _very end_. Used only _once_ after all training and tuning is done to get a final, unbiased score of how well the model performs on truly unseen data.

This splitting ensures we get a realistic idea of how well our pixel model will perform on new, unseen sprites.

## The Magic Splitter: `torch.utils.data.random_split`

PyTorch has a handy spell for this: `random_split`!

- **Ingredients:** You give it your original, complete `Dataset` (holding all your sprites) and a list/tuple saying how many sprites you want in each slice (e.g., `[num_train, num_val, num_test]`).
- **Requirement:** The total number of sprites requested for the slices _must_ exactly equal the total number of sprites in the original dataset.
- **Result:** It returns a tuple containing new `Dataset`-like objects (called `Subset`s), one for each slice.

## How to Cast `random_split`

1.  **Create Your Full Sprite Collection `Dataset`:** Make an instance of your `SimplePixelSpriteDataset` (or whatever you called it) containing _all_ your sprites.

    ```python
    # Spell Snippet:
    # Assuming SimplePixelSpriteDataset is defined
    # Let's say we have 100 total sprites
    all_sprites = [torch.randn(1, 8, 8) for _ in range(100)] # Dummy sprites
    full_pixel_dataset = SimplePixelSpriteDataset(all_sprites)
    print(f"Total sprites in full dataset: {len(full_pixel_dataset)}") # Output: 100
    ```

2.  **Decide on Slice Sizes:** Calculate how many sprites go into each slice (e.g., 80% for training, 20% for validation).

    ```python
    # Spell Snippet:
    TOTAL_SPRITES = len(full_pixel_dataset)
    TRAIN_FRACTION = 0.8

    train_count = int(TRAIN_FRACTION * TOTAL_SPRITES)
    val_count = TOTAL_SPRITES - train_count # The rest go to validation

    print(f"Splitting into: {train_count} training, {val_count} validation sprites.")
    # Output: Splitting into: 80 training, 20 validation sprites.
    ```

3.  **Cast the Spell!** Call `random_split` with the full dataset and the list of counts.

    ```python
    # Spell Snippet:
    from torch.utils.data import random_split

    # Get back two new dataset objects (Subsets)
    train_pixel_dataset, val_pixel_dataset = random_split(
        full_pixel_dataset, [train_count, val_count]
    )
    ```

    - _If you wanted a test set too:_ `train_set, val_set, test_set = random_split(full_set, [train_len, val_len, test_len])`

## Meet the `Subset`: A Slice of the Pie

The things returned by `random_split` (`train_pixel_dataset`, `val_pixel_dataset`) are special `Subset` objects. Think of them as windows onto the original `full_pixel_dataset`. They don't _copy_ the sprites, they just remember which _indices_ from the original dataset belong to them.

You can use a `Subset` just like a regular `Dataset`:

```python
# Spell Snippet:
print(f"\nLength of training dataset slice: {len(train_pixel_dataset)}") # -> 80
print(f"Length of validation dataset slice: {len(val_pixel_dataset)}")   # -> 20

# Get the first sprite *from the training slice*
# (This accesses the correct underlying sprite from full_pixel_dataset)
first_train_sprite = train_pixel_dataset[0]
print(f"Shape of first training sprite: {first_train_sprite.shape}")
```

## Using Slices with `DataLoader`

The next step (covered in the next guides) is to wrap each `Subset` in its own `DataLoader`. This lets you easily loop through batches of training sprites (often shuffled) and batches of validation sprites (usually not shuffled).

```python
# Preview of DataLoader usage:
# from torch.utils.data import DataLoader
# BATCH_SIZE = 16
# train_loader = DataLoader(train_pixel_dataset, batch_size=BATCH_SIZE, shuffle=True)
# val_loader = DataLoader(val_pixel_dataset, batch_size=BATCH_SIZE, shuffle=False)
```

## Keeping Splits Consistent (Reproducibility)

`random_split` shuffles the dataset randomly before slicing. If you need the _exact same_ sprites to end up in the training/validation sets every time you run your code (e.g., for comparing experiments fairly), set PyTorch's random seed _just before_ calling `random_split`:

```python
# Example for consistent splits:
torch.manual_seed(42) # Set the seed
train_pixel_dataset, val_pixel_dataset = random_split(full_pixel_dataset, [train_count, val_count])
# Now the split will be the same every time this code runs.
```

## Summary

Use `torch.utils.data.random_split` to slice your full pixel art `Dataset` into separate training, validation, and (optionally) test `Subset`s. Give it the full dataset and the desired number of sprites for each slice. This is essential for properly training and evaluating your pixel models to ensure they generalize well to new sprites!
