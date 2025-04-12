# Guide: 06 Pixel Summaries: Aggregation Functions!

Ever wonder what the _average_ color of your sprite is? Or the single _brightest_ pixel value? Aggregation functions are your magic magnifying glass 🔎 to get these kinds of summary stats from your pixel tensors! This guide explores the summarizing spells in `06_aggregation_functions.py`.

**Core Concept:** Aggregation is about taking a whole bunch of pixel values (maybe even your entire sprite!) and crunching them down into fewer numbers, often just one single summary value. Think `sum()`, `mean()`, `min()`, `max()`.

## Why Summarize Pixels?

- **Metrics & Analysis:** Calculate the average brightness (`mean`) to see if a generated sprite is too dark/light. Find the `max` pixel value to ensure it's within the expected range (e.g., <= 255 for `uint8`). Calculate total energy/intensity (`sum`).
- **Understanding Data:** Get a quick feel for the overall properties of a sprite or dataset.
- **Feature Engineering:** Sometimes, the average color or max intensity might be a useful feature itself.

## Sprite-Wide Summaries (The Big Picture)

Simplest case: summarize _all_ the pixel values in a tensor into one single number (a scalar).

Let's use a small grayscale sprite (using floats for `mean`):

```python
# Potion Ingredients:
import torch

grayscale_sprite = torch.tensor([ # 2x3 sprite
    [0.0, 0.5, 1.0],
    [0.2, 0.7, 0.3]
])
print(f"Original Grayscale Sprite:\n{grayscale_sprite}")

# Total Brightness (Sum)
total_intensity = grayscale_sprite.sum()
# Alternative: torch.sum(grayscale_sprite)
print(f"\nTotal Intensity (Sum): {total_intensity:.2f}, Scalar? {total_intensity.ndim == 0}") # Output: 2.70, True

# Average Brightness (Mean)
average_intensity = grayscale_sprite.mean()
# Alternative: torch.mean(grayscale_sprite)
print(f"Average Intensity (Mean): {average_intensity:.2f}, Scalar? {average_intensity.ndim == 0}") # Output: 0.45, True

# Darkest / Brightest Pixel (Min / Max)
darkest_pixel = grayscale_sprite.min()
brightest_pixel = grayscale_sprite.max()
# Alternatives: torch.min(grayscale_sprite), torch.max(grayscale_sprite)
print(f"Darkest Pixel Value: {darkest_pixel:.2f}") # Output: 0.00
print(f"Brightest Pixel Value: {brightest_pixel:.2f}") # Output: 1.00
```

- **Method vs. Function:** Just like arithmetic, you can use `sprite.sum()` or `torch.sum(sprite)` – dealer's choice!

## Summarizing Along Dimensions (Rows, Columns, Channels)

What if you want the average color _per channel_ in an RGB sprite? Or the average brightness _per row_? Use the `dim` argument!

**Key Idea:** `dim` tells PyTorch which dimension to **collapse** or **reduce**. The aggregation happens _across_ this dimension.

Let's summon our 3x3 RGB sprite again (Shape: `[H=3, W=3, C=3]`):

```python
# Spell Snippet:
sprite_rgb = torch.tensor([
  [[255, 0, 0],   [0, 255, 0],   [0, 0, 255]],  # R, G, B
  [[255, 255, 0], [255, 0, 255], [0, 255, 255]],  # Y, M, C
  [[0, 0, 0],   [128, 128, 128], [255, 255, 255]] # Black, Gray, White
], dtype=torch.float32) # Use float for mean calculation!

print(f"\nOriginal 3x3 RGB Sprite (float32):\n{sprite_rgb}")
```

### Average Color Across the Whole Sprite (`dim=[0, 1]`)

We want to average across Height (dim 0) AND Width (dim 1), leaving only the Channel dimension.

```python
# Spell Snippet:
# Average across Height (0) and Width (1) dimensions
average_color = sprite_rgb.mean(dim=[0, 1])
print(f"\nAverage Color (R, G, B) across sprite (averaged over dims 0, 1):\n{average_color.round()}, Shape: {average_color.shape}")
# Output:
# Average Color (R, G, B) across sprite (averaged over dims 0, 1):
# tensor([155., 110., 129.]), Shape: torch.Size([3])
# (Calculation: Sum all Red values / 9, Sum all Green / 9, Sum all Blue / 9)
```

- Input shape `(3, 3, 3)`, `dim=[0, 1]` -> Dims 0 and 1 removed -> Output shape `(3,)` (just the channel averages).

### Maximum Brightness Per Row (`dim=1`, then maybe `max` again)

Let's find the max R, G, B values within each row. We aggregate across the Width (dim 1).

```python
# Spell Snippet:
# Find max R, G, B value in each row (collapsing columns, dim=1)
max_per_row = sprite_rgb.max(dim=1) # This returns values and indices
max_values_per_row = max_per_row.values # We only care about the values here
print(f"\nMax R,G,B value per ROW (dim=1 collapsed):\n{max_values_per_row}, Shape: {max_values_per_row.shape}")
# Output:
# Max R,G,B value per ROW (dim=1 collapsed):
# tensor([[255., 255., 255.],  <- Max values in Row 0
#         [255., 255., 255.],  <- Max values in Row 1
#         [255., 255., 255.]]) <- Max values in Row 2
# Shape: torch.Size([3, 3]) # Shape is [H=3, C=3] now
```

- Input shape `(3, 3, 3)`, `dim=1` -> Dimension 1 (Width) removed -> Output shape `(3, 3)` (Height, Channels).
- _Note: `.max()` and `.min()` return both the values and their indices. We often just grab the `.values`._

## More Summarizing Spells!

PyTorch has others:

- `torch.std()`: Standard deviation (how spread out are the pixel values?).
- `torch.prod()`: Product (multiply all pixel values together).
- `torch.argmin()`, `torch.argmax()`: _Where_ is the min/max pixel value located (returns index)?

## Summary

Aggregation spells (`sum`, `mean`, `min`, `max`, etc.) let you condense your pixel data into meaningful summaries. You can get the overall picture (across the whole sprite) or zoom in on specific dimensions (like rows, columns, or channels) using the `dim` argument. This is vital for calculating metrics, understanding your sprite data, and even creating new features!
