# Guide: 01 Pixel Surgery: Indexing and Slicing Sprites!

Welcome, pixel surgeon! 🧑‍⚕️ Today, we learn how to precisely grab specific pixels, rows, columns, or even entire color channels from our sprite tensors. This guide dissects the techniques shown in `01_indexing_slicing.py`.

**Core Concept:** Your sprite tensor holds a grid of pixel data. Often, you don't want the _whole_ thing – maybe just the character's eye, the background color, or just the red channel. Indexing and slicing are your scalpels and tweezers for extracting exactly the pixel data you need. The syntax feels a lot like Python lists or NumPy arrays, so it might seem familiar!

## Our Patient: A 3x3 Rainbow Sprite

Let's work with a slightly bigger, more colorful sprite tensor (3x3 pixels, RGB channels). Remember: `[Height, Width, Channels]`.

```python
# Potion Ingredients:
import torch

# A 3x3 sprite with different colors
# Shape: [3, 3, 3]
sprite = torch.tensor([
  [[255, 0, 0],   [0, 255, 0],   [0, 0, 255]],  # R, G, B
  [[255, 255, 0], [255, 0, 255], [0, 255, 255]],  # Y, M, C
  [[0, 0, 0],   [128, 128, 128], [255, 255, 255]] # Black, Gray, White
], dtype=torch.uint8) # Using uint8 for realistic pixel values!

print(f"Original 3x3 RGB Sprite:\n{sprite}")
print(f"Sprite Shape: {sprite.shape}")
```

## Indexing: Grabbing Specific Pixels or Rows/Channels

Indexing lets you pinpoint individual items or whole slices along one dimension.

### 1. Snatching a Single Pixel's RGB Value

Need the color of the pixel at row 1, column 2 (the Cyan one)? Use `[row_index, col_index]`.

```python
# Spell Snippet:
pixel_1_2 = sprite[1, 2] # Row index 1, Column index 2
print(f"\nPixel at [1, 2] (Cyan): {pixel_1_2}, Shape: {pixel_1_2.shape}")
# Output:
# Pixel at [1, 2] (Cyan): tensor([  0, 255, 255], dtype=torch.uint8), Shape: torch.Size([3])
```

- **Decoding the Runes:** `sprite[1, 2]` gives us the RGB values `[0, 255, 255]` for that pixel. The result is a 1D tensor (vector) of shape `[3]`, holding the R, G, B values.

### 2. Extracting a Whole Row of Pixels

Want the middle row (Yellow, Magenta, Cyan)? Just provide the row index.

```python
# Spell Snippet:
middle_row = sprite[1] # Get the row at index 1
print(f"\nMiddle Row:\n{middle_row}, Shape: {middle_row.shape}")
# Output:
# Middle Row:
# tensor([[255, 255,   0],
#         [255,   0, 255],
#         [  0, 255, 255]], dtype=torch.uint8), Shape: torch.Size([3, 3])
```

- **Decoding the Runes:** `sprite[1]` grabs the entire second row. Notice the shape is `[3, 3]`, representing 3 pixels wide, each with 3 color channels.

## Slicing: Carving Out Regions and Channels

Slicing uses the mighty colon `:` (`start:stop:step`) to select ranges.

- `start`: Index to start at (included, default: beginning).
- `stop`: Index to stop **before** (excluded, default: end).
- `step`: How many to jump by (default: 1).
- Just `:` means "grab everything along this dimension".

### 1. Isolating a Color Channel (e.g., just the Green values)

Want to see how green everything is? Select all rows (`:`), all columns (`:`), and only the green channel (index `1` in the R,G,B dimension).

```python
# Spell Snippet:
green_channel = sprite[:, :, 1] # All rows, all columns, channel index 1 (Green)
print(f"\nGreen Channel:\n{green_channel}, Shape: {green_channel.shape}")
# Output:
# Green Channel:
# tensor([[  0, 255,   0],
#         [255,   0, 255],
#         [  0, 128, 255]], dtype=torch.uint8), Shape: torch.Size([3, 3])
```

- **Decoding the Runes:** `sprite[:, :, 1]` extracts only the green component of every pixel. The result is a 2D tensor (like a grayscale image) showing green intensity.

### 2. Grabbing a Column of Pixels

Need the middle column (Green, Magenta, Gray)? Select all rows (`:`), column index `1`.

```python
# Spell Snippet:
middle_column = sprite[:, 1] # All rows, column index 1
print(f"\nMiddle Column:\n{middle_column}, Shape: {middle_column.shape}")
# Output:
# Middle Column:
# tensor([[  0, 255,   0],
#         [255,   0, 255],
#         [128, 128, 128]], dtype=torch.uint8), Shape: torch.Size([3, 3])
```

- **Decoding the Runes:** `sprite[:, 1]` grabs the full RGB values for the 3 pixels in the middle column. Shape `[3, 3]` = 3 pixels high, 3 channels each.

### 3. Cutting Out a Rectangular Patch

Want the bottom-right 2x2 block (Magenta, Cyan, Gray, White)? Slice rows 1 onwards (`1:`), and columns 1 onwards (`1:`).

```python
# Spell Snippet:
# Rows from index 1 to end
# Columns from index 1 to end
bottom_right_corner = sprite[1:, 1:]
print(f"\nBottom-Right 2x2 block:\n{bottom_right_corner}, Shape: {bottom_right_corner.shape}")
# Output:
# Bottom-Right 2x2 block:
# tensor([[[255,   0, 255], # Magenta
#          [  0, 255, 255]], # Cyan
#
#         [[128, 128, 128], # Gray
#          [255, 255, 255]]], dtype=torch.uint8) # White
# , Shape: torch.Size([2, 2, 3])
```

- **Decoding the Runes:** `sprite[1:, 1:]` carves out the desired 2x2 pixel region, keeping all 3 color channels. Shape `[2, 2, 3]`.

## Pixel Surgery Reminders!

- **Start at 0:** Indices always start counting from zero!
- **`stop` is Exclusive:** The `stop` index in `start:stop` is where you stop _before_.
- **Watch the Shape:** Notice how slicing changes the shape! Getting a single pixel's RGB gives `[3]`. Getting a channel gives `[H, W]`. Getting a block gives `[BlockH, BlockW, C]`.

## Summary

You're now equipped with the indexing (`[]`) and slicing (`:`) tools to perform precise pixel surgery on your sprites! You can grab individual pixels, rows, columns, color channels, or rectangular patches. This is crucial for analyzing sprites, preparing data for specific model layers, or applying targeted effects!
