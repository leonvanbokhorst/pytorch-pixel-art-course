# Guide: 02 Inspecting Your Pixel-Tensor's Stats! 🕵️‍♀️

Alright, pixel investigator! We've conjured some tensors. Now, how do we check their vitals? This guide explores the inspection spells used in `02_checking_tensor_attributes.py`.

**Core Concept:** Every pixel-tensor you summon has secret properties – its size, dimensions, and the _type_ of pixel data it holds. Knowing these is like knowing your sprite's height, width, and color depth – super important before sending it into battle (or a neural network)!

## Our Specimen: A Tiny RGB Sprite

Instead of a boring old matrix, let's create a tiny 2x3 pixel RGB sprite. Remember, for color images, we usually need 3 dimensions: Height, Width, and Color Channels (Red, Green, Blue).

```python
# Potion Ingredients:
import torch

# Let's imagine a 2-pixel high, 3-pixel wide sprite
# Each pixel has [R, G, B] values (0-255)
# Shape will be [Height, Width, Channels] -> [2, 3, 3]
tiny_rgb_sprite = torch.tensor([
  [[255, 0, 0],   [0, 255, 0],   [0, 0, 255]],  # Row 1: Red, Green, Blue pixels
  [[255, 255, 0], [0, 255, 255], [255, 0, 255]]   # Row 2: Yellow, Cyan, Magenta pixels
])

print(f"Our Tiny RGB Sprite:\\n{tiny_rgb_sprite}")
```

This little guy has 2 rows of pixels, 3 columns, and each pixel holds 3 color values.

## Inspecting the Sprite's Stats!

Now, let's use our PyTorch inspection tools (`.shape`, `.ndim`, `.dtype`) to reveal its secrets:

### 1. `.shape`: The Sprite's Dimensions (Height, Width, Channels)

- **What it is:** This magical attribute tells you the size of your tensor in each dimension. It returns a `torch.Size` object (think tuple!).
- **Pixel Analogy:** Exactly what it sounds like! For our RGB sprite, it's Height x Width x Color Channels. For a grayscale sprite, it would just be Height x Width.
- **The Reveal:**

  ```python
  # Spell Snippet:
  print(f"Sprite Shape (H, W, C): {tiny_rgb_sprite.shape}")
  # Expected Output:
  # Sprite Shape (H, W, C): torch.Size([2, 3, 3])
  ```

- **Decoding the Runes:** `torch.Size([2, 3, 3])` confirms our sprite has 2 pixels vertically (Height), 3 pixels horizontally (Width), and 3 values per pixel (Channels: R, G, B). No guesswork needed!

### 2. `.ndim`: How Many Dimensions? (Grayscale vs. Color)

- **What it is:** Tells you the _number_ of dimensions your tensor has. It's just the count of numbers in the `.shape`.
- **Pixel Analogy:** This often distinguishes grayscale from color!
  - A single pixel value (scalar) = 0 dimensions (`ndim=0`).
  - An RGB color vector `[R, G, B]` = 1 dimension (`ndim=1`).
  - A grayscale sprite (Height x Width) = 2 dimensions (`ndim=2`).
  - An RGB sprite (Height x Width x Channels) = 3 dimensions (`ndim=3`).
  - A _batch_ of RGB sprites? That'd be 4 dimensions (`ndim=4`)!
- **The Reveal:**

  ```python
  # Spell Snippet:
  print(f"Number of Dimensions (Rank): {tiny_rgb_sprite.ndim}")
  # Expected Output:
  # Number of Dimensions (Rank): 3
  ```

- **Decoding the Runes:** The output `3` confirms our `tiny_rgb_sprite` is a 3-dimensional tensor, perfect for holding Height, Width, and Color info.

### 3. `.dtype`: Pixel Precision (Data Type)

- **What it is:** Reveals the _type_ of data stored in the tensor. Are they whole numbers (integers)? Numbers with decimals (floats)? True/False flags (booleans)?
- **Pixel Analogy:** This is like the color depth or format!
  - `torch.uint8`: Unsigned 8-bit integer. Perfect for standard 0-255 pixel values. Very common for storing/loading images.
  - `torch.float32`: 32-bit floating-point. Often used as _input_ to neural networks, usually scaled to be between 0.0 and 1.0, or sometimes -1.0 and 1.0. Offers more precision.
  - `torch.int64`: 64-bit integer. What PyTorch often defaults to if you feed it Python ints. Can be overkill for pixel values, using more memory.
- **Importance:** Affects memory use, calculation precision, and whether your GPU smiles or frowns upon the data.
- **The Reveal:**

  ```python
  # Spell Snippet:
  print(f"Pixel Data Type: {tiny_rgb_sprite.dtype}")
  # Expected Output:
  # Pixel Data Type: torch.int64
  ```

- **Decoding the Runes:** `torch.int64` tells us PyTorch stored our 0-255 values as big integers. Why? Because the Python lists we used contained standard Python integers. We'll soon learn how to _specify_ the `dtype` if we want `torch.uint8` or `torch.float32` instead!

## Quick Recap!

You're now a Tensor Inspector! Checking `.shape`, `.ndim`, and `.dtype` lets you verify your pixel data's dimensions (H, W, C), dimensionality (grayscale/color/batch), and data format (uint8/float32/etc.). These checks are your best friends for debugging and making sure your pixel sprites are ready for the deep learning magic ahead!

Next up: Taking control of that `dtype`!
