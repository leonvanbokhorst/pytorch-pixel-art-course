# Guide: 04 Broadcasting: Pixel Magic with Mismatched Shapes!

Prepare to bend the rules of reality (or at least tensor shapes)! This guide unveils the powerful magic of **Broadcasting**, the technique behind `04_broadcasting.py` that lets PyTorch cleverly handle math between tensors of _different_ sizes.

**Core Concept:** Broadcasting is PyTorch's smart way of letting you, say, add a single brightness value to _every_ pixel in a sprite, or add a color tint vector to _every_ row, without you having to manually copy that value or vector over and over again. PyTorch _pretends_ the smaller tensor is stretched out to match the bigger one, performs the math element-wise, and does it **without actually using extra memory** for the stretched-out version. It's pure magic!

## Why is Broadcasting a Pixel Artist's Best Friend?

- **Less Code, More Art:** Forget writing loops or using `.repeat()` to manually make shapes match. Broadcasting makes common adjustments (like changing overall brightness or tinting) super concise.
- **Memory Saver:** That "stretching" is imaginary! No extra memory is wasted copying brightness values or color vectors for every single pixel.
- **Super Convenient:** Adding a single color offset to an entire RGB image? Subtracting the average brightness from every pixel? Broadcasting makes it a one-liner.

## The Secret Broadcasting Incantation (Rules)

How does PyTorch know if two differently shaped sprites can be broadcast together? It follows these rules, comparing shapes from **right-to-left** (trailing dimensions):

1.  **Line 'em Up:** Imagine the shapes aligned by their rightmost dimension.
2.  **Compare Dimensions:** For each dimension pair (moving right to left):
    - If the sizes are the **same**, great! Move to the next dimension leftwards.
    - If one of the sizes is **1**, PyTorch magically stretches this dimension to match the other tensor's size for the operation. Cool!
    - If the sizes are **different** and **neither is 1**, the spell fails! `RuntimeError`! They are not broadcastable.
3.  **Missing Dimensions?** If one tensor runs out of dimensions while comparing, PyTorch pretends it has dimensions of size 1 on its left until the number of dimensions matches. Then Rule 2 applies.

The final sprite's shape will be the element-wise _maximum_ size along each dimension from the inputs.

## Broadcasting Spells in Action!

### 1. Adding Brightness (Scalar Broadcasting)

- **Goal:** Add 50 brightness points to every pixel of a 2x2 sprite.
- **Operation:** `Sprite (2, 2) + Brightness (Scalar)`
- **Shapes:** `[2, 2]` vs `[]` (Scalar)
- **Magic:** Scalar treated as `[1, 1]`. Stretched to `[2, 2]`. Added element-wise.
- **Result Shape:** `[2, 2]`

```python
# Spell Snippet:
import torch
sprite = torch.tensor([[0, 50], [100, 150]])
brightness_boost = 50

result_sprite = sprite + brightness_boost
print(f"Original Sprite:\n{sprite}")
print(f"\nSprite + 50 Brightness:\n{result_sprite}")
# Output:
# Original Sprite:
# tensor([[  0,  50],
#         [100, 150]])
#
# Sprite + 50 Brightness:
# tensor([[ 50, 100],
#         [150, 200]])
```

### 2. Adding a Horizontal Gradient (Row Vector)

- **Goal:** Add a different offset to each _column_ of a 2x3 sprite.
- **Operation:** `Sprite (2, 3) + Gradient (3,)`
- **Shapes:** `[2, 3]` vs `[3]`
- **Magic:** Gradient `[3]` treated as `[1, 3]`. Dimension 0 (size 1) stretched to match sprite's dimension 0 (size 2). Result `[2, 3]` vs `[2, 3]`.
- **Result Shape:** `[2, 3]`

```python
# Spell Snippet:
sprite_2x3 = torch.tensor([[1, 2, 3], [4, 5, 6]])
col_offsets = torch.tensor([10, 20, 30]) # Shape (3,)

result_gradient = sprite_2x3 + col_offsets
print(f"\nOriginal 2x3 Sprite:\n{sprite_2x3}")
print(f"Column Offsets: {col_offsets}")
print(f"\nSprite + Column Offsets:\n{result_gradient}")
# Output:
# Original 2x3 Sprite:
# tensor([[1, 2, 3],
#         [4, 5, 6]])
# Column Offsets: tensor([10, 20, 30])
#
# Sprite + Column Offsets:
# tensor([[11, 22, 33],
#         [14, 25, 36]])
```

*(The `[10, 20, 30]` vector was added to *each* row)*

### 3. Adding a Vertical Gradient (Column Vector)

- **Goal:** Add a different offset to each _row_ of a 3x2 sprite.
- **Operation:** `Sprite (3, 2) + Gradient (3, 1)`
- **Shapes:** `[3, 2]` vs `[3, 1]`
- **Magic:** Dimension 1 (size 1) of the gradient is stretched to match sprite's dimension 1 (size 2). Result `[3, 2]` vs `[3, 2]`.
- **Result Shape:** `[3, 2]`

```python
# Spell Snippet:
sprite_3x2 = torch.tensor([[1, 2], [3, 4], [5, 6]])
row_offsets = torch.tensor([[10], [20], [30]]) # Shape (3, 1)

result_v_gradient = sprite_3x2 + row_offsets
print(f"\nOriginal 3x2 Sprite:\n{sprite_3x2}")
print(f"Row Offsets:\n{row_offsets}")
print(f"\nSprite + Row Offsets:\n{result_v_gradient}")
# Output:
# Original 3x2 Sprite:
# tensor([[1, 2],
#         [3, 4],
#         [5, 6]])
# Row Offsets:
# tensor([[10],
#         [20],
#         [30]])
#
# Sprite + Row Offsets:
# tensor([[11, 12],
#         [23, 24],
#         [35, 36]])
```

*(The column `[[10], [20], [30]]` was added to *each* column)*

### 4. Tinting an RGB Image

- **Goal:** Add a specific RGB tint `[R_offset, G_offset, B_offset]` to every pixel of an HxW image.
- **Operation:** `Image (H, W, 3) + Tint (3,)`
- **Shapes:** `[H, W, 3]` vs `[3]`
- **Magic:** Tint `[3]` treated as `[1, 1, 3]`. Dim 1 (size 1) stretched to W. Dim 0 (size 1) stretched to H. Result `[H, W, 3]` vs `[H, W, 3]`.
- **Result Shape:** `[H, W, 3]`

```python
# Conceptual Example (Code in 04_broadcasting.py is similar)
image = torch.zeros((2, 2, 3)) # Example 2x2 black image
tint = torch.tensor([50, 0, -20]) # Add some red, remove some blue (conceptually)

tinted_image = image + tint # Broadcasting does the magic!
# tinted_image will now have shape [2, 2, 3]
# where every pixel had [50, 0, -20] added.
```

### 5. When Magic Fails (Incompatible Shapes)

If the rules aren't met, PyTorch complains!

- **Operation:** `Sprite (2, 3) + Sprite (2, 1)`
- **Shapes:** `[2, 3]` vs `[2, 1]`
- **Magic Attempt:** Align right. Dim 1: Stretch size 1 to match 3 -> OK. Dim 0: Sizes are 2 and 2 -> OK.
- **Wait!** My example was wrong. This actually works!
- **Correct Failing Example:** `Sprite (3, 2) + Sprite (2, 3)`
- **Shapes:** `[3, 2]` vs `[2, 3]`
- **Magic Attempt:** Align right. Dim 1: Sizes 2 vs 3. Different, neither is 1. **FAIL!**
- **Result:** `RuntimeError`

```python
# Example of FAILURE:
sprite_a = torch.ones((3, 2))
sprite_b = torch.ones((2, 3))
try:
    result = sprite_a + sprite_b
except RuntimeError as e:
    print(f"\nError adding (3, 2) and (2, 3): {e}")
```

## Summary

Broadcasting is PyTorch's superpower for operating on tensors with compatible but different shapes. It saves memory and code by implicitly stretching dimensions of size 1. Master the rules (compare shapes right-to-left, dimensions must match or one must be 1), and you can easily add scalars (brightness), vectors (gradients/tints), and more to your pixel sprites without manual effort!
