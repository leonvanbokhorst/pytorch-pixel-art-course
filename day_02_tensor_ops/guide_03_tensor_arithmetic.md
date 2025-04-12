# Guide: 03 Pixel Math: Basic Arithmetic with Sprites!

Time for some pixel alchemy! ✨ We'll learn how to perform basic math operations (add, subtract, multiply, divide) on our sprite tensors. This guide demystifies the pixel math shown in `03_tensor_arithmetic.py`.

**Core Concept:** PyTorch lets you do math directly on tensors! When two tensors have the _same shape_, these operations usually happen _element-wise_. Think of it like overlaying two sprites perfectly and applying the math to the corresponding pixels at each `[row, column]` location (and for each color channel if it's RGB!).

## The Shape Shackle!

For these basic operations, your pixel tensors generally need to be **identically shaped**. If you have a 16x16 sprite, you can directly add it to another 16x16 sprite. Adding it to an 8x8 sprite directly? Not gonna work (yet!).

_(Pssst... there's a powerful spell called **Broadcasting** that *can* handle different shapes sometimes. We'll learn that next! But for now, assume shapes must match.)_

## Our Experimental Sprites

Let's create two simple 2x2 grayscale sprites (pretend they are `uint8` 0-255, but we'll use small ints for clarity):

```python
# Potion Ingredients:
import torch

# Sprite 1: A simple gradient
sprite1 = torch.tensor([[0, 50], [100, 150]])
# Sprite 2: A constant value patch
sprite2 = torch.tensor([[10, 10], [20, 20]])

print(f"Sprite 1:\n{sprite1}")
print(f"Sprite 2:\n{sprite2}")
# Output:
# Sprite 1:
# tensor([[  0,  50],
#         [100, 150]])
# Sprite 2:
# tensor([[10, 10],
#         [20, 20]])
```

## Element-wise Pixel Magic

Let's see what happens when we combine them!

### 1. Addition (`+` or `torch.add`): Brightness Boost / Blending

Adds the values of corresponding pixels. Great for increasing brightness or blending sprites together.

```python
# Spell Snippet:
brighter_sprite = sprite1 + sprite2
# alternative: torch.add(sprite1, sprite2)
print(f"\nAdded Sprites (Brightness Boost):\n{brighter_sprite}")
# Output:
# Added Sprites (Brightness Boost):
# tensor([[ 10,  60],
#         [120, 170]])
```

- _(Pixel [0,0]: 0+10=10, Pixel [0,1]: 50+10=60, Pixel [1,0]: 100+20=120, Pixel [1,1]: 150+20=170)_

### 2. Subtraction (`-` or `torch.sub`): Finding Differences

Subtracts corresponding pixel values. Useful for seeing the difference between two sprites.

```python
# Spell Snippet:
difference_sprite = sprite1 - sprite2
# alternative: torch.sub(sprite1, sprite2)
print(f"\nSubtracted Sprites (Difference):\n{difference_sprite}")
# Output:
# Subtracted Sprites (Difference):
# tensor([[ -10,   40],
#         [  80,  130]])
```

- _(Pixel [0,0]: 0-10=-10, Pixel [0,1]: 50-10=40, Pixel [1,0]: 100-20=80, Pixel [1,1]: 150-20=130)_
- _(Note: If using `uint8`, negative results would wrap around or clamp!)_

### 3. Element-wise Multiplication (`*` or `torch.mul`): Masking / Scaling Contrast

Multiplies corresponding pixels. **Important: This is NOT matrix multiplication!** It's often used for applying masks (where one sprite has 0s and 1s) or scaling contrast.

```python
# Spell Snippet:
scaled_sprite = sprite1 * sprite2
# alternative: torch.mul(sprite1, sprite2)
print(f"\nElement-wise Multiplied Sprites (Scaling/Masking):\n{scaled_sprite}")
# Output:
# Element-wise Multiplied Sprites (Scaling/Masking):
# tensor([[    0,   500],
#         [ 2000,  3000]])
```

- *(Pixel [0,0]: 0*10=0, Pixel [0,1]: 50*10=500, Pixel [1,0]: 100*20=2000, Pixel [1,1]: 150*20=3000)*

### 4. Division (`/` or `torch.div`): Normalization / Ratio

Divides corresponding pixels. Useful for normalization or finding ratios.

```python
# Spell Snippet:
# Let's use slightly different values to avoid division by zero
sprite_a = torch.tensor([[10, 20], [30, 40]])
sprite_b = torch.tensor([[2, 5], [10, 8]])

divided_sprite = sprite_a / sprite_b
# alternative: torch.div(sprite_a, sprite_b)
print(f"\nDivided Sprites (Ratio):\n{divided_sprite}")
# Output:
# Divided Sprites (Ratio):
# tensor([[5.0000, 4.0000],
#         [3.0000, 5.0000]])
```

- _(Pixel [0,0]: 10/2=5, Pixel [0,1]: 20/5=4, Pixel [1,0]: 30/10=3, Pixel [1,1]: 40/8=5)_
- **Heads Up!** Watch out for dividing by zero! Also, division usually turns integer pixels into float pixels (`5.0000` instead of `5`), changing the `dtype`.

### 5. Exponentiation (`**` or `torch.pow`): Adjusting Gamma / Curves

Raises pixel values to a power. Can be used for gamma correction or adjusting contrast curves.

```python
# Spell Snippet:
# Square the pixel values of sprite1
squared_sprite = sprite1**2
# alternative: torch.pow(sprite1, 2)
print(f"\nSquared Sprite (Curve Adjustment):\n{squared_sprite}")
# Output:
# Squared Sprite (Curve Adjustment):
# tensor([[    0,  2500],
#         [10000, 22500]])
```

- _(Pixel [0,0]: 0^2=0, Pixel [0,1]: 50^2=2500, Pixel [1,0]: 100^2=10000, Pixel [1,1]: 150^2=22500)_

## Summary

You can use standard math symbols (`+`, `-`, `*`, `/`, `**`) to perform element-wise operations on identically shaped sprite tensors. This is fantastic for blending, adjusting brightness/contrast, masking, finding differences, and more! Just make sure the shapes match (for now 😉) and be aware of how operations like division might change the `dtype`. Remember `*` is element-wise, not matrix multiplication!
