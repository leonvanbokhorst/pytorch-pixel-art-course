# Guide: 02 Autograd with Multiple Pixel Ingredients!

What happens when your final pixel color depends on mixing _multiple_ learnable base colors or parameters? This guide shows how Autograd handles feedback for multiple inputs, based on `02_autograd_multiple_variables.py`.

**Core Concept:** Often, your target pixel value or the overall "look" isn't just influenced by one thing. Maybe it's a mix of a base color and a highlight color, or a character sprite placed on a background layer. Autograd is smart enough to figure out how _each_ contributing learnable input affects the final result, all at once!

## Partial Gradients: Who Influenced What?

When your output depends on multiple inputs (say, `final_color` depends on `base_color` and `highlight_tint`), we talk about **partial derivatives**. The partial derivative with respect to `base_color` tells you "How much does the `final_color` change if I tweak _only_ the `base_color`?" Similarly for `highlight_tint`.

Autograd calculates these partial derivatives for _all_ inputs marked with `requires_grad=True`.

## The Setup: Mixing Two Learnable Colors

Let's imagine we have two base colors, `color1` and `color2`, that we want to learn or optimize. We mark both for gradient tracking.

```python
# Potion Ingredients:
import torch

# Let's say these represent weights for two primary colors (0.0 to 1.0)
color1_weight = torch.tensor([0.2], requires_grad=True)
color2_weight = torch.tensor([0.7], requires_grad=True)

print(f"Learnable Color 1 Weight: {color1_weight}")
print(f"Learnable Color 2 Weight: {color2_weight}")
```

## The Pixel Calculation: A Simple Mix

Let's create a final mixed color based on a weighted sum of our learnable weights. We also calculate a simple "loss" – how far is our mix from a target value (e.g., 0.5)?

```python
# Spell Snippet:
# Mix the colors: 60% of color1, 40% of color2
mixed_color = color1_weight * 0.6 + color2_weight * 0.4

# Calculate a simple loss: how far is the mix from 0.5?
target_value = 0.5
loss = (mixed_color - target_value)**2 # Squared difference

print(f"\nMixed Color: {mixed_color}")
print(f"Loss (Squared Difference from {target_value}): {loss}")
# Note: 'loss' requires gradients because color1_weight and color2_weight do.
```

## One `backward()` Spell, Many Gradients!

Even though the `loss` depends on both `color1_weight` and `color2_weight`, a _single_ call to `loss.backward()` is enough! Autograd traces the recipe back and calculates the gradient for _both_ inputs simultaneously.

```python
# Spell Snippet:
print(f"\nCalling loss.backward()...")
loss.backward() # Calculates d(loss)/d(color1) AND d(loss)/d(color2)
```

## Reading the Feedback for Each Color

The partial derivatives (gradients) are now waiting in the `.grad` attribute of each original input tensor.

```python
# Spell Snippet:
# How does changing color1_weight affect the loss?
grad_c1 = color1_weight.grad
print(f"Gradient w.r.t. color1_weight: {grad_c1}")
# Output: Gradient w.r.t. color1_weight: tensor([-0.1200])

# How does changing color2_weight affect the loss?
grad_c2 = color2_weight.grad
print(f"Gradient w.r.t. color2_weight: {grad_c2}")
# Output: Gradient w.r.t. color2_weight: tensor([-0.0800])
```

## Mathematical Verification (Optional Brain Teaser!)

Let $c_1$ = `color1_weight`, $c_2$ = `color2_weight`.
Mix $M = 0.6c_1 + 0.4c_2$.
Loss $L = (M - 0.5)^2 = (0.6c_1 + 0.4c_2 - 0.5)^2$.

1.  **Partial derivative w.r.t. $c_1$ (treat $c_2$ as constant):**
    Use chain rule: $\frac{\partial L}{\partial c_1} = \frac{\partial L}{\partial M} \times \frac{\partial M}{\partial c_1}$
    $\frac{\partial L}{\partial M} = 2(M - 0.5)^1 = 2(0.6c_1 + 0.4c_2 - 0.5)$
    $\frac{\partial M}{\partial c_1} = 0.6$
    So, $\frac{\partial L}{\partial c_1} = 2(0.6c_1 + 0.4c_2 - 0.5) \times 0.6$
    At $c_1=0.2, c_2=0.7$: $M = 0.6(0.2) + 0.4(0.7) = 0.12 + 0.28 = 0.4$.
    $\frac{\partial L}{\partial c_1} = 2(0.4 - 0.5) \times 0.6 = 2(-0.1) \times 0.6 = -0.12$. Matches `grad_c1`. ✅

2.  **Partial derivative w.r.t. $c_2$ (treat $c_1$ as constant):**
    $\frac{\partial L}{\partial c_2} = \frac{\partial L}{\partial M} \times \frac{\partial M}{\partial c_2}$
    $\frac{\partial M}{\partial c_2} = 0.4$
    So, $\frac{\partial L}{\partial c_2} = 2(0.6c_1 + 0.4c_2 - 0.5) \times 0.4$
    At $c_1=0.2, c_2=0.7$: $M=0.4$.
    $\frac{\partial L}{\partial c_2} = 2(0.4 - 0.5) \times 0.4 = 2(-0.1) \times 0.4 = -0.08$. Matches `grad_c2`. ✅

## Summary

Autograd is smart! When your pixel outcome depends on multiple learnable inputs (colors, weights, parameters), a single `.backward()` call figures out the individual influence (partial derivative) of _each_ input. The feedback lands in each input's `.grad` attribute, ready to guide the learning process. This is fundamental for training models with many adjustable parts!
