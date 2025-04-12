# Guide: 03 Gradient Piling Up! (Accumulation)

Ever felt like your feedback signals are just piling up instead of giving you fresh advice? That's exactly what PyTorch gradients do by default! This guide explains the crucial concept of gradient accumulation (and why we need to clean the slate!), based on `03_gradient_accumulation.py`.

**Core Concept:** When you cast the `.backward()` spell to get feedback (`.grad`), PyTorch doesn't _replace_ any old feedback that might be lingering there. Instead, it **adds** the _new_ feedback (gradient) to whatever was already in the `.grad` attribute. It accumulates!

## Why Does PyTorch Pile Up the Feedback?

It sounds weird, right? Why not just give the latest news? This piling-up behavior is actually useful for advanced spells, like pretending you processed a HUGE batch of sprites even if your GPU memory can only handle small batches (by processing small batches one after another, calling `backward()` each time to accumulate the feedback, and _then_ updating the model based on the combined feedback).

But for our typical training loop where we process one batch, calculate feedback, and update immediately, we need to make sure we're only using the feedback from the _current_ batch.

## Seeing the Pile-Up in Action!

Let's revisit our learnable pixel `p` from Guide 1.

```python
# Potion Ingredients:
import torch

p = torch.tensor([0.5], requires_grad=True)

# --- Feedback Spell 1 --- #
target1 = 0.8
penalty1 = (p - target1)**2 # d(penalty1)/dp = 2(p-0.8) = 2(0.5-0.8) = -0.6
penalty1.backward()
print(f"p.grad after first backward(): {p.grad}")
# Output: p.grad after first backward(): tensor([-0.6000])

# --- Feedback Spell 2 (WITHOUT Clearing Old Feedback!) --- #
# Let's calculate penalty based on a *different* target now
target2 = 0.1
penalty2 = (p - target2)**2 # d(penalty2)/dp = 2(p-0.1) = 2(0.5-0.1) = 0.8

# !! We call backward() AGAIN without clearing p.grad !!
penalty2.backward()

# The new gradient (0.8) is ADDED to the old one (-0.6)!
print(f"p.grad after second backward() (ACCUMULATED!): {p.grad}")
# Output: p.grad after second backward() (ACCUMULATED!): tensor([0.2000]) # -0.6 + 0.8 = 0.2
```

Whoa! The second `backward()` call didn't set `p.grad` to `0.8`. It added `0.8` to the `-0.6` that was already there, giving `0.2`. This combined feedback is confusing and not what we usually want for simple batch training!

## Wiping the Slate Clean: `.grad.zero_()`

To get _fresh_ feedback specific to the latest calculation, you need to manually reset the `.grad` attribute to zero **before** casting `.backward()`.

The magic wand for this is the `.zero_()` method, called directly _on the gradient tensor_ (`p.grad`).

```python
# Spell Snippet:
print(f"\nManually cleaning the slate with p.grad.zero_()...")
# Safety check: Make sure .grad exists before trying to zero it!
if p.grad is not None:
    p.grad.zero_()
print(f"p.grad after zeroing: {p.grad}")
# Output: p.grad after zeroing: tensor([0.])

# --- Feedback Spell 3 (AFTER Wiping the Slate!) --- #
# Let's use penalty1 again
penalty3 = (p - target1)**2 # d(penalty3)/dp = -0.6 again
penalty3.backward()
print(f"p.grad after third backward() (Fresh Feedback!): {p.grad}")
# Output: p.grad after third backward() (Fresh Feedback!): tensor([-0.6000])
```

Success! After using `.grad.zero_()`, the third `backward()` call stored the correct, fresh gradient of `-0.6`.

## The Training Loop Shortcut: `optimizer.zero_grad()`

Doing `p.grad.zero_()` manually for every single parameter in a big pixel model would be a nightmare! Luckily, the `optimizer` (which we'll meet properly in Day 6) handles this for us.

In a standard training loop, the very **first thing** you usually do in each step (for each batch) is call:

```python
optimizer.zero_grad()
```

The optimizer knows about all the learnable parameters (tensors) in your model. This one command tells it to go through every single one and zero out its `.grad` attribute. It perfectly wipes the slate clean, ready for the feedback from the current batch of sprites.

## Summary

Gradients accumulate (add up) in `.grad` every time you call `.backward()`! This is useful sometimes, but for normal training, we need fresh feedback for each batch. **Always zero out gradients before your `backward()` call in each training step.** The easiest way is using `optimizer.zero_grad()` at the start of your loop iteration.
