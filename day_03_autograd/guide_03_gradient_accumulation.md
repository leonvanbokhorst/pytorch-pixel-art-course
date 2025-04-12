# Guide: 03 Gradient Accumulation

This guide explains a crucial default behavior of PyTorch's `autograd`: gradient accumulation, and why it necessitates zeroing gradients during training, as shown in `03_gradient_accumulation.py`.

**Core Concept:** By default, whenever you call `.backward()` to compute gradients, PyTorch **adds** the newly computed gradients to the values already present in the `.grad` attributes of the relevant tensors. Gradients are _accumulated_, not overwritten.

## Why Does PyTorch Accumulate Gradients?

While it might seem counter-intuitive at first, accumulation is the desired behavior for certain advanced scenarios, most notably **gradient accumulation steps**. This technique allows simulating larger batch sizes than can fit into memory:

1. Process a small mini-batch.
2. Compute loss and call `.backward()` (gradients accumulate).
3. Repeat steps 1-2 for several mini-batches.
4. Finally, call `optimizer.step()` to update parameters using the accumulated gradients from multiple mini-batches.

However, for standard training where you process one batch per optimizer step, this accumulation needs to be explicitly cleared.

## Demonstration of Accumulation

The script illustrates this clearly:

```python
# Script Snippet:
import torch

x = torch.tensor([2.0], requires_grad=True)

# --- First backward pass --- #
y = x**2 # dy/dx = 2x = 4
y.backward()
print(f"x.grad after first backward(): {x.grad}")
# Output: x.grad after first backward(): tensor([4.])

# --- Second backward pass (WITHOUT zeroing) --- #
z = x**3 # dz/dx = 3x^2 = 12
z.backward()
# New gradient (12) is ADDED to the previous gradient (4)
print(f"x.grad after second backward() (accumulated): {x.grad}")
# Output: x.grad after second backward() (accumulated): tensor([16.])
```

As you can see, the second call to `backward()` (for `z`) didn't set `x.grad` to 12; it added 12 to the existing 4, resulting in 16.

## Zeroing Gradients Manually: `.grad.zero_()`

To prevent accumulation and get a fresh gradient calculation, you must explicitly set the `.grad` attribute back to zero before calling `.backward()` again.

The `.zero_()` method, called _on the gradient tensor itself_, does this in-place.

```python
# Script Snippet:
print(f"\nZeroing the gradient with x.grad.zero_()...")
if x.grad is not None: # Check if grad exists before zeroing
    x.grad.zero_()
print(f"x.grad after zeroing: {x.grad}")
# Output: x.grad after zeroing: tensor([0.])

# --- Third backward pass (AFTER zeroing) --- #
w = x**2 + x # dw/dx = 2x + 1 = 5
w.backward()
print(f"x.grad after third backward() (fresh gradient): {x.grad}")
# Output: x.grad after third backward() (fresh gradient): tensor([5.])
```

After zeroing, the third `backward()` call correctly stores the gradient of `w` (which is 5) in `x.grad`.

## The Training Loop Context: `optimizer.zero_grad()`

In a standard neural network training loop, you process data in batches. You want the parameter updates for the current batch to be based _only_ on the gradients calculated from that batch's loss, not contaminated by gradients from previous batches.

This is why the **first step** inside a typical PyTorch training loop iteration is:

```python
optimizer.zero_grad()
```

The `optimizer` (e.g., `torch.optim.Adam`, `torch.optim.SGD`) holds references to all the model parameters (tensors) it's supposed to update. Calling `optimizer.zero_grad()` iterates through all these parameters and calls `.grad.zero_()` on each one for you, effectively clearing the slate before the current batch's forward and backward passes.

## Summary

PyTorch accumulates gradients in the `.grad` attribute with each `.backward()` call by default. While useful for advanced techniques, this means for standard batch-wise training, you **must** explicitly zero the gradients before each `backward()` pass to avoid interference from previous iterations. This is typically done using `optimizer.zero_grad()` at the beginning of each training step.
