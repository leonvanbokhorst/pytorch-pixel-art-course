# Guide: 04 Disabling Gradients with `torch.no_grad()`

This guide explains how and why to temporarily disable gradient computation tracking using the `torch.no_grad()` context manager, as demonstrated in `04_disabling_gradients.py`.

**Core Concept:** While `autograd` is essential for training, tracking the computation graph requires memory and computational overhead. When you don't need gradients (e.g., during model evaluation, inference, or specific calculations), you should disable tracking for efficiency.

## Why Disable Gradient Tracking?

1. **Memory Savings:** PyTorch doesn't need to store intermediate values and context required for the backward pass, reducing memory consumption.
2. **Speed Improvement:** Avoiding the overhead of tracking operations makes computations faster.
3. **Correctness:** Prevents accidental calls to `.backward()` or gradient accumulation in code sections where it's not intended (like evaluation loops).

## The `torch.no_grad()` Context Manager

The primary way to disable gradient tracking for a block of code is using the `with torch.no_grad():` context manager.

```python
with torch.no_grad():
    # Operations inside this block will not be tracked by autograd
    # Tensors created here will have requires_grad=False
    ...
# Outside the block, gradient tracking resumes its previous state
```

## How it Works

Any tensor operation performed _inside_ the `with torch.no_grad():` block behaves as if none of the input tensors had `requires_grad=True`. Consequently:

- Output tensors created within the context will have `requires_grad=False`.
- These operations will not be recorded in the computation graph.
- Their `grad_fn` attribute will be `None`.

## Walkthrough Example

The script contrasts operations inside and outside the `no_grad` context:

```python
# Script Snippet:
import torch

x = torch.tensor([2.0], requires_grad=True)
print(f"Tensor x: {x}, requires_grad: {x.requires_grad}")
# Output: Tensor x: tensor([2.], requires_grad=True), requires_grad: True

# --- Tracking Enabled --- #
y = x * 2
print(f"\ny = x * 2: {y}")
print(f"y.requires_grad: {y.requires_grad}") # Output: True (because x requires grad)
print(f"y.grad_fn: {y.grad_fn}")          # Output: <MulBackward0 ...>

# --- Tracking Disabled --- #
print(f"\nEntering torch.no_grad() context...")
with torch.no_grad():
    z = x * 3
    print(f"  Inside context: z = x * 3: {z}")
    print(f"  Inside context: z.requires_grad: {z.requires_grad}") # Output: False
    print(f"  Inside context: z.grad_fn: {z.grad_fn}")          # Output: None
print("Exited torch.no_grad() context.")
```

Even though `x` requires grad, `z` does not because it was created inside the `no_grad` block.

## Consequences for `.backward()`

Since `z` doesn't track its history (`grad_fn` is `None`), you cannot call `backward()` on it or any computation derived solely from it within the `no_grad` block.

```python
# Script Snippet:
# Backward for y works (history was tracked)
y.backward()
print(f"x.grad after y.backward(): {x.grad}") # Output: tensor([2.])

# Backward for z fails (no history)
try:
    if x.grad is not None: x.grad.zero_() # Clear previous gradient
    z.backward()
except RuntimeError as e:
    print(f"\nError calling z.backward(): {e}")
# Output: Error calling z.backward(): element 0 of tensors does not require grad...
```

## Key Use Case: Model Evaluation / Inference

It is standard practice and highly recommended to wrap your model evaluation or inference loop with `torch.no_grad()`:

```python
model.eval() # Set model to evaluation mode (disables dropout, batchnorm updates etc.)
with torch.no_grad():
    for inputs, labels in test_dataloader:
        outputs = model(inputs)
        # Calculate accuracy, loss (without tracking gradients), etc.
        ...
```

This ensures you get the performance benefits and prevents accidental gradient calculations during testing.

## Alternative: `.detach()`

Another related method is `tensor.detach()`. This creates a _new_ tensor that shares the same data as the original but is explicitly detached from the computation graph (it has `requires_grad=False` and no `grad_fn`). This is useful when you need a version of a tensor without gradient history for specific calculations, but want the original tensor to remain part of the graph. `torch.no_grad()` is generally preferred for disabling tracking over blocks of code.

## Summary

Use the `with torch.no_grad():` context manager to temporarily disable gradient tracking for performance and memory efficiency when gradients are not required, most commonly during model evaluation and inference phases. Operations performed within this context will not build up the autograd graph.
