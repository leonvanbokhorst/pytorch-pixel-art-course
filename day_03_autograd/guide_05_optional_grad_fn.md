# Guide: 05 (Optional) The `grad_fn` Attribute

This guide provides a brief look at the `grad_fn` attribute, which offers insight into how PyTorch's `autograd` tracks operations, as shown in `05_optional_grad_fn.py`.

**Core Concept:** When PyTorch performs an operation involving tensors that require gradients, it not only computes the result but also records the function (operation) that created the output. The `grad_fn` attribute of the output tensor holds a reference to this function object, which contains the logic needed to compute gradients during the backward pass.

## Leaf vs. Non-Leaf Tensors

Understanding `grad_fn` involves distinguishing between two types of tensors in the computation graph:

1. **Leaf Tensors:** These are tensors created directly by the user (e.g., using `torch.tensor()`, `torch.randn()`) or model parameters (like weights and biases). They are the starting points or parameters of your computation.

    - Leaf tensors **do not have** a `grad_fn` (it's `None`), even if they have `requires_grad=True`.
    - Gradients (`.grad`) are typically accumulated _only_ for leaf tensors that have `requires_grad=True`.

2. **Non-Leaf Tensors:** These are tensors created as the result of an operation involving at least one tensor that requires gradients.
    - Non-leaf tensors **do have** a `grad_fn` attribute, pointing to the backward function associated with the operation that created them.
    - Their `requires_grad` status is inherited from their inputs.

## Observing `grad_fn`

The script demonstrates this distinction:

```python
# Script Snippet:
import torch

# Leaf tensor created by user
x = torch.tensor([2.0], requires_grad=True)
print(f"\nTensor x: {x}")
print(f"x.requires_grad: {x.requires_grad}") # Output: True
print(f"x.grad_fn: {x.grad_fn}")          # Output: None (Leaf tensor)

# --- Operations create grad_fn --- #

# y = x + 3 (Non-leaf)
y = x + 3
print(f"\ny = x + 3: {y}")
print(f"y.requires_grad: {y.requires_grad}") # Output: True
print(f"y.grad_fn: {y.grad_fn}")          # Output: <AddBackward0 object ...>

# z = y * y (Non-leaf)
z = y * y
print(f"\nz = y * y: {z}")
print(f"z.requires_grad: {z.requires_grad}") # Output: True
print(f"z.grad_fn: {z.grad_fn}")          # Output: <MulBackward0 object ...>

# w = z.mean() (Non-leaf)
w = z.mean()
print(f"\nw = z.mean(): {w}")
print(f"w.requires_grad: {w.requires_grad}") # Output: True
print(f"w.grad_fn: {w.grad_fn}")          # Output: <MeanBackward0 object ...>
```

## `grad_fn` Links the Graph

These `grad_fn` objects are interconnected, forming the backward graph. Each `grad_fn` knows how to compute the gradient with respect to its inputs, and it holds references to the `grad_fn`s of _those_ inputs (via an internal `next_functions` attribute). When you call `.backward()`, PyTorch traverses this chain of `grad_fn`s using the chain rule.

```python
# Script Snippet (Conceptual):
# z was created by multiplication (MulBackward0)
# The input to that multiplication was y (twice)
# y was created by addition (AddBackward0)
print(f"z.grad_fn: {z.grad_fn}") # MulBackward0
# Shows the function(s) that created the inputs for MulBackward0
print(f"z.grad_fn.next_functions: {z.grad_fn.next_functions}") # ((<AddBackward0...>, 0), ...)
```

## Interaction with `torch.no_grad()`

As expected, if a tensor is created within a `torch.no_grad()` context, gradient tracking is disabled, and therefore no `grad_fn` is created.

```python
# Script Snippet:
with torch.no_grad():
    d = x / 2
    print(f"\nInside no_grad: d = x / 2: {d}")
    print(f"Inside no_grad: d.requires_grad: {d.requires_grad}") # Output: False
    print(f"Inside no_grad: d.grad_fn: {d.grad_fn}")          # Output: None
```

## Why is `grad_fn` Relevant?

You typically don't interact directly with `grad_fn` in everyday use. Its main relevance is:

- **Understanding Autograd:** It helps visualize how PyTorch builds and uses the computation graph for backpropagation.
- **Debugging (Rarely):** In complex scenarios, inspecting `grad_fn` might offer clues about the computation history, but it's usually not the primary debugging tool.

## Summary

The `grad_fn` attribute is present on non-leaf tensors that require gradients. It acts as a pointer to the backward function associated with the operation that created the tensor. These `grad_fn` objects link together to form the computation graph that `autograd` traverses during the `.backward()` call. While not typically used directly, understanding its role clarifies how PyTorch tracks operations for automatic differentiation.
