# Guide: 02 Autograd with Multiple Variables

This guide explains how PyTorch's `autograd` handles computations involving multiple input tensors that require gradients, as demonstrated in `02_autograd_multiple_variables.py`.

**Core Concept:** Real-world computations, especially in neural networks, often involve outputs that depend on multiple inputs (e.g., weights, biases, input data). `autograd` seamlessly calculates the gradient of the output with respect to _each_ of these inputs simultaneously.

## Partial Derivatives

When a function has multiple input variables (like $c(a, b)$ ), we talk about **partial derivatives**. The partial derivative of $c$ with respect to $a$ (denoted $\dfrac{\partial c}{\partial a}$) measures how $c$ changes when $a$ changes slightly, assuming $b$ is held constant. Similarly, $\dfrac{\partial c}{\partial b}$ measures how $c$ changes with $b$, assuming $a$ is constant.

`autograd` automatically computes these partial derivatives for all inputs that have `requires_grad=True`.

## The Setup

The script defines two input tensors, `a` and `b`, both flagged for gradient tracking:

```python
# Script Snippet:
import torch

a = torch.tensor([2.0], requires_grad=True)
b = torch.tensor([3.0], requires_grad=True)
print(f"Tensor a: {a}")
print(f"Tensor b: {b}")
```

## The Computation

An output `c` is computed based on both `a` and `b`:

```python
# Script Snippet:
c = a**2 * b # c = a^2 * b
print(f"Tensor c = a**2 * b: {c}")
# Output: Tensor c = a**2 * b: tensor([12.], grad_fn=<MulBackward0>)
# Note: c has requires_grad=True and a grad_fn because its inputs did.
```

## Single `backward()` Call for All Gradients

A single call to `c.backward()` calculates all relevant partial derivatives ($\dfrac{\partial c}{\partial a}$ and $\dfrac{\partial c}{\partial b}$) in one pass through the computation graph.

```python
# Script Snippet:
print(f"\nCalling c.backward()...")
c.backward()
```

## Accessing Individual Gradients

The computed partial derivatives are stored in the `.grad` attribute of their respective input tensors.

```python
# Script Snippet:
# Check dc/da
grad_a = a.grad
print(f"Gradient of c with respect to a (dc/da): {grad_a}")
# Output: Gradient of c with respect to a (dc/da): tensor([12.])

# Check dc/db
grad_b = b.grad
print(f"Gradient of c with respect to b (dc/db): {grad_b}")
# Output: Gradient of c with respect to b (dc/db): tensor([4.])
```

## Mathematical Verification

Let's manually calculate the partial derivatives for $`c = a^2 b`$:

1. **Partial derivative with respect to $a$ (treat $b$ as a constant):**

    $$\frac{\partial c}{\partial a} = \frac{\partial}{\partial a}(a^2 b) = b \cdot \frac{\partial}{\partial a}(a^2) = b \cdot (2a) = 2ab$$

    At $`a = 2.0`$, $`b = 3.0`$: The result $`2(2.0)(3.0) = 12.0`$. Matches `a.grad`.✅

2. **Partial derivative with respect to $b$ (treat $a$ as a constant):**

    $$\frac{\partial c}{\partial b} = \frac{\partial}{\partial b}(a^2 b) = a^2 \cdot \frac{\partial}{\partial b}(b) = a^2 \cdot 1 = a^2$$

    At $`a = 2.0`$, $`b = 3.0`$: The result $`(2.0)^2 = 4.0`$. Matches `b.grad`. ✅

## Summary

PyTorch's `autograd` efficiently handles computations involving multiple inputs requiring gradients. Calling `.backward()` on the scalar output computes the partial derivatives with respect to _all_ such inputs simultaneously, storing each result in the corresponding input tensor's `.grad` attribute. This capability is essential for optimizing complex functions like neural network loss functions, which depend on many parameters (weights and biases).
