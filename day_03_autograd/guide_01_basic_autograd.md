# Guide: 01 Basic Autograd

This guide introduces the fundamental concepts of PyTorch's automatic differentiation engine, `autograd`, as demonstrated in `01_basic_autograd.py`.

**Core Concept:** `autograd` is the system that powers gradient-based learning in PyTorch. It automatically calculates the derivative (gradient) of the output of a sequence of operations with respect to its inputs. This is crucial for optimization algorithms like Gradient Descent, which adjust model parameters based on gradients.

## The Core Workflow

The basic `autograd` process involves these key steps:

1. **Flag Inputs:** Mark the input tensors for which you need gradients by setting their `requires_grad` attribute to `True` when creating them.
2. **Perform Computations:** Execute the sequence of operations (your model's forward pass, loss calculation, etc.) using these flagged tensors.
3. **Trigger Backpropagation:** Call the `.backward()` method on the final _scalar_ output tensor (typically the loss).
4. **Access Gradients:** Retrieve the computed gradients from the `.grad` attribute of the input tensors you flagged.

## 1. `requires_grad=True`

This attribute tells PyTorch that operations involving this tensor should be recorded. If any input tensor in an operation has `requires_grad=True`, the output tensor will also have it (unless gradient tracking is disabled).

```python
# Script Snippet:
import torch

x = torch.tensor([2.0], requires_grad=True) # Flag x for gradient tracking
print(f"Tensor x: {x}")
```

## 2. The Computation Graph

Behind the scenes, as you perform operations on tensors with `requires_grad=True`, PyTorch builds a directed acyclic graph (DAG). Tensors are the nodes, and the functions (operations like `+`, `*`, `**`) that create output tensors from input tensors are also nodes (specifically, `Function` objects). `autograd` uses this graph to trace calculations backward.

```python
# Script Snippet:
y = x**2 + 3 * x + 1 # Operations involving x are recorded
print(f"Tensor y = x**2 + 3*x + 1: {y}")
# y implicitly has requires_grad=True because x does
# y also has a grad_fn attribute pointing to the last operation (AddBackward0)
```

## 3. `tensor.backward()`

Calling `.backward()` on a scalar tensor (a tensor with only one element) initiates the gradient computation. `autograd` starts from that tensor and works backward through the graph, applying the chain rule at each step to calculate the gradients of that scalar output with respect to the tensors that were flagged with `requires_grad=True`.

```python
# Script Snippet:
print(f"\nCalling y.backward()...")
y.backward() # Computes dy/dx
```

- **Note:** If `backward()` is called on a non-scalar tensor, you typically need to provide a `gradient` argument representing the gradient of the final function with respect to the tensor `backward()` was called on.

## 4. `tensor.grad`

After `.backward()` completes, the computed gradients are _accumulated_ (added) into the `.grad` attribute of the respective input tensors.

```python
# Script Snippet:
gradient = x.grad
print(f"Gradient of y with respect to x (dy/dx) at x=2.0: {gradient}")
# Output: Gradient of y with respect to x (dy/dx) at x=2.0: tensor([7.])
```

## Mathematical Verification

`autograd` uses calculus (specifically the chain rule) internally. We can verify the result manually:

- Function: $( y = x^2 + 3x + 1 )$
- Derivative: $( \frac{dy}{dx} = \frac{d}{dx}(x^2) + \frac{d}{dx}(3x) + \frac{d}{dx}(1) = 2x + 3 + 0 = 2x + 3 )$
- At $( x = 2.0 ): ( \frac{dy}{dx} = 2(2.0) + 3 = 4 + 3 = 7.0 )$

The result `x.grad = tensor([7.])` matches the manual calculation.

## Another Example

The script reinforces this with $( w = z^3 )$ :

- Function: $( w = z^3 )$
- Derivative: $( \frac{dw}{dz} = 3z^2 )$
- At $( z = 3.0 ): ( \frac{dw}{dz} = 3(3.0)^2 = 3 \cdot 9 = 27.0 )$

```python
# Script Snippet:
z = torch.tensor([3.0], requires_grad=True)
w = z**3
w.backward()
print(f"Gradient of w with respect to z (dw/dz) at z=3.0: {z.grad}")
# Output: Gradient of w with respect to z (dw/dz) at z=3.0: tensor([27.])
```

## Summary

`autograd` automates the process of gradient calculation. The core workflow involves setting `requires_grad=True` on inputs, performing calculations to build a computation graph, calling `backward()` on the final scalar output, and accessing the computed gradients via the `.grad` attribute of the inputs. This mechanism is the foundation for training neural networks in PyTorch.
