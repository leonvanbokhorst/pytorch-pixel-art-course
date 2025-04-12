# Guide: 03 Adding Non-Linearity (Activation Functions)

This guide explains why non-linear activation functions are essential in neural networks and how to incorporate them into your `nn.Module` definitions, as demonstrated in `03_adding_non_linearity.py`.

**Core Concept:** Stacking only linear layers (`nn.Linear`) results in a model that can only represent linear transformations of the input data, regardless of how many layers you add. To enable networks to learn complex, non-linear patterns (which are present in most real-world data), we introduce **non-linear activation functions** between layers.

## Why Non-Linearity?

Imagine trying to separate data points that form a circle using only straight lines – it's impossible! Linear layers produce straight lines/planes/hyperplanes. Activation functions bend and warp the space, allowing the network to create complex decision boundaries.

## Activation Functions

Activation functions are typically simple mathematical functions applied element-wise to the output tensor of a layer (often called the _pre-activation_).

Common activation functions include:

- **ReLU (Rectified Linear Unit):** `f(x) = max(0, x)`. Very popular, computationally efficient. Zeros out negative values.
- **Sigmoid:** `f(x) = 1 / (1 + exp(-x))`. Squashes values between 0 and 1. Often used in output layers for binary classification.
- **Tanh (Hyperbolic Tangent):** `f(x) = tanh(x)`. Squashes values between -1 and 1.
- And many others (LeakyReLU, GeLU, etc.).

## Incorporating Activations in `nn.Module`

There are two common ways:

1. **As separate layers (Module approach - shown in the script):**

    - Instantiate the activation function module in `__init__` (e.g., `self.relu = nn.ReLU()`). Most activations don't have learnable parameters, but treating them as modules is consistent.
    - Apply the activation module in the `forward` method after the layer whose output you want to transform.

2. **Using the `torch.nn.functional` API (Functional approach):**
    - Import `torch.nn.functional` (usually as `F`).
    - Call the functional version directly in `forward` (e.g., `x = F.relu(x)`).
    - This avoids needing to define the activation in `__init__` if it has no parameters.

Both approaches are valid. The module approach makes the activation explicit as a "layer" in the model definition.

## Example: `SimpleNetWithReLU`

The script modifies the previous `SimpleNet` to include a ReLU activation:

```python
# Script Snippet (Class Definition):
import torch
import torch.nn as nn

class SimpleNetWithReLU(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNetWithReLU, self).__init__()
        # Define layers
        self.linear_layer = nn.Linear(input_size, output_size)
        # Define the activation function instance
        self.relu = nn.ReLU()

    def forward(self, x):
        # 1. Pass input through the linear layer
        linear_output = self.linear_layer(x)
        # 2. Apply the activation function
        output = self.relu(linear_output)
        return output
```

- In `__init__`, `self.relu = nn.ReLU()` is added.
- In `forward`, the output of `self.linear_layer` is now passed through `self.relu` before being returned.

## ReLU in Action

The script demonstrates that after applying ReLU, all negative values in the output of the linear layer become zero:

```python
# Conceptual Output from Script:
# ...
# Output values before ReLU:
# tensor([[-0.5,  1.2, -2.3],
#         [ 3.1, -0.1,  0.8]])
# ...
# Output values after ReLU (negatives zeroed):
# tensor([[ 0. ,  1.2,  0. ],
#         [ 3.1,  0. ,  0.8]])
```

## Functional Alternative (Not in script, but common)

You could achieve the same result without defining `self.relu` in `__init__` by using the functional API:

```python
import torch.nn.functional as F

class SimpleNetWithReLUFunctional(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear_layer = nn.Linear(input_size, output_size)

    def forward(self, x):
        linear_output = self.linear_layer(x)
        output = F.relu(linear_output) # Apply functional ReLU
        return output
```

## Summary

Non-linear activation functions are essential components placed between linear layers (or other types of layers) to allow neural networks to learn complex patterns. They are easily integrated into `nn.Module` either by instantiating their module counterparts (e.g., `nn.ReLU`) in `__init__` and applying them in `forward`, or by using their functional equivalents from `torch.nn.functional` directly within the `forward` method.
