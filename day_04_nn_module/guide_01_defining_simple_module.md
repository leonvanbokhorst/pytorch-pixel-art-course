# Guide: 01 Defining a Simple nn.Module

This guide explains the fundamental structure for defining custom neural network architectures in PyTorch using `torch.nn.Module`, as demonstrated in `01_defining_simple_module.py`.

**Core Concept:** `torch.nn.Module` is the base class for all neural network modules in PyTorch. Every model you build, whether it's a single layer or a complex network, should inherit from `nn.Module`. This provides a standardized way to structure your code, manage parameters, and integrate with other PyTorch components like optimizers and dataloaders.

## The Blueprint: Creating a Custom Module

Defining your own network module follows a consistent pattern:

1. **Create a Class:** Define a new Python class that inherits from `torch.nn.Module`.

    ```python
    import torch.nn as nn

    class YourModelName(nn.Module):
        # ... definition follows ...
    ```

2. **Define the `__init__` Method:**
    - The constructor is where you **initialize the layers** your model will use.
    - **Crucially**, the _first_ line must be `super(YourModelName, self).__init__()` (or the more common `super().__init__()`) to ensure the parent `nn.Module` class is properly initialized.
    - Instantiate PyTorch layer objects (e.g., `nn.Linear`, `nn.Conv2d`, `nn.ReLU`) and assign them as attributes of `self` (e.g., `self.layer1 = nn.Linear(...)`). Assigning layers as attributes automatically registers them (and their parameters) with the module.

    ```python
    def __init__(self, arg1, arg2, ...):
        super().__init__() # Essential first step
        # Define layers here
        self.layer1 = nn.Linear(in_features=arg1, out_features=arg2)
        self.activation = nn.ReLU()
        # ... other layers ...
    ```

3. **Define the `forward` Method:**
    - This method defines the **computation performed at every call**. It dictates how input data flows through the layers defined in `__init__`.
    - It takes `self` and the input tensor(s) (commonly named `x`) as arguments.
    - Inside `forward`, you use the layer attributes defined in `__init__` as if they were functions, passing the data through them sequentially or in whatever structure you design.
    - It must return the output tensor(s).

    ```python
    def forward(self, x):
        # Define the data flow
        x = self.layer1(x)
        x = self.activation(x)
        # ... pass through other layers ...
        return x
    ```

## Example: `SimpleNet`

The script defines a very basic network with just one linear layer:

```python
# Script Snippet (Class Definition):
import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNet, self).__init__() # Call parent constructor
        # Define one linear layer
        self.linear_layer = nn.Linear(in_features=input_size, out_features=output_size)

    def forward(self, x):
        # Pass input directly through the linear layer
        output = self.linear_layer(x)
        return output
```

- **`__init__`:** Initializes the parent `nn.Module` and defines a single `nn.Linear` layer, storing it as `self.linear_layer`.
- **`forward`:** Takes an input `x` and passes it directly through `self.linear_layer`, returning the result.

## Why Use `nn.Module`?

- **Organization:** Encapsulates layers and their computational flow logically within a class.
- **Parameter Tracking:** `nn.Module` automatically detects and tracks all learnable parameters (weights, biases) defined within its layers (via `nn.Parameter`). This makes it easy to access them using methods like `model.parameters()` for use with optimizers.
- **Sub-modules:** Modules can contain other modules, allowing for complex, nested architectures.
- **Helper Methods:** Provides useful methods for managing the model state (e.g., `model.train()`, `model.eval()`), moving the model to different devices (`model.to(device)`), saving/loading (`torch.save`, `torch.load_state_dict`), etc.

## Summary

Inheriting from `nn.Module` is the standard way to define networks in PyTorch. The core pattern involves defining layers as attributes within the `__init__` method (remembering to call `super().__init__()`) and defining the data flow logic through these layers in the `forward` method. This structure provides organization, automatic parameter tracking, and access to PyTorch's rich ecosystem of tools.
