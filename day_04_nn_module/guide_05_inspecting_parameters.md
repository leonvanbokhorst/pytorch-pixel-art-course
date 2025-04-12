# Guide: 05 Inspecting Model Parameters

This guide explains how `nn.Module` automatically tracks the learnable parameters (weights and biases) of the layers defined within it, and how to access them, as demonstrated in `05_inspecting_parameters.py`.

**Core Concept:** One of the most powerful features of `nn.Module` is its built-in mechanism for registering and accessing parameters. When you define layers like `nn.Linear` as attributes within your module's `__init__`, PyTorch automatically recognizes their internal `nn.Parameter` objects (which hold the weights and biases) and makes them easily accessible.

## Why Access Parameters?

- **Optimization:** The primary reason is to pass the model's parameters to an optimizer (e.g., `Adam`, `SGD`) so it knows which tensors to update during training based on the computed gradients.
- **Inspection/Debugging:** Examining parameter shapes, values, or gradients can be crucial for understanding model behavior and debugging issues.
- **Initialization/Modification:** You might want to apply custom weight initializations or perform specific operations on certain parameters (e.g., weight decay, freezing layers).
- **Model Size:** Counting the total number of parameters is a common way to estimate a model's complexity and memory footprint.

## Accessing Parameters in PyTorch

`nn.Module` provides convenient methods:

1. **`model.parameters()`:**

    - Returns an _iterator_ that yields all `torch.Tensor` objects considered learnable parameters of the module and all its sub-modules recursively.
    - This is the most common way to pass parameters to an optimizer: `optimizer = torch.optim.Adam(model.parameters(), lr=0.001)`.

2. **`model.named_parameters()`:**
    - Similar to `.parameters()`, but returns an _iterator_ yielding tuples of `(name, parameter)`, where `name` is a string identifying the parameter (e.g., `'layer_1.weight'`, `'layer_2.bias'`) and `parameter` is the tensor itself.
    - Useful when you need to identify specific parameters for logging, debugging, or applying different logic (like different learning rates for different layers).

## Inspecting Parameter Details

The script uses `named_parameters()` to iterate and display information about each parameter:

```python
# Script Snippet (Looping through parameters):
model = MultiLayerNet(...) # Instantiate the model
print("\nInspecting model parameters:")
total_params = 0

for name, param in model.named_parameters():
    if param.requires_grad: # Check if parameter is trainable
        num_elements = param.numel() # Total elements in the tensor
        print(f"Parameter Name: {name}")
        print(f"  Shape: {param.shape}")
        print(f"  Requires Grad: {param.requires_grad}")
        print(f"  Number of Elements: {num_elements}")
        # Access data directly (e.g., for initialization or viewing)
        # print(f"  Data (first few): {param.data.flatten()[:5]}")
        total_params += num_elements

print(f"\nTotal number of learnable parameters: {total_params}")
```

- **`param.shape`**: Shows the dimensions of the weight matrix or bias vector.
- **`param.requires_grad`**: Indicates if `autograd` should track operations and compute gradients for this parameter (should be `True` for learnable parameters).
- **`param.numel()`**: Gives the total count of individual numbers in the parameter tensor.
- **`param.data`**: Accesses the underlying data tensor directly, bypassing the gradient tracking mechanism (useful for modifying values outside of autograd, like during custom initialization).

## Calculating Total Parameters

Summing `param.numel()` for all parameters gives the total count of learnable numbers in the model. The script verifies this against a manual calculation based on the layer dimensions:

- `nn.Linear(in, out)` has `in * out` weights and `out` biases.
- For the example (`in=10, hidden=7, out=3`):
  - Layer 1: `(10 * 7) + 7 = 77` parameters
  - Layer 2: `(7 * 3) + 3 = 24` parameters
  - Total: `77 + 24 = 101` parameters.

## Summary

`nn.Module` automatically keeps track of all parameters defined within its constituent layers. You can easily access these parameters for inspection, modification, or passing to an optimizer using `model.parameters()` (for the tensors) or `model.named_parameters()` (for `(name, tensor)` tuples). This automatic tracking significantly simplifies the process of building and training neural networks.
