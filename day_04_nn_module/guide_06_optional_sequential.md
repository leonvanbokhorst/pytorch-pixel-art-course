# Guide: 06 (Optional) Using nn.Sequential

This guide introduces `torch.nn.Sequential`, a container module that provides a convenient way to build simple neural networks composed of a linear stack of layers, as demonstrated in `06_optional_sequential.py`.

**Core Concept:** `nn.Sequential` is itself an `nn.Module` that takes an ordered sequence of other modules (layers, activations) as input. When data is passed to an `nn.Sequential` container, it automatically passes the data through each module in the exact order they were defined.

## Use Case: Simple Feed-Forward Networks

`nn.Sequential` is particularly useful when your network architecture is a straightforward pipeline where the output of one layer feeds directly into the input of the next, without any branching, skipping, or complex routing. It allows you to define such models concisely without needing to explicitly write a `forward` method.

## Defining Models with `nn.Sequential`

There are two main ways to define a sequential model:

### 1. Simple Argument List

You can pass the layer instances as ordered arguments directly to the `nn.Sequential` constructor. PyTorch automatically assigns numerical indices (starting from 0) as keys for these layers.

```python
# Script Snippet (Method 1):
import torch
import torch.nn as nn

input_size = 10
hidden_size = 7
output_size = 3

model_sequential_simple = nn.Sequential(
    nn.Linear(input_size, hidden_size),  # Index '0'
    nn.ReLU(),                           # Index '1'
    nn.Linear(hidden_size, output_size)  # Index '2'
)

print("--- Simple Sequential Model --- ")
print(model_sequential_simple)
# Output:
# Sequential(
#   (0): Linear(in_features=10, out_features=7, bias=True)
#   (1): ReLU()
#   (2): Linear(in_features=7, out_features=3, bias=True)
# )
```

### 2. Using `OrderedDict` for Named Layers

For better readability and the ability to access layers by meaningful names, you can pass an `OrderedDict` from Python's `collections` module. The keys of the dictionary become the names of the layers within the sequential container.

```python
# Script Snippet (Method 2):
from collections import OrderedDict

model_sequential_named = nn.Sequential(
    OrderedDict(
        [
            ("input_layer", nn.Linear(input_size, hidden_size)),
            ("activation", nn.ReLU()),
            ("output_layer", nn.Linear(hidden_size, output_size)),
        ]
    )
)

print("\n--- Named Sequential Model --- ")
print(model_sequential_named)
# Output:
# Sequential(
#   (input_layer): Linear(in_features=10, out_features=7, bias=True)
#   (activation): ReLU()
#   (output_layer): Linear(in_features=7, out_features=3, bias=True)
# )

# Accessing named layers:
print(f"Input layer weight shape: {model_sequential_named.input_layer.weight.shape}")
```

## Equivalence and Usage

Both `model_sequential_simple` and `model_sequential_named` define the exact same network architecture as the `MultiLayerNet` class we built manually in the previous examples. You use them in the same way: pass input data through the instance, and it automatically executes the layers in sequence.

```python
# Script Snippet (Usage):
batch_size = 4
dummy_input = torch.randn(batch_size, input_size)
with torch.no_grad():
    output_simple = model_sequential_simple(dummy_input)
    output_named = model_sequential_named(dummy_input)

print(f"Output shape (simple): {output_simple.shape}") # torch.Size([4, 3])
print(f"Output shape (named): {output_named.shape}")   # torch.Size([4, 3])
```

## Limitations of `nn.Sequential`

While convenient for simple cases, `nn.Sequential` has limitations:

- **Strictly Sequential:** It only works for architectures where data flows linearly from one layer to the next.
- **No Complex Logic:** You cannot implement custom logic within the forward pass, such as:
  - Skip connections (like in ResNets).
  - Multiple inputs or outputs.
  - Branching paths.
  - Reusing layers multiple times with different inputs.

**For any architecture more complex than a simple stack of layers, you must subclass `nn.Module` and define your custom `forward` method.**

## Summary

`nn.Sequential` provides a concise way to define simple feed-forward neural networks where layers are applied in a fixed order. You can pass layers as arguments or use an `OrderedDict` for named layers. However, for models requiring non-sequential data flow or custom logic in the forward pass, subclassing `nn.Module` directly remains the necessary and more flexible approach.
