# Guide: 06 (Optional) Express Pixel Models: `nn.Sequential`

Need to build a simple pixel generator where the noise just flows straight through a few layers like an assembly line? `torch.nn.Sequential` is your shortcut! This guide introduces this handy container, as seen in `06_optional_sequential.py`.

**Core Concept:** `nn.Sequential` is a special type of `nn.Module` that acts like a container. You feed it a list of layers (and activations) in the order you want them applied. When you pass data into the `nn.Sequential` object, it automatically sends the data through each layer in that exact order, one after the other. Quick and easy!

## When is `nn.Sequential` Perfect for Pixels?

It shines when your pixel model is a simple pipeline:

Noise -> Layer 1 -> Activation 1 -> Layer 2 -> Activation 2 -> Output Pixels

If your data flow is just a straight shot like this, `nn.Sequential` saves you from writing the `forward` method explicitly!

## Building Sequential Pixel Generators

Two ways to cook this up:

### 1. Simple List of Layers

Just give the `nn.Sequential` constructor your layers, in order. PyTorch names them with numbers (`0`, `1`, `2`, ...).

```python
# Potion Ingredients:
import torch
import torch.nn as nn

NOISE_DIM = 10
HIDDEN_DIM = 32
NUM_PIXELS = 16 # For a 4x4 output

# Method 1: Simple Sequential Generator
# Matches the structure of MultiLayerPixelGenerator from Guide 4
simple_pixel_generator = nn.Sequential(
    nn.Linear(NOISE_DIM, HIDDEN_DIM), # Layer 0
    nn.ReLU(),                        # Layer 1
    nn.Linear(HIDDEN_DIM, NUM_PIXELS),# Layer 2
    nn.Sigmoid()                      # Layer 3 (final activation)
)

print("--- Simple Sequential Pixel Generator --- ")
print(simple_pixel_generator)
# Output:
# Sequential(
#   (0): Linear(in_features=10, out_features=32, bias=True)
#   (1): ReLU()
#   (2): Linear(in_features=32, out_features=16, bias=True)
#   (3): Sigmoid()
# )
```

### 2. Using `OrderedDict` for Named Layers

Want more descriptive names than `0`, `1`, `2`? Use an `OrderedDict` from Python's `collections` module. The dictionary keys become the layer names.

```python
# Spell Snippet (Method 2):
from collections import OrderedDict

named_pixel_generator = nn.Sequential(
    OrderedDict(
        [
            ("input_layer", nn.Linear(NOISE_DIM, HIDDEN_DIM)),
            ("hidden_activation", nn.ReLU()),
            ("output_layer", nn.Linear(HIDDEN_DIM, NUM_PIXELS)),
            ("final_activation", nn.Sigmoid()),
        ]
    )
)

print("\n--- Named Sequential Pixel Generator --- ")
print(named_pixel_generator)
# Output:
# Sequential(
#   (input_layer): Linear(in_features=10, out_features=32, bias=True)
#   (hidden_activation): ReLU()
#   (output_layer): Linear(in_features=32, out_features=16, bias=True)
#   (final_activation): Sigmoid()
# )

# Now you can access layers by name if needed (less common with Sequential)
# print(f"Output layer weight shape: {named_pixel_generator.output_layer.weight.shape}")
```

## Using Your Sequential Generator

Both `simple_pixel_generator` and `named_pixel_generator` represent the _exact same_ network structure as the `MultiLayerPixelGenerator` we built manually. You use them just like any other `nn.Module`:

```python
# Spell Snippet (Usage):
BATCH_SIZE = 1
dummy_noise = torch.randn(BATCH_SIZE, NOISE_DIM)

with torch.no_grad():
    pixels_simple = simple_pixel_generator(dummy_noise)
    pixels_named = named_pixel_generator(dummy_noise)

print(f"\nOutput shape (simple): {pixels_simple.shape}") # torch.Size([1, 16])
print(f"Output shape (named): {pixels_named.shape}")   # torch.Size([1, 16])
```

## When `nn.Sequential` Isn't Enough

`nn.Sequential` is great for simple pipelines, but it can't handle:

- **Complex Routing:** If data needs to skip layers (skip connections), branch off, or take different paths.
- **Multiple Inputs/Outputs:** If your model takes more than one input tensor or produces multiple output tensors.
- **Reusing Layers:** If you need to apply the same layer multiple times in different parts of the flow.

**For anything more complex than a straight line of layers, you MUST go back to subclassing `nn.Module` and writing your own `forward` method.**

## Summary

`nn.Sequential` is a convenient container for defining simple pixel models where data flows linearly through a stack of layers. It saves you writing the `forward` method. Use simple arguments for quick models, or an `OrderedDict` for named layers. But remember, for complex pixel architectures with branches, skips, or custom logic, you need the full power of defining a custom `nn.Module` class.
