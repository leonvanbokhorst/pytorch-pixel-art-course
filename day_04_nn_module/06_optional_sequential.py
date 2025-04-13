from collections import OrderedDict

import torch
import torch.nn as nn

# --- Define Model Parameters --- #
input_size = 10
hidden_size = 7
output_size = 3
batch_size = 4

# --- Method 1: Simple nn.Sequential --- #
# Pass layers as separate arguments
model_sequential_simple = nn.Sequential(
    nn.Linear(input_size, hidden_size),  # Layer 0
    nn.ReLU(),  # Layer 1
    nn.Linear(hidden_size, output_size),  # Layer 2
)

print("--- Simple Sequential Model --- ")
print(model_sequential_simple)

# Test the simple sequential model
dummy_input = torch.randn(batch_size, input_size)
with torch.no_grad():
    output_simple = model_sequential_simple(dummy_input)
print(f"Input shape: {dummy_input.shape}")
print(f"Output shape (simple): {output_simple.shape}")
assert output_simple.shape == (batch_size, output_size)

# --- Method 2: nn.Sequential with OrderedDict --- #
# Use an OrderedDict to give names to the layers
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

# Test the named sequential model
with torch.no_grad():
    output_named = model_sequential_named(dummy_input)
print(f"Output shape (named): {output_named.shape}")
assert output_named.shape == (batch_size, output_size)

# You can access layers by name
print(f"\nAccessing named layers:")
print(f" - Input layer weight shape: {model_sequential_named.input_layer.weight.shape}")

# --- Comparison with nn.Module subclass --- #
# This nn.Sequential model is equivalent to the MultiLayerNet defined earlier,
# but nn.Sequential is less flexible if you have complex forward passes
# (e.g., skip connections, multiple inputs/outputs).

print("\n`nn.Sequential` is convenient for simple linear stacks of layers.")
print("For more complex architectures, subclassing `nn.Module` is preferred.")
