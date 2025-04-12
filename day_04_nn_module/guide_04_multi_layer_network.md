# Guide: 04 Stacking Layers: A Deeper Pixel Generator!

Our pixel generator is okay, but a single layer might be too simple to create truly awesome sprites. Let's stack layers to build a slightly deeper network, giving it more capacity to learn! This guide explores stacking, based on `04_multi_layer_network.py`.

**Core Concept:** Real magic often involves multiple steps! Similarly, powerful neural networks usually stack multiple layers. By combining linear layers (`nn.Linear`) and non-linear activations (`nn.ReLU`, `nn.Sigmoid`), each layer can build upon the previous one, learning increasingly complex features and transformations from the input noise to the final pixel output.

## The Layer Cake Structure (MLP for Pixels)

Even for generating pixels, a common structure (like a simple Multi-Layer Perceptron or MLP) involves:

1.  **Input Layer:** Usually a `nn.Linear` layer taking our input noise vector and transforming it into an intermediate "hidden" representation.
2.  **Hidden Layer(s):** One or more layers doing the heavy lifting. Each typically includes:
    - A `nn.Linear` layer to transform the features.
    - A non-linear activation (like `nn.ReLU`) to add complexity.
3.  **Output Layer:** A final `nn.Linear` layer that transforms the last hidden representation into the desired number of output pixel values.
4.  **Final Activation:** Often, an activation like `nn.Sigmoid` or `nn.Tanh` is applied after the output layer to constrain pixel values to the desired range (e.g., [0, 1]).

## Connecting the Layers: The Dimensions Must Flow!

This is crucial! The output size of one layer must match the input size of the next:

- The `out_features` of a `nn.Linear` layer determines how many features come _out_.
- The `in_features` of the _next_ `nn.Linear` layer must match that number.
- Activation functions like `ReLU` or `Sigmoid` usually don't change the number of features.

## Example: `MultiLayerPixelGenerator`

Let's build a generator with one hidden layer (Linear + ReLU) and an output layer (Linear + Sigmoid):

```python
# Script Snippet (Class Definition):
import torch
import torch.nn as nn

class MultiLayerPixelGenerator(nn.Module):
    def __init__(self, noise_dim, hidden_dim, num_pixels):
        super().__init__() # Initialize base Module!

        # Define the layers needed
        # Layer 1: Input noise -> Hidden dimension
        self.layer_1 = nn.Linear(noise_dim, hidden_dim)
        # Activation for hidden layer
        self.relu = nn.ReLU()
        # Layer 2: Hidden dimension -> Output pixels
        self.layer_2 = nn.Linear(hidden_dim, num_pixels)
        # Final activation to constrain pixel values (e.g., 0 to 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, noise_vector):
        # Define the assembly line for noise -> pixels
        # 1. Pass noise through the first linear layer
        hidden_output = self.layer_1(noise_vector)
        # 2. Apply ReLU activation
        activated_hidden = self.relu(hidden_output)
        # 3. Pass activated hidden state through the second linear layer
        linear_output = self.layer_2(activated_hidden)
        # 4. Apply final Sigmoid activation
        pixel_output = self.sigmoid(linear_output)
        return pixel_output
```

- **`__init__` Breakdown:**
  - `self.layer_1`: Maps input noise (`noise_dim`) to an intermediate `hidden_dim`.
  - `self.relu`: ReLU activation for the hidden layer.
  - `self.layer_2`: Maps the `hidden_dim` representation to the final `num_pixels`.
  - `self.sigmoid`: Final activation to get pixels between 0 and 1.
- **`forward` Breakdown:**
  - Noise goes into `layer_1`.
  - Result goes through `relu`.
  - Activated result goes into `layer_2`.
  - Final linear output goes through `sigmoid` to produce the pixel values.

## `forward` Flexibility: Beyond Simple Stacks

This `forward` method shows a simple sequence. But you have complete control! You could build models with:

- Many hidden layers.
- Skip connections (feeding an earlier layer's output further down the line).
- Multiple output branches (like the multi-task template in Day 9).

## Summary

Stacking layers (like `nn.Linear` + `nn.ReLU`) within an `nn.Module` allows you to build deeper, more powerful pixel models. Define the layers in `__init__` and orchestrate their interaction in `forward`, making sure the output dimension of one layer matches the input dimension of the next. This layered approach is key to learning complex pixel representations!
