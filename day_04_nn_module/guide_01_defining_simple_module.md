# Guide: 01 Blueprinting Your First Pixel Model (`nn.Module`)

Alright, pixel architect! It's time to draw the blueprints for our pixel-generating (or classifying, or transforming) machines! This guide explains how to define a basic model structure using `torch.nn.Module`, the foundation shown in `01_defining_simple_module.py`.

**Core Concept:** Think of `torch.nn.Module` as the ultimate LEGO baseplate for building neural networks in PyTorch. Every model, big or small, starts by inheriting from `nn.Module`. It gives us a standard way to organize our model's parts (layers), manage its learnable knobs (parameters), and connect it to the rest of the PyTorch universe (like optimizers).

## The Pixel Model Blueprint

Creating your own pixel-munching model follows this sacred pattern:

1.  **Create a Class (Inheriting `nn.Module`):** Define a new Python class, making sure it inherits from `torch.nn.Module` (or just `nn.Module` if you `import torch.nn as nn`).

    ```python
    import torch
    import torch.nn as nn

    class YourPixelModel(nn.Module):
        # Model innards go here!
        ...
    ```

2.  **Define `__init__` (The Construction Yard):**

    - This is the model's constructor. It's where you **declare and initialize all the building blocks (layers)** your model will use.
    - **Job #1: Call `super().__init__()`!** You _must_ call the parent `nn.Module`'s constructor first thing. Don't forget!
    - Create instances of PyTorch layers (like `nn.Linear` to map inputs to outputs, `nn.ReLU` for non-linearity, maybe `nn.Conv2d` later for spatial stuff) and assign them to `self` (e.g., `self.input_layer = nn.Linear(...)`). This automatically tells the `nn.Module` baseplate about these layers and their learnable parameters.

    ```python
    def __init__(self, input_noise_dim, output_pixel_count):
        super().__init__() # <- ESSENTIAL!
        # Define the layers needed to turn noise into pixels
        self.generator_layer = nn.Linear(in_features=input_noise_dim, out_features=output_pixel_count)
        # Maybe add an activation later like self.activation = nn.Sigmoid()
    ```

3.  **Define `forward` (The Assembly Line):**

    - This method dictates **what happens every time you feed data into the model**. It defines how the input `x` flows through the layers you set up in `__init__`.
    - It takes `self` and the input tensor(s) (usually `x`) as arguments.
    - Inside `forward`, you call your layers (e.g., `self.generator_layer(x)`) like functions, passing the data along.
    - It **must return** the final output tensor(s) (e.g., the generated pixels).

    ```python
    def forward(self, input_noise):
        # Define how noise flows to become pixels
        pixel_output = self.generator_layer(input_noise)
        # Later we might add: pixel_output = self.activation(pixel_output)
        return pixel_output
    ```

## Example: `SimplePixelGenerator`

The script (`01...`) defines a super basic model that takes some input features (like a random noise vector) and directly maps them to output pixel values using one linear layer.

```python
# Script Snippet (Class Definition):
import torch
import torch.nn as nn

class SimplePixelGenerator(nn.Module):
    # Takes the size of the input random vector and how many pixels to output
    def __init__(self, noise_dim, num_pixels):
        super().__init__() # Step 1: Initialize the base Module
        # Step 2: Define the layer(s) - just one linear layer here
        self.generator_layer = nn.Linear(in_features=noise_dim, out_features=num_pixels)

    # Step 3: Define the data flow
    def forward(self, noise_vector):
        # Pass the noise directly through the linear layer
        generated_pixels = self.generator_layer(noise_vector)
        return generated_pixels
```

- **`__init__`:** Calls `super().__init__()`, then creates one `nn.Linear` layer that will learn to map `noise_dim` inputs to `num_pixels` outputs.
- **`forward`:** Takes the `noise_vector` and simply pushes it through the defined `generator_layer` to get the `generated_pixels`.

## Why Bother with `nn.Module`?

- **Organization:** Keeps your model's layers and logic neatly packed in a class.
- **Parameter Magic:** `nn.Module` automatically finds all the learnable weights and biases inside the layers you defined (like `self.generator_layer`). You can easily get them all using `model.parameters()` to feed to an optimizer – no manual tracking needed!
- **Nesting Dolls:** You can put `nn.Module`s inside other `nn.Module`s to build really complex pixel models layer by layer.
- **Handy Tools:** Comes with built-in helpers like `model.train()` / `model.eval()` (to switch modes), `model.to(device)` (to move to GPU), and ways to save/load your precious trained models.

## Summary

Building your pixel models starts with inheriting from `nn.Module`. Define your layers (like `nn.Linear` or later `nn.Conv2d`) in `__init__` (and call `super().__init__()`!), and define how data flows through them in `forward`. This structure keeps things tidy, tracks parameters automatically, and plugs right into the rest of PyTorch for training and evaluation!
