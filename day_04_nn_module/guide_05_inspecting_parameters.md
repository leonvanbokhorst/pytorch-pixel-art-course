# Guide: 05 Peeking Inside: Inspecting Pixel Model Parameters!

We've built our pixel generator blueprints (`nn.Module`). But how does it actually _learn_? By tweaking its internal "knobs" – the **parameters** (weights and biases) inside its layers! This guide shows how `nn.Module` cleverly keeps track of these knobs and how we can peek at them, based on `05_inspecting_parameters.py`.

**Core Concept:** A huge benefit of using `nn.Module` is that it automatically finds and manages all the learnable parameters within the layers you define (like the weights and biases in `nn.Linear`). You don't have to manually collect them – PyTorch does it for you!

## Why Peek at the Parameters?

- **Feeding the Optimizer:** This is the #1 reason! When we train (Day 6), we need to tell the optimizer (like Adam or SGD) exactly _which_ tensors in our model it should be adjusting. `model.parameters()` gives the optimizer this list.
- **Sanity Checks & Debugging:** Is the shape of that weight matrix correct? Are the gradients flowing (we'll see `.grad` later)? Inspecting parameters helps understand what's going on.
- **Custom Spells:** Maybe you want to initialize weights in a special way, or apply different learning rules to different layers (advanced magic!).
- **Model Size Check:** How complex is our pixel generator? Counting the total number of parameters gives a rough idea of its size and potential memory usage.

## Accessing the Magic Knobs

`nn.Module` gives us easy ways to get the parameters:

1.  **`model.parameters()`:**

    - Returns a handy _iterator_ that gives you every single learnable parameter tensor within the model (and any sub-models it contains).
    - This is what you'll almost always use to tell the optimizer what to optimize: `optimizer = torch.optim.Adam(my_pixel_generator.parameters(), lr=0.001)`.

2.  **`model.named_parameters()`:**
    - Similar, but the iterator gives you pairs: `(parameter_name, parameter_tensor)`.
    - The `parameter_name` is a string like `'layer_1.weight'` or `'layer_2.bias'`, helping you identify _which_ knob you're looking at.
    - Useful for logging, debugging, or applying specific rules to certain layers.

## Inspecting Our Generator's Knobs

The script uses `named_parameters()` to loop through and display info about each parameter in our `MultiLayerPixelGenerator`:

```python
# Script Snippet (Looping through parameters):

# Assume MultiLayerPixelGenerator is defined and instantiated:
# generator = MultiLayerPixelGenerator(noise_dim=10, hidden_dim=32, num_pixels=16)

print("\nInspecting generator parameters:")
total_params = 0

for name, param in generator.named_parameters():
    # parameters() only yields learnable ones, but checking is good practice
    if param.requires_grad:
        num_elements = param.numel() # How many numbers in this knob?
        print(f"--- Parameter: {name} ---")
        print(f"  Shape: {param.shape}")
        print(f"  Requires Grad?: {param.requires_grad}") # Should be True!
        print(f"  Number of Elements: {num_elements}")
        # We can even look at the initial random values (first 5):
        # print(f"  Data (first 5): {param.data.flatten()[:5]}")
        total_params += num_elements

print(f"\n---> Total number of learnable parameters: {total_params}")
```

- **`param.shape`**: Tells you the dimensions (e.g., `[hidden_dim, noise_dim]` for `layer_1.weight`).
- **`param.requires_grad`**: Should be `True` for these, indicating Autograd will calculate gradients for them.
- **`param.numel()`**: Counts the total individual numbers in that specific weight matrix or bias vector.
- **`param.data`**: Lets you access the raw numbers directly, bypassing gradient tracking (useful for manual initialization, but use with care).

## Counting the Knobs (Total Parameters)

Adding up `param.numel()` for all parameters gives the total number of learnable values the optimizer needs to tune.

Let's manually verify for our `MultiLayerPixelGenerator(noise_dim=10, hidden_dim=32, num_pixels=16)`:

- `layer_1` (`nn.Linear(10, 32)`):
  - Weight: `10 * 32 = 320`
  - Bias: `32`
  - Subtotal: `320 + 32 = 352`
- `layer_2` (`nn.Linear(32, 16)`):
  - Weight: `32 * 16 = 512`
  - Bias: `16`
  - Subtotal: `512 + 16 = 528`
- Total: `352 + 528 = 880` parameters. The script should print this number!

## Summary

`nn.Module` is awesome because it automatically finds all the learnable parameters (weights, biases) in your pixel model's layers. Use `model.parameters()` to easily give them to an optimizer, or `model.named_parameters()` to inspect specific knobs by name and shape. This auto-tracking makes building and training much simpler!
