# Guide: 04 Saving Your Pixel Masterpiece: Model Parameters (`state_dict`)!

Your pixel model has trained hard and learned amazing things! ✨ You don't want to lose all that progress, right? This guide explains the best way to save the model's learned knowledge – its **parameters** – using the `state_dict`, as shown in `04_saving_model_parameters.py`.

**Core Concept:** After training, the most important part of your model is the set of learned values for its internal knobs (weights and biases). We need to save these so we can load them back later to generate more pixels, classify sprites, or even continue training without starting from scratch.

## The Best Spell: Saving the `state_dict`

You _could_ try to save the entire model object, but that often causes headaches later if you change your code. The **recommended and safest way** is to save just the model's **`state_dict`**.

Think of it like saving just the _learned settings_ of your pixel art software, not the entire software installation. It's much more portable!

## What's a `state_dict`, Really?

It's just a Python dictionary! It holds the current state of your model's learnable parts:

- **Keys:** Names of layers (e.g., `'generator_layer.weight'`, `'output_layer.bias'`).
- **Values:** The actual `torch.Tensor` objects containing the learned weights and biases for each layer.
- It also includes **buffers** (like running averages in BatchNorm layers), which are part of the state but not directly trained by the optimizer.

Saving this dictionary saves all the essential learned information.

## How to Save Your Model's Brain (`state_dict`)

It's a simple three-step spell:

1.  **Extract the Brain (`.state_dict()`):** Get the state dictionary from your _trained_ model instance.

    ```python
    # Spell Snippet:
    # Assume 'trained_pixel_model' is your nn.Module after training
    learned_parameters = trained_pixel_model.state_dict()
    ```

2.  **Choose a Save Location:** Pick a folder and filename. Using `.pth` or `.pt` is the common convention for PyTorch files.

    ```python
    # Spell Snippet:
    import os
    SAVE_FOLDER = "trained_pixel_models"
    FILENAME = "my_generator_v1.pth"
    SAVE_PATH = os.path.join(SAVE_FOLDER, FILENAME)

    # Make sure the folder exists!
    os.makedirs(SAVE_FOLDER, exist_ok=True)
    ```

3.  **Cast `torch.save()`:** Use this function to save the `learned_parameters` dictionary to your chosen file path.

    ```python
    # Spell Snippet:
    print(f"Saving learned parameters to {SAVE_PATH}...")
    torch.save(learned_parameters, SAVE_PATH)
    print("Parameters saved successfully!")
    ```

## The Catch: Blueprint Not Included!

Super important: Saving the `state_dict` **only saves the parameters (the learned settings)**. It does **NOT** save the model's architecture (the Python class definition, like `MultiLayerPixelGenerator`).

To use these saved parameters later, you'll need both:

1. The **Python code defining your model class** (the blueprint).
2. The **saved `.pth` file** containing the `state_dict` (the learned settings).

## Summary

To save your trained pixel model's progress, save its `state_dict`: `torch.save(model.state_dict(), PATH)`. This is the standard, portable way to store the learned weights and biases. Just remember you'll need the model's class definition code handy when you want to load these parameters back in later (next guide!).
