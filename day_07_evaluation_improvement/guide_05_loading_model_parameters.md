# Guide: 05 Reawakening Your Pixel Model: Loading Parameters (`state_dict`)!

We saved our model's learned parameters (`state_dict`) to a file. Now, how do we bring that knowledge back to life in a fresh model instance? This guide explains how to load the saved `state_dict`, based on `05_loading_model_parameters.py`.

**Core Concept:** Loading the `state_dict` is the reverse of saving. It takes the dictionary of learned weights and biases from your file and carefully copies them into a newly created instance of your model architecture, restoring its trained state.

## What You Need Before Loading

To reawaken your model, you absolutely need:

1.  **The Model Blueprint:** The Python code defining your exact `nn.Module` class (e.g., `MultiLayerPixelGenerator`). PyTorch needs this blueprint to build the empty shell.
2.  **The Saved Brain (`.pth` File):** The file containing the `state_dict` dictionary that you saved using `torch.save()`.

## The Reawakening Spell: Loading the `state_dict`

Follow these steps:

1.  **Build an Empty Shell (Instantiate):** Create a _new_ instance of your model class. It will start with random, untrained parameters.

    ```python
    # Spell Snippet:
    # Import or define your model class first!
    # from your_model_file import MultiLayerPixelGenerator

    # Create a new instance with the SAME architecture arguments as the saved one
    new_generator = MultiLayerPixelGenerator(noise_dim=10, hidden_dim=32, num_pixels=16)
    # At this point, new_generator has random weights!
    ```

2.  **Load the Brain from File (`torch.load`):** Use `torch.load()` to read the saved dictionary from your `.pth` file back into memory.

    ```python
    # Spell Snippet:
    LOAD_PATH = "trained_pixel_models/my_generator_v1.pth"
    print(f"Loading saved parameters from {LOAD_PATH}...")
    loaded_state_dict = torch.load(LOAD_PATH)
    # loaded_state_dict is now a Python dictionary
    ```

    - **Moving Between CPU/GPU?** If you saved weights from a GPU model but want to load onto a CPU machine (or vice-versa), use `map_location`: `loaded_state_dict = torch.load(LOAD_PATH, map_location=torch.device('cpu'))`.

3.  **Implant the Brain (`.load_state_dict()`):** Call the `.load_state_dict()` method _on your newly created model instance_, passing it the dictionary you just loaded.

    ```python
    # Spell Snippet:
    print("Applying loaded parameters to the model...")
    new_generator.load_state_dict(loaded_state_dict)
    print("Model parameters loaded successfully!")
    # Now, new_generator contains the learned weights!
    ```

    - **How it Works:** PyTorch carefully matches the keys (parameter names like `'layer_1.weight'`) in the loaded dictionary to the layers in your `new_generator` instance and copies the values over.
    - **Strict Matching (`strict=True`):** By default, PyTorch is strict. If the dictionary contains names that aren't in your model, or if your model has parameters not in the dictionary, it throws an error! This is usually good – it catches mistakes if you accidentally try to load weights into the wrong model architecture. (You _can_ use `strict=False` to load only matching parts, but be careful!).

4.  **Switch to Evaluation Mode (`model.eval()`):** If you're loading the model just to generate pixels or classify sprites (i.e., not for further training), immediately switch it to evaluation mode! This ensures layers like Dropout/BatchNorm behave correctly for inference.

    ```python
    # Spell Snippet:
    print("Setting model to evaluation mode...")
    new_generator.eval()
    # Now it's ready to generate pixels!
    ```

## Architecture MUST Match!

This cannot be stressed enough: The model class definition you use in Step 1 **must exactly match** the architecture used when the `state_dict` was saved. If you added a layer, changed a size, or renamed something, `load_state_dict` (with `strict=True`) will fail!

## Summary

To load your saved pixel model parameters:

1. Instantiate a new model object using the **exact same class definition**.
2. Load the saved parameter dictionary using `torch.load(PATH)`.
3. Apply the loaded dictionary to your model instance using `model.load_state_dict(loaded_dictionary)`.
4. If using for inference/evaluation, call `model.eval()`.
   This process breathes life back into your trained model, ready for more pixel adventures!
