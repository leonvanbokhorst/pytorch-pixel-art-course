# Day 4: `nn.Module` - Blueprints for Pixel Art Models

**Topics:**

- `torch.nn.Module`: The base class for building our pixel generation/processing models.
- Defining a Custom Pixel Model:
  - Subclassing `nn.Module`.
  - Defining layers in `__init__`: Examples might include `nn.Linear` (for mapping latent vectors to pixels), `nn.Conv2d` (essential for image processing, _briefly introduce idea_), activation functions like `nn.ReLU` or `nn.Sigmoid` (to keep pixel values in range).
  - Implementing the `forward` method: Defining how input data (like noise or coordinates) flows through layers to produce pixel outputs.
- Model Parameters: How `nn.Module` keeps track of the learnable weights and biases that define the model's pixel-generating abilities.
- Using Built-in Layers for Pixels: Linear layers, activation functions. Mentioning convolutional layers (`nn.Conv2d`) as the standard tool for spatial data like images, even if not fully implemented yet.
- Using the Pixel Model: Creating an instance (`generator = PixelGenerator()`) and generating pixel data (`output_pixels = generator(input_noise)`).
- `nn.Sequential`: A simpler way to define models where data flows straight through a sequence of layers (good for simple generators or classifiers).

**Focus:** Learning how to structure neural network models for pixel art tasks using `nn.Module`, defining layers, and the forward computation path.

## Key Resources

- **PyTorch Official Tutorials - Build the Model:** [https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html](https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html)
- **`torch.nn` Module Documentation:** [https://pytorch.org/docs/stable/nn.html](https://pytorch.org/docs/stable/nn.html)
- **`nn.Module` Documentation:** [https://pytorch.org/docs/stable/generated/torch.nn.Module.html](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)
- **`nn.Linear` Documentation:** [https://pytorch.org/docs/stable/generated/torch.nn.Linear.html](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)
- **`nn.ReLU`, `nn.Sigmoid` Documentation:** [https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity](https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity)
- **`nn.Conv2d` Documentation:** [https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html) (_Important for later image work_)
- **`nn.Sequential` Documentation:** [https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html)

## Hands-On Pixel Model Examples

- **Defining a Simple Pixel Generator (`nn.Module`):** ([`01_defining_simple_module.py`](./01_defining_simple_module.py))
  - **Pixel Idea:** Create a class `SimplePixelGenerator` inheriting `nn.Module`. In `__init__`, define a linear layer mapping a small input noise vector (e.g., size 10) to a small number of output pixels (e.g., 16 for a 4x4 grayscale image).
  - **Purpose:** Show the basic structure of a custom `nn.Module` for a pixel-related task.
- **Instantiating and Using the Generator:** ([`02_instantiating_using_model.py`](./02_instantiating_using_model.py))
  - **Pixel Idea:** Create an instance `generator = SimplePixelGenerator()`. Create dummy input noise `noise = torch.randn(1, 10)`. Generate output pixels: `pixels = generator(noise)`. Print the output shape (should be `[1, 16]`). _Optional: Reshape to 4x4 to visualize conceptually._
  - **Purpose:** Show how to create and use the defined pixel generator model.
- **Adding Non-linearity (Sigmoid for Pixels):** ([`03_adding_non_linearity.py`](./03_adding_non_linearity.py))
  - **Pixel Idea:** Modify `SimplePixelGenerator`. Add a `nn.Sigmoid()` activation. In `forward`, pass the input through the linear layer _then_ the Sigmoid activation to constrain output values between 0 and 1 (suitable for representing normalized pixel intensities).
  - **Purpose:** Introduce activation functions relevant to pixel generation (like Sigmoid/Tanh) and incorporate them.
- **Building a Multi-Layer Pixel Network:** ([`04_multi_layer_network.py`](./04_multi_layer_network.py))
  - **Pixel Idea:** Define a `MultiLayerPixelGenerator` with an input layer, one hidden linear layer with ReLU, and an output linear layer followed by Sigmoid. Define the `forward` pass. The output could still be a flattened pixel vector.
  - **Purpose:** Show stacking layers to create a slightly more complex generator.
- **Inspecting Generator Parameters:** ([`05_inspecting_parameters.py`](./05_inspecting_parameters.py))
  - **Pixel Idea:** Create an instance of `MultiLayerPixelGenerator`. Iterate through `model.parameters()` and print their shapes. Count the total number of learnable parameters.
  - **Purpose:** Demonstrate how `nn.Module` manages the learnable parameters of the generator.
- **(Optional) Using `nn.Sequential` for Simple Generation:** ([`06_optional_sequential.py`](./06_optional_sequential.py))
  - **Pixel Idea:** Recreate the `MultiLayerPixelGenerator` using `nn.Sequential`, passing the layers and activations in order.
  - **Purpose:** Introduce `nn.Sequential` for defining straightforward pixel generation pipelines concisely.
