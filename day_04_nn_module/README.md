# Day 4: Building Networks with nn.Module

**Topics:**

- `torch.nn.Module`: The base class for all neural network modules.
- Defining a Custom Model:
  - Subclassing `nn.Module`.
  - Defining layers (e.g., `nn.Linear`, `nn.ReLU`) in `__init__`.
  - Implementing the `forward` method to define computation flow.
- Model Parameters: How `nn.Module` tracks parameters (`model.parameters()`).
- Using Built-in Layers: Common layers like linear, activation functions.
  - _Beyond `nn.Linear`/`nn.ReLU`: Awareness of other crucial layers like `nn.Conv2d` (for images), `nn.RNN`/`nn.LSTM` (for sequences), `nn.Embedding` (for categorical data) - See PyTorch docs._
- Using the Model: Instantiating and calling the model (`output = model(input)`).
- `nn.Sequential`: A container for simple, sequential models.

**Focus:** Organizing model architecture and parameters using PyTorch's standard building blocks.

## Key Resources

- **PyTorch Official Tutorials - Build the Model:** [https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html](https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html) (Covers defining a network class, `nn.Module`, layers, `forward` pass, `model.parameters()`)
- **`torch.nn` Module Documentation:** [https://pytorch.org/docs/stable/nn.html](https://pytorch.org/docs/stable/nn.html) (Overview of all neural network modules, layers, losses, containers, etc.)
- **`nn.Module` Documentation:** [https://pytorch.org/docs/stable/generated/torch.nn.Module.html](https://pytorch.org/docs/stable/generated/torch.nn.Module.html) (Detailed API for the base class)
- **`nn.Linear` Documentation:** [https://pytorch.org/docs/stable/generated/torch.nn.Linear.html](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)
- **`nn.ReLU` Documentation:** [https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html)
- **`nn.Sequential` Documentation:** [https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html)

## Hands-On Examples

- **Defining a Simple `nn.Module`:** ([`01_defining_simple_module.py`](./01_defining_simple_module.py))
  - **Code Idea:** Create a class `SimpleNet` that inherits from `nn.Module`. In `__init__`, define one linear layer (`nn.Linear(in_features=10, out_features=5)`). In `forward`, pass the input through this layer.
  - **Purpose:** Demonstrate the basic structure of an `nn.Module`: subclassing, defining layers in `__init__`, and defining the computation in `forward`.
- **Instantiating and Using the Model:** ([`02_instantiating_using_model.py`](./02_instantiating_using_model.py))
  - **Code Idea:** Create an instance of `SimpleNet`. Create a dummy input tensor `x = torch.randn(1, 10)` (batch size 1, 10 features). Pass the input through the model: `output = model(x)`. Print the output shape.
  - **Purpose:** Show how to create an object from the model class and call it like a function to get predictions.
- **Adding Non-linearity:** ([`03_adding_non_linearity.py`](./03_adding_non_linearity.py))
  - **Code Idea:** Modify `SimpleNet`. Add a `nn.ReLU()` activation function. In `forward`, pass the input through the linear layer _then_ the ReLU activation. Rerun the instantiation and prediction steps.
  - **Purpose:** Introduce activation functions and show how they are incorporated into the `forward` pass.
- **Building a Multi-Layer Network:** ([`04_multi_layer_network.py`](./04_multi_layer_network.py))
  - **Code Idea:** Define a `MultiLayerNet` with an input layer, one hidden layer (`nn.Linear`), a ReLU activation, and an output layer (`nn.Linear`). Define the `forward` pass accordingly. Instantiate and test with dummy data.
  - **Purpose:** Show how to stack multiple layers and activations to create a deeper network.
- **Inspecting Parameters:** ([`05_inspecting_parameters.py`](./05_inspecting_parameters.py))
  - **Code Idea:** Create an instance of `MultiLayerNet`. Iterate through `model.parameters()` and print the shape of each parameter tensor (weights and biases). Use `sum(p.numel() for p in model.parameters())` to count total parameters.
  - **Purpose:** Demonstrate how `nn.Module` automatically tracks all learnable parameters within its defined layers, making them accessible for optimizers.
- **(Optional) Using `nn.Sequential`:** ([`06_optional_sequential.py`](./06_optional_sequential.py))
  - **Code Idea:** Recreate the `MultiLayerNet` using `nn.Sequential` to define the layers and activations in order. Instantiate and test.
  - **Purpose:** Introduce `nn.Sequential` as a convenient way to define simple feed-forward architectures concisely.
