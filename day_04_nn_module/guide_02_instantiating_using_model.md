# Guide: 02 Instantiating and Using an nn.Module

This guide explains how to create an instance (object) of your custom `nn.Module` class and pass data through it to get predictions, building on the definition in `01_defining_simple_module.py` and demonstrated in `02_instantiating_using_model.py`.

**Core Concept:** Defining the class using `nn.Module` creates the blueprint. To actually use the network, you need to instantiate this class, creating a model object with its own set of initialized parameters (weights and biases). You then pass input data to this object to perform the forward pass defined in the class.

## 1. Instantiating the Model

Creating an instance is like creating any other Python object. You call the class name with any arguments required by its `__init__` method.

```python
# Script Snippet:
# Assuming SimpleNet class is defined
input_features = 10
output_features = 5

# Create the model object
model = SimpleNet(input_size=input_features, output_size=output_features)
print(f"\nModel instantiated: {model}")
# Output (example):
# Model instantiated: SimpleNet(
#   (linear_layer): Linear(in_features=10, out_features=5, bias=True)
# )
```

- **What happens here?** The `SimpleNet.__init__` method is executed for this specific `model` object.
- The `super().__init__()` call sets up the `nn.Module` base.
- `self.linear_layer = nn.Linear(...)` creates the linear layer _within_ this `model` instance. The weights and biases of this layer are automatically initialized (usually with random values according to standard schemes).

## 2. Preparing Input Data

Neural networks typically process data in batches. For a simple feed-forward network like this, the input tensor needs a shape compatible with the first layer's `in_features`. The standard shape is `(batch_size, input_features)`.

```python
# Script Snippet:
batch_size = 4 # Process 4 samples at once
dummy_input = torch.randn(batch_size, input_features)
print(f"\nCreated dummy input data with shape: {dummy_input.shape}")
# Output: Created dummy input data with shape: torch.Size([4, 10])
```

## 3. Calling the Model (Performing the Forward Pass)

This is the most direct way to use the model. You call the `model` object itself as if it were a function, passing the input tensor.

**Important:** You **do not** call `model.forward(dummy_input)` directly. Calling `model(dummy_input)` handles calling the `forward` method internally, along with other necessary hooks managed by PyTorch.

Since this is just for getting predictions (inference) and we don't need gradients, we wrap the call in `torch.no_grad()` for efficiency.

```python
# Script Snippet:
with torch.no_grad():  # Disable gradient calculations
    print(f"\nCalling model(dummy_input)... (runs the forward pass)")
    output = model(dummy_input)
```

- **`model.eval()` (Good Practice):** Although not strictly needed for this specific simple model, if your model included layers like `nn.Dropout` or `nn.BatchNorm2d`, you would typically call `model.eval()` before inference. This puts the model in evaluation mode, ensuring these layers behave correctly during prediction (e.g., dropout is turned off). `model.train()` switches it back.

## 4. Inspecting the Output

The `output` variable now holds the result of passing the `dummy_input` through the `forward` method of the `model`.

```python
# Script Snippet:
print(f"\nOutput received from model:")
print(f" - Output shape: {output.shape}")
print(f" - Output tensor:\n{output}")
# Output shape: torch.Size([4, 5]) (batch_size, output_features)
```

The output shape is `(batch_size, output_features)`, matching the `out_features` defined for the `linear_layer`.

## Summary

Using a defined `nn.Module` involves:

1. **Instantiating** the class to create a model object (`model = YourModel(...)`). This runs `__init__` and initializes parameters.
2. **Preparing** input data with the expected shape (e.g., `(batch_size, input_features)`).
3. **Calling** the model instance directly with the input (`output = model(input)`), ideally within a `torch.no_grad()` context for inference.
4. The **output** tensor contains the results of the defined `forward` pass.

This process separates the model definition (the class) from its application (the instance and its use).
