import torch

# Assume SimpleNet is defined in the previous file or available here
# For simplicity, let's redefine it briefly (normally you'd import)
import torch.nn as nn


class SimpleNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNet, self).__init__()
        self.linear_layer = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear_layer(x)


# --- Setup --- #
input_features = 10
output_features = 5
batch_size = 4  # Let's use a batch size > 1

print(f"Input Features: {input_features}")
print(f"Output Features: {output_features}")
print(f"Batch Size: {batch_size}")

# --- Instantiate the Model --- #
# Create an object (instance) of our network class
model = SimpleNet(input_size=input_features, output_size=output_features)
print(f"\nModel instantiated: {model}")

# --- Create Dummy Input Data --- #
# Create a batch of random data with the correct input shape
# Shape: (batch_size, input_features)
dummy_input = torch.randn(batch_size, input_features)
print(f"\nCreated dummy input data with shape: {dummy_input.shape}")

# --- Pass Input Through Model (Inference/Prediction) --- #
# Call the model instance like a function, passing the input tensor.
# This implicitly calls the model's forward() method.
# Put the model in evaluation mode if it had layers like Dropout/BatchNorm
# model.eval() # Not strictly necessary here, but good practice
with torch.no_grad():  # Disable gradient calculation for inference
    print(f"\nCalling model(dummy_input)... (runs the forward pass)")
    output = model(dummy_input)

# --- Inspect the Output --- #
print(f"\nOutput received from model:")
print(f" - Output shape: {output.shape}")  # Should be (batch_size, output_features)
print(f" - Output tensor:\n{output}")

# Verify output shape
assert output.shape == (
    batch_size,
    output_features,
), f"Expected output shape {(batch_size, output_features)}, but got {output.shape}"

print("\nSuccessfully instantiated the model and passed data through it!")
