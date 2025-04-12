import torch
import torch.nn as nn


# Define a network with multiple layers
class MultiLayerNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MultiLayerNet, self).__init__()

        # Define the layers
        self.layer_1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer_2 = nn.Linear(hidden_size, output_size)

        print("MultiLayerNet initialized!")
        print(f" - Input size: {input_size}")
        print(f" - Hidden size: {hidden_size}")
        print(f" - Output size: {output_size}")
        print(f" - Layer 1: {self.layer_1}")
        print(f" - Activation: {self.relu}")
        print(f" - Layer 2: {self.layer_2}")

    def forward(self, x):
        # Define the forward pass: input -> linear1 -> relu -> linear2 -> output
        print(f"\nExecuting forward pass...")
        print(f"  Input shape: {x.shape}")

        # Pass through first layer
        out = self.layer_1(x)
        print(f"  Shape after layer 1: {out.shape}")

        # Apply activation function
        out = self.relu(out)
        print(f"  Shape after ReLU: {out.shape}")

        # Pass through second layer
        out = self.layer_2(out)
        print(f"  Shape after layer 2 (final output): {out.shape}")

        return out


# --- Example Usage --- #
if __name__ == "__main__":
    input_dim = 20
    hidden_dim = 15  # Size of the hidden layer
    output_dim = 5
    batch_size = 3

    # Instantiate the model
    model = MultiLayerNet(
        input_size=input_dim, hidden_size=hidden_dim, output_size=output_dim
    )
    print(f"\nModel: {model}")

    # Create dummy input data
    dummy_input = torch.randn(batch_size, input_dim)
    print(f"\nCreated dummy input with shape: {dummy_input.shape}")

    # Pass the dummy input through the model
    with torch.no_grad():
        dummy_output = model(dummy_input)

    print(f"\nFinal output shape: {dummy_output.shape}")
    print(f"Final output tensor:\n{dummy_output}")

    # Verify output shape
    assert dummy_output.shape == (
        batch_size,
        output_dim,
    ), f"Expected output shape {(batch_size, output_dim)}, but got {dummy_output.shape}"
    print("\nAssertion passed: Output shape is correct.")
