import torch
import torch.nn as nn


# Define the network class with non-linearity
class SimpleNetWithReLU(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNetWithReLU, self).__init__()

        # Define layers
        self.linear_layer = nn.Linear(input_size, output_size)
        # Define the activation function instance
        self.relu = nn.ReLU()

        print("SimpleNetWithReLU initialized!")
        print(f" - Layer defined: {self.linear_layer}")
        print(f" - Activation defined: {self.relu}")

    def forward(self, x):
        # Define the forward pass
        print(f"\nExecuting forward pass...")
        print(f"  Input shape: {x.shape}")
        # 1. Pass input through the linear layer
        linear_output = self.linear_layer(x)
        print(f"  Output shape after linear layer: {linear_output.shape}")
        print(f"  Output values before ReLU:\n{linear_output}")
        # 2. Apply the activation function
        output = self.relu(linear_output)
        print(f"  Output shape after ReLU: {output.shape}")
        print(f"  Output values after ReLU (negatives zeroed):\n{output}")
        return output


# --- Example Usage --- #
if __name__ == "__main__":
    input_dim = 6
    output_dim = 3
    batch_size = 2

    # Instantiate the model
    model = SimpleNetWithReLU(input_size=input_dim, output_size=output_dim)
    print(f"\nModel: {model}")

    # Create dummy input data
    dummy_input = torch.randn(batch_size, input_dim)
    # Let's make sure some inputs to ReLU will be negative
    # Note: This input change doesn't guarantee negative values *after* the linear layer
    # but increases the chance for demonstration.
    dummy_input[0, 0] = -5.0
    print(f"\nCreated dummy input with shape: {dummy_input.shape}")

    # Pass the dummy input through the model
    with torch.no_grad():
        dummy_output = model(dummy_input)

    print(f"\nFinal output shape: {dummy_output.shape}")
    print(f"Final output tensor:\n{dummy_output}")

    # Check: All output values should be >= 0 due to ReLU
    assert torch.all(dummy_output >= 0), "ReLU failed, found negative values!"
    print("\nAssertion passed: All output values are non-negative.")
