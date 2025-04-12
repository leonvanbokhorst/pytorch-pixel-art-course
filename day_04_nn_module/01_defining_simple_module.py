import torch
import torch.nn as nn


# Define the network class
class SimpleNet(nn.Module):
    def __init__(self, input_size, output_size):
        # Call the parent class constructor
        super(SimpleNet, self).__init__()

        # Define the layers
        # A single linear layer (fully connected layer)
        # It takes input_size features and outputs output_size features
        self.linear_layer = nn.Linear(in_features=input_size, out_features=output_size)

        print("SimpleNet initialized!")
        print(f" - Input size: {input_size}")
        print(f" - Output size: {output_size}")
        print(f" - Layer defined: {self.linear_layer}")

    def forward(self, x):
        # Define the forward pass - how input flows through the layers
        # In this case, just pass the input through the linear layer
        print(f"\nExecuting forward pass...")
        print(f"  Input shape: {x.shape}")
        output = self.linear_layer(x)
        print(f"  Output shape after linear layer: {output.shape}")
        return output


# --- Example Usage (demonstration purposes, more in the next file) --- #
if __name__ == "__main__":
    print("Demonstrating SimpleNet definition:")
    input_dim = 10
    output_dim = 5

    # Instantiate the model
    model = SimpleNet(input_size=input_dim, output_size=output_dim)

    # Create some dummy input data
    # Batch size of 1, input features = input_dim
    dummy_input = torch.randn(1, input_dim)
    print(f"\nCreated dummy input with shape: {dummy_input.shape}")

    # Pass the dummy input through the model (calls the forward method)
    dummy_output = model(dummy_input)
    print(f"\nReceived dummy output with shape: {dummy_output.shape}")
    print(f"Dummy output tensor:\n{dummy_output}")

    print("\nThis file primarily defines the SimpleNet class.")
    print("See the next example for more focused usage.")
