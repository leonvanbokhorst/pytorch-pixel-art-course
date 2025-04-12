import torch
import torch.nn as nn


# Assume MultiLayerNet is defined or available (redefining for clarity)
class MultiLayerNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()  # Modern way to call super()
        self.layer_1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer_2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.layer_1(x)
        out = self.relu(out)
        out = self.layer_2(out)
        return out


# --- Instantiate the Model --- #
input_dim = 10
hidden_dim = 7
output_dim = 3
model = MultiLayerNet(
    input_size=input_dim, hidden_size=hidden_dim, output_size=output_dim
)
print(f"Model instantiated: {model}")

# --- Accessing Parameters --- #
print("\nInspecting model parameters:")
total_params = 0

# model.parameters() returns an iterator over all learnable parameters
# (weights and biases) of the model and its submodules.
for name, param in model.named_parameters():
    # 'param' is the tensor containing the parameter values
    # 'name' is the string name (e.g., 'layer_1.weight', 'layer_2.bias')
    if param.requires_grad:  # Check if it's learnable (should be True by default)
        num_elements = param.numel()  # Get the total number of elements in the tensor
        print(f"Parameter Name: {name}")
        print(f"  Shape: {param.shape}")
        print(f"  Requires Grad: {param.requires_grad}")
        print(f"  Number of Elements: {num_elements}")
        # print(f"  Values (first few): {param.data.flatten()[:5]}...") # Uncomment to peek
        total_params += num_elements

# --- Calculating Total Parameters --- #
print(f"\nTotal number of learnable parameters: {total_params}")

# Let's manually calculate for verification:
# Layer 1 weights: input_dim * hidden_dim = 10 * 7 = 70
# Layer 1 bias: hidden_dim = 7
# Layer 2 weights: hidden_dim * output_dim = 7 * 3 = 21
# Layer 2 bias: output_dim = 3
# Total = 70 + 7 + 21 + 3 = 101
manual_total = (
    (input_dim * hidden_dim) + hidden_dim + (hidden_dim * output_dim) + output_dim
)
print(f"Manually calculated total parameters: {manual_total}")
assert total_params == manual_total, "Parameter count mismatch!"

print("\n`model.parameters()` is crucial for passing weights to the optimizer.")
