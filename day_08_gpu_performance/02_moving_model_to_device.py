import torch
import torch.nn as nn

# --- 1. Define the Target Device (copied from previous example) --- #
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Selected device: CUDA (NVIDIA GPU)")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Selected device: MPS (Apple Silicon GPU)")
else:
    device = torch.device("cpu")
    print("Selected device: CPU")
print(f"Using device: {device}")


# --- 2. Define a Simple Model --- #
class SimpleNet(nn.Module):
    def __init__(self, input_size=10, hidden_size=5, output_size=2):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)
        print("SimpleNet initialized.")

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return x


# --- 3. Instantiate the Model --- #
model = SimpleNet()
print(f"\nModel instantiated: {model}")

# --- 4. Check Initial Parameter Device --- #
# Parameters are on CPU by default when the model is initialized
initial_param = next(model.parameters())
print(f"Device of initial parameter (e.g., layer1.weight): {initial_param.device}")
# Check if the model itself reports a device (it usually doesn't directly)
try:
    print(f"Model's device attribute (if exists): {model.device}")
except AttributeError:
    print("Model instance does not have a direct '.device' attribute.")

# --- 5. Move Model to the Target Device --- #
# model.to(device) moves all parameters and buffers of the model to the specified device.
# IMPORTANT: This operation is NOT typically in-place for nn.Module.
# You need to reassign the result back to the variable.
print(f"\nMoving model to device: {device} using model.to(device)...")
model = model.to(device)
print("Model move operation complete.")

# --- 6. Verify Parameter Device After Move --- #
# Check the device of the parameters again
moved_param = next(model.parameters())
print(f"Device of parameter after move: {moved_param.device}")

if moved_param.device == device:
    print(f"Successfully verified that model parameters are now on {device}.")
else:
    print(f"Error: Model parameters are still on {moved_param.device} after move!")

# --- 7. Moving Tensors (Example) --- #
# Remember that tensors also need to be moved to the device
cpu_tensor = torch.randn(4, 10)  # Input batch
print(f"\nCreated CPU tensor with device: {cpu_tensor.device}")

# Move tensor to the target device
device_tensor = cpu_tensor.to(device)
print(f"Moved tensor to device: {device_tensor.device}")

# Now you can pass the tensor through the model (if both are on the same device)
try:
    model.eval()  # Set to eval mode
    with torch.no_grad():
        output = model(device_tensor)
    print(f"Successfully passed device tensor through model on {device}.")
    print(f"Output tensor device: {output.device}")
except Exception as e:
    print(f"Error during model inference on {device}: {e}")
    print(
        "This usually happens if the model and input tensor are on different devices."
    )

print("\nMoving the model is a one-time setup step before training/evaluation.")
print("Moving data batches happens repeatedly inside the loops (next examples).")
