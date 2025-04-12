import torch
import torch.nn as nn
import os


# --- Re-define/Import Model Definition --- #
class SimpleClassificationNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.layer_1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer_2 = nn.Linear(hidden_size, num_classes)
        print("SimpleClassificationNet initialized for saving.")

    def forward(self, x):
        x = self.layer_1(x)
        x = self.relu(x)
        x = self.layer_2(x)
        return x


# ----------------------------------------- #

# --- Configuration & Instantiate Model --- #
INPUT_FEATURES = 10
HIDDEN_FEATURES = 15
NUM_CLASSES = 3

# Instantiate the model (imagine it has been trained)
model = SimpleClassificationNet(INPUT_FEATURES, HIDDEN_FEATURES, NUM_CLASSES)
print(f"\nModel instantiated: {model}")

# --- Define Save Path --- #
SAVE_DIR = "saved_models"
MODEL_FILENAME = "simple_classifier_weights.pth"
SAVE_PATH = os.path.join(SAVE_DIR, MODEL_FILENAME)

# Create the directory if it doesn't exist
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
    print(f"Created directory: {SAVE_DIR}")

print(f"Will save model state_dict to: {SAVE_PATH}")

# --- Saving the Model's State Dictionary --- #
# `model.state_dict()` returns an OrderedDict containing all learnable parameters
# (weights and biases) and registered buffers (like BatchNorm running means).
# This is the recommended way to save model parameters.
print("\nAccessing model.state_dict()...")
state_dict = model.state_dict()

# Optional: Print the keys in the state dictionary
print("Keys in state_dict:")
for key in state_dict.keys():
    print(f" - {key}")

# Use torch.save() to serialize the state_dict to a file
try:
    torch.save(state_dict, SAVE_PATH)
    print(f"\nSuccessfully saved model state_dict to {SAVE_PATH}")

    # Verify file exists
    if os.path.exists(SAVE_PATH):
        print(f"File verification successful: {SAVE_PATH} exists.")
    else:
        print(f"Error: File {SAVE_PATH} was not created after saving.")

except Exception as e:
    print(f"Error saving model: {e}")

print(
    "\nNote: This saves only the model parameters (weights and biases)."
    " The model's architecture (the class definition) must be available separately to load these weights."
)
print("See the next example for how to load these saved parameters.")
