import os

import torch
import torch.nn as nn


# --- Re-define/Import Model Definition --- #
# CRITICAL: The architecture MUST match the one used when the state_dict was saved.
class SimpleClassificationNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.layer_1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer_2 = nn.Linear(hidden_size, num_classes)
        print("SimpleClassificationNet initialized for loading.")

    def forward(self, x):
        x = self.layer_1(x)
        x = self.relu(x)
        x = self.layer_2(x)
        return x


# ----------------------------------------- #

# --- Configuration (must match the saved model's config) --- #
INPUT_FEATURES = 10
HIDDEN_FEATURES = 15
NUM_CLASSES = 3

# --- Define Load Path --- #
LOAD_DIR = "saved_models"
MODEL_FILENAME = "simple_classifier_weights.pth"  # Must match the saved filename
LOAD_PATH = os.path.join(LOAD_DIR, MODEL_FILENAME)

print(f"Attempting to load model state_dict from: {LOAD_PATH}")

# --- Instantiate a NEW Model Instance --- #
# Create an instance of the model architecture.
# Its weights are initialized randomly at this point.
loaded_model = SimpleClassificationNet(INPUT_FEATURES, HIDDEN_FEATURES, NUM_CLASSES)
print("\nNew model instance created (with random initial weights).")

# --- Loading the State Dictionary --- #
if os.path.exists(LOAD_PATH):
    try:
        # 1. Load the saved state_dict from the file
        state_dict = torch.load(LOAD_PATH)
        print(f"Successfully loaded state_dict from {LOAD_PATH}")

        # Optional: Inspect keys if needed
        # print("Keys in loaded state_dict:", state_dict.keys())

        # 2. Load the state_dict into the model instance
        # `load_state_dict` copies the parameters from the state_dict into the model,
        # matching keys between the state_dict and the model's parameters.
        # `strict=True` (default) ensures all keys match exactly.
        loaded_model.load_state_dict(state_dict)
        print("Successfully loaded state_dict into the model instance.")

        # --- Set to Evaluation Mode --- #
        # Always set the model to eval() mode after loading for inference/evaluation
        # to disable dropout, use batchnorm running stats etc.
        loaded_model.eval()
        print("Model set to evaluation mode: loaded_model.eval()")

        # --- Optional: Test Loaded Model --- #
        print("\nTesting loaded model with dummy input...")
        dummy_input = torch.randn(1, INPUT_FEATURES)  # Batch size 1
        with torch.no_grad():
            output = loaded_model(dummy_input)
        print(f" - Dummy input shape: {dummy_input.shape}")
        print(f" - Output shape: {output.shape}")
        print(f" - Output (logits): {output}")
        print("Dummy inference successful.")

    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        print("Ensure the LOAD_PATH is correct and the file is not corrupted.")
        print(
            "Also ensure the model architecture definition matches the saved weights."
        )
else:
    print(f"Error: Saved model file not found at {LOAD_PATH}")
    print("Please run the '04_saving_model_parameters.py' script first.")

print("\nModel loading process demonstrated.")
