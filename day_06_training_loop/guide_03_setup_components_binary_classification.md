# Guide: 03 Setup for Sprite Classification!

Okay, let's switch gears! Instead of generating pixels, what if we want to teach a model to _recognize_ different types of sprites? For example, telling apart a 'player' sprite from an 'enemy' sprite (a binary classification task). This guide covers setting up the ingredients for this specific task, based on `03_setup_components_binary_classification.py`.

**Core Concept:** The overall 5-ingredient setup (Model, Data, Loss, Optimizer, Device) is the same, but we need to tweak the _specific kind_ of Model output and Loss Function to suit classification.

## Ingredient Adjustments for Sprite Classification

Here's what needs changing compared to a simple pixel generator:

1.  **Model Output:** The final layer should output a _single number_ per sprite. This number will represent the model's raw score or "logit" indicating how strongly it believes the sprite belongs to class 1 (e.g., 'enemy') versus class 0 (e.g., 'player').
2.  **Model `forward`:** Should return this single raw logit directly, _without_ a final Sigmoid activation (because our chosen loss function handles that).
3.  **Sprite Labels:** The `Dataset` needs to provide labels for each sprite, typically `0.0` for one class and `1.0` for the other.
4.  **Loss Function:** We need a loss function designed for binary classification, like `nn.BCEWithLogitsLoss`.

Let's break it down:

### 1. & 2. Model Output: One Score Per Sprite (Logit)

- **Single Output Neuron:** The very last `nn.Linear` layer in your classifier model must have `out_features=1`.
- **Raw Score (Logit):** This single output isn't a probability yet; it's a raw score. Higher scores mean the model leans towards class 1.
- **No Final Sigmoid in Model:** Don't put `nn.Sigmoid()` at the very end of your model's `forward` method if you plan to use `nn.BCEWithLogitsLoss` (our recommended loss). The loss function includes the Sigmoid internally for better numerical stability.

```python
# Script Snippet (Model Definition - Example Classifier):
import torch.nn as nn

class SimpleSpriteClassifier(nn.Module):
    # Takes flattened sprite size and optional hidden layer size
    def __init__(self, input_pixel_count, hidden_size=64):
        super().__init__()
        self.layer_1 = nn.Linear(input_pixel_count, hidden_size)
        self.relu = nn.ReLU()
        # Output layer has only 1 output neuron!
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, flattened_sprite):
        x = self.layer_1(flattened_sprite)
        x = self.relu(x)
        # Return the raw logit directly from the last linear layer
        logits = self.output_layer(x)
        return logits
```

### 3. Sprite Labels: Is it Class 0 or Class 1?

Your `Dataset`'s `__getitem__` method should now return not just the sprite tensor, but also its corresponding label. For binary classification with `BCEWithLogitsLoss`, these labels should be **floats**: `0.0` or `1.0`.

```python
# Conceptual Dataset __getitem__:
def __getitem__(self, idx):
    sprite = self.load_sprite(idx) # Load sprite tensor
    label = self.get_label(idx)    # Get label (e.g., 0.0 for 'player', 1.0 for 'enemy')

    # Apply transforms if needed
    # if self.transform:
    #    sprite = self.transform(sprite)

    # Make sure label is a float tensor with shape [1]
    label_tensor = torch.tensor([label], dtype=torch.float32)

    # Return sprite and its label
    return sprite, label_tensor
```

### 4. Loss Function: `nn.BCEWithLogitsLoss` - The Binary Judge

This is the go-to loss function for binary classification in PyTorch.

- **BCE:** Binary Cross-Entropy measures the difference between the _predicted probability_ (derived from the logit) and the _true label_ (0.0 or 1.0).
- **WithLogits:** This is the important part! It tells the loss function: "Expect raw logits from the model. I will apply the Sigmoid _myself_ before calculating the loss." This is numerically more stable than you doing it separately.
- **Input Expectations:**
  - Expects Model Output (Logits): Shape `[batch_size, 1]`
  - Expects Target Labels: Shape `[batch_size, 1]`, dtype `torch.float32`, values `0.0` or `1.0`.

```python
# Script Snippet (Loss Instantiation):
# Use the numerically stable version!
criterion = nn.BCEWithLogitsLoss()
```

### 5. Optimizer: No Changes Needed!

Good news! The optimizer setup is exactly the same as before. Just point it at your _classifier_ model's parameters.

```python
# Script Snippet (Optimizer):
# Assume 'classifier_model' is an instance of SimpleSpriteClassifier
# Use Adam or SGD, just like before
optimizer = torch.optim.Adam(classifier_model.parameters(), lr=0.001)
```

## Summary

Setting up for binary sprite classification involves:

1. Designing a model that outputs **one raw logit** per sprite.
2. Ensuring your `Dataset` provides sprite tensors and corresponding **float labels (0.0 or 1.0)**.
3. Using **`nn.BCEWithLogitsLoss`** as the criterion.
4. Setting up the optimizer as usual, pointing it to the classifier model's parameters.

With these ingredients adjusted, you're ready for the classification training loop!
