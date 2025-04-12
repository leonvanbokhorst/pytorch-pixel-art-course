# Guide: 03 Setup Components for Binary Classification

This guide outlines how to adapt the training setup components (Model, Data, Loss) for a **binary classification** task, as demonstrated in `03_setup_components_binary_classification.py`.

**Core Concept:** While the overall training loop structure remains the same, the specifics of the model's output layer and the choice of loss function change when moving from regression (predicting a continuous value) to binary classification (predicting one of two classes, typically labeled 0 or 1).

## Component Adjustments for Binary Classification

Compared to the regression setup, we need to modify:

1. **Model Output Layer:** The final layer needs to produce a single output value per input sample.
2. **Model `forward` Return:** The model should return this single raw output value (logit).
3. **Data Labels:** Target labels should represent the two classes, typically as floats `0.0` and `1.0`.
4. **Loss Function:** Use a loss function suitable for binary classification, like Binary Cross Entropy.

Let's look at each:

### 1. & 2. Model Output and `forward`

- **Single Output:** The last `nn.Linear` layer in the model should have `out_features=1`.
- **Logits:** This single output value represents the raw, unnormalized prediction score, often called a **logit**. A higher logit suggests a higher probability of belonging to the positive class (class 1).
- **No Sigmoid in Model:** Crucially, for use with `nn.BCEWithLogitsLoss`, the model's `forward` method should return this raw logit directly, _without_ applying a Sigmoid activation function at the end.

```python
# Script Snippet (Model Definition):
class SimpleBinaryClassifier(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.layer_1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer_2 = nn.Linear(hidden_size, 1) # Output size is 1

    def forward(self, x):
        x = self.layer_1(x)
        x = self.relu(x)
        x = self.layer_2(x)
        return x # Return raw logit
```

### 3. Data Labels

The target labels (`y`) in your dataset should be floating-point numbers, either `0.0` or `1.0`, corresponding to the two classes.

```python
# Script Snippet (Data Generation):
# ... generate features X_data ...
true_boundary = ... # Logic to create boolean True/False labels
# Convert boolean to float 0.0 or 1.0 and ensure shape [num_samples, 1]
y_data_binary = true_boundary.float().unsqueeze(1)
```

### 4. Loss Function: `nn.BCEWithLogitsLoss`

This is the standard and recommended loss function for binary classification problems in PyTorch.

- **BCE:** Stands for Binary Cross-Entropy, a measure of the difference between two probability distributions (the predicted probability and the true label).
- **WithLogits:** This part signifies that the loss function _internally applies a Sigmoid function_ to the raw logits received from the model before calculating the BCE loss. This combination is numerically more stable than manually applying `nn.Sigmoid` in the model and then using `nn.BCELoss`.
- **Expectations:**
  - It expects the model's output to be raw logits (shape `[batch_size, 1]`).
  - It expects the target labels to be floats (`0.0` or `1.0`) with the same shape as the model output (`[batch_size, 1]`).

```python
# Script Snippet (Loss Instantiation):
criterion = nn.BCEWithLogitsLoss()
```

### 5. Optimizer

The optimizer setup remains unchanged. You still initialize it with the model's parameters and a learning rate.

```python
# Script Snippet (Optimizer):
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
```

## Summary

To set up for binary classification, ensure your model outputs a single raw logit per sample. Prepare your data with float labels (0.0 and 1.0). Use `nn.BCEWithLogitsLoss` as your criterion, feeding it the raw model logits and the float labels. The optimizer setup remains the same, taking `model.parameters()`.
