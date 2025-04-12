# Guide: 03 Calculating Accuracy

This guide explains how to calculate a common performance metric, accuracy, within the PyTorch evaluation loop, focusing on the multi-class classification example in `03_calculating_accuracy.py`.

**Core Concept:** While the loss function guides the training process, metrics like accuracy provide a more interpretable measure of how well the model is actually performing its classification task. Accuracy represents the fraction of samples that the model classifies correctly.

## Calculating Accuracy in the Evaluation Loop

Calculating accuracy typically involves these steps within the `with torch.no_grad():` block of your evaluation loop:

1. **Get Model Outputs:** Obtain the model's predictions for the current batch. For classification, these are often raw scores (logits).

   ```python
   # Inside loop:
   outputs = model(batch_X) # Shape e.g., [batch_size, num_classes]
   ```

2. **Determine Predicted Labels:** Convert the model's raw outputs into predicted class labels.
   The method depends on the output format and task:
   _**Multi-Class Classification (Logits Output):** Find the index of the highest score along the class dimension. The `torch.argmax()` function is perfect for this.

   ```python
    # dim=1 because classes are typically along the second dimension
    predicted_labels = torch.argmax(outputs, dim=1) # Shape: [batch_size]
   ```

   _ **Binary Classification (Logits Output):** Convert logits to probabilities using `torch.sigmoid()`, then threshold at 0.5.

   ```python
    # predicted_probs = torch.sigmoid(outputs) # Shape: [batch_size, 1]
    # predicted_labels = (predicted_probs >= 0.5).float() # Shape: [batch_size, 1]
    # Ensure true labels batch_y also have shape [batch_size, 1]
   ```

3. **Compare Predictions to True Labels:** Use the equality operator (`==`) to compare the tensor of predicted labels with the tensor of true labels (`batch_y`) for the current batch. This results in a boolean tensor.

   ```python
   # Assumes predicted_labels and batch_y have compatible shapes
   correct_predictions = (predicted_labels == batch_y)
   ```

4. **Count Correct Predictions:** Sum the boolean tensor. `True` values are treated as 1 and `False` as 0, so the sum gives the number of correct predictions in the batch. Use `.item()` to get the result as a Python number.

   ```python
   correct_in_batch = correct_predictions.sum().item()
   ```

5. **Accumulate Total Correct:** Keep a running total of correct predictions across all batches in the evaluation set.

   ```python
   # Before the loop: total_correct = 0
   # Inside the loop: total_correct += correct_in_batch
   ```

6. **Calculate Final Accuracy:** After iterating through all batches, divide the `total_correct` count by the total number of samples evaluated.

   ```python
   # After the loop:
   # num_samples_processed = len(val_loader.dataset)
   # accuracy = total_correct / num_samples_processed
   ```

## Code Walkthrough (Multi-Class)

The script `03_calculating_accuracy.py` implements these steps for multi-class classification:

```python
# Script Snippet (Inside Evaluation Loop):
...
total_correct = 0
num_samples_processed = 0
with torch.no_grad():
    for batch_idx, (batch_X, batch_y) in enumerate(val_loader):
        outputs = model(batch_X) # Logits: [batch_size, num_classes]
        loss = criterion(outputs, batch_y)
        total_val_loss += loss.item() * batch_X.size(0)

        # --- Accuracy Calculation --- #
        # Get predicted class index (highest logit score)
        predicted_labels = torch.argmax(outputs, dim=1) # Shape: [batch_size]
        # Compare with true labels (assuming batch_y shape is [batch_size])
        correct_in_batch = (predicted_labels == batch_y).sum().item()
        total_correct += correct_in_batch
        # -------------------------- #

        num_samples_processed += batch_X.size(0)
...
# After loop:
accuracy = total_correct / num_samples_processed
print(f"Validation Accuracy: {accuracy:.4f}")
```

## Other Metrics

Accuracy is common but might be misleading for imbalanced datasets. Other metrics often used, especially for classification, include:

- Precision
- Recall
- F1-Score
- AUC (Area Under the ROC Curve)

Libraries like `scikit-learn` or PyTorch-specific libraries like `torchmetrics` provide implementations for these and many other metrics, simplifying their calculation.

## Summary

Calculating metrics like accuracy within the evaluation loop provides interpretable insights into model performance. For multi-class classification with logit outputs, this typically involves using `torch.argmax` to find the predicted class indices, comparing them to the true labels, and aggregating the results over the entire evaluation dataset. Remember to perform these calculations within the `torch.no_grad()` context and after setting `model.eval()`.
