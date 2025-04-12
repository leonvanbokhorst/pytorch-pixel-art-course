# Guide: 03 Grading the Quiz: Calculating Sprite Classification Accuracy!

Okay, our model took the pop quiz (evaluation loop). We calculated the overall loss, but how many answers did it actually get _right_? This guide focuses on calculating **accuracy** – the percentage of sprites correctly classified – within the evaluation loop, as shown in `03_calculating_accuracy.py` (often adapted for classification).

**Core Concept:** While validation loss tells you the _average error_, accuracy gives a more direct answer to "How often does my model correctly identify the sprite type (e.g., player vs. enemy)?". It's often a more intuitive metric for classification tasks.

## Calculating Accuracy in the Evaluation Loop

We perform this _inside_ the `with torch.no_grad():` block of the evaluation loop (from Guide 2), right after getting the model's outputs for a batch:

1.  **Get Model Outputs (Logits):** Get the raw scores from your classifier model.

    ```python
    # Inside evaluation loop:
    # outputs shape: [batch_size, 1] for binary, or [batch_size, num_classes] for multi-class
    outputs = model_to_evaluate(sprite_batch)
    ```

2.  **Convert Outputs to Predicted Labels:** Turn the raw scores into actual class predictions (0 or 1 for binary; 0, 1, 2... for multi-class).

    - **Binary Classification (using `BCEWithLogitsLoss`):**
      - Convert logits to probabilities: `predicted_probs = torch.sigmoid(outputs)`
      - Threshold probabilities at 0.5: `predicted_labels = (predicted_probs >= 0.5).float()` (Result is 0.0 or 1.0)
    - **Multi-Class Classification (using `CrossEntropyLoss`):**
      - Find the index (class number) with the highest score using `argmax`: `predicted_labels = torch.argmax(outputs, dim=1)` (Result is 0, 1, 2...)

3.  **Compare Predictions to True Labels:** Use `==` to see where the `predicted_labels` match the actual `label_batch`.

    ```python
    # Assumes predicted_labels and label_batch have compatible shapes and dtypes!
    # For binary, label_batch should be [batch_size, 1] float
    # For multi-class, label_batch should be [batch_size] integer
    correct_predictions_mask = (predicted_labels == label_batch)
    ```

4.  **Count Correct in Batch:** Sum the `True` values in the comparison mask. `.sum()` treats `True` as 1 and `False` as 0. `.item()` gets the Python number.

    ```python
    correct_in_batch = correct_predictions_mask.sum().item()
    ```

5.  **Accumulate Total Correct:** Keep a running total across all batches.

    ```python
    # Before the loop: total_correct_predictions = 0
    # Inside the loop, after counting: total_correct_predictions += correct_in_batch
    ```

6.  **Calculate Final Accuracy:** After the loop finishes (all batches processed), divide the total correct count by the total number of validation sprites.

    ```python
    # After the loop:
    total_validation_samples = len(val_loader.dataset)
    validation_accuracy = total_correct_predictions / total_validation_samples
    ```

## Code Example (Adding to Evaluation Loop)

Let's integrate the binary classification accuracy steps into the loop from Guide 2:

```python
# ... (Setup: model, val_loader, criterion, device) ...
model_to_evaluate.eval()

total_validation_loss = 0.0
total_correct_predictions = 0 # <<< Initialize accumulator

with torch.no_grad():
    for sprite_batch, label_batch in val_loader:
        sprite_batch = sprite_batch.to(device)
        label_batch = label_batch.to(device)

        # --- Forward Pass & Loss --- #
        outputs = model_to_evaluate(sprite_batch) # Logits output
        loss = criterion(outputs, label_batch)
        total_validation_loss += loss.item() * sprite_batch.size(0)

        # === Accuracy Calculation Steps === #
        predicted_probs = torch.sigmoid(outputs)             # Logits -> Probabilities
        predicted_labels = (predicted_probs >= 0.5).float()  # Probabilities -> Labels (0.0/1.0)
        correct_in_batch = (predicted_labels == label_batch).sum().item() # Count correct
        total_correct_predictions += correct_in_batch       # Accumulate
        # =============================== #

# --- After the loop --- #
average_validation_loss = total_validation_loss / len(val_loader.dataset)
validation_accuracy = total_correct_predictions / len(val_loader.dataset) # Final Accuracy!

print(f"Validation Loss: {average_validation_loss:.4f}")
print(f"Validation Accuracy: {validation_accuracy:.4f}")
```

## Beyond Accuracy: Other Pixel Metrics

Accuracy is great, but sometimes insufficient:

- **Imbalanced Data:** If 95% of your sprites are 'background' and 5% are 'player', a model predicting 'background' always gets 95% accuracy but is useless! Look at metrics like **Precision**, **Recall**, and **F1-Score** in these cases.
- **Pixel Generation:** Accuracy doesn't apply. Besides loss (like MSE), **visual inspection** is key! You might also look at image quality metrics like **PSNR** (Peak Signal-to-Noise Ratio) or **SSIM** (Structural Similarity Index Measure), though these have their own limitations for perceptual quality.

Libraries like `scikit-learn` and `torchmetrics` can help calculate many of these advanced metrics.

## Summary

Calculating accuracy for sprite classification during evaluation involves getting model logits, converting them to predicted labels (via `sigmoid`+thresholding for binary, or `argmax` for multi-class), comparing to true labels, and averaging the correct count over the dataset. Remember to do this inside `torch.no_grad()` and after `model.eval()`. While accuracy is intuitive, consider other metrics or visual checks depending on your specific pixel art task!
