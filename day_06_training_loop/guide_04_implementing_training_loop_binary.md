# Guide: 04 The Sprite Classification Loop!

We have the ingredients prepped for classifying sprites (Model outputting 1 logit, Dataset with float labels, `BCEWithLogitsLoss`). Now, let's run the training loop and teach the model to tell players from enemies (or circles from squares!), based on `04_implementing_training_loop_binary.py`.

**Core Concept:** The 5-step training loop (`zero_grad` -> `forward` -> `loss` -> `backward` -> `step`) works exactly the same way! The main difference is _how_ we interpret the model's output and often, we add an extra step inside the loop to track **accuracy** (how many sprites did it classify correctly?).

## The Classification Loop Structure (Recap + Accuracy)

```python
# Conceptual Structure:
num_epochs = ...
classifier_model = ...
train_loader = ... # DataLoader yields (sprite_batch, label_batch)
criterion = nn.BCEWithLogitsLoss() # Our binary classification judge
optimizer = ...
device = ...

classifier_model.to(device)

for epoch in range(num_epochs):
    classifier_model.train() # Set to training mode
    epoch_loss = 0.0
    epoch_correct_predictions = 0

    for sprite_batch, label_batch in train_loader:
        # Move data to device
        sprite_batch = sprite_batch.to(device)
        label_batch = label_batch.to(device)

        # 1. Zero Gradients
        optimizer.zero_grad()

        # 2. Forward Pass -> Get Raw Logits
        logits = classifier_model(sprite_batch)

        # 3. Calculate Loss (Using BCEWithLogitsLoss)
        loss = criterion(logits, label_batch)

        # 4. Backward Pass
        loss.backward()

        # 5. Update Parameters
        optimizer.step()

        # --- (Optional but Recommended) Calculate Accuracy --- #
        # a) Convert logits to probabilities (0 to 1)
        predicted_probs = torch.sigmoid(logits)
        # b) Convert probabilities to predicted labels (0 or 1)
        predicted_labels = (predicted_probs >= 0.5).float()
        # c) Count how many predictions match the true labels
        correct_in_batch = (predicted_labels == label_batch).sum().item()
        epoch_correct_predictions += correct_in_batch
        # ---------------------------------------------------- #

        epoch_loss += loss.item()

    # --- Epoch Finished --- #
    avg_epoch_loss = epoch_loss / len(train_loader)
    epoch_accuracy = epoch_correct_predictions / len(train_loader.dataset)
    print(f"Epoch {epoch+1} Avg Loss: {avg_epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

```

## Key Differences & Steps for Classification Loop

1.  **Forward Pass Output (`logits`):** Remember, `classifier_model(sprite_batch)` returns the raw _logits_ (shape `[batch_size, 1]`), not probabilities.
2.  **Loss Calculation:** `loss = criterion(logits, label_batch)` works perfectly because `nn.BCEWithLogitsLoss` _expects_ raw logits and float `0.0`/`1.0` labels.
3.  **Accuracy Calculation (The Extra Bit):** Loss tells us the average error, but accuracy gives a more intuitive feel for performance (% correct). To get this from logits:
    - **Logits to Probabilities:** Apply `torch.sigmoid(logits)`. This squeezes the raw scores into the [0, 1] probability range.
    - **Probabilities to Labels:** Decide on a threshold (usually 0.5). If `predicted_probs >= 0.5`, predict class 1, otherwise predict class 0. The `.float()` converts the resulting True/False boolean into `1.0`/`0.0`.
    - **Compare & Count:** Check how many `predicted_labels` match the `label_batch` using `(predicted_labels == label_batch)`. The `.sum().item()` counts the number of `True` matches in the batch.
    - **Accumulate:** Add `correct_in_batch` to `epoch_correct_predictions`.
    - **Calculate Epoch Accuracy:** After looping through all batches, divide `epoch_correct_predictions` by the total number of sprites in the training set (`len(train_loader.dataset)`).

## Training vs. Validation Accuracy

It's good to track accuracy on the training set during the loop to see if the model is learning _something_. However, don't be fooled if training accuracy gets very high! The _real_ test is the accuracy on the **validation set** (using `model.eval()` and `torch.no_grad()`, see Day 7), as that shows if the model can classify sprites it hasn't seen before.

## Summary

The sprite classification training loop uses the same 5 core steps. The key differences are using `nn.BCEWithLogitsLoss` (which takes raw logits) and often adding steps _within_ the loop to calculate training accuracy by converting logits -> probabilities -> predicted labels and comparing with true labels.
