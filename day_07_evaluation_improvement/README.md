# Day 7: Evaluation and Model Improvement

**Topics:**

- Model Evaluation:
  - Switching to evaluation mode (`model.eval()`)
  - Using `torch.no_grad()` context.
  - Evaluating on a validation/test set.
- Performance Metrics: Calculating accuracy or other relevant metrics beyond loss.
- Diagnosing Training Issues:
  - Identifying underfitting vs. overfitting.
  - Recognizing optimization problems (e.g., learning rate issues).
  - Importance of checking data quality.
- Tuning Hyperparameters: Experimenting with learning rate, batch size, etc.
- Preventing Overfitting: Regularization techniques (brief mention), early stopping concept.
- Saving and Loading Models: Using `torch.save` and `model.load_state_dict`.

**Focus:** Assessing model generalization, understanding common training problems and how to address them, and practical skills like saving models.

## Key Resources

- **PyTorch Official Tutorials - Save and Load the Model:** [https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html](https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html) (Covers saving/loading `state_dict`, model architecture, inference)
- **`torch.no_grad` Documentation:** [https://pytorch.org/docs/stable/generated/torch.no_grad.html](https://pytorch.org/docs/stable/generated/torch.no_grad.html) (Essential for evaluation)
- **`model.eval()` Documentation:** [https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.eval](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.eval) (Setting the module to evaluation mode)
- **Saving & Loading Models Guide:** [https://pytorch.org/tutorials/beginner/saving_loading_models.html](https://pytorch.org/tutorials/beginner/saving_loading_models.html) (More detailed guide on different saving approaches)
- **TorchMetrics Library (Optional but recommended for metrics):** [https://torchmetrics.readthedocs.io/en/stable/](https://torchmetrics.readthedocs.io/en/stable/) (While not strictly core PyTorch, very useful for calculating accuracy, F1, etc.)

## Hands-On Examples

- **Setting up for Evaluation:** ([`01_setup_for_evaluation.py`](./01_setup_for_evaluation.py))
  - **Code Idea:** Assume you have a trained `model` from Day 6. Create a separate dummy validation dataset (`X_val`, `y_val`) and wrap it in a `Dataset` and `DataLoader` (`val_loader`). Keep the `criterion` (loss function) from training.
  - **Purpose:** Prepare the necessary components for evaluating the model on unseen data.
- **Implementing the Evaluation Loop:** ([`02_implementing_evaluation_loop.py`](./02_implementing_evaluation_loop.py))

  - **Code Idea:**

    ```python
    model.eval() # Set model to evaluation mode
    total_val_loss = 0.0
    correct_predictions = 0
    with torch.no_grad(): # Disable gradient calculations
        for batch_X, batch_y in val_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y) # Calculate loss
            total_val_loss += loss.item() * batch_X.size(0)

            # --- Add accuracy calculation (assuming classification) ---
            # Example for logits output and integer labels:
            # predicted_labels = outputs.argmax(dim=1)
            # correct_predictions += (predicted_labels == batch_y).sum().item()
            # ---------------------------------------------------------

    avg_val_loss = total_val_loss / len(val_loader.dataset)
    # accuracy = correct_predictions / len(val_loader.dataset) # If calculating accuracy
    print(f"Validation Loss: {avg_val_loss:.4f}")
    # print(f"Validation Accuracy: {accuracy:.4f}") # If calculating accuracy
    ```

  - **Purpose:** Demonstrate how to loop through validation data using `model.eval()` and `torch.no_grad()`, calculate validation loss, and optionally compute other metrics like accuracy. Highlight the key differences from the training loop (no optimizer steps, no backward pass).

- **Calculating Accuracy (Classification Example):** ([`03_calculating_accuracy.py`](./03_calculating_accuracy.py))
  - **Code Idea:** Integrate the accuracy calculation snippet (shown commented above) into the evaluation loop. This part depends on the model's output format (e.g., logits) and label format. Show how to get predicted labels (e.g., using `argmax`) and compare them to true labels.
  - **Purpose:** Provide a concrete example of calculating a common performance metric beyond loss.
- **Saving Model Parameters:** ([`04_saving_model_parameters.py`](./04_saving_model_parameters.py))
  - **Code Idea:** After training (or evaluation), save the model's learned parameters: `torch.save(model.state_dict(), 'my_model_weights.pth')`.
  - **Purpose:** Show the standard way to save the trained state of a model for later use.
- **Loading Model Parameters:** ([`05_loading_model_parameters.py`](./05_loading_model_parameters.py))
  - **Code Idea:**
    1. Create a _new_ instance of the same model architecture: `new_model = SimpleNet(...)`.
    2. Load the saved parameters: `new_model.load_state_dict(torch.load('my_model_weights.pth'))`.
    3. (Optional) Set `new_model.eval()` and run a prediction to show it works.
  - **Purpose:** Demonstrate how to load previously saved weights into a model instance, allowing reuse without retraining. Emphasize that the model architecture must match the saved state dict.
