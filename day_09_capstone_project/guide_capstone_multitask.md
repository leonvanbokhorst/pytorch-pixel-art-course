# Guide: Capstone Project - Multi-Task Learning Template

This guide explores the `capstone_multitask.py` script, which serves as a capstone project template, integrating concepts from Days 1-8 of the PyTorch Fundamentals course. It demonstrates a multi-task learning setup but is designed for easy adaptation.

**Core Concept:** This capstone provides a structured template for a complete PyTorch project, including configuration, data loading, model definition, training loop, evaluation loop, TensorBoard logging, and model saving. It uses a multi-task learning (MTL) approach as an example, where a single model learns to perform multiple tasks simultaneously.

## Project Structure Overview

The `capstone_multitask.py` script organizes the project into several key parts:

1.  **Device Setup:** Detects and selects the appropriate compute device (CUDA/MPS/CPU).
2.  **Configuration:** Defines key parameters (dimensions, learning rate, epochs, paths) at the top for easy modification.
3.  **`MultiTaskDataset`:** A custom `Dataset` class (adaptable for user data).
4.  **`MultiTaskNet`:** An `nn.Module` defining the model architecture (adaptable).
5.  **`train_epoch` Function:** Encapsulates the logic for one training epoch.
6.  **`evaluate_model` Function:** Encapsulates the logic for evaluating the model.
7.  **Main Execution Block (`if __name__ == '__main__':`)**: Orchestrates the setup, training, evaluation, logging, and saving process.

## Multi-Task Learning (MTL) Example

The default example implements MTL:

- **Goal:** Train one model to perform both regression and binary classification using the same input features.
- **Model Architecture:** The `MultiTaskNet` has a _shared body_ (initial layers processing the input) and then splits into two _task-specific heads_ (separate final layers) - one for regression output, one for classification logits.
  The idea is that the shared body learns representations useful for both tasks.
- **Combined Loss:** The script calculates the loss for each task separately (`regression_criterion`, `classification_criterion`). It then combines these into a single `combined_loss` using `loss_weights`. This combined loss is what `autograd` uses for the `backward()` pass, updating parameters in the shared body based on errors from _both_ tasks.

```python
# Snippet from train_epoch:
loss_reg = regression_criterion(output_reg, target_reg)
loss_cls = classification_criterion(output_cls, target_cls)

reg_weight, cls_weight = loss_weights
combined_loss = (reg_weight * loss_reg) + (cls_weight * loss_cls)

combined_loss.backward() # Gradients based on combined error
optimizer.step()
```

## Key Components and Adaptability

This script is designed as a **TEMPLATE**. Look for comments like `*** ADAPT THIS ... ***` to guide your modifications:

- **Configuration:** Change `INPUT_DIM`, `HIDDEN_DIM`, `NUM_CLASSES`, `LEARNING_RATE`, `NUM_EPOCHS`, `SAVE_MODEL_PATH`, etc., to match your project.
- **`MultiTaskDataset` (`__init__`)**: **Crucially**, replace the synthetic data generation with code to load **your data** (e.g., read CSVs, load images). Ensure `self.features` and your target tensor(s) (`self.target_regression`, `self.target_classification`) are correctly populated. If doing a single task, remove the unused target.
- **`MultiTaskNet` (`__init__`, `forward`)**: Adjust layer dimensions (`input_dim`, `hidden_dim`, head output sizes) based on your data. Modify the body/head architectures (add/remove layers, change activations) as needed. If doing a single task, remove the unused head layer and its corresponding output from the `forward` method.
- **Loss Functions & Weights (Main Block):** Select the appropriate `criterion` instances for your task(s) (e.g., `nn.CrossEntropyLoss` for multi-class). If using MTL, adjust `loss_weights` to balance task importance.
- **`evaluate_model` (Metrics):** Modify the metric calculations inside this function to compute metrics relevant to _your_ specific task(s) (e.g., R-squared, F1-score, Precision/Recall). Update the final print statements.
- **Model Saving (Main Block):** Change the condition for saving the `best_model` to monitor the most important validation metric for your primary task (e.g., `val_reg_loss`, `val_accuracy`).

## Running the Capstone

1.  Adapt the script sections as described above for your specific data and task.
2.  Ensure your environment with PyTorch is active.
3.  Run from the terminal: `python day_09_capstone_project/capstone_multitask.py`
4.  Monitor progress via console output and TensorBoard (`tensorboard --logdir runs`).

## Summary

The Day 9 Capstone script provides a comprehensive, adaptable template integrating the concepts learned throughout the course. It demonstrates a multi-task learning setup with a shared model body and task-specific heads, combined loss calculation, separate training/evaluation functions, TensorBoard logging, and best-model saving. By modifying the indicated sections, you can use this structure as a robust starting point for your own PyTorch projects, whether single-task or multi-task.
