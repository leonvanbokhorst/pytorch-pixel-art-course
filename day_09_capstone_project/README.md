# Day 9: Capstone Project - Multi-Task Learning Template

This directory contains a **template** for a capstone project designed to integrate many of the concepts learned throughout the PyTorch Fundamentals course (Days 1-8). It's set up for multi-task learning but can be adapted for single tasks as well.

## Project Overview

The goal of this project (`capstone_multitask.py`) is to provide a reusable structure for training a neural network on one or more tasks simultaneously using the same input data. The default example demonstrates:

1. **Regression:** Predict a continuous value based on the input.
2. **Binary Classification:** Predict a class label (0 or 1) based on the input.

This showcases **multi-task learning**, where a model leverages shared representations (learned in the model's "body") to perform well on multiple related tasks. You can easily adapt it to focus on a single task by removing the components related to the second task.

## Adaptability

This script is designed as a **template**. To use it for your own domain or problem, you will primarily need to modify the sections marked with `*** ADAPT THIS ... ***` comments in the `capstone_multitask.py` file:

1. **Configuration:** Adjust parameters like `INPUT_DIM`, `HIDDEN_DIM`, `NUM_CLASSES`, `LEARNING_RATE`, `NUM_EPOCHS`, etc., at the top of the script.
2. **`MultiTaskDataset`:**
    - **Crucially:** Modify the `__init__` method to load _your_ specific data (e.g., from CSV files, image folders) instead of generating synthetic data.
    - Ensure the `self.features` tensor has the correct shape based on your input.
    - Define how your specific target(s) (`self.target_regression`, `self.target_classification`) are loaded or calculated.
    - _If doing a single task_, remove the unused target tensor and references to it.
3. **`MultiTaskNet`:**
    - Adjust the `input_dim`, `hidden_dim`, and `num_classes` in the `__init__` method to match your data.
    - Modify the architecture (number of layers, layer sizes, activation functions) within the shared body or the task-specific heads as needed for your problem's complexity.
    - _If doing a single task_, remove the unused head (e.g., `self.regression_head`) and its usage in the `forward` method.
4. **Loss Functions & Weights:**
    - In the main execution block (`if __name__ == "__main__":`), select appropriate loss functions (`regression_criterion`, `classification_criterion`) for your task(s). PyTorch offers many options (e.g., `nn.CrossEntropyLoss` for multi-class classification).
    - Adjust the `loss_weights` tuple to control the influence of each task's loss on the final combined loss used for training.
5. **Evaluation (`evaluate_model` function):**
    - Keep the loss calculations if using the same criteria.
    - **Crucially:** Add or modify the calculation of relevant evaluation _metrics_ for your specific tasks (e.g., R-squared, F1-score, precision, recall). The current example only calculates binary accuracy.
    - Update the print statements to report your chosen metrics.
6. **Model Saving:**
    - In the main loop, change the condition for saving the best model to use the most relevant validation metric for your primary task (e.g., `val_reg_loss` if focusing on regression).

## Components (Default Example)

- **Synthetic Data (`MultiTaskDataset`):** Generates random inputs and derives regression/classification targets.
- **Multi-Task Model (`MultiTaskNet`):** An `nn.Module` with a shared body and two separate heads (regression and classification).
- **Combined Loss:** Uses `nn.MSELoss` and `nn.BCEWithLogitsLoss`, combined with weights.
- **Training Loop (`train_epoch`):** Standard loop performing backpropagation on the combined loss.
- **Evaluation Loop (`evaluate_model`):** Calculates MSE and binary accuracy on validation data.

## How to Run (After Adapting)

1. **Ensure Prerequisites:** Activate your Python virtual environment with PyTorch installed.
2. **Navigate:** `cd` to the root of the `pytorch-course-base` project.
3. **Execute:** Run the adapted script:

    ```bash
    python day_09_capstone_project/capstone_multitask.py
    ```
