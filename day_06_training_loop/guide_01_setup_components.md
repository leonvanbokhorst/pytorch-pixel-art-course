# Guide: 01 Setup Components for Training

This guide outlines the essential components that need to be prepared _before_ implementing the main PyTorch training loop, as demonstrated in `01_setup_components.py`.

**Core Concept:** Training a neural network requires several key ingredients working together. Before iterating through data and updating weights, you need to define your model architecture, prepare your data pipeline, choose how to measure error (loss), and select how to update the model based on that error (optimizer).

## The 5 Core Components

Think of setting up for training like preparing for cooking (_mise en place_):

1. **The Model (`nn.Module`):**

    - **What:** Your neural network architecture, defined as a class inheriting from `nn.Module`.
    - **Action:** Instantiate your model class (e.g., `model = SimpleRegressionNet(...)`). This creates the network object with its initial (usually random) weights and biases.

2. **The Data (`Dataset` / `DataLoader`):**

    - **What:** Your training data, accessible through a standard interface.
    - **Action:**
      - Create or load your raw data.
      - Wrap it in a `torch.utils.data.Dataset` instance (either a built-in one or your custom class).
      - Wrap the `Dataset` instance in a `torch.utils.data.DataLoader` instance to handle batching and shuffling efficiently.

3. **The Loss Function (Criterion):**

    - **What:** A way to measure how wrong the model's predictions are compared to the true target values.
    - **Action:** Instantiate a loss function class from `torch.nn`. The choice depends on the task:
      - Regression (predicting continuous values): `nn.MSELoss()` (Mean Squared Error), `nn.L1Loss()` (Mean Absolute Error).
      - Binary Classification (two classes): `nn.BCEWithLogitsLoss()` (Binary Cross Entropy - recommended as it combines Sigmoid + BCE for numerical stability), `nn.BCELoss()` (requires Sigmoid applied to model output first).
      - Multi-class Classification (more than two classes): `nn.CrossEntropyLoss()` (recommended as it combines LogSoftmax + NLLLoss).
    - Store the instance (e.g., `criterion = nn.MSELoss()`).

4. **The Optimizer (`torch.optim`):**

    - **What:** The algorithm used to update the model's parameters (weights and biases) based on the gradients computed during backpropagation.
    - **Action:** Instantiate an optimizer class from `torch.optim`, passing it the model's parameters and a learning rate (`lr`):
      - `optimizer = optim.SGD(model.parameters(), lr=0.01)`
      - `optimizer = optim.Adam(model.parameters(), lr=0.001)`
    - **`model.parameters()`:** This is crucial! It tells the optimizer which tensors it needs to manage and update.
    - **`lr` (Learning Rate):** A hyperparameter controlling the step size of the parameter updates.

5. **The Device (CPU/GPU) (Conceptual):**
    - **What:** Where the computations will happen (CPU or a specific GPU).
    - **Action:** Typically defined early (e.g., `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`). The `model` needs to be moved to this device (`model.to(device)`), and data batches also need to be moved to the device inside the training loop (`features.to(device)`, `labels.to(device)`).

## Script Example Walkthrough

The script `01_setup_components.py` performs these steps for a simple regression task:

- **Model:** Defines and instantiates `SimpleRegressionNet`.
- **Data:** Generates synthetic `X_data`, `y_data`; creates `RegressionDataset`; creates `train_loader` (a `DataLoader`).
- **Loss:** Instantiates `criterion = nn.MSELoss()`.
- **Optimizer:** Instantiates `optimizer = optim.SGD(model.parameters(), lr=0.01)`.

## Ready to Train

Once these components (Model, DataLoader, Criterion, Optimizer, and Device awareness) are set up, you have everything you need to implement the core training loop, which iteratively processes data batches and updates the model.

## Summary

Setting up for PyTorch training involves instantiating five key components: your `nn.Module` model, a `DataLoader` wrapping your `Dataset`, an appropriate loss function (criterion), an optimizer from `torch.optim` linked to your model's parameters, and defining the target computation device. With these pieces in place, you're ready to write the training loop.
