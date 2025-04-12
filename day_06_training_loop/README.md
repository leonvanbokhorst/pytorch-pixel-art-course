# Day 6: Training Loop and Optimization Basics

**Topics:**

- The Training Loop Structure:
  1. Forward pass (`model(data)`)
  2. Loss computation (`criterion(outputs, labels)`)
  3. Backward pass (`loss.backward()`)
  4. Parameter update (`optimizer.step()`)
  5. Zero gradients (`optimizer.zero_grad()`)
- Loss Functions: Choosing appropriate loss (e.g., `nn.CrossEntropyLoss`, `nn.MSELoss`).
- Optimizers: Using `torch.optim` (e.g., `optim.SGD`, `optim.Adam`).
- Learning Rate: Understanding its role and impact.
- Epochs and Batches: Iterating through the dataset.
- Setting Model Mode: Using `model.train()`.
- Monitoring Training: Tracking loss to observe learning progress.

**Focus:** Implementing the core training cycle in PyTorch, connecting the model, data, loss, and optimizer.

## Key Resources

- **PyTorch Official Tutorials - Optimization Loop:** [https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html) (Covers the full training loop: forward pass, loss, backward pass, optimizer step, hyperparameters)
- **`torch.optim` Documentation:** [https://pytorch.org/docs/stable/optim.html](https://pytorch.org/docs/stable/optim.html) (Overview of different optimizers like SGD, Adam, etc., and how to use them)
- **Loss Functions (`torch.nn`) Documentation:** [https://pytorch.org/docs/stable/nn.html#loss-functions](https://pytorch.org/docs/stable/nn.html#loss-functions) (List of common loss functions like CrossEntropyLoss, MSELoss, BCEWithLogitsLoss)
- **`optimizer.zero_grad()` Explanation:** [https://pytorch.org/docs/stable/optim.html#torch.optim.Optimizer.zero_grad](https://pytorch.org/docs/stable/optim.html#torch.optim.Optimizer.zero_grad) (Why zeroing gradients is necessary)
- **`model.train()` vs `model.eval()`:** [https://pytorch.org/docs/stable/notes/serialization.html#saving-and-loading-models-for-inference](https://pytorch.org/docs/stable/notes/serialization.html#saving-and-loading-models-for-inference) (Briefly explains the importance of switching modes, though more relevant for Day 7/8)

## Hands-On Examples

- **Setup (Regression Task):** ([`01_setup_components_regression.py`](./01_setup_components_regression.py))
  - **Code Idea:** Set up components for a regression task (model, data, `nn.MSELoss`, optimizer).
  - **Purpose:** Prepare components for a regression training loop.
- **Implementing the Regression Training Loop:** ([`02_implementing_training_loop_regression.py`](./02_implementing_training_loop_regression.py))
  - **Code Idea:** Implement the standard 5-step training loop for the regression setup.
  - **Purpose:** Show the basic training loop for a regression problem.
- **Setup (Binary Classification Task):** ([`03_setup_components_binary_classification.py`](./03_setup_components_binary_classification.py))
  - **Code Idea:** Set up components for a binary classification task (model with 1 output, `nn.BCEWithLogitsLoss`).
  - **Purpose:** Prepare components for a binary classification training loop.
- **Implementing the Binary Classification Training Loop:** ([`04_implementing_training_loop_binary.py`](./04_implementing_training_loop_binary.py))
  - **Code Idea:** Implement the training loop for binary classification, including accuracy calculation.
  - **Purpose:** Show the training loop applied to a binary classification problem.
- **Visualizing Loss Curves (Matplotlib):** ([`05_visualizing_loss.py`](./05_visualizing_loss.py))
  - **Code Idea:** Modify a training loop to store epoch losses and plot them using `matplotlib`.
  - **Purpose:** Demonstrate how to plot training loss to visually assess convergence.
- **Experimenting with Learning Rate:** ([`06_experimenting_learning_rate.py`](./06_experimenting_learning_rate.py))
  - **Code Idea:** Rerun a training loop with different learning rates and compare/plot loss curves.
  - **Purpose:** Demonstrate the impact of the learning rate hyperparameter.
- **(Optional) Using a Different Optimizer:** ([`07_optional_different_optimizer.py`](./07_optional_different_optimizer.py))
  - **Code Idea:** Repeat setup with `optim.Adam` vs `optim.SGD` and compare/plot loss curves.
  - **Purpose:** Show how to easily swap optimizers.
- **(Optional) Logging with TensorBoard:** ([`08_optional_tensorboard.py`](./08_optional_tensorboard.py))
  - **Code Idea:** Use `torch.utils.tensorboard.SummaryWriter` to log training loss during the loop.
  - **Purpose:** Introduce basic TensorBoard logging for enhanced monitoring.
