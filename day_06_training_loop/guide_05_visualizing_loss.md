# Guide: 05 Visualizing Loss Curves (Matplotlib)

This guide demonstrates a simple way to monitor training progress by plotting the loss curve using the `matplotlib` library, as shown in `05_visualizing_loss.py`.

**Core Concept:** While printing the loss value at the end of each epoch gives you numerical feedback, visualizing the loss over time as a curve provides much richer insight into the training dynamics. You can quickly see if the model is learning (loss decreasing), stalling (loss plateauing), or becoming unstable (loss increasing or jumping erratically).

## Tool: `matplotlib`

`matplotlib` is the most widely used data visualization library in Python. We use its `pyplot` module (commonly imported as `plt`) for creating simple plots.

_Note: `matplotlib` is not a core PyTorch dependency, so you might need to install it separately (`pip install matplotlib` or `uv pip install matplotlib`)._

## Steps for Plotting Epoch Loss

Adding basic loss plotting to your training script involves these steps:

1. **Import:** Add `import matplotlib.pyplot as plt` at the top of your script.
2. **Initialize List:** Before starting the main epoch loop, create an empty list to store the average loss from each epoch (e.g., `epoch_losses = []`).
3. **Record Loss:** Inside the epoch loop, _after_ calculating the average loss for the completed epoch (`avg_epoch_loss`), append this value to your list: `epoch_losses.append(avg_epoch_loss)`.
4. **Plot After Training:** Once the entire training loop (all epochs) is finished:
    _Call `plt.figure()` to create a plotting area (optional, but good practice).
    _ Call `plt.plot(range(1, num_epochs + 1), epoch_losses)` to plot epoch number (starting from 1) on the x-axis and the recorded average losses on the y-axis.
    You can add markers and line styles (e.g., `marker='o', linestyle='-'`).
    _Add labels and a title for clarity using `plt.xlabel("Epoch")`, `plt.ylabel("Average Loss")`, `plt.title("Training Loss Curve")`.
    _ Optionally add a grid with `plt.grid(True)`. \* Finally, display the plot using `plt.show()` or save it to a file using `plt.savefig("filename.png")`.

## Code Example Walkthrough

The script implements exactly these steps:

```python
# Script Snippets:
import matplotlib.pyplot as plt # Step 1

# ... (Setup components) ...

num_epochs = 30
epoch_losses = [] # Step 2

for epoch in range(num_epochs):
    # ... (Inner training loop to calculate avg_loss) ...
    avg_loss = running_loss / num_batches
    epoch_losses.append(avg_loss) # Step 3
    print(f"Epoch {epoch+1}/{num_epochs} completed. Average Loss: {avg_loss:.4f}")

# --- Plotting the Loss Curve --- # (Step 4)
plt.figure(figsize=(10, 5))
plt.plot(
    range(1, num_epochs + 1), epoch_losses, marker="o", linestyle="-"
)
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Average Loss")
plt.grid(True)
plt.xticks(range(1, num_epochs + 1, max(1, num_epochs // 10)))
# plt.savefig("training_loss_curve.png") # Optional save
plt.show()
```

## Interpreting the Loss Curve

- **Steadily Decreasing Loss:** Generally indicates successful learning. The model is getting better at minimizing the error on the training data.
- **Plateauing Loss:** The loss stops decreasing significantly. This might mean:
  - The model has converged.
  - The learning rate is too small.
  - The model architecture isn't complex enough.
  - The data doesn't contain enough signal.
- **Increasing/Jumpy Loss:** Often indicates instability, possibly due to:
  - A learning rate that is too high.
  - Numerical issues.
  - Problems with the data or model architecture.

## Beyond Basic Plots

While `matplotlib` is great for simple end-of-training plots, tools like **TensorBoard** (integrated with PyTorch via `torch.utils.tensorboard`) or other experiment tracking libraries (Weights & Biases, MLflow) offer more advanced features like real-time plotting, comparison of multiple runs, and logging other metrics (like accuracy, learning rate changes). See script `08_optional_tensorboard.py` for a basic TensorBoard example.

## Summary

Visualizing the training loss curve provides valuable insight into the learning process. By simply recording the average loss per epoch in a list and using `matplotlib.pyplot.plot` after training, you can generate informative plots to help diagnose training issues or confirm convergence.
