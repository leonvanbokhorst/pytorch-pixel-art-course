# Guide: 05 Charting Your Pixel Progress: Visualizing Loss Curves!

Staring at loss numbers printing out is okay, but seeing them drawn as a **curve**? That's how you _really_ see if your pixel model is learning the magic or just stumbling around! This guide shows how to plot the training loss using `matplotlib`, based on `05_visualizing_loss.py`.

**Core Concept:** A picture is worth a thousand loss numbers! Plotting the average training loss after each epoch creates a **loss curve**. This curve visually tells you the story of your model's training journey: Is it steadily improving? Getting stuck? Going haywire?

## Our Charting Tool: `matplotlib`

We'll use `matplotlib`, the trusty paintbrush of Python data visualization. Specifically, its `pyplot` module (usually imported as `plt`).

_(Side Quest: If you don't have it, you might need to install it: `pip install matplotlib` or `uv pip install matplotlib`)_

## Steps to Draw Your Learning Curve

It's pretty straightforward to add plotting to your training script:

1.  **Import the Brush:** Add `import matplotlib.pyplot as plt` at the start.
2.  **Prepare the Canvas:** Before your main epoch loop, create an empty list to hold the loss values: `epoch_average_losses = []`.
3.  **Record the Progress:** Inside the epoch loop, _after_ you calculate the average loss for the epoch (like `avg_epoch_loss` from the previous guide), add it to your list: `epoch_average_losses.append(avg_epoch_loss)`.
4.  **Paint the Picture (After Training):** Once all epochs are finished:
    - Create a figure: `plt.figure()`.
    - Plot the data: `plt.plot(range(1, num_epochs + 1), epoch_average_losses)`. This puts epoch number (1, 2, 3...) on the x-axis and your recorded losses on the y-axis.
    - Add labels and title: `plt.xlabel("Epoch")`, `plt.ylabel("Average Training Loss")`, `plt.title("Pixel Model Learning Curve")`.
    - Make it pretty (optional): Add markers (`marker='o'`), grid (`plt.grid(True)`).
    - Show or Save: Use `plt.show()` to display the plot, or `plt.savefig("pixel_loss.png")` to save it as an image file.

## Code Example Walkthrough

The script follows these exact steps:

```python
# Spell Snippets:
import matplotlib.pyplot as plt # Step 1
# ... (Other imports and setup) ...

num_epochs = ...
epoch_average_losses = [] # Step 2: Prepare the list

# --- The Training Loop --- #
for epoch in range(num_epochs):
    # ... (Inner loop calculating avg_epoch_loss) ...
    # Example from previous guide:
    # avg_epoch_loss = epoch_loss / len(train_loader)

    epoch_average_losses.append(avg_epoch_loss) # Step 3: Record loss
    print(f"Epoch {epoch+1} Avg Loss: {avg_epoch_loss:.4f}")

# --- Plotting Time (After the loop finishes) --- # (Step 4)
print("\nTraining finished! Plotting loss curve...")
plt.figure(figsize=(10, 5)) # Create a plot figure (optional size)
plt.plot(
    range(1, num_epochs + 1), # X-axis: Epoch numbers (1 to num_epochs)
    epoch_average_losses,     # Y-axis: Recorded average losses
    marker="o",               # Add circles at each data point
    linestyle="-"             # Connect points with a solid line
)
plt.title("Pixel Model Training Loss") # Chart title
plt.xlabel("Epoch")                   # X-axis label
plt.ylabel("Average Loss")            # Y-axis label
plt.grid(True)                        # Add a grid for readability

# Optional: Adjust x-axis ticks if many epochs
# plt.xticks(range(1, num_epochs + 1, max(1, num_epochs // 10)))

plt.show() # Display the plot
# Or plt.savefig("pixel_training_loss.png") # Save to a file
```

## Reading the Tea Leaves (Interpreting the Curve)

- 📉 **Steadily Downhill:** Awesome! Your model is learning. Loss is decreasing.
- <0xE2><0x80><0x94> **Flat Plateau:** Hmm. Learning has stalled. Maybe the learning rate is too small? Model too simple? Or it has learned all it can from the data?
- 📈 / <0xE2><0x9A><0xA1>️ **Uphill or Crazy Jumps:** Danger! Training might be unstable. Is the learning rate WAY too high? Are there weird values (NaNs) happening? Problems with the data?

## Level Up: Advanced Visualization Tools

`matplotlib` is great for a quick look after training. For more power (real-time plots, comparing different training runs, tracking accuracy too), check out tools like:

- **TensorBoard:** Integrates directly with PyTorch (`torch.utils.tensorboard`). See `08_optional_tensorboard.py` for a taste.
- **Weights & Biases (Wandb), MLflow:** External experiment tracking platforms.

## Summary

Plotting your training loss is like having a progress bar for your model's learning! Just save the average loss from each epoch into a list, and use `matplotlib.pyplot.plot` after the training loop finishes to create an informative curve. It's a simple but powerful way to see if your pixel model is truly learning the magic!
