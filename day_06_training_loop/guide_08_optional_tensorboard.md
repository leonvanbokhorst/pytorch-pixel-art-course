# Guide: 08 (Optional) Logging with TensorBoard

This guide introduces basic usage of TensorBoard for logging and visualizing training metrics within PyTorch using `torch.utils.tensorboard.SummaryWriter`, as demonstrated in `08_optional_tensorboard.py`.

**Core Concept:** While simple `print` statements or `matplotlib` plots are useful, dedicated experiment tracking tools like TensorBoard provide a much richer, interactive way to monitor training progress, compare different runs, and visualize various aspects of your models and data.

## What is TensorBoard?

TensorBoard is a visualization toolkit originally developed for TensorFlow, but widely adopted and integrated with PyTorch. It provides a web-based dashboard where you can visualize:

- Scalar metrics over time (loss, accuracy, learning rate)
- Histograms of weights, biases, or activations
- Model graph structure (visualizing your `nn.Module`)
- Images, text, audio data
- Embedding projections

## Using `SummaryWriter` in PyTorch

PyTorch integrates with TensorBoard through the `torch.utils.tensorboard.SummaryWriter` class.

### Steps for Basic Scalar Logging

1. **Import:**

    ```python
    from torch.utils.tensorboard import SummaryWriter
    ```

2. **Initialize Writer:** Before the training loop, create an instance of `SummaryWriter`. You typically specify a `log_dir` where TensorBoard event files will be saved. It's good practice to create subdirectories within your `log_dir` (often called `runs`) for different experiments.

    ```python
    # Example: Create a unique log directory for this run
    import os
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join("runs", f"day6_demo_{timestamp}")
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs will be saved in: {log_dir}")
    ```

3. **Log Values:** Inside your training loop (commonly at the end of each epoch, but can also be per-batch), call methods on the `writer` object. For logging scalar values like loss or accuracy, use `add_scalar`:

    ```python
    # Inside the epoch loop, after calculating avg_epoch_loss:
    # writer.add_scalar(tag, scalar_value, global_step)
    writer.add_scalar("Loss/train", avg_epoch_loss, epoch + 1)
    # You can log other scalars too:
    # writer.add_scalar("Accuracy/train", epoch_accuracy, epoch + 1)
    # writer.add_scalar("LearningRate", current_lr, epoch + 1)
    ```

    - `tag`: A string identifier for the plot (e.g., `Loss/train`). The `/` creates hierarchical groupings in the TensorBoard UI.
    - `scalar_value`: The Python number (or 0-dim tensor) to log.
    - `global_step`: The value for the x-axis, typically the epoch number or the total number of batches processed.
4. **Close Writer:** After the training loop finishes, it's crucial to close the writer to ensure all pending data is flushed to the log file.

    ```python
    writer.close()
    ```

## Launching and Viewing TensorBoard

1. **Open Terminal:** Navigate your terminal to the directory _containing_ your `log_dir` (e.g., if your logs are in `runs/day6_demo_...`, `cd` to the directory containing the `runs` folder).
2. **Run Command:** Execute `tensorboard --logdir runs` (replace `runs` with your top-level log directory if different).
3. **Open Browser:** TensorBoard will print a URL (usually `http://localhost:6006/` or similar). Open this URL in your web browser.

## Benefits of TensorBoard

- **Interactive Visualization:** Zoom, pan, smooth plots.
- **Experiment Comparison:** Launch TensorBoard pointing to a directory containing multiple run logs (`tensorboard --logdir runs`) to overlay plots from different experiments (e.g., different learning rates, optimizers, architectures).
- **Rich Data Types:** Visualize more than just scalars.
- **Standard Tool:** Widely used in the ML community.

## Summary

`torch.utils.tensorboard.SummaryWriter` provides a simple interface for logging training metrics from PyTorch scripts. By adding `writer.add_scalar()` calls within your training loop and launching the `tensorboard` command-line tool pointing to the log directory, you can access an interactive dashboard to monitor loss curves, compare runs, and gain deeper insights into your model's training process compared to static plots or print statements.
