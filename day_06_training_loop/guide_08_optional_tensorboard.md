# Guide: 08 (Optional) The Crystal Ball: Logging with TensorBoard!

Want a fancier way to watch your pixel model learn than just printing loss numbers or making a basic plot at the end? **TensorBoard** is like a crystal ball 🔮 for your training runs! This guide introduces basic logging with `torch.utils.tensorboard.SummaryWriter`, as seen in `08_optional_tensorboard.py`.

**Core Concept:** While `print()` and `matplotlib` are okay, tools like TensorBoard give you an interactive web dashboard to monitor training live, compare different experiments easily, and visualize more than just loss curves (like accuracy, learning rate, or even generated sprite examples!).

## What's This TensorBoard Magic?

It's a separate tool (originally from TensorFlow, but works great with PyTorch) that reads special log files created by your script and displays them in a nice web interface. You can see:

- How loss and accuracy change over epochs (or even batches!).
- How model weights or gradients are distributed (histograms).
- Visualizations of your model's structure.
- Samples of images being generated or processed!

## Using PyTorch's `SummaryWriter` Spell

PyTorch talks to TensorBoard using the `SummaryWriter` class.

### Steps for Basic Loss/Accuracy Logging

1.  **Import the Scribe:**

    ```python
    from torch.utils.tensorboard import SummaryWriter
    ```

2.  **Summon the Scribe (`SummaryWriter`):** Before your training loop, create a `SummaryWriter`. Tell it where to save the log files (the `log_dir`). It's good practice to put logs for different runs in separate subfolders (often under a main `runs/` directory) perhaps named with a timestamp.

    ```python
    # Spell Snippet:
    import os
    from datetime import datetime

    # Create a unique directory name for this specific training run
    run_name = f"pixel_gen_lr0.001_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = os.path.join("runs", run_name) # e.g., runs/pixel_gen_lr0.001_20231027_103000

    # Summon the writer, telling it where to save logs
    writer = SummaryWriter(log_dir=log_dir)
    print(f"✨ TensorBoard logs will be saved in: {log_dir}")
    ```

3.  **Record the Events (`add_scalar`):** Inside your training loop (usually at the end of each epoch), tell the `writer` what to record using methods like `add_scalar` for single number metrics (loss, accuracy).

    ```python
    # Spell Snippet (Inside Epoch Loop, after calculating avg loss & accuracy):
    # writer.add_scalar("Chart Title/Metric Name", value_to_log, current_epoch_number)

    # Log average training loss for this epoch
    writer.add_scalar("Loss/Train", avg_epoch_loss, epoch + 1)

    # Log training accuracy for this epoch (if calculated)
    # writer.add_scalar("Accuracy/Train", epoch_accuracy, epoch + 1)

    # You could also log the learning rate if it changes
    # current_lr = optimizer.param_groups[0]['lr']
    # writer.add_scalar("LearningRate", current_lr, epoch + 1)
    ```

    - `tag`: The name for the chart. Using `/` like `Loss/Train` helps organize charts in TensorBoard.
    - `scalar_value`: The Python number you want to plot.
    - `global_step`: The x-axis value, typically the epoch number (starting from 1 is common).

4.  **Dismiss the Scribe (`close`):** After your _entire_ training process finishes, **it's important to `close()` the writer**. This ensures all the information is actually written to the log file.

    ```python
    # Spell Snippet (After the main training loop):
    writer.close()
    print("TensorBoard writer closed.")
    ```

## Gazing into the Crystal Ball (Launching TensorBoard)

1.  **Open Terminal/Command Prompt:** Navigate to the directory that _contains_ your `runs` folder (the parent directory of your `log_dir`).
2.  **Cast the Command:** Type `tensorboard --logdir runs` (or your top-level log directory name).
3.  **Open Your Browser:** TensorBoard will start a local web server and print a URL (like `http://localhost:6006/`). Paste this into your browser.

Now you can explore the interactive charts for your training run(s)!

## Why TensorBoard is Awesome for Pixel Training

- **Live Monitoring:** See the loss curve update while training!
- **Comparing Pixel Experiments:** Run TensorBoard on the main `runs` directory to see plots from different learning rates or model architectures overlaid on the same chart!
- **Visualize Images:** Use `writer.add_image()` or `writer.add_images()` inside your loop to save samples of generated sprites at different epochs and view them directly in TensorBoard!

## Summary

Using `torch.utils.tensorboard.SummaryWriter` lets you easily log metrics like loss and accuracy during training. Initialize the `writer` pointing to a log directory, use `writer.add_scalar("Tag", value, epoch)` inside your loop to record data, and `writer.close()` at the end. Launch TensorBoard using `tensorboard --logdir runs` in your terminal to view the interactive dashboard. It's a fantastic tool for monitoring, comparing, and debugging your pixel model training!
