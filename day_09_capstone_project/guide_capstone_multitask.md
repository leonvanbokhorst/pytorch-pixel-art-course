# Guide: Capstone Pixel Project - Your Grand Creation!

This is it! The final frontier! This guide explores the `capstone_multitask.py` script – not just code, but a **launchpad** for your _own_ awesome Pixel Art + PyTorch project! It combines everything from Days 1-8.

**Core Concept:** Forget generic examples! This capstone template gives you a complete, structured PyTorch project skeleton (config, data loading, model, training, evaluation, logging, saving) that you can **adapt** to bring your _own_ pixel art ideas to life. Want to build a sprite classifier? A pixel art generator? A style transfer tool? This is your starting point!

## Project Structure: The Magic Components

The `capstone_multitask.py` template organizes the key parts:

1.  **Device Setup:** Automatically picks GPU (CUDA/MPS) or CPU.
2.  **Configuration:** All the important settings (image size, learning rate, epochs, save paths) are at the top – easy to tweak!
3.  **Pixel Art Dataset:** A placeholder `MultiTaskDataset` class you **MUST** adapt to load _your_ sprites and labels.
4.  **Pixel Art Model:** A placeholder `MultiTaskNet` (`nn.Module`) you **MUST** adapt to _your_ desired architecture (e.g., using `nn.Conv2d` for images!).
5.  **`train_epoch` Function:** The training loop logic, neatly packaged.
6.  **`evaluate_model` Function:** The evaluation loop logic, also packaged.
7.  **Main Block (`if __name__ == '__main__':`)**: The conductor that brings everything together – sets up, trains, evaluates, logs to TensorBoard, and saves the best model!

## Why Multi-Task? (It's Just an Example!)

The template shows a **Multi-Task Learning (MTL)** setup (classifying AND predicting a value). Why?

- **Demonstration:** It shows how one model can learn related things (e.g., classify a sprite _and_ estimate its main color).
- **Shared Learning:** The model has a _shared body_ (early layers) learning general pixel features, and _separate heads_ (later layers) specializing in each task. This can sometimes be more efficient than training two separate models.
- **Easy Adaptation:** You can **easily remove** the multi-task parts to make it a **single-task** project (like just classification, or just generation).

**Don't feel obligated to do multi-tasking! Focus on adapting this template to _your_ pixel art idea.**

## Your Quest: Adapting the Template!

This is where YOU become the pixel wizard! Look for the comments `*** ADAPT THIS ... ***` in `capstone_multitask.py` – these are your main modification points:

- **Configuration:** Set your desired `IMAGE_HEIGHT`, `IMAGE_WIDTH`, `CHANNELS` (1 for grayscale, 3 for RGB), `NUM_CLASSES` (if classifying), `LEARNING_RATE`, `NUM_EPOCHS`, `SAVE_MODEL_PATH`, etc.
- **Dataset Class (`MultiTaskDataset` -> Your `PixelArtDataset`):**
  - **MOST IMPORTANT:** Replace the fake data generation in `__init__` with code to load **YOUR pixel art**! Load PNGs, sprite sheets, whatever you have. Use libraries like `PIL` (Pillow) or `torchvision.io`.
  - Make `__getitem__` return your processed `sprite_tensor` and its corresponding `target` (e.g., class label for classification, or maybe the sprite itself for generation tasks like autoencoders). Remember transforms!
  - _Single Task?_ Remove the code related to the second, unused target.
- **Model Class (`MultiTaskNet` -> Your `PixelArtModel`):**
  - Rename the class!
  - Adjust layer dimensions in `__init__` to match your sprite size and task. **Consider using `nn.Conv2d` layers** in the body for image data – they are much better at understanding spatial patterns than `nn.Linear` alone!
  - Design the head(s) for your specific output (e.g., one output neuron for binary classification, `NUM_CLASSES` neurons for multi-class, `IMAGE_HEIGHT * IMAGE_WIDTH * CHANNELS` neurons for generation).
  - Modify the `forward` method to define the data flow through your chosen layers.
  - _Single Task?_ Remove the unused head layer and its output from `forward`.
- **Loss Function(s) & Weights (Main Block):**
  - Choose the `criterion` that fits your pixel task (e.g., `nn.CrossEntropyLoss` for multi-class classification, `nn.MSELoss` for generation).
  - _Single Task?_ Remove the second criterion and the `loss_weights`. Just calculate and use `loss = criterion(outputs, targets)`.
- **Evaluation Metrics (`evaluate_model`):**
  - **CRUCIAL:** Add calculations for metrics relevant to YOUR task! For classification, calculate accuracy, precision, recall, F1. For generation, you might log the validation loss (MSE), but **visual inspection** of generated samples (maybe saved using `torchvision.utils.save_image` or logged to TensorBoard with `writer.add_images`) is often the best evaluation!
  - Update the print statements to show your relevant metrics.
- **Model Saving (Main Block):**
  - Change the `if best_metric > current_best:` logic to monitor _your_ most important validation metric (e.g., `val_accuracy`, or maybe just lowest `val_loss` for generation) to decide when to save the `best_model.pth`.

## Launching Your Creation!

1.  **Adapt the Code:** Modify `capstone_multitask.py` as needed for YOUR pixel project.
2.  **Activate Environment:** Ensure your PyTorch environment is active.
3.  **Run:** `python day_09_capstone_project/capstone_multitask.py`
4.  **Monitor:** Watch the console output and check TensorBoard (`tensorboard --logdir runs`) for loss curves and maybe even image samples!

## Summary

The Day 9 Capstone is your flexible launchpad! It provides a solid structure integrating everything learned. Your mission, should you choose to accept it, is to **adapt** the Dataset, Model, Loss, and Evaluation sections to fit _your_ specific pixel art deep learning idea. Go create something awesome!
