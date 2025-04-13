# Day 7: Scrying Pools & Alchemical Tomes - Evaluation & Improvement

**Assessing and Refining the Magic**

Our Pixel Paladin has successfully performed the Grand Enchantment (Training Loop)! A magical artifact (our model) has been created. But the journey isn't over. Is the enchantment potent? Does it achieve the desired effect? It's time to **Evaluate** our work. We'll use Scrying Pools (evaluation metrics, validation datasets) to gaze critically at our model's performance on unseen data. Does it generalize well, or did it just memorize the training runes? Based on what the Scrying Pool reveals, we consult the Alchemical Tomes for techniques to **Improve** our magic – tweaking hyperparameters (like adjusting the focus of a magical lens), applying regularization (adding stabilizing runes), or employing validation loops to guide our refinement process. This iterative cycle of evaluation and improvement is how we forge truly powerful pixel magic!

---

## 🎯 Objectives

**Topics:**

- Evaluating Pixel Model Performance:
  - Switching to evaluation mode (`model.eval()`): Important for layers like Dropout/BatchNorm if used.
  - Using `torch.no_grad()`: Disabling gradients for faster evaluation and no unwanted learning.
  - Evaluating on a validation set of unseen pixel art.
- Pixel Art Metrics: Assessing quality beyond just the loss.
  - Classification: Accuracy (how many sprites correctly identified?).
  - Generation: Validation loss (e.g., MSE). Visual inspection of generated sprites is crucial! Mention concepts like PSNR/SSIM if relevant.
- Diagnosing Pixel Training Issues:
  - Overfitting: Model generates/classifies training sprites perfectly but fails on new ones.
  - Underfitting: Model struggles even on training sprites (blurry generations, low accuracy).
  - Learning Rate Problems: Loss explodes or plateaus too quickly.
  - Data Issues: Check if your pixel art dataset is clean and representative.
- Tuning for Better Pixels: Experimenting with learning rate, model complexity, batch size.
- Preventing Pixel Overfitting: Mention regularization techniques, early stopping (stopping training when validation performance degrades).
- Saving and Loading Pixel Models: Using `torch.save` and `model.load_state_dict` to store and reuse trained generators/classifiers.

**Focus:** Evaluating how well pixel art models generalize, identifying and fixing common training problems, and saving trained models for later pixel generation or classification.

## Key Resources

- **PyTorch Official Tutorials - Save and Load the Model:** [https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html](https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html)
- **`torch.no_grad` Documentation:** [https://pytorch.org/docs/stable/generated/torch.no_grad.html](https://pytorch.org/docs/stable/generated/torch.no_grad.html)
- **`model.eval()` Documentation:** [https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.eval](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.eval)
- **Saving & Loading Models Guide:** [https://pytorch.org/tutorials/beginner/saving_loading_models.html](https://pytorch.org/tutorials/beginner/saving_loading_models.html)
- **TorchMetrics Library (Optional):** [https://torchmetrics.readthedocs.io/en/stable/](https://torchmetrics.readthedocs.io/en/stable/) (Useful for standard metrics like accuracy)

## Hands-On Pixel Evaluation Examples

- **Setting up for Pixel Evaluation:** ([`01_setup_for_evaluation.py`](./01_setup_for_evaluation.py))
  - **Pixel Idea:** Assume a trained pixel model (generator or classifier). Create a validation `DataLoader` (`val_loader`) with a separate set of unseen sprites. Keep the appropriate `criterion` (loss function).
  - **Purpose:** Prepare components to test the model on pixel data it hasn't seen during training.
- **Implementing the Pixel Evaluation Loop:** ([`02_implementing_evaluation_loop.py`](./02_implementing_evaluation_loop.py))
  - **Pixel Idea:** Implement the loop:
    Set `model.eval()`. Use `with torch.no_grad():`. Iterate through `val_loader`. Perform forward pass (`outputs = model(batch_sprites)`). Calculate validation loss (`loss = criterion(outputs, batch_targets)`). Accumulate total validation loss.
  - **Purpose:** Demonstrate the standard evaluation loop for pixel models, highlighting `model.eval()` and `torch.no_grad()`.
- **Calculating Sprite Classification Accuracy:** ([`03_calculating_accuracy.py`](./03_calculating_accuracy.py))
  - **Pixel Idea:** If doing classification, modify the evaluation loop to calculate accuracy: get predicted class labels (e.g., `outputs.argmax(dim=1)`), compare to true labels, count correct predictions, and calculate the overall accuracy on the validation set.
  - **Purpose:** Provide a concrete example of calculating a performance metric for pixel classification.
- **Saving Pixel Model Parameters:** ([`04_saving_model_parameters.py`](./04_saving_model_parameters.py))
  - **Pixel Idea:** After training a pixel generator or classifier, save its learned weights: `torch.save(model.state_dict(), 'pixel_model_weights.pth')`.
  - **Purpose:** Show how to save the trained state (the important part!) of your pixel model.
- **Loading Pixel Model Parameters:** ([`05_loading_model_parameters.py`](./05_loading_model_parameters.py))
  - **Pixel Idea:**
    1. Define the _exact same_ pixel model architecture: `new_pixel_model = YourPixelModel(...)`.
    2. Load the saved weights: `new_pixel_model.load_state_dict(torch.load('pixel_model_weights.pth'))`.
    3. Set `new_pixel_model.eval()`.
    4. Generate/classify some sample pixels using `new_pixel_model` to confirm it loaded correctly.
  - **Purpose:** Demonstrate loading saved weights to reuse a trained pixel model without retraining.
