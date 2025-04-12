# Guide: 01 Pixel Model Checkup: Setting Up for Evaluation!

Our pixel model _thinks_ it learned something during training. But how do we _really_ know? We need to give it a pop quiz using sprites it's never seen before! This guide covers setting up the stage for this evaluation, based on `01_setup_for_evaluation.py`.

**Core Concept:** Just checking the final _training_ loss isn't enough. A model might ace the homework (training data) but fail the final exam (new data). We need to evaluate it on a separate **validation dataset** (the slice of the pie we kept aside in Day 5) to see how well it _generalizes_ to new, unseen pixel art.

## Ingredients for the Pixel Pop Quiz

Setting up for evaluation is similar to training setup, but with a focus on testing:

1.  **The (Trained) Pixel Model (`nn.Module`):** You need an instance of the _exact same model architecture_ you trained. **Crucially**, this instance should be loaded with the **learned parameters** (weights and biases) that resulted from your successful training run. (We cover loading in guide 5).
2.  **The Quiz Sprites (`Dataset` / `DataLoader`):** A `DataLoader` specifically for your **validation set** (or test set) – the sprites the model _never_ saw during training.
3.  **The Grading Rubric (Loss Function):** Usually, the _same_ loss function (`criterion`) you used for training. This lets you calculate the model's loss score on the new sprites, providing a comparable measure of error.

## Setting the Stage for Evaluation

1.  **Summon the Trained Model:** Create an instance of your pixel model class. **In a real workflow, this is where you'd immediately load the saved weights** from your training run (using `model.load_state_dict(...)` - see guide 5).

    ```python
    # Spell Snippet (Model Instantiation):
    # Assume YourPixelModel class is defined
    model_to_evaluate = YourPixelModel(noise_dim=..., num_pixels=...)

    # --- VERY IMPORTANT --- #
    # In a real script, load the weights you saved after training!
    # print("Loading trained weights...")
    # model_to_evaluate.load_state_dict(torch.load('best_pixel_model.pth'))
    # -------------------- #
    ```

2.  **Prepare the Validation Sprites:** You should already have your `val_pixel_dataset` (a `Subset` object) from when you split the data in Day 5 using `random_split`.

3.  **Create the Validation `DataLoader`:** Wrap your `val_pixel_dataset` in a `DataLoader`. **Crucially, set `shuffle=False`!** We want to evaluate on the validation sprites in the same, consistent order every time.

    ```python
    # Spell Snippet (DataLoader):
    # Assume val_pixel_dataset is the Subset from random_split
    BATCH_SIZE = 16 # Can often use a larger batch size for evaluation

    val_loader = DataLoader(
        dataset=val_pixel_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False # <<< Important: NO shuffling for evaluation!
    )
    ```

4.  **Get the Grading Rubric (Loss Function):** Instantiate the same loss function you used during training.

    ```python
    # Spell Snippet (Loss):
    # Use the same criterion as training
    # e.g., if you trained a generator with MSE:
    criterion = nn.MSELoss()
    # e.g., if you trained a classifier with BCEWithLogits:
    # criterion = nn.BCEWithLogitsLoss()
    ```

## Ready for the Evaluation Loop!

With your trained model, the validation `DataLoader` (serving unseen sprites, unshuffled), and the criterion ready, you have everything needed for the evaluation loop (Guide 2). This loop will run the validation sprites through the model, calculate the loss (and maybe accuracy or other metrics), but **will not** perform backpropagation or update the model's weights.

## Summary

Setting up for pixel model evaluation requires:

1. Your trained **Model** instance (with loaded weights).
2. A **`DataLoader`** for the validation/test set (with `shuffle=False`).
3. The same **Loss Function** (Criterion) used during training.
   These components allow you to measure how well your model performs on sprites it hasn't encountered before.
