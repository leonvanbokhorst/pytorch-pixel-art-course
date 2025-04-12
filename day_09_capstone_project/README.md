# Day 9: The Capstone Citadel - Forging Your Legend!

**The Final Challenge: Your Masterpiece**

Our Pixel Paladin stands at the precipice, gazing up at the imposing, glittering **Capstone Citadel**. The journey has been long and arduous. You've deciphered the ancient Runes (Tensors), mastered the Incantations (Operations), wielded the Enchanted Quill (Autograd), drafted Magical Blueprints (`nn.Module`), gathered precious Resources (`Dataset`/`DataLoader`), toiled at the Grand Enchantment Forge (Training Loop), peered into the Scrying Pools (Evaluation), and channeled Raw Power (GPU).

Now, the ultimate test awaits. The Citadel's gates swing open, revealing not a final boss, but a grand workshop – _your_ workshop. This is your chance to combine **all** the skills, all the magic learned on this quest. Choose your challenge, define your goal, and forge your own **Legendary Artifact**! Will you create a breathtaking pixel landscape generator? A sprite classifier of unparalleled accuracy? A tool to transform images into pixel art masterpieces?

The path is yours to choose, Paladin. Take everything you have learned, unleash your creativity, overcome the final challenges (debugging!), and build something magnificent. Your capstone project is the culmination of your adventure, the proof of your mastery. Enter the Citadel, and forge your legend in the annals of Pixel Magic!

---

## 🎯 Objectives

This directory contains a **template** for a capstone project designed to integrate the PyTorch concepts learned throughout Days 1-8, applied specifically to the **pixel art domain**. It's set up as a flexible starting point for various pixel art tasks, including multi-task learning.

## Project Overview

The goal of this project template (`capstone_multitask.py`) is to provide a reusable structure for training a neural network on pixel art data. The default structure (which you **must adapt**) suggests a potential multi-task scenario, but can easily be simplified to a single task. For example, you could adapt this to:

1.  **Sprite Classification:** Train a model to recognize different types of pixel art sprites (e.g., characters, items, tiles).
2.  **Pixel Art Generation:** Build a simple generator (perhaps conditional) to create novel pixel sprites.
3.  **Multi-Task Learning:** Train a model to perform multiple tasks simultaneously, e.g.:
    - Classify a sprite's category **AND** predict its average color (regression).
    - Generate a sprite **AND** ensure it matches a certain style.

The template demonstrates how a shared model body (potentially using convolutional layers learned conceptually earlier) can learn features useful for one or more pixel-related tasks.

## Adaptability

This script is a **template**! To make it your own pixel art project, you **must** modify the sections marked with `*** ADAPT THIS ... ***` comments in the `capstone_multitask.py` file:

1.  **Configuration:** Adjust parameters like image dimensions (`IMG_HEIGHT`, `IMG_WIDTH`, `CHANNELS`), number of classes (`NUM_CLASSES`), learning rate, epochs, etc.
2.  **`PixelArtDataset` (or your chosen name):**
    - **Crucially:** Modify the `__init__` method to load _your_ pixel art dataset (e.g., from folders of PNGs, sprite sheets) instead of the placeholder synthetic data.
    - Ensure `__getitem__` loads and returns a sprite tensor (potentially with transforms) and its corresponding target(s) (e.g., class label, target color vector, or even the sprite itself for generation tasks like autoencoders).
    - _If doing a single task_, remove references to the unused target data.
3.  **`PixelArtModel` (or your chosen name):**
    - Adjust the architecture in `__init__`. For pixel art, consider using `nn.Conv2d` layers in the shared body to process spatial information effectively. Define appropriate input/output dimensions.
    - Define task-specific heads if doing multi-task learning (e.g., a classification head, a color prediction head).
    - Modify the `forward` method to pass data through the body and then the relevant head(s).
    - _If doing a single task_, simplify the model to have only the necessary components.
4.  **Loss Functions & Weights:**
    - In the main execution block, choose appropriate loss functions for your pixel task(s). Examples:
      - Classification: `nn.CrossEntropyLoss`
      - Generation/Regression (pixel values/colors): `nn.MSELoss`, `nn.L1Loss`
    - If multi-tasking, adjust `loss_weights` to balance the contribution of each task.
5.  **Evaluation (`evaluate_model` function):**
    - Keep loss calculations relevant to your chosen criteria.
    - **Crucially:** Add or modify the calculation of relevant evaluation _metrics_ for your pixel art tasks (e.g., classification accuracy, PSNR for generation, Mean Absolute Error for color prediction). **Visual inspection of generated outputs is often essential for generation tasks!**
    - Update print statements to report your chosen pixel art metrics.
6.  **Model Saving:**
    - In the main loop, change the condition for saving the best model to use the most relevant validation metric for your primary pixel task (e.g., `val_accuracy` or `val_generation_loss`).

## Components (Needs Adaptation)

- **Dataset (`PixelArtDataset`):** Needs modification to load _your_ sprites and targets.
- **Model (`PixelArtModel`):** Needs architecture adjustments for your specific pixel task (consider `nn.Conv2d`).
- **Loss Function(s):** Choose based on your task (classification, generation, etc.).
- **Training Loop (`train_epoch`):** Mostly standard, uses chosen loss.
- **Evaluation Loop (`evaluate_model`):** Calculates loss and needs _your_ specific pixel art metrics added.

## How to Run (After Adapting)

1.  **Ensure Prerequisites:** Activate your virtual environment with PyTorch (and potentially `torchvision`, `matplotlib`) installed.
2.  **Adapt the Code:** Modify `capstone_multitask.py` as described above for your specific pixel art project.
3.  **Navigate:** `cd` to the root of the project.
4.  **Execute:** Run your adapted script:

    ```bash
    python day_09_capstone_project/capstone_multitask.py
    ```
