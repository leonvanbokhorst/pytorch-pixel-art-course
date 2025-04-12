# Guide: 06 Experimenting with Learning Rate

This guide demonstrates the crucial impact of the **learning rate** hyperparameter on the training process by comparing training runs with different learning rates, as shown in `06_experimenting_learning_rate.py`.

**Core Concept:** The learning rate (`lr`) is arguably one of the most important hyperparameters in training neural networks. It scales the magnitude of the parameter updates made by the optimizer based on the computed gradients. Setting it appropriately is critical for effective and efficient training.

## What is Learning Rate?

Recall the optimizer's role is to update parameters to minimize loss. It uses the gradient (which indicates the direction of steepest ascent of the loss) to decide which way to move the parameters. The learning rate determines _how big_ of a step to take in the opposite direction of the gradient.

`parameter_update = -learning_rate * gradient` (Simplified view for SGD)

## Why is Learning Rate Critical?

Finding a good learning rate is often described as a "Goldilocks" problem:

- **Too High:** If the `lr` is too large, the parameter updates overshoot the optimal minimum of the loss landscape. The loss might oscillate wildly, fail to decrease, or even increase (diverge). Training becomes unstable.
  - _Loss Curve:_ Jumpy, potentially increasing.
- **Too Low:** If the `lr` is too small, the parameter updates are tiny. Training will be very slow, requiring many epochs to converge. It might also get stuck in poor local minima more easily.
  - _Loss Curve:_ Decreases very slowly, might plateau prematurely at a high loss value.
- **Just Right:** An appropriate `lr` allows the loss to decrease consistently and efficiently towards a good minimum without becoming unstable.
  - _Loss Curve:_ Steadily decreasing, potentially with some minor fluctuations.

## Experimenting with Learning Rate

The script `06_experimenting_learning_rate.py` sets up an experiment to observe these effects:

1. **Reusable Training Function:** A `train_model` function encapsulates the training loop to run multiple experiments easily.
2. **Independent Runs:** It trains the _same_ model architecture three times:
    - **Resetting Weights:** Crucially, `copy.deepcopy(base_model)` is used before each run. This ensures each experiment starts from the _exact same_ initial random parameter values, making the comparison fair and isolating the effect of the learning rate.
    - **Different Optimizers:** An optimizer (`optim.SGD`) is created for each run, linked to the fresh model copy, but initialized with a different `learning_rate` (e.g., `lr_standard=0.01`, `lr_high=0.5`, `lr_low=0.0001`).
3. **Record Losses:** Each call to `train_model` returns the list of average epoch losses for that run.

```python
# Script Snippet (Experiment Setup):
base_model = SimpleRegressionNet(...)
criterion = nn.MSELoss()
num_epochs_short = 10

# Run 1: Standard LR
lr_standard = 0.01
model_standard = copy.deepcopy(base_model)
optimizer_standard = optim.SGD(model_standard.parameters(), lr=lr_standard)
losses_standard = train_model(..., optimizer_standard, ...)

# Run 2: High LR
lr_high = 0.5
model_high = copy.deepcopy(base_model)
optimizer_high = optim.SGD(model_high.parameters(), lr=lr_high)
losses_high = train_model(..., optimizer_high, ...)

# Run 3: Low LR
lr_low = 0.0001
model_low = copy.deepcopy(base_model)
optimizer_low = optim.SGD(model_low.parameters(), lr=lr_low)
losses_low = train_model(..., optimizer_low, ...)
```

## Observing the Results

By comparing the `losses_standard`, `losses_high`, and `losses_low` lists (or plotting them, although plotting multiple runs isn't shown in this specific script), you can directly see the impact of the learning rate:

- The high LR run likely results in a high final loss, or maybe even `NaN` (Not a Number) if it diverged completely.
- The low LR run likely results in a higher final loss than the standard run after the same number of epochs because convergence is much slower.
- The standard LR run (assuming it's reasonably chosen) should show the best convergence within the given epochs.

(Conceptual Plot)

```bash
  Loss
   |
   |       |--*------- High LR (unstable/diverging)
   |      /|  *
   |     / |
   |    /  *
   |   *   |
   |  /|\ / \*------- Standard LR (converging well)
   | / | \
   |*  |  \ / *------ Low LR (converging slowly)
   |\  *
   | \/
   *----------*-------*
  -----------------------> Epoch
```

## Finding a Good Learning Rate

- **Rule of Thumb:** Start with default values often suggested for specific optimizers (e.g., `0.01` or `0.1` for SGD, `0.001` for Adam) and adjust based on observation.
- **Experimentation:** Run short training experiments with different orders of magnitude (e.g., 0.1, 0.01, 0.001, 0.0001).
- **Learning Rate Finders:** Tools/techniques (like fastai's `lr_find` or PyTorch Lightning's tuner) that automatically test a range of LRs.
- **Learning Rate Schedulers:** Gradually decrease the learning rate during training (e.g., `torch.optim.lr_scheduler`).
- **Adaptive Optimizers:** Optimizers like Adam automatically adapt the learning rate for each parameter.

## Summary

The learning rate (`lr`) passed to the optimizer is a critical hyperparameter. It dictates the step size for parameter updates. An `lr` that's too high leads to instability, while one that's too low leads to slow training. Experimentation, often visualized by plotting loss curves from runs with different learning rates (starting from identical initial weights), is essential for finding a value that allows the model to learn effectively and efficiently.
