# Guide: 07 (Optional) Using a Different Optimizer (SGD vs Adam)

This guide compares the performance of two different optimizers, Stochastic Gradient Descent (SGD) and Adam, on the same task, as shown in `07_optional_different_optimizer.py`.

**Core Concept:** PyTorch offers various optimization algorithms in `torch.optim`. While SGD is fundamental, more advanced optimizers like Adam often provide faster convergence or better performance on complex problems by adapting the learning rate or incorporating momentum in more sophisticated ways.

## Optimizers Overview

- **`optim.SGD`:** Stochastic Gradient Descent. Updates parameters based on the negative gradient scaled by a single, fixed learning rate. May include optional momentum and weight decay.
  - _Pros:_ Simple, well-understood, can generalize well (sometimes better than Adam if tuned carefully).
  - _Cons:_ Can be slow to converge, highly sensitive to learning rate choice and scaling of parameters, can get stuck in local minima/saddle points.
- **`optim.Adam`:** Adaptive Moment Estimation. Computes adaptive learning rates for _each parameter_ based on estimates of the first moment (mean) and second moment (uncentered variance) of the gradients.
  - _Pros:_ Often converges much faster than SGD, less sensitive to the initial learning rate choice (though tuning still helps), works well on a wide range of problems.
  - _Cons:_ Can sometimes converge to slightly worse minima than well-tuned SGD, uses more memory to store moment estimates.

## Experiment Setup

The script compares SGD and Adam fairly:

1. **Reusable Training Function:** Uses the same `train_model` function for both runs.
2. **Identical Start:** Uses `copy.deepcopy(base_model)` to ensure both optimizers start optimizing the model from the _exact same_ initial weights.
3. **Optimizer Instantiation:** Creates two optimizer instances, one `optim.SGD` and one `optim.Adam`. Both are linked to their respective fresh model copies.
4. **Same Initial LR:** For a direct comparison in the script, both optimizers are initialized with the _same_ learning rate (`lr_common = 0.01`). Note that Adam's default `lr` is typically smaller (0.001), and using different, individually tuned LRs might show different results.

```python
# Script Snippet (Experiment Setup):
base_model = SimpleRegressionNet(...)
criterion = nn.MSELoss()
lr_common = 0.01
num_epochs_compare = 20

# Run 1: SGD
model_sgd = copy.deepcopy(base_model)
optimizer_sgd = optim.SGD(model_sgd.parameters(), lr=lr_common)
losses_sgd = train_model(..., optimizer_sgd, ...)

# Run 2: Adam
model_adam = copy.deepcopy(base_model)
optimizer_adam = optim.Adam(model_adam.parameters(), lr=lr_common)
losses_adam = train_model(..., optimizer_adam, ...)
```

## Observing the Results

By comparing the lists of losses (`losses_sgd`, `losses_adam`) or plotting them, we can see the difference in convergence speed and final loss.

- **Typical Outcome:** Adam often decreases the loss much more rapidly in the initial epochs compared to basic SGD with the same learning rate. It may also reach a lower final loss within the same number of epochs.
- **Task Dependence:** While Adam is a strong default, there are cases where well-tuned SGD (perhaps with momentum) can achieve better final results, although it might require more careful tuning of the learning rate and potentially a learning rate schedule.

(Conceptual Plot)

```bash
  Loss
   |
   | *------------- SGD
   |  \ \
   |   \ \
   |    \ *---------- Adam (faster convergence)
   |     \ /
   |      *
   |     / \
   |    /   \
   |   *     *
   |  /
   | /
   *--------------*---
  -----------------------> Epoch
```

## Switching Optimizers

PyTorch makes it trivial to swap optimizers. You only need to change the line where the optimizer is instantiated, ensuring you pass it `model.parameters()` and appropriate hyperparameters (like `lr`).

## Other Optimizers in `torch.optim`

PyTorch provides many other optimizers, including:

- `RMSprop`: Also uses adaptive learning rates based on moving averages of squared gradients.
- `Adagrad`: Adapts learning rates based on historical gradient information (larger updates for infrequent features).
- `AdamW`: A variant of Adam that decouples weight decay from the adaptive learning rate calculation, often preferred in modern transformer models.

## Summary

Different optimizers employ different strategies for updating model parameters based on gradients. Adam is a popular adaptive optimizer known for often converging faster than basic SGD and being less sensitive to the initial learning rate. PyTorch makes it easy to experiment with various optimizers from `torch.optim` by simply changing the optimizer instantiation while keeping the rest of the training loop structure the same.
