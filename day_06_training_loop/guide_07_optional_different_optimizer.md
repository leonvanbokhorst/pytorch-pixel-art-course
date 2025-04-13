# Guide: 07 (Optional) Swapping Magic Spoons: Trying Different Optimizers!

We've been using the `Adam` optimizer as our go-to magic stirring spoon for updating our pixel model. But PyTorch offers other spoons, like the classic `SGD` (Stochastic Gradient Descent)! Does the spoon matter? Let's compare them, based on `07_optional_different_optimizer.py`.

**Core Concept:** The optimizer is the _algorithm_ that decides _how_ to use the gradients (calculated by `loss.backward()`) to update the model's parameters. Different optimizers use different strategies, which can affect how quickly the model learns and how good the final result is.

## Meet the Spoons: SGD vs. Adam

- **`optim.SGD` (Stochastic Gradient Descent):**
  - The OG, the classic! It takes simple steps downhill based on the current batch's gradient, scaled by the learning rate.
  - _Pros:_ Simple, uses less memory, sometimes finds slightly better final solutions if tuned carefully.
  - _Cons:_ Can be slow, very sensitive to the learning rate, might get easily stuck on flat areas or bounce around in ravines.
- **`optim.Adam` (Adaptive Moment Estimation):**
  - The popular, adaptive wizard! It cleverly adapts the learning rate _for each parameter individually_, using estimates of the gradient's recent average (momentum) and squared average (variance).
  - _Pros:_ Often learns _much_ faster than SGD, less fussy about the initial learning rate (though tuning still helps!), generally a great starting point.
  - _Cons:_ Uses a bit more memory (to store those gradient averages), might occasionally settle for a slightly less optimal solution than perfectly tuned SGD.

## The Optimizer Showdown Experiment

The script runs a fair comparison:

1.  **Same Training Spell:** Uses the exact same `train_model` function.
2.  **Identical Starting Point:** Crucially uses `copy.deepcopy(base_model)` so both SGD and Adam start optimizing the _exact same_ initial random weights.
3.  **Different Spoons:** Creates one `optim.SGD` instance and one `optim.Adam` instance, each linked to its own fresh model copy.
4.  **Same Initial Stirring Speed (LR):** For this specific comparison, the script often initializes both optimizers with the _same_ learning rate to highlight the algorithm differences. (Note: Adam's default `lr` is usually smaller, so this isn't necessarily the _best_ setting for both, just a way to compare algorithms directly).

```python
# Spell Snippet (Experiment Setup):
import copy

# Assume base_pixel_model is defined
base_pixel_model = YourPixelModel(...)
criterion = ...
lr_common = 0.01 # Use same LR for direct comparison here
num_epochs = 20

# --- Run 1: SGD --- #
model_sgd = copy.deepcopy(base_pixel_model)
optimizer_sgd = torch.optim.SGD(model_sgd.parameters(), lr=lr_common)
losses_sgd = train_model(model_sgd, ..., optimizer_sgd, ...)

# --- Run 2: Adam --- #
model_adam = copy.deepcopy(base_pixel_model)
optimizer_adam = torch.optim.Adam(model_adam.parameters(), lr=lr_common)
losses_adam = train_model(model_adam, ..., optimizer_adam, ...)

# Compare losses_sgd and losses_adam (e.g., by plotting)
```

## Observing the Race

By comparing the loss lists (or plotting them):

- **Likely Outcome:** You'll probably see the `losses_adam` curve drop much faster initially than the `losses_sgd` curve when using the same LR. Adam often gets you to a decent result quicker.
- **The Catch:** It's _possible_ that with very careful tuning (and maybe extra tricks like momentum), SGD _might_ eventually find a slightly better final loss, but it often takes more effort and epochs.
- **Conclusion:** Adam is usually a great default optimizer choice due to its speed and robustness, but knowing about SGD is good too!

(See the conceptual loss curve plot in the original guide text comparing SGD and Adam - it illustrates the typical speed difference.)

## Swapping Spoons is Easy!

Want to try a different optimizer? PyTorch makes it super simple! Just change the line where you create the optimizer object. That's it! The rest of the training loop stays the same.

```python
# Swap from Adam to SGD:
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01) # Just change this line!
```

## Other Magic Spoons in `torch.optim`

PyTorch has a whole drawer full of optimizers:

- `RMSprop`: Another popular adaptive method.
- `Adagrad`: Good for sparse data (where many inputs are zero).
- `AdamW`: A tweaked version of Adam often preferred for modern architectures like Transformers.

## Summary

PyTorch offers various optimizers (magic spoons!) like `SGD` and `Adam` that use different strategies to update your pixel model's parameters. `Adam` is often faster and easier to get working initially due to its adaptive nature. Experimenting is easy – just change the optimizer instantiation line! `Adam` is usually a solid first choice for many pixel art tasks.
