# Guide: 04 Saving Model Parameters (state_dict)

This guide explains the standard and recommended method for saving the learned parameters of your trained PyTorch model using its `state_dict`, as demonstrated in `04_saving_model_parameters.py`.

**Core Concept:** After investing time and computation to train a model, you need a way to save its learned state (primarily its weights and biases) so you can reuse it later for inference, fine-tuning, or analysis without having to retrain from scratch.

## Recommended Method: Saving the `state_dict`

While it's possible to save the entire model object using `torch.save(model, PATH)`, the **recommended and most robust approach** is to save only the model's **state dictionary (`state_dict`)**. Saving the entire model pickles the specific class structure and file paths used during saving, which can easily break if you refactor your code or move the project.

Saving the `state_dict` only saves the _parameters_ and _buffers_, making it more portable and less prone to breaking.

## What is a `state_dict`?

A PyTorch `state_dict` is simply a Python dictionary object (specifically, an `OrderedDict`) that maps each layer to its parameter tensors.

- It includes all **learnable parameters** (weights and biases) of the model and its submodules.
- It also includes registered **buffers** (like the running mean and variance tracked by BatchNorm layers) that are part of the model's state but are not updated by the optimizer.

## Steps to Save the `state_dict`

Saving the parameters is straightforward:

1. **Get the `state_dict`:** Access the state dictionary from your trained model instance using the `.state_dict()` method.

   ```python
   # Assuming 'model' is your trained nn.Module instance
   state_dict = model.state_dict()
   ```

2. **Define Save Path:** Choose a location and filename for saving the parameters. The convention is to use `.pth` or `.pt` file extensions.

   ```python
   # Script Snippet:
   import os
   SAVE_DIR = "saved_models"
   MODEL_FILENAME = "simple_classifier_weights.pth"
   SAVE_PATH = os.path.join(SAVE_DIR, MODEL_FILENAME)
   # Ensure directory exists
   os.makedirs(SAVE_DIR, exist_ok=True)
   ```

3. **Save using `torch.save()`:** Use the `torch.save()` function to serialize the `state_dict` object to the specified file path. PyTorch uses Python's `pickle` utility behind the scenes for serialization.

   ```python
   # Script Snippet:
   torch.save(state_dict, SAVE_PATH)
   print(f"Successfully saved model state_dict to {SAVE_PATH}")
   ```

## Important Limitation

Remember, this method **only saves the model's parameters and buffers**. It **does not save the model's architecture** (the Python class definition, like `SimpleClassificationNet`). To load these weights later, you will need:

1. The definition of the model class itself.
2. The saved `state_dict` file.

## Summary

The standard and recommended way to save a trained PyTorch model is to save its `state_dict` using `torch.save(model.state_dict(), PATH)`. This saves the learnable parameters and buffers, providing a portable way to store the model's learned state. Remember that the model's class definition needs to be available separately to load these parameters later.
