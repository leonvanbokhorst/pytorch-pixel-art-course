# Guide: 01 Setting Up for Evaluation

This guide covers the setup required before evaluating a trained PyTorch model on unseen data, as demonstrated in `01_setup_for_evaluation.py`.

**Core Concept:** After training a model, simply looking at the final training loss isn't enough to know how well it will perform on new, real-world data. We need to evaluate it on a separate dataset (validation or test set) that was held out during training to get an unbiased estimate of its generalization ability.

## Components Needed for Evaluation

Similar to training setup, evaluation requires:

1. **The Model (`nn.Module`):** An instance of the _same architecture_ that was trained. Crucially, this instance should contain the **learned parameters** (weights and biases) obtained from the training process. (Loading these parameters is covered later in script `05`).
2. **Validation/Test Data (`Dataset` / `DataLoader`):** A `DataLoader` that provides batches from a dataset the model **did not see** during training.
3. **Loss Function (Criterion):** Usually the _same_ loss function used during training, allowing you to calculate the model's loss on the evaluation data.

## Evaluation Setup Steps

1. **Instantiate Model Architecture:** Create an instance of your model class. In a typical workflow, you would immediately load the saved trained weights into this instance.

   ```python
   # Script Snippet (Model Instantiation):
   # Assuming SimpleClassificationNet class is defined
   model = SimpleClassificationNet(INPUT_FEATURES, HIDDEN_FEATURES, NUM_CLASSES)
   # IMPORTANT: In practice, load trained weights here!
   # model.load_state_dict(torch.load('path/to/trained_weights.pth'))
   ```

2. **Prepare Validation/Test Data:** Load or generate your separate validation or test dataset (`X_val`, `y_val`). Ensure this data follows the same format and preprocessing steps as the training data, but contains different samples.

   ```python
   # Script Snippet (Data Generation):
   NUM_VAL_SAMPLES = 500
   X_val = torch.randn(NUM_VAL_SAMPLES, INPUT_FEATURES)
   y_val = torch.randint(0, NUM_CLASSES, (NUM_VAL_SAMPLES,))
   ```

3. **Create Evaluation `Dataset` & `DataLoader`:** Wrap the evaluation data in your `Dataset` class and then create a `DataLoader` for it.

   ```python
   # Script Snippet (DataLoader):
   val_dataset = ClassificationDataset(X_val, y_val)
   # No need to shuffle for evaluation
   val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
   ```

   - **`shuffle=False`:** It's standard practice to _not_ shuffle validation or test data. This ensures consistent evaluation results across runs and makes it easier to compare specific predictions if needed.

4. **Instantiate Loss Function:** Create an instance of the same loss function used during training.

   ```python
   # Script Snippet (Loss):
   # Use the same criterion as training (e.g., CrossEntropyLoss)
   criterion = nn.CrossEntropyLoss()
   ```

## Ready for the Evaluation Loop

With the model (containing trained weights), the evaluation `DataLoader`, and the criterion prepared, you have the necessary components to run the evaluation loop. This loop will iterate through the `val_loader`, pass the data through the model, and calculate the loss and other performance metrics (like accuracy) without updating the model's weights.

## Summary

Setting up for evaluation involves preparing the model instance (ideally loading trained weights), creating a `DataLoader` for the unseen validation or test dataset (typically with `shuffle=False`), and instantiating the same loss function used during training. These components are then used in the evaluation loop to assess the model's generalization performance.
