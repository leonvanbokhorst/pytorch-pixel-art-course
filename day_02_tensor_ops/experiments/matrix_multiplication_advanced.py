import torch

# Let's create a batch of feature matrices (like having multiple sprites)
batch_size = 3
features_per_sprite = 4
hidden_units = 5

# A batch of feature matrices: (batch_size, features_per_sprite)
batch_features = torch.randn(batch_size, features_per_sprite)
print(f"Batch of features shape: {batch_features.shape}")
print(f"Batch features:\n{batch_features}\n")

# A transformation matrix: (features_per_sprite, hidden_units)
transformation = torch.randn(features_per_sprite, hidden_units)
print(f"Transformation matrix shape: {transformation.shape}")
print(f"Transformation:\n{transformation}\n")

# Matrix multiplication with batch
transformed_batch = batch_features @ transformation
print(f"Transformed batch shape: {transformed_batch.shape}")
print(f"Transformed batch:\n{transformed_batch}\n")

# Now let's try something more complex - 3D tensors!
# Imagine we have multiple channels of features (like RGB)
channels = 3
batch_features_3d = torch.randn(batch_size, channels, features_per_sprite)
print(f"3D batch features shape: {batch_features_3d.shape}")
print(f"3D batch features:\n{batch_features_3d}\n")

# This will fail! Can you guess why?
try:
    result = batch_features_3d @ transformation
except RuntimeError as e:
    print("Expected error when trying to multiply 3D tensor with 2D matrix:")
    print(f"Error: {e}\n")

# To make it work, we need to reshape or use broadcasting
# Let's try reshaping first
reshaped_features = batch_features_3d.reshape(
    batch_size * channels, features_per_sprite
)
print(f"Reshaped features shape: {reshaped_features.shape}")
transformed_reshaped = reshaped_features @ transformation
print(f"Transformed reshaped shape: {transformed_reshaped.shape}\n")

# Or we can use broadcasting with permute
# This is more efficient and preserves the structure
transformed_3d = torch.matmul(batch_features_3d, transformation)
print(f"Transformed 3D shape: {transformed_3d.shape}")
print(f"Transformed 3D:\n{transformed_3d}")
