import torch
import matplotlib.pyplot as plt
import numpy as np

# Let's create some example sprite features
# Each row represents a different sprite with 4 features
sprite_features = torch.tensor(
    [
        # [brightness, contrast, edge_intensity, color_saturation]
        [0.8, 0.6, 0.9, 0.7],  # High contrast sprite with strong edges
        [0.3, 0.2, 0.1, 0.4],  # Dark, low contrast sprite
        [0.5, 0.7, 0.3, 0.8],  # Medium brightness with high saturation
    ]
)

# Our transformation matrix - each column represents a "pattern detector"
# These weights would normally be learned during training
transformation = torch.tensor(
    [
        # Each column detects a different pattern
        # [diagonal_lines, red_tones, light_dark_contrast, symmetry, gradient]
        [
            0.8,
            0.1,
            0.6,
            0.3,
            0.4,
        ],  # How much each input feature contributes to detecting diagonal lines
        [0.2, 0.9, 0.3, 0.1, 0.5],  # Contribution to detecting red tones
        [0.7, 0.2, 0.8, 0.4, 0.3],  # Contribution to light/dark contrast
        [0.3, 0.4, 0.5, 0.9, 0.2],  # Contribution to symmetry
    ]
)

print("Original sprite features (4 features per sprite):")
print(sprite_features)
print("\nTransformation matrix (4 input features -> 5 hidden units):")
print(transformation)

# Transform the features
transformed_features = sprite_features @ transformation
print("\nTransformed features (5 hidden units per sprite):")
print(transformed_features)

# Let's interpret what these hidden units mean for each sprite
print("\nInterpretation of hidden units for each sprite:")
for i, sprite in enumerate(transformed_features):
    print(f"\nSprite {i+1}:")
    print(f"Diagonal lines strength: {sprite[0]:.2f}")
    print(f"Red tones presence: {sprite[1]:.2f}")
    print(f"Light/dark contrast: {sprite[2]:.2f}")
    print(f"Symmetry score: {sprite[3]:.2f}")
    print(f"Gradient effect: {sprite[4]:.2f}")
