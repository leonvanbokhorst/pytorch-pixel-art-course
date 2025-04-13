import torch

# Layer 1: Basic features to simple patterns
basic_features = torch.tensor(
    [
        # [brightness, contrast]
        [0.8, 0.6],  # Bright, high contrast sprite
        [0.3, 0.2],  # Dark, low contrast sprite
    ]
)

# First layer: Detects simple patterns
layer1 = torch.tensor(
    [
        # [edges, color_pops]
        [0.9, 0.1],  # How brightness and contrast contribute to edges
        [0.2, 0.8],  # How they contribute to color popping
    ]
)

# Second layer: Combines simple patterns into complex ones
layer2 = torch.tensor(
    [
        # [character_shape, background]
        [0.7, 0.3],  # How edges and color contribute to character shape
        [0.1, 0.9],  # How they contribute to background
    ]
)

print("Original sprite features:")
print(basic_features)
print("\nFirst layer weights (simple patterns):")
print(layer1)
print("\nSecond layer weights (complex patterns):")
print(layer2)

# Pass through first layer
simple_patterns = basic_features @ layer1
print("\nAfter first layer (simple patterns):")
print(simple_patterns)

# Pass through second layer
complex_patterns = simple_patterns @ layer2
print("\nAfter second layer (complex patterns):")
print(complex_patterns)

# Let's interpret what each layer learned
print("\nWhat the network learned:")
for i, sprite in enumerate(complex_patterns):
    print(f"\nSprite {i+1}:")
    print(f"Character shape strength: {sprite[0]:.2f}")
    print(f"Background presence: {sprite[1]:.2f}")
