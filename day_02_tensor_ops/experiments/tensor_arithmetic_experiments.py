import torch

print("=== Tensor Arithmetic Experiments ===\n")

# 1. Basic Operations with Random Vectors
print("1. Basic Operations with Random Vectors")
vector = torch.randn(1, 8)  # Create a random vector
print(f"Original vector:\n{vector}")

# Let's try different operations
print("\nA. Absolute Value (multiple ways)")
print(f"Using abs(): {vector.abs()}")
print(f"Using **2 and **0.5: {(vector**2)**0.5}")
print(f"Using pow(2).pow(0.5): {vector.pow(2).pow(0.5)}")

# 2. Power Operations
print("\n2. Power Operations")
print("A. Square and Cube")
print(f"Square: {vector**2}")
print(f"Cube: {vector**3}")

print("\nB. Square Root and Cube Root")
print(f"Square Root: {vector.abs().sqrt()}")  # Note: sqrt needs positive numbers
print(f"Cube Root: {vector.pow(1/3)}")  # Can handle negative numbers

# 3. Combining Operations
print("\n3. Combining Operations")
print("A. Normalize and Square")
normalized = vector / vector.abs().max()  # Normalize to [-1, 1]
print(f"Normalized: {normalized}")
print(f"Squared normalized: {normalized**2}")

print("\nB. Create a 'glow' effect")
# Create a base value and add a fraction of the squared value
glow = normalized + 0.5 * (normalized**2)
print(f"Glow effect: {glow}")

# 4. Pixel Art Example
print("\n4. Pixel Art Example")
# Create a simple 4x4 grayscale sprite
sprite = torch.tensor(
    [[0, 64, 128, 192], [32, 96, 160, 224], [64, 128, 192, 255], [96, 160, 224, 255]],
    dtype=torch.float32,
)

print("A. Original Sprite:")
print(sprite)

print("\nB. Normalized Sprite (0-1):")
normalized_sprite = sprite / 255.0
print(normalized_sprite)

print("\nC. Gamma Correction (power of 2):")
gamma_corrected = normalized_sprite**2
print(gamma_corrected)

print("\nD. Back to 0-255 range:")
final_sprite = (gamma_corrected * 255).to(torch.uint8)
print(final_sprite)

# 5. Detailed Glow Effect Explanation
print("\n5. Detailed Glow Effect Explanation")
# Create a simple gradient sprite
gradient = torch.tensor(
    [
        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    ],
    dtype=torch.float32,
)

print("A. Original Gradient:")
print(gradient)

# Basic glow formula: output = input + (glow_strength * input^2)
# Where glow_strength controls how strong the effect is
print("\nB. Different Glow Strengths:")
glow_strength_weak = 0.25
glow_weak = gradient + (glow_strength_weak * gradient**2)
print(f"\nWeak Glow (strength={glow_strength_weak}):")
print(glow_weak)

glow_strength_medium = 0.5
glow_medium = gradient + (glow_strength_medium * gradient**2)
print(f"\nMedium Glow (strength={glow_strength_medium}):")
print(glow_medium)

glow_strength_strong = 1.0
glow_strong = gradient + (glow_strength_strong * gradient**2)
print(f"\nStrong Glow (strength={glow_strength_strong}):")
print(glow_strong)

# Explanation of how it works:
print("\nC. How the Glow Effect Works:")
print("1. Original value: The base pixel value (0-1)")
print("2. Squared value: Makes bright areas brighter (0-1)")
print("3. Scaled squared: Controls glow intensity (0 to glow_strength)")
print("4. Final result: Original + scaled squared")

# Example calculation for one pixel:
pixel_value = 0.5
print(f"\nD. Example Calculation for pixel value {pixel_value}:")
print(f"Original: {pixel_value}")
print(f"Squared: {pixel_value**2}")
print(f"Scaled squared (0.5): {0.5 * pixel_value**2}")
print(f"Final result: {pixel_value + 0.5 * pixel_value**2}")

# 6. Glow Effect Visualization
print("\n6. Glow Effect Visualization")
# Create a single row gradient for clearer visualization
simple_gradient = torch.tensor([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
print(f"\nOriginal Gradient:\n{simple_gradient}")

# Show how the glow effect changes with brightness
print("\nGlow Effect (strength=0.5) at different brightness levels:")
for value in simple_gradient:
    original = value.item()
    squared = original**2
    glow = original + 0.5 * squared
    print(f"Value: {original:.1f} -> Squared: {squared:.2f} -> Glow: {glow:.2f}")
