import torch

print("--- Pixel Art Tensor Examples ---")

# Scalar: Representing a single grayscale pixel value (0-255)
# Use torch.uint8 for byte representation commonly used for images
grayscale_pixel = torch.tensor(128, dtype=torch.uint8)
print(f"Grayscale Pixel Value (Scalar): {grayscale_pixel}")
print(
    f"Shape: {grayscale_pixel.shape}, Dimensions: {grayscale_pixel.ndim}, Dtype: {grayscale_pixel.dtype}\n"
)

# Vector: Representing an RGB color [R, G, B]
# Use torch.uint8 for 0-255 color values
rgb_color = torch.tensor([255, 0, 255], dtype=torch.uint8)  # Magenta
print(f"RGB Color (Vector): {rgb_color}")
print(
    f"Shape: {rgb_color.shape}, Dimensions: {rgb_color.ndim}, Dtype: {rgb_color.dtype}\n"
)

# Matrix: Representing a small 2x2 grayscale image
# Values could be 0-255 (uint8) or 0.0-1.0 (float32, more common for models)
# Let's use uint8 for simplicity here.
small_grayscale_image = torch.tensor(
    [[0, 255], [255, 0]],  # Top row: Black, White  # Bottom row: White, Black
    dtype=torch.uint8,
)
print(f"Small 2x2 Grayscale Image (Matrix):\n{small_grayscale_image}")
print(
    f"Shape: {small_grayscale_image.shape}, Dimensions: {small_grayscale_image.ndim}, Dtype: {small_grayscale_image.dtype}"
)

# Example of a small RGB image (3D Tensor)
# Shape: [Height, Width, Channels]
small_rgb_image = torch.tensor(
    [
        [[255, 0, 0], [0, 255, 0]],  # Row 1: Red, Green
        [[0, 0, 255], [255, 255, 0]],  # Row 2: Blue, Yellow
    ],
    dtype=torch.uint8,
)
print(f"\nSmall 2x2 RGB Image (3D Tensor):\n{small_rgb_image}")
print(
    f"Shape: {small_rgb_image.shape}, Dimensions: {small_rgb_image.ndim}, Dtype: {small_rgb_image.dtype}"
)
