import torch

# Create tensors

# Scalar
grayscale_pixel = torch.tensor(128, dtype=torch.uint8)

# Vector
rgb_color = torch.tensor([255, 0, 255], dtype=torch.uint8)

# Matrix
small_grayscale_image = torch.tensor([[0, 255], [255, 0]], dtype=torch.uint8)

# 3D Matrix
rgb_color_cube = torch.tensor(
    [
        [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
        [[255, 255, 0], [0, 255, 255], [255, 0, 255]],
        [[128, 128, 128], [128, 128, 128], [128, 128, 128]],
    ],
    dtype=torch.uint8,
)


def print_tensor_properties(name, description, tensor):
    print(
        f"\n{'-' * 100}\n"
        f"{name}\n{description}\n\n"
        f"Shape: \t\t{tensor.shape}\n"
        f"Dimensions: \t{tensor.ndim}\n"
        f"Dtype: \t\t{tensor.dtype}\n"
        f"\n{tensor}"
    )

# Print tensor properties

print_tensor_properties(
    "SCALAR", "Representing a single grayscale pixel value (0-255)", grayscale_pixel
)

print_tensor_properties(
    "VECTOR", "Representing an RGB color [R, G, B] (0-255)", rgb_color
)

print_tensor_properties(
    "MATRIX", "Representing a small 2x2 grayscale image (0-255)", small_grayscale_image
)

print_tensor_properties(
    "3D MATRIX", "Representing a 3x3x3 RGB color cube", rgb_color_cube
)
