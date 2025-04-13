import torch

# Create tensors
# Everything in PyTorch is a tensor.

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


def _print_tensor_properties(name: str, description: str, tensor: torch.Tensor):
    """
    Print the properties of a tensor.
    """
    separator = f"\n{'-' * 100}\n"
    tensor_info = f"{name}\n{description}\n\n"
    shape_info = f"Shape: \t\t{tensor.shape}\n"
    dim_info = f"Dimensions: \t{tensor.ndim}\n"
    dtype_info = f"Dtype: \t\t{tensor.dtype}\n"
    tensor_representation = f"\n{tensor}"

    print(
        f"{separator}"
        f"{tensor_info}"
        f"{shape_info}"
        f"{dim_info}"
        f"{dtype_info}"
        f"{tensor_representation}"
    )


# Print tensor properties

_print_tensor_properties(
    "SCALAR", "Representing a single grayscale pixel value (0-255)", grayscale_pixel
)

_print_tensor_properties(
    "VECTOR", "Representing an RGB color [R, G, B] (0-255)", rgb_color
)

_print_tensor_properties(
    "MATRIX", "Representing a small 2x2 grayscale image (0-255)", small_grayscale_image
)

_print_tensor_properties(
    "3D MATRIX", "Representing a 3x3x3 RGB color cube (0-255)", rgb_color_cube
)
