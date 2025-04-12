# Day 1: The Basic Runes - Understanding Tensors

**Our Quest Begins!**

Welcome, Pixel Paladin, to the very start of your adventure! Before you can enchant sprites or battle `gradient` dragons, you must master the fundamental magic of this land: **Tensors**. Think of them as the clay, the wood, the raw arcane energy from which all pixel creations are formed. Today, we gather our first supplies and learn to read the basic runes that represent everything from a single pixel's color to the layout of an entire sprite sheet. Master these, and the path to the Capstone Citadel opens before you!

---

**Topics:**

- What is PyTorch? (A library for deep learning, especially with GPUs)
- Tensors: Representing pixel data (colors, coordinates, images).
- PyTorch vs NumPy: Similarities, but PyTorch has GPU power and automatic differentiation (crucial for learning!).
- Tensor Basics for Pixel Art:
  - Creating tensors (`torch.tensor`) to hold pixel values.
  - Understanding dimensions: Scalar (single value, like brightness), Vector (1D list, like a color palette index or coordinates), Matrix (2D grid, like a small grayscale sprite), 3D+ Tensor (like an RGB sprite).
  - Tensor shape (`.shape`): How many pixels high/wide? How many color channels?
  - Tensor data type (`.dtype`): Are pixel values integers (0-255) or floats (0.0-1.0)?
- Device Awareness: Using the CPU or GPU for pixel processing.

**Focus:** Understanding how to represent and inspect pixel art data using PyTorch's fundamental building block, the tensor.

## Key Resources

- **PyTorch Official Tutorials - Tensors:** [https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html](https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)
- **`torch.tensor` Documentation:** [https://pytorch.org/docs/stable/generated/torch.tensor.html](https://pytorch.org/docs/stable/generated/torch.tensor.html)

## Hands-On Pixel Examples

- **Creating Pixel Data Tensors:** ([`01_creating_basic_tensors.py`](./01_creating_basic_tensors.py))
  - **Pixel Idea:** Create a tensor representing a single pixel's grayscale value (scalar), a 1D tensor for RGB color `[R, G, B]` (vector), and a 2D tensor for a tiny 2x2 grayscale image (matrix).
  - **Purpose:** Show how different pixel art elements map to tensor dimensions.
- **Inspecting Sprite Attributes:** ([`02_checking_tensor_attributes.py`](./02_checking_tensor_attributes.py))
  - **Pixel Idea:** Create a small image tensor (e.g., 3x3 pixels) and print its `.shape` (height, width), `.ndim` (always 2 for grayscale, 3 for color), and `.dtype` (e.g., `torch.uint8` or `torch.float32`).
  - **Purpose:** Show how to check the size, dimensions, and data format of your pixel representation.
- **Specifying Pixel Data Type:** ([`03_specifying_data_type.py`](./03_specifying_data_type.py))
  - **Pixel Idea:** Create a tensor representing pixel values, explicitly setting `dtype=torch.float32` (common for model inputs) or `dtype=torch.uint8` (common for image file storage).
  - **Purpose:** Illustrate controlling the precision and range of pixel values.
- **(Optional) Pixel Tensor from NumPy:** ([`04_optional_tensor_from_numpy.py`](./04_optional_tensor_from_numpy.py))
  - **Pixel Idea:** Create a NumPy array representing a small image and convert it to a PyTorch tensor using `torch.from_numpy()`. Useful if loading image data with libraries like Pillow or OpenCV.
  - **Purpose:** Show how to integrate PyTorch with standard Python image handling.
