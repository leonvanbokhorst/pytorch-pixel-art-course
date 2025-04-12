# Day 2: Manipulating the Arcane Glyphs - Tensor Operations

**Learning the Incantations**

Our Pixel Paladin has successfully identified the basic runes (Tensors)! But knowing the symbols isn't enough; true power lies in manipulating them. Today, we learn the fundamental incantations – the **Tensor Operations**. These are the actions, the verbs, the spells that allow us to reshape, combine, and transform the raw energy of tensors. Mastering these operations is like learning to wield a blacksmith's hammer or a wizard's staff; it's how we turn basic ingredients into useful tools and potent magic!

---

## 🎯 Objectives

**Topics:**

- Tensor Attributes Review: `shape` (image height/width/channels), `dtype` (pixel data type), `device` (CPU/GPU).
- Indexing and Slicing Pixels: Accessing specific pixels, rows/columns of pixels, or color channels.
- Reshaping Sprite Data: Changing the layout of pixel data (e.g., flattening an image for a neural network).
- Broadcasting for Pixel Effects: Applying operations (like adding brightness) to images of different sizes/dimensions.
- Basic Pixel Arithmetic: Element-wise operations for tasks like adjusting brightness (`+`), contrast (`*`), or color overlays.
- Mathematical Functions on Pixels: Calculating average pixel intensity (`torch.mean`), total brightness (`torch.sum`), etc.
- In-place Pixel Modifications: Directly changing pixel values in a tensor (e.g., `image.add_(value)`).

**Focus:** Learning how to manipulate and transform pixel art data represented as tensors using PyTorch operations.

## Key Resources

- **PyTorch Official Tutorials - Tensor Operations:** [https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html#operations-on-tensors](https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html#operations-on-tensors)
- **PyTorch `torch` Module Documentation:** [https://pytorch.org/docs/stable/torch.html](https://pytorch.org/docs/stable/torch.html)
- **Broadcasting Semantics:** [https://pytorch.org/docs/stable/notes/broadcasting.html](https://pytorch.org/docs/stable/notes/broadcasting.html)

## Hands-On Pixel Operations

- **Indexing and Slicing Sprites:** ([`01_indexing_slicing.py`](./01_indexing_slicing.py))
  - **Pixel Idea:** Create a small RGB image tensor (e.g., 3x3x3). Show how to get a specific pixel's RGB value (`image[y, x]`), the entire red channel (`image[:, :, 0]`), a row of pixels (`image[row_index, :]`), or a rectangular patch (`image[y_start:y_end, x_start:x_end]`).
  - **Purpose:** Demonstrate how to access and extract specific parts of pixel art data.
- **Reshaping Pixel Data:** ([`02_reshaping_tensors.py`](./02_reshaping_tensors.py))
  - **Pixel Idea:** Create a tensor representing a small image (e.g., 4x4 grayscale). Use `.view(-1)` or `.reshape(-1)` to flatten it into a 1D vector (like preparing input for a simple dense layer). Reshape it back to 4x4.
  - **Purpose:** Illustrate changing the tensor shape, crucial for feeding data into different types of neural network layers.
- **Pixel Arithmetic (Brightness/Contrast):** ([`03_tensor_arithmetic.py`](./03_tensor_arithmetic.py))
  - **Pixel Idea:** Create a simple grayscale image tensor. Add a scalar value to increase brightness (`image + 10`). Multiply by a scalar to adjust contrast (`image * 1.5`). Add two images together for blending.
  - **Purpose:** Show how basic arithmetic applies to common image manipulation tasks.
- **Broadcasting Color Adjustments:** ([`04_broadcasting.py`](./04_broadcasting.py))
  - **Pixel Idea:** Create an RGB image (e.g., shape `[H, W, 3]`). Create a color adjustment tensor `adjust = torch.tensor([10, -5, 20])` (shape `[3]`). Add them (`image + adjust`). PyTorch should broadcast the adjustment across all pixels.
  - **Purpose:** Demonstrate applying a single color adjustment vector to every pixel in an image efficiently.
- **Matrix Multiplication (Simple Filters):** ([`05_matrix_multiplication.py`](./05_matrix_multiplication.py))
  - **Pixel Idea:** Represent a simple 1D signal (like a row of pixels) and a small kernel (like `[0.5, 0.5]`). Show how `torch.matmul` _could_ be used (though convolution is more standard for filters). _Note: This might be a bit contrived; convolution (`nn.Conv2d`) is the proper way later._ Maybe focus on matrix multiplication in a different context relevant later?
  - **Purpose:** Introduce matrix multiplication, even if its direct application to filtering here is simplified. It's foundational for linear layers.
- **Pixel Aggregations (Average Color):** ([`06_aggregation_functions.py`](./06_aggregation_functions.py))
  - **Pixel Idea:** Create an image tensor. Calculate the overall average pixel value (`.mean()`) or the average value per color channel (`.mean(dim=[0, 1])`).
  - **Purpose:** Demonstrate calculating summary statistics from pixel data.
- **(Optional) In-place Pixel Changes:** ([`07_optional_inplace_ops.py`](./07_optional_inplace_ops.py))
  - **Pixel Idea:** Create an image tensor. Use `image[:, :, 0].add_(50)` to add 50 to the red channel directly.
  - **Purpose:** Introduce in-place operations for direct modification, noting potential side effects.
