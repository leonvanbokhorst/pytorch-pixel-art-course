# Day 1: Introduction to PyTorch and Tensors

**Topics:**

- What is PyTorch? (Tensor library, Autograd, NN library)
- Tensors: The fundamental data structure.
- PyTorch vs NumPy: Similarities and key differences (GPU support, autograd).
- Tensor Basics:
  - Creating tensors (`torch.tensor`)
  - Understanding rank/dimensions (scalar, vector, matrix)
  - Tensor shape (`.shape`)
  - Tensor data type (`.dtype`)
- Device Awareness: Introduction to CPU vs GPU (briefly).

**Focus:** Understanding the core concepts of PyTorch and its primary data structure, the tensor.

## Key Resources

- **PyTorch Official Tutorials - Tensors:** [https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html](https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html) (Covers creation, attributes, NumPy bridge, device)
- **`torch.tensor` Documentation:** [https://pytorch.org/docs/stable/generated/torch.tensor.html](https://pytorch.org/docs/stable/generated/torch.tensor.html) (Detailed API for the main tensor creation function)

## Hands-On Examples

- **Creating Basic Tensors:** ([`01_creating_basic_tensors.py`](./01_creating_basic_tensors.py))
  - **Code Idea:** Show `torch.tensor(7)` for a scalar, `torch.tensor([1, 2, 3])` for a vector, and `torch.tensor([[1, 2], [3, 4]])` for a matrix.
  - **Purpose:** Demonstrate the basic syntax for creating 0D, 1D, and 2D tensors.
- **Checking Tensor Attributes:** ([`02_checking_tensor_attributes.py`](./02_checking_tensor_attributes.py))
  - **Code Idea:** Create a tensor (e.g., the matrix above) and print its `.shape`, `.ndim` (number of dimensions), and `.dtype`.
  - **Purpose:** Show how to inspect the fundamental properties of a tensor.
- **Specifying Data Type:** ([`03_specifying_data_type.py`](./03_specifying_data_type.py))
  - **Code Idea:** Create a tensor with a specific `dtype`, e.g., `torch.tensor([1.0, 2.0], dtype=torch.float32)`. Print the `dtype` to confirm.
  - **Purpose:** Illustrate how to control the data type, contrasting with default type inference.
- **(Optional) Tensor from NumPy:** ([`04_optional_tensor_from_numpy.py`](./04_optional_tensor_from_numpy.py))
  - **Code Idea:** Create a NumPy array and convert it to a PyTorch tensor using `torch.from_numpy()`.
  - **Purpose:** Show the interoperability with NumPy, reinforcing the similarities and differences.
