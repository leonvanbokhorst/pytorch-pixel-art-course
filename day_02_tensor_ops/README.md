# Day 2: Tensor Operations and Manipulation

**Topics:**

- Tensor Attributes: `shape`, `dtype`, `device`.
- Indexing and Slicing: Accessing elements and sub-tensors (similar to NumPy).
- Reshaping: Changing tensor dimensions (`view`, `reshape`).
- Broadcasting: How PyTorch handles operations on tensors with different shapes.
- Basic Arithmetic: Element-wise operations (`+`, `-`, `*`, `/`).
- Mathematical Functions: Using `torch` functions (`torch.sum`, `torch.mean`, etc.).
- In-place Operations: Modifying tensors directly (e.g., `add_`).

**Focus:** Getting comfortable with common ways to manipulate and compute with tensors.

## Key Resources

- **PyTorch Official Tutorials - Tensor Operations:** [https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html#operations-on-tensors](https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html#operations-on-tensors) (Section within the Tensor Quickstart covering indexing, slicing, reshaping, arithmetic, broadcasting)
- **PyTorch `torch` Module Documentation:** [https://pytorch.org/docs/stable/torch.html](https://pytorch.org/docs/stable/torch.html) (Comprehensive list of all tensor operations like `matmul`, `sum`, `mean`, etc.)
- **Broadcasting Semantics:** [https://pytorch.org/docs/stable/notes/broadcasting.html](https://pytorch.org/docs/stable/notes/broadcasting.html) (Detailed explanation of broadcasting rules)

## Hands-On Examples

- **Indexing and Slicing:** ([`01_indexing_slicing.py`](./01_indexing_slicing.py))
  - **Code Idea:** Create a 2D tensor (e.g., `torch.tensor([[1, 2, 3], [4, 5, 6]])`). Show how to get the first row (`tensor[0]`), the second column (`tensor[:, 1]`), and a specific element (`tensor[1, 2]`).
  - **Purpose:** Demonstrate NumPy-like indexing and slicing syntax to access parts of a tensor.
- **Reshaping Tensors:** ([`02_reshaping_tensors.py`](./02_reshaping_tensors.py))
  - **Code Idea:** Create a tensor (e.g., `torch.arange(6)` which gives `[0, 1, 2, 3, 4, 5]`). Use `.view(2, 3)` or `.reshape(2, 3)` to change its shape to 2x3. Print the original and reshaped tensors.
  - **Purpose:** Illustrate how to change a tensor's shape without changing its data. Briefly mention `view` vs `reshape`.
- **Tensor Arithmetic:** ([`03_tensor_arithmetic.py`](./03_tensor_arithmetic.py))
  - **Code Idea:** Create two tensors of the same shape (e.g., `torch.tensor([[1, 1], [2, 2]])` and `torch.tensor([[3, 3], [4, 4]])`). Perform element-wise addition (`+`) and multiplication (`*`).
  - **Purpose:** Show basic element-wise arithmetic operations.
- **Broadcasting:** ([`04_broadcasting.py`](./04_broadcasting.py))
  - **Code Idea:** Create a tensor `A = torch.tensor([[1], [2], [3]])` (shape 3x1) and `B = torch.tensor([10, 20])` (shape 2). Try adding them (`A + B`). Observe the result (should broadcast to 3x2).
  - **Purpose:** Demonstrate how PyTorch automatically expands tensor dimensions during operations under certain rules.
- **Matrix Multiplication:** ([`05_matrix_multiplication.py`](./05_matrix_multiplication.py))
  - **Code Idea:** Create two compatible matrices (e.g., `A = torch.randn(2, 3)` and `B = torch.randn(3, 2)`). Perform matrix multiplication using `torch.matmul(A, B)` or `A @ B`. Check the resulting shape (should be 2x2).
  - **Purpose:** Show how to perform standard matrix multiplication, essential for neural networks.
- **Aggregation Functions:** ([`06_aggregation_functions.py`](./06_aggregation_functions.py))
  - **Code Idea:** Create a tensor (e.g., `torch.tensor([1., 2., 3., 4.])`). Calculate its sum (`.sum()`) and mean (`.mean()`).
  - **Purpose:** Demonstrate common aggregation functions.
- **(Optional) In-place Operations:** ([`07_optional_inplace_ops.py`](./07_optional_inplace_ops.py))
  - **Code Idea:** Create a tensor `x = torch.tensor([1., 2.])`. Show `x.add_(5)` and print `x` afterward to see it was modified directly.
  - **Purpose:** Introduce in-place operations and note they modify the tensor directly (use with caution, especially with autograd later).
