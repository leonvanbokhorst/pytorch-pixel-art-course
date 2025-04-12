# Day 3: Automatic Differentiation with Autograd

**Topics:**

- Concept: How PyTorch tracks operations using a computational graph.
- `requires_grad`: Setting this attribute to `True` on tensors to track history for gradients.
- `backward()`: Calling this on a scalar output (like loss) to compute gradients.
- `.grad` Attribute: Accessing the computed gradients for tensors with `requires_grad=True`.
- Gradient Accumulation: Understanding that gradients add up by default and the need for zeroing.
- Chain Rule: The underlying mathematical principle enabling autograd.
- Disabling Gradient Tracking: Using `torch.no_grad()` for inference or when gradients aren't needed.

**Focus:** Understanding PyTorch's automatic differentiation mechanism, which is crucial for training neural networks.

## Key Resources

- **PyTorch Official Tutorials - Autograd:** [https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html](https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html) (Covers the basics of automatic differentiation, `requires_grad`, `backward()`, and disabling tracking)
- **`torch.autograd` Documentation:** [https://pytorch.org/docs/stable/autograd.html](https://pytorch.org/docs/stable/autograd.html) (Detailed API for the autograd engine)
- **Autograd Mechanics (Advanced):** [https://pytorch.org/docs/stable/notes/autograd.html](https://pytorch.org/docs/stable/notes/autograd.html) (Deeper dive into how autograd works internally)
- **`torch.no_grad` Documentation:** [https://pytorch.org/docs/stable/generated/torch.no_grad.html](https://pytorch.org/docs/stable/generated/torch.no_grad.html) (API for the context manager to disable gradient calculation)

## Hands-On Examples

- **Basic Autograd:** ([`01_basic_autograd.py`](./01_basic_autograd.py))
  - **Code Idea:** Create a tensor `x = torch.tensor([2.0], requires_grad=True)`. Define `y = x**2 + 3*x + 1`. Call `y.backward()`. Print `x.grad`.
  - **Purpose:** Demonstrate the core autograd workflow: set `requires_grad`, perform operations, call `backward()`, check the resulting gradient. Verify the gradient matches manual calculation (dy/dx = 2x + 3 = 2\*2 + 3 = 7).
- **Autograd with Multiple Variables:** ([`02_autograd_multiple_variables.py`](./02_autograd_multiple_variables.py))
  - **Code Idea:** Create `a = torch.tensor([2.0], requires_grad=True)` and `b = torch.tensor([3.0], requires_grad=True)`. Define `c = a**2 * b`. Call `c.backward()`. Print `a.grad` and `b.grad`.
  - **Purpose:** Show that autograd computes gradients with respect to multiple input variables involved in the computation. Verify gradients (dc/da = 2ab = 12, dc/db = a^2 = 4).
- **Gradient Accumulation:** ([`03_gradient_accumulation.py`](./03_gradient_accumulation.py))
  - **Code Idea:** Using the `x` and `y` from the first example: Call `y.backward()` once, print `x.grad`. Call `y.backward()` again _without zeroing gradients_, print `x.grad` again. Then, zero the grad with `x.grad.zero_()` and call `y.backward()` again, print `x.grad`.
  - **Purpose:** Clearly demonstrate that gradients accumulate unless explicitly zeroed using `.grad.zero_()`. This highlights the need for `optimizer.zero_grad()` in training loops.
- **Disabling Gradients with `torch.no_grad()`:** ([`04_disabling_gradients.py`](./04_disabling_gradients.py))
  - **Code Idea:** Create `x = torch.tensor([2.0], requires_grad=True)`. Define `y = x * 2`. Print `y.requires_grad`. Then, using `with torch.no_grad():`, define `z = x * 3`. Print `z.requires_grad`.
  - **Purpose:** Show how the `torch.no_grad()` context manager prevents gradient tracking for operations within its scope, useful for evaluation/inference.
- **(Optional) `grad_fn`:** ([`05_optional_grad_fn.py`](./05_optional_grad_fn.py))
  - **Code Idea:** Using `x` and `y` from the first example, print `y.grad_fn`.
  - **Purpose:** Show the `grad_fn` attribute, which points to the function that created the tensor as part of the backward graph (helps visualize the computation graph conceptually).
