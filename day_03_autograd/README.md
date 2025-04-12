# Day 3: Autograd - Learning Pixel Patterns Automatically

**Topics:**

- Concept: How PyTorch tracks operations on pixel data to figure out how to improve.
- `requires_grad`: Telling PyTorch which tensors (e.g., model parameters, maybe even input pixels if generating) need gradients calculated.
- `backward()`: Kicking off the gradient calculation, usually starting from a measure of error (loss) like "how different are the generated pixels from the target?".
- `.grad` Attribute: Seeing the calculated gradients – how much changing each value would affect the final result (e.g., how much changing a weight improves the generated sprite).
- Gradient Accumulation: Why gradients add up, and why we need to reset them (using `optimizer.zero_grad()` later) in each learning step.
- Chain Rule for Pixels: The math behind how PyTorch calculates gradients through multiple pixel processing steps.
- Disabling Gradient Tracking (`torch.no_grad()`): Turning off tracking when just generating pixels (inference) or not actively learning.

**Focus:** Understanding PyTorch's automatic differentiation (Autograd) engine, the core mechanism that enables models to learn from pixel art data.

## Key Resources

- **PyTorch Official Tutorials - Autograd:** [https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html](https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)
- **`torch.autograd` Documentation:** [https://pytorch.org/docs/stable/autograd.html](https://pytorch.org/docs/stable/autograd.html)
- **Autograd Mechanics (Advanced):** [https://pytorch.org/docs/stable/notes/autograd.html](https://pytorch.org/docs/stable/notes/autograd.html)
- **`torch.no_grad` Documentation:** [https://pytorch.org/docs/stable/generated/torch.no_grad.html](https://pytorch.org/docs/stable/generated/torch.no_grad.html)

## Hands-On Gradient Examples

- **Basic Pixel Gradient:** ([`01_basic_autograd.py`](./01_basic_autograd.py))
  - **Pixel Idea:** Create a single pixel value `p = torch.tensor([0.5], requires_grad=True)` (representing maybe a learned color channel). Define a simple operation like `brightness_penalty = (p - 0.8)**2` (penalty for being too far from target brightness 0.8). Call `brightness_penalty.backward()`. Print `p.grad`.
  - **Purpose:** Demonstrate the basic autograd flow in a pixel context: how does changing `p` affect the penalty?
- **Gradients for Color Mixing:** ([`02_autograd_multiple_variables.py`](./02_autograd_multiple_variables.py))
  - **Pixel Idea:** Create two learnable color components `c1 = torch.tensor([0.2], requires_grad=True)` and `c2 = torch.tensor([0.7], requires_grad=True)`. Define a combined color `mix = c1 * 0.6 + c2 * 0.4`. Define a simple target difference `loss = (mix - 0.5)**2`. Call `loss.backward()`. Print `c1.grad` and `c2.grad`.
  - **Purpose:** Show autograd calculating gradients for multiple inputs affecting a final pixel value or loss.
- **Accumulating Pixel Error Signals:** ([`03_gradient_accumulation.py`](./03_gradient_accumulation.py))
  - **Pixel Idea:** Use the setup from the first example. Call `brightness_penalty.backward()` multiple times without zeroing `p.grad`. Observe how the gradient accumulates. Then, show zeroing with `p.grad.zero_()` before calling `backward()` again.
  - **Purpose:** Clearly illustrate gradient accumulation and the necessity of zeroing gradients between learning steps when optimizing pixel generation or model parameters.
- **Disabling Gradients for Pixel Generation:** ([`04_disabling_gradients.py`](./04_disabling_gradients.py))
  - **Pixel Idea:** Create a tensor `latent = torch.randn(10, requires_grad=True)` (representing some input to a generator). Define a simple transformation `pixels = latent * 2`. Check `pixels.requires_grad`. Then, use `with torch.no_grad():` to define `final_pixels = pixels + 1`. Check `final_pixels.requires_grad` (should be False).
  - **Purpose:** Show how `torch.no_grad()` prevents tracking, useful when simply using a trained model to generate pixels without further learning.
- **(Optional) Tracing Pixel Operations (`grad_fn`):** ([`05_optional_grad_fn.py`](./05_optional_grad_fn.py))
  - **Pixel Idea:** Using `p` and `brightness_penalty` from the first example, print `brightness_penalty.grad_fn`.
  - **Purpose:** Show the `grad_fn` attribute to understand the history of operations that led to the final pixel value or loss, conceptually visualizing the computation graph.
