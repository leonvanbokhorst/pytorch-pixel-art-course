import torch

# Create multiple tensors that require gradients
a = torch.tensor([2.0], requires_grad=True)
b = torch.tensor([3.0], requires_grad=True)
print(f"Tensor a: {a}")
print(f"Tensor b: {b}")

# Define a function involving both a and b
# c = a^2 * b
c = a**2 * b
print(f"Tensor c = a**2 * b: {c}")

# --- Compute Gradients --- #
print(f"\nCalling c.backward()...")
c.backward()

# --- Check the Gradients --- #
# c = a^2 * b
# Partial derivative dc/da = 2ab
# At a=2.0, b=3.0: dc/da = 2 * 2.0 * 3.0 = 12.0
grad_a = a.grad
print(f"Gradient of c with respect to a (dc/da): {grad_a}")

# Partial derivative dc/db = a^2
# At a=2.0, b=3.0: dc/db = (2.0)^2 = 4.0
grad_b = b.grad
print(f"Gradient of c with respect to b (dc/db): {grad_b}")
