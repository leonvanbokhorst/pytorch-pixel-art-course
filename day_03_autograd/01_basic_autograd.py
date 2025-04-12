import torch

# Create a tensor and specify requires_grad=True to track computation
x = torch.tensor([2.0], requires_grad=True)
print(f"Tensor x: {x}")

# Define a function involving x
# y = x^2 + 3x + 1
y = x**2 + 3 * x + 1
print(f"Tensor y = x**2 + 3*x + 1: {y}")

# --- Compute Gradients --- #
# When you finish your computation you can call .backward() and have all the gradients computed automatically.
# The gradient for this tensor will be accumulated into .grad attribute.
# It needs to be a scalar output or have a gradient passed to backward().
print(f"\nCalling y.backward()...")
y.backward()

# --- Check the Gradient --- #
# y = x^2 + 3x + 1
# dy/dx = 2x + 3
# At x = 2.0, dy/dx = 2*(2.0) + 3 = 4 + 3 = 7
gradient = x.grad
print(f"Gradient of y with respect to x (dy/dx) at x=2.0: {gradient}")

# Let's try another one
z = torch.tensor([3.0], requires_grad=True)
w = z**3  # w = z^3 -> dw/dz = 3z^2
w.backward()  # dw/dz at z=3 is 3*(3^2) = 27
print(f"\nTensor z: {z}")
print(f"Tensor w = z**3: {w}")
print(f"Gradient of w with respect to z (dw/dz) at z=3.0: {z.grad}")
