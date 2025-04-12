import torch

# Create a tensor requiring gradient
x = torch.tensor([2.0], requires_grad=True)
print(f"Tensor x: {x}")

# --- First backward pass --- #
y = x**2  # y = x^2 -> dy/dx = 2x
print(f"\nCalling y.backward() for the first time...")
y.backward()  # Computes gradient dy/dx = 2*2 = 4
print(f"x.grad after first backward(): {x.grad}")

# --- Second backward pass (WITHOUT zeroing) --- #
# Create another computation (or use the same one)
z = x**3  # z = x^3 -> dz/dx = 3x^2
print(f"\nCalling z.backward() for the second time (no zeroing)...")
z.backward()  # Computes gradient dz/dx = 3*(2^2) = 12
# The new gradient (12) is ADDED to the existing gradient (4)
print(
    f"x.grad after second backward() (accumulated): {x.grad}"
)  # Should be 4 + 12 = 16

# --- Zeroing the gradient --- #
print(f"\nZeroing the gradient with x.grad.zero_()...")
if x.grad is not None:
    x.grad.zero_()
print(f"x.grad after zeroing: {x.grad}")

# --- Third backward pass (AFTER zeroing) --- #
w = x**2 + x  # w = x^2 + x -> dw/dx = 2x + 1
print(f"\nCalling w.backward() after zeroing...")
w.backward()  # Computes gradient dw/dx = 2*2 + 1 = 5
print(f"x.grad after third backward() (fresh gradient): {x.grad}")  # Should be 5

# --- Important: Optimizer zero_grad() --- #
print("\nIn a typical training loop, you call `optimizer.zero_grad()`")
print("at the start of each iteration to clear gradients from the previous step.")
