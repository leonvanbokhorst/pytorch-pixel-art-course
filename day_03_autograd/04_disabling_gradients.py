import torch

# Create a tensor requiring gradient
x = torch.tensor([2.0], requires_grad=True)
print(f"Tensor x: {x}, requires_grad: {x.requires_grad}")

# --- Operation WITH gradient tracking --- #
y = x * 2
print(f"\ny = x * 2: {y}")
# y was created from an operation involving a tensor that requires grad,
# so it also requires grad by default.
print(f"y.requires_grad: {y.requires_grad}")
print(f"y.grad_fn: {y.grad_fn}")  # Shows the function that created y (MulBackward0)

# --- Operation WITHOUT gradient tracking using torch.no_grad() --- #
print(f"\nEntering torch.no_grad() context...")
with torch.no_grad():
    # Operations inside this block will not be tracked
    z = x * 3
    print(f"  Inside context: z = x * 3: {z}")
    # z does not require grad because it was created inside the no_grad context
    print(f"  Inside context: z.requires_grad: {z.requires_grad}")
    print(f"  Inside context: z.grad_fn: {z.grad_fn}")  # grad_fn is None

    # You can still perform operations, they just won't be part of the graph
    w = y * z  # y requires grad, z does not
    print(f"  Inside context: w = y * z: {w}")
    # If any operand requires grad, the output might require grad *outside* the context
    # but *inside* the context, tracking is temporarily disabled.
    print(f"  Inside context: w.requires_grad: {w.requires_grad}")

print("Exited torch.no_grad() context.")

# Outside the context, z still doesn't require grad
print(f"\nOutside context: z.requires_grad: {z.requires_grad}")

# Can we compute gradients for y? Yes.
y.backward()  # Computes dy/dx = 2
print(f"x.grad after y.backward(): {x.grad}")

# Can we compute gradients for z? No.
try:
    # Need to zero grad first if we want to call backward again
    if x.grad is not None:
        x.grad.zero_()
    z.backward()  # This will fail because z does not require grad
except RuntimeError as e:
    print(f"\nError calling z.backward(): {e}")

print("\n`torch.no_grad()` is essential for model evaluation/inference")
print("to prevent unnecessary gradient computations and save memory.")
