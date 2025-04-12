import torch

# Tensors created by the user have requires_grad=False by default
a = torch.tensor([1.0, 2.0])
print(f"Tensor a: {a}")
print(f"a.requires_grad: {a.requires_grad}")
print(f"a.grad_fn: {a.grad_fn}")  # No grad_fn because it wasn't created by an op

# Let's enable gradient tracking
x = torch.tensor([2.0], requires_grad=True)
print(f"\nTensor x: {x}")
print(f"x.requires_grad: {x.requires_grad}")
print(f"x.grad_fn: {x.grad_fn}")  # Still None, x is a leaf node created by user

# --- Operations create grad_fn --- #

# y = x + 3
y = x + 3
print(f"\ny = x + 3: {y}")
print(f"y.requires_grad: {y.requires_grad}")
# y has a grad_fn because it resulted from an operation (add) involving x
# The grad_fn points to the backward function for addition (AddBackward0)
print(f"y.grad_fn: {y.grad_fn}")

# z = y * y
z = y * y
print(f"\nz = y * y: {z}")
print(f"z.requires_grad: {z.requires_grad}")
# z's grad_fn points to the backward function for multiplication (MulBackward0)
print(f"z.grad_fn: {z.grad_fn}")

# You can follow the chain back
print(f"z.grad_fn.next_functions: {z.grad_fn.next_functions}")
# Shows the function(s) that created the input(s) to MulBackward0, which is y's AddBackward0

# w = z.mean()
w = z.mean()
print(f"\nw = z.mean(): {w}")
print(f"w.requires_grad: {w.requires_grad}")
# w's grad_fn points to MeanBackward0
print(f"w.grad_fn: {w.grad_fn}")

# --- Tensors not requiring grad --- #

b = torch.tensor([4.0])  # requires_grad is False
c = x * b  # x requires grad, b does not
print(f"\nc = x * b: {c}")
# c requires grad because x did
print(f"c.requires_grad: {c.requires_grad}")
print(f"c.grad_fn: {c.grad_fn}")  # MulBackward0

# --- Using torch.no_grad() --- #
with torch.no_grad():
    d = x / 2
    print(f"\nInside no_grad: d = x / 2: {d}")
    print(f"Inside no_grad: d.requires_grad: {d.requires_grad}")
    print(f"Inside no_grad: d.grad_fn: {d.grad_fn}")  # None

print("\nThe `grad_fn` attribute allows PyTorch to build the backward graph.")
print("It links each tensor to the function that created it.")
