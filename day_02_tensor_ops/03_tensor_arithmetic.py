import torch

# Create two tensors of the same shape
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[10, 20], [30, 40]])

print(f"Tensor a:\n{a}")
print(f"Tensor b:\n{b}")

# --- Element-wise operations --- #

# Addition
addition = a + b
# alternative: torch.add(a, b)
print(f"\nAddition (a + b):\n{addition}")

# Subtraction
subtraction = b - a
# alternative: torch.sub(b, a)
print(f"\nSubtraction (b - a):\n{subtraction}")

# Multiplication (element-wise, Hadamard product)
multiplication = a * b
# alternative: torch.mul(a, b)
print(f"\nElement-wise Multiplication (a * b):\n{multiplication}")

# Division
division = b / a  # Be careful about division by zero if using integers!
# alternative: torch.div(b, a)
print(f"\nDivision (b / a):\n{division}")

# Exponentiation
exponentiation = a**2
# alternative: torch.pow(a, 2)
print(f"\nExponentiation (a ** 2):\n{exponentiation}")
