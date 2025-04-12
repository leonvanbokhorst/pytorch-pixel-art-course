import torch

# Create two matrices compatible for multiplication
# Matrix A: shape (2, 3)
# Matrix B: shape (3, 2)
# Result C = A @ B will have shape (2, 2)

# Use random data for variety
torch.manual_seed(42)  # for reproducibility
A = torch.randn(2, 3)
B = torch.randn(3, 2)

print(f"Matrix A (2x3):\n{A}")
print(f"Matrix B (3x2):\n{B}")

# --- Using torch.matmul --- #
C_matmul = torch.matmul(A, B)
print(f"\nResult using torch.matmul(A, B) (2x2):\n{C_matmul}")

# --- Using the @ operator --- #
# This is Python's dedicated infix operator for matrix multiplication
C_operator = A @ B
print(f"\nResult using A @ B (2x2):\n{C_operator}")

# Check if results are the same
print(f"\nAre results equal? {torch.allclose(C_matmul, C_operator)}")

# --- Incompatible shapes --- #
# Trying to multiply incompatible matrices will raise an error
A_wrong = torch.randn(2, 3)
B_wrong = torch.randn(2, 2)  # Incorrect inner dimension
# try:
#     C_wrong = A_wrong @ B_wrong
# except RuntimeError as e:
#     print(f"\nError multiplying shapes {A_wrong.shape} and {B_wrong.shape}: {e}")
