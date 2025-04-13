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

# --- Pixel Art Exponentiation Examples --- #
print("\n=== Pixel Art Exponentiation Examples ===")

# Create a simple 4x4 grayscale sprite (values 0-15)
sprite = torch.arange(16).reshape(4, 4)
print(f"\nOriginal Sprite (4x4):\n{sprite}")

# Square the values (power of 2)
squared = sprite**2
print(f"\nSquared Sprite (values^2):\n{squared}")

# Cube the values (power of 3)
cubed = sprite**3
print(f"\nCubed Sprite (values^3):\n{cubed}")

# Square root (power of 0.5)
sqrt = sprite**0.5
print(f"\nSquare Root Sprite (values^0.5):\n{sqrt}")

# Let's try with some actual pixel values (0-255)
pixel_sprite = torch.tensor(
    [[0, 64, 128, 192], [32, 96, 160, 224], [64, 128, 192, 255], [96, 160, 224, 255]]
)
print(f"\nOriginal Pixel Sprite (0-255):\n{pixel_sprite}")

# Square the pixel values
squared_pixels = pixel_sprite**2
print(f"\nSquared Pixel Sprite:\n{squared_pixels}")

# Notice how the values grow much larger when squared!
# This is why we often normalize or scale the results
# Let's try with normalized values (0-1)
normalized_sprite = pixel_sprite / 255.0
print(f"\nNormalized Sprite (0-1):\n{normalized_sprite}")

# Now square the normalized values
squared_normalized = normalized_sprite**2
print(f"\nSquared Normalized Sprite:\n{squared_normalized}")

# Convert back to 0-255 range
squared_pixels_scaled = (squared_normalized * 255).to(torch.uint8)
print(f"\nSquared and Scaled Back to 0-255:\n{squared_pixels_scaled}")

# Square root of vector (-1, 1)
random_vector = torch.randn(1, 8, dtype=torch.float32)
print(f"\nRandom Vector:\n{random_vector}")
sqrt_random_vector = (random_vector**2.0)**0.5
print(f"\nSquare Root of Random Vector:\n{sqrt_random_vector}")

# same as
sqrt_random_vector_2 = random_vector.pow(2.0).pow(0.5)
print(f"\nSquare Root of Random Vector 2:\n{sqrt_random_vector_2}")

# same as
sqrt_random_vector_3 = random_vector.abs()
print(f"\nSquare Root of Random Vector 3:\n{sqrt_random_vector_3}")

# same as
sqrt_random_vector_4 = random_vector.pow(2.0).sqrt()
print(f"\nSquare Root of Random Vector 4:\n{sqrt_random_vector_4}")

















