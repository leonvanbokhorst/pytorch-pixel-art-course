import torch

# Create a 6x9 RGBA frame (like a tiny movie frame or game sprite)
# Shape: [Height, Width, Channels]
frame = torch.tensor(
    [
        # Row 0: Top border - semi-transparent red
        [
            [255, 0, 0, 128],
            [255, 0, 0, 128],
            [255, 0, 0, 128],
            [255, 0, 0, 128],
            [255, 0, 0, 128],
            [255, 0, 0, 128],
            [255, 0, 0, 128],
            [255, 0, 0, 128],
            [255, 0, 0, 128],
        ],
        # Row 1: Left and right borders - opaque green
        [
            [0, 255, 0, 255],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 255, 0, 255],
        ],
        # Row 2: Left and right borders - opaque green
        [
            [0, 255, 0, 255],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 255, 0, 255],
        ],
        # Row 3: Left and right borders - opaque green
        [
            [0, 255, 0, 255],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 255, 0, 255],
        ],
        # Row 4: Left and right borders - opaque green
        [
            [0, 255, 0, 255],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 255, 0, 255],
        ],
        # Row 5: Bottom border - semi-transparent blue
        [
            [0, 0, 255, 128],
            [0, 0, 255, 128],
            [0, 0, 255, 128],
            [0, 0, 255, 128],
            [0, 0, 255, 128],
            [0, 0, 255, 128],
            [0, 0, 255, 128],
            [0, 0, 255, 128],
            [0, 0, 255, 128],
        ],
    ],
    dtype=torch.uint8,
)

print(f"Frame Shape: {frame.shape}")

# Get the top-left corner pixel
top_left = frame[0, 0]
print(f"\nTop-left pixel: {top_left}")

# Get the middle-right pixel
middle_right = frame[2, 8]
print(f"\nMiddle-right pixel: {middle_right}")

# Get the bottom row
bottom_row = frame[5]
print(f"\nBottom row shape: {bottom_row.shape}")

# Get just the alpha channel of the whole frame
alpha_channel = frame[:, :, 3]
print(f"\nAlpha channel shape: {alpha_channel.shape}")

# Find all transparent pixels (where alpha = 0)
transparent_pixels = frame[:, :, 3] == 0  # Creates a boolean mask
print(f"\nTransparent pixels mask:\n{transparent_pixels}")

# Count how many transparent pixels we have
num_transparent = transparent_pixels.sum()
print(f"\nNumber of transparent pixels: {num_transparent}")

# Get the coordinates of all transparent pixels
transparent_coords = torch.where(transparent_pixels)
print(f"\nCoordinates of transparent pixels (row, col):")
for row, col in zip(transparent_coords[0], transparent_coords[1]):
    print(f"Pixel at ({row}, {col}) is transparent")

# Find all fully opaque red pixels
red_pixels = (frame[:, :, 0] == 255) & (frame[:, :, 1] == 0) & (frame[:, :, 2] == 0)
opaque_pixels = frame[:, :, 3] == 255
red_opaque_pixels = red_pixels & opaque_pixels

print(f"\nFully opaque red pixels mask:\n{red_opaque_pixels}")
print(f"Number of fully opaque red pixels: {red_opaque_pixels.sum()}")

# Get the coordinates of all fully opaque red pixels
red_opaque_coords = torch.where(red_opaque_pixels)
print(f"\nCoordinates of fully opaque red pixels (row, col):")
for row, col in zip(red_opaque_coords[0], red_opaque_coords[1]):
    print(f"Pixel at ({row}, {col}) is fully opaque red")

# Find all fully opaque green pixels
bright_green_pixies = (
    (frame[:, :, 0] == 0) & (frame[:, :, 1] == 255) & (frame[:, :, 2] == 0)
)
green_opaqies = bright_green_pixies & (frame[:, :, 3] == 255)

print(f"\nFully opaque green pixels mask:\n{green_opaqies}")
print(f"Number of fully opaque green pixels: {green_opaqies.sum()}")

# Get the coordinates of all fully opaque green pixels
green_opaque_coords = torch.where(green_opaqies)
print(f"\nCoordinates of fully opaque green pixels (row, col):")
for row, col in zip(green_opaque_coords[0], green_opaque_coords[1]):
    print(f"Pixel at ({row}, {col}) is fully opaque green")

# Find pixels that are either red OR green
red_or_green_pixeloties = (frame[:, :, 0] == 255) | (frame[:, :, 1] == 255)
print(f"\nRed or Green pixels mask:\n{red_or_green_pixeloties}")
print(f"Number of red or green pixels: {red_or_green_pixeloties.sum()}")

# Get the coordinates of all red or green pixels
red_green_coords = torch.where(red_or_green_pixeloties)
print(f"\nCoordinates of red or green pixels (row, col):")
for row, col in zip(red_green_coords[0], red_green_coords[1]):
    print(f"Pixel at ({row}, {col}) is either red or green")
