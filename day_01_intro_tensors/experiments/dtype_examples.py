import torch


def main():
    print("🎮 PyTorch Data Type Experiments 🎮\n")

    # 1. Loading Raw Image Data (uint8)
    print("1️⃣ Loading Raw Image Data (uint8)")
    # Simulate loading a 2x2 RGB image
    raw_image = torch.tensor(
        [
            [[255, 0, 0], [0, 255, 0]],  # Red, Green
            [[0, 0, 255], [255, 255, 255]],  # Blue, White
        ],
        dtype=torch.uint8,
    )
    print(
        f"Raw Image (uint8):\n{raw_image}\nShape: {raw_image.shape}\nDtype: {raw_image.dtype}\n"
    )

    # 2. Normalizing to float32 (Deep Learning Standard)
    print("2️⃣ Normalizing to float32 (Deep Learning Standard)")
    # Convert to float32 and normalize to [0, 1]
    normalized_image = raw_image.to(torch.float32) / 255.0
    print(
        f"Normalized Image (float32):\n{normalized_image}\nShape: {normalized_image.shape}\nDtype: {normalized_image.dtype}\n"
    )

    # 3. Creating a Binary Mask (bool)
    print("3️⃣ Creating a Binary Mask (bool)")
    # Create a mask where red channel > 128
    mask = raw_image[:, :, 0] > 128
    print(f"Binary Mask (bool):\n{mask}\nShape: {mask.shape}\nDtype: {mask.dtype}\n")

    # 4. Neural Network Weights (float32)
    print("4️⃣ Neural Network Weights (float32)")
    # Simulate a small weight matrix
    weights = torch.randn(3, 3, dtype=torch.float32)
    print(
        f"Neural Network Weights (float32):\n{weights}\nShape: {weights.shape}\nDtype: {weights.dtype}\n"
    )

    # 5. Memory Comparison
    print("5️⃣ Memory Usage Comparison")
    uint8_tensor = torch.zeros(1000, 1000, dtype=torch.uint8)
    float32_tensor = torch.zeros(1000, 1000, dtype=torch.float32)
    print("Memory usage for 1000x1000 tensor:")
    print(f"uint8: {uint8_tensor.element_size() * uint8_tensor.nelement()} bytes")
    print(f"float32: {float32_tensor.element_size() * float32_tensor.nelement()} bytes")
    print(
        f"float32 uses {float32_tensor.element_size() / uint8_tensor.element_size()}x more memory!"
    )


if __name__ == "__main__":
    main()
