import torch
from torch.utils.data import Dataset

# Import transforms (even if not used heavily here, good practice)
import torchvision.transforms as transforms # type: ignore

# Note: torchvision is not a core dependency but often used for transforms.
# We'll use a simple lambda here, but show the structure.


# --- Dataset with Transform Option --- #
class SimpleTensorDatasetWithTransform(Dataset):
    """A simple dataset that can apply a transform to features."""

    def __init__(self, num_samples=100, feature_dim=10, transform=None):
        """Initialize the dataset.

        Args:
            num_samples (int): Number of data points.
            feature_dim (int): Dimension of the feature vectors.
            transform (callable, optional): A function/transform to apply to the features.
                                            Defaults to None.
        """
        self.num_samples = num_samples
        self.transform = transform

        # Generate random data (features and dummy labels)
        print(f"Generating {num_samples} random samples...")
        self.features = torch.randn(num_samples, feature_dim)
        self.labels = torch.randint(0, 2, (num_samples, 1)).float()
        print(f" - Features shape: {self.features.shape}")
        print(f" - Labels shape: {self.labels.shape}")
        if self.transform:
            print(f" - Transform provided: {self.transform}")
        else:
            print(" - No transform provided.")

    def __len__(self):
        """Returns the total number of samples."""
        return self.num_samples

    def __getitem__(self, idx):
        """Returns a single sample, applying the transform if it exists."""
        if not 0 <= idx < self.num_samples:
            raise IndexError(
                f"Index {idx} is out of bounds for dataset size {self.num_samples}"
            )

        feature = self.features[idx]
        label = self.labels[idx]

        # Apply the transform to the feature *before* returning
        if self.transform:
            feature = self.transform(feature)

        return feature, label


# --- Example Usage --- #
if __name__ == "__main__":
    print("--- Dataset WITHOUT Transform ---")
    dataset_no_transform = SimpleTensorDatasetWithTransform(
        num_samples=5, feature_dim=3
    )
    feature_0_orig, label_0_orig = dataset_no_transform[0]
    print(f"Sample 0 feature (original): {feature_0_orig}")
    print(f"Sample 0 label (original):   {label_0_orig}")

    print("\n--- Dataset WITH Simple Lambda Transform --- ")
    # Define a simple lambda function as a transform (adds 10 to features)
    add_10_transform = lambda x: x + 10

    dataset_with_transform = SimpleTensorDatasetWithTransform(
        num_samples=5, feature_dim=3, transform=add_10_transform
    )
    feature_0_transformed, label_0_transformed = dataset_with_transform[0]
    print(
        f"Sample 0 feature (original): {dataset_with_transform.features[0]} (Stored value)"
    )  # Show original
    print(
        f"Sample 0 feature (transformed): {feature_0_transformed} (Value after __getitem__)"
    )
    print(
        f"Sample 0 label (transformed):   {label_0_transformed} (Labels usually untransformed)"
    )

    # --- Mentioning torchvision.transforms --- #
    print("\n--- Note on torchvision.transforms --- ")
    print("For common image or tensor operations, use `torchvision.transforms`:")
    # Example (not applied here, needs data in correct format):
    # composed_transform = transforms.Compose([
    #     transforms.ToTensor(), # If data wasn't already tensors
    #     transforms.Normalize(mean=(0.5,), std=(0.5,)) # Example normalization
    # ])
    # print(f"Example composed transform: {composed_transform}")
    print("Transforms are powerful for preprocessing and data augmentation.")
