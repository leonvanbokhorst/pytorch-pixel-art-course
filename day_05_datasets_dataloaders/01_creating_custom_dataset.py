import torch
from torch.utils.data import Dataset


# Define a custom Dataset class
class SimpleTensorDataset(Dataset):
    """A simple dataset wrapping features and labels (random tensors)."""

    def __init__(self, num_samples=100, feature_dim=10, label_dim=1):
        """Initialize the dataset.

        Args:
            num_samples (int): Number of data points.
            feature_dim (int): Dimension of the feature vectors.
            label_dim (int): Dimension of the label vectors (e.g., 1 for regression).
        """
        super().__init__()  # Not strictly necessary for Dataset, but good practice
        self.num_samples = num_samples

        # Generate some random data for features and labels
        # In a real scenario, you would load data from files or other sources here.
        print(f"Generating {num_samples} random samples...")
        self.features = torch.randn(num_samples, feature_dim)
        # Example: labels are derived from features (e.g., sum + noise)
        # self.labels = self.features.sum(dim=1, keepdim=True) + torch.randn(num_samples, label_dim) * 0.1
        self.labels = torch.randint(
            0, 2, (num_samples, label_dim)
        ).float()  # Example: binary labels

        print(f" - Features shape: {self.features.shape}")
        print(f" - Labels shape: {self.labels.shape}")

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        # This method is required by DataLoader to know the size of the dataset.
        return self.num_samples

    def __getitem__(self, idx):
        """Returns a single sample (feature and label) at the given index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: (feature_tensor, label_tensor) for the requested sample.
        """
        # This method is required by DataLoader to fetch individual data points.
        if not 0 <= idx < self.num_samples:
            raise IndexError(
                f"Index {idx} is out of bounds for dataset size {self.num_samples}"
            )

        feature = self.features[idx]
        label = self.labels[idx]
        return feature, label


# --- Example Usage --- #
if __name__ == "__main__":
    print("Demonstrating the Custom Dataset:")

    # Instantiate the dataset
    dataset = SimpleTensorDataset(num_samples=50, feature_dim=5, label_dim=1)
    print(f"\nDataset created. Total length: {len(dataset)}")

    # Get the first sample
    first_idx = 0
    feature_0, label_0 = dataset[first_idx]  # Calls __getitem__(0)
    print(f"\nSample at index {first_idx}:")
    print(f"  Feature shape: {feature_0.shape}, Label shape: {label_0.shape}")
    print(f"  Feature: {feature_0}")
    print(f"  Label: {label_0}")

    # Get the last sample
    last_idx = len(dataset) - 1
    feature_last, label_last = dataset[last_idx]  # Calls __getitem__(last_idx)
    print(f"\nSample at index {last_idx}:")
    print(f"  Feature: {feature_last}")
    print(f"  Label: {label_last}")

    # Try accessing an invalid index
    try:
        invalid_idx = len(dataset)
        _ = dataset[invalid_idx]
    except IndexError as e:
        print(f"\nSuccessfully caught error for invalid index {invalid_idx}: {e}") # type: ignore

    print("\nThis file defines and demonstrates the SimpleTensorDataset class.")
