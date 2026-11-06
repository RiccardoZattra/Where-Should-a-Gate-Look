"""MNIST dataset loader.

This module defines the DatasetMNIST class for loading MNIST image and label
tensors stored in `.pt` files.
"""

import torch
from torch.utils.data import Dataset


class DatasetMNIST(Dataset):
    """Custom dataset class for loading MNIST tensors.

    This class inherits from `torch.utils.data.Dataset` that loads pre-saved
    MNIST tensors (.pt files) containing images and labels.
    It implements the `__len__` and `__getitem__` methods.and implements
    the `__len__` and `__getitem__` methods so it can be used directly
    with PyTorch `DataLoader`.

    Parameters:
      data : str
        Path to the `.pt` file containing image tensors.
      labels : str
        Path to the `.pt` file containing label tensors.

    Attributes:
      data : torch.Tensor
        Tensor containing all loaded image samples.
      labels : torch.Tensor
        Tensor containing all loaded label samples.

    Examples:
        dataset = DatasetMNIST("train_data.pt", "train_labels.pt")
        image, label = dataset[0]
    """

    def __init__(self, data: str, labels: str):
        super().__init__()

        self.data = torch.load(f=data)
        self.labels = torch.load(f=labels)

    def __len__(self) -> int:
        """Return the total number of samples in the dataset.

        Returns:
          int: Total number of examples in the dataset.
        """
        return self.data.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieve an image and label sample by index.

        Parameters:
          index : int
            Index of the desired sample.

        Returns:
          tuple[torch.Tensor, torch.Tensor]: A tuple containing the image tensor
            and the corresponding label tensor.
        """
        image = self.data[index, :]
        label = self.labels[index]

        return image, label
