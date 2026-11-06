"""Preprocess the MNIST dataset and save it as PyTorch 2D tensors.

Specifically, it:
1. Downloads the dataset using the torchvision library.
2. Unpickles the original binary files.
3. Converts the images into PyTorch tensors.
4. Preserves the 28x28 spatial dimensions of the images (no flattening).
5. Splits the dataset into training and test sets.
6. Saves the resulting tensors (.pt files) for later use in PyTorch models.
"""

from pathlib import Path

import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import MNIST

from utils.directories import (
    MNIST_DIR,
    MNIST_TEST_DATA_PATH,
    MNIST_TEST_LABELS_PATH,
    MNIST_TRAIN_DATA_PATH,
    MNIST_TRAIN_LABELS_PATH,
)

# Train/Test Split
TRAIN_CUTOFF = 60000


def preprocess(
    TrainDataset: MNIST, ValDataset: MNIST
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Process the MNIST training and validation datasets into 2D tensors.

    This function iterates over the provided datasets, removes the channel
    dimension while preserving the 28x28 spatial structure, and converts the
    labels to unsigned 8-bit integers.

    Parameters
    ----------
    TrainDataset : torchvision.datasets.MNIST
        The MNIST dataset object containing the training data.
    ValDataset : torchvision.datasets.MNIST
        The MNIST dataset object containing the validation/test data.

    Returns
    -------
    train_data : torch.Tensor
        Tensor containing the training images with shape [N, 28, 28].
    train_labels : torch.Tensor
        Tensor containing the training labels with shape [N].
    test_data : torch.Tensor
        Tensor containing the test images with shape [N, 28, 28].
    test_labels : torch.Tensor
        Tensor containing the test labels with shape [N].
    """
    # List to contain all images and labels
    all_images = []
    all_labels = []

    for k in range(len(TrainDataset)):
        img, label = TrainDataset[k]  # img shape: [1, 28, 28], label: int

        # Delete image first channel to keep shape [28, 28]
        img = img.squeeze(0)
        label_tensor = torch.tensor(label).to(torch.uint8)

        # Saving the 2D image tensors
        all_images.append(img)
        all_labels.append(label_tensor)

    # At this point the list contains all training images (60000)

    for k in range(len(ValDataset)):
        img, label = ValDataset[k]  # img shape: [1, 28, 28], label: int

        # Delete image first channel to keep shape [28, 28]
        img = img.squeeze(0)
        label_tensor = torch.tensor(label).to(torch.uint8)

        # Saving the newly created grayscale images
        all_images.append(img)
        all_labels.append(label_tensor)

    # At this point the list contains all training and test images (60000 + 10000 = 70000)

    # Creating a unique tensor
    all_images = torch.stack(all_images)
    all_labels = torch.stack(all_labels)

    # Separating training data and test data
    train_data = all_images[:TRAIN_CUTOFF, :, :]
    train_labels = all_labels[:TRAIN_CUTOFF]
    test_data = all_images[TRAIN_CUTOFF:, :, :]
    test_labels = all_labels[TRAIN_CUTOFF:]

    return train_data, train_labels, test_data, test_labels


def main() -> None:
    """Execute the dataset download, preprocessing, and saving pipeline."""
    # Created Transformations to apply to raw images
    trainTransform = transforms.Compose(
        [transforms.Lambda(lambda x: torch.from_numpy(np.array(x)))]
    )
    valTransform = transforms.Compose([transforms.Lambda(lambda x: torch.from_numpy(np.array(x)))])

    # Download the dataset and transform every image in a tensor
    TrainDataset = MNIST(root=MNIST_DIR, train=True, download=True, transform=trainTransform)
    ValDataset = MNIST(root=MNIST_DIR, train=False, download=True, transform=valTransform)

    # Preprocessing
    train_data, train_labels, test_data, test_labels = preprocess(TrainDataset, ValDataset)

    # Creating folder before saving
    MNIST_2D_path = Path(MNIST_TRAIN_DATA_PATH)
    MNIST_2D_path.parent.mkdir(parents=True, exist_ok=True)

    # Saving
    torch.save(obj=train_data, f=MNIST_TRAIN_DATA_PATH)
    torch.save(obj=train_labels, f=MNIST_TRAIN_LABELS_PATH)
    torch.save(obj=test_data, f=MNIST_TEST_DATA_PATH)
    torch.save(obj=test_labels, f=MNIST_TEST_LABELS_PATH)


if __name__ == "__main__":
    main()
