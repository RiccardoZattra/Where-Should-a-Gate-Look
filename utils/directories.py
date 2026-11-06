from pathlib import Path

ROOT_DIR = Path()
print(f"Executing from '{ROOT_DIR.absolute()}'")

DATA_DIR = ROOT_DIR / "data"

# MNIST Directories
MNIST_DIR = DATA_DIR / "MNIST"

# Paths for MNIST 2D
MNIST_TEST_DATA_PATH = MNIST_DIR / "2D" / "test_data.pt"
MNIST_TEST_LABELS_PATH = MNIST_DIR / "2D" / "test_labels.pt"
MNIST_TRAIN_DATA_PATH = MNIST_DIR / "2D" / "train_data.pt"
MNIST_TRAIN_LABELS_PATH = MNIST_DIR / "2D" / "train_labels.pt"
