"""Dataset factory utility.

This module provides an helper for building training, validation, and
testing data loaders from a Hydra configuration object. It supports the three
tasks reported in the paper: fixed-trigger Induction Head, Selective Copying and
MNIST (with a stratified train/validation split).
"""

import logging

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from datasets.ih import Seq2SeqDataset
from datasets.MNIST import DatasetMNIST
from datasets.selective_copying import SelectiveCopyingDataset
from utils.SeedManager import SeedManager, seed_data_worker
from utils.directories import (
    MNIST_TEST_DATA_PATH,
    MNIST_TEST_LABELS_PATH,
    MNIST_TRAIN_DATA_PATH,
    MNIST_TRAIN_LABELS_PATH,
)

logger = logging.getLogger(__name__)


def get_data_loaders(cfg) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return training, validation, and testing data loaders based on configuration.

    This function loads the specified dataset defined in the configuration object
    and splits it into training, validation and testing subsets using a stratified
    approach via scikit-learn's `train_test_split`. The stratification ensures
    that all subsets maintain the same class distribution as the original dataset.

    Parameters:
      cfg : DictConfig
        Configuration object containing dataset and training parameters.

    Returns:
      tuple[DataLoader, DataLoader, DataLoader]: Three data loaders for training,
        validation, and testing.

    Raises:
      NotImplementedError: If the dataset name in `cfg.dataset.name` is not recognized.
    """

    ds_cfg = cfg.dataset
    logger.info(f"Retrieving '{ds_cfg.name}'")


    if ds_cfg.name == "mnist":
        # Load the full training dataset
        dataset = DatasetMNIST(MNIST_TRAIN_DATA_PATH, MNIST_TRAIN_LABELS_PATH)

        # Perform stratified split using scikit-learn train_test_split
        indices = np.arange(len(dataset))
        train_idx, val_idx = train_test_split(
            indices,
            test_size=ds_cfg.splitting_proportion,
            stratify=dataset.labels,
            random_state=SeedManager.dataset_sklearn_random_state(ds_cfg.splitting_seed),
        )

        # Create training, validation and testing subsets
        train_dataset = Subset(dataset, train_idx)
        validation_dataset = Subset(dataset, val_idx)
        test_dataset = DatasetMNIST(MNIST_TEST_DATA_PATH, MNIST_TEST_LABELS_PATH)

    elif ds_cfg.name == "ih":
        # Initialize Seq2Seq datasets
        train_dataset = Seq2SeqDataset(
            vocabulary_size=ds_cfg.vocab_size,
            sequence_length=ds_cfg.sequence_length,
            sequence_trigger_length=ds_cfg.sequence_trigger_length,
            sequence_noise_length=ds_cfg.sequence_noise_length,
            sequence_target_length=ds_cfg.sequence_target_length,
            padding_idx=ds_cfg.padding_idx,
            batches_per_epoch=ds_cfg.batches_per_epoch,
            batch_size=cfg.training_batch_size,
            ds_cfg=ds_cfg,
        )
        validation_dataset = Seq2SeqDataset.from_dataset(train_dataset)
        test_dataset = Seq2SeqDataset.from_dataset(train_dataset)

    elif ds_cfg.name == "selective_copying":
        train_dataset = SelectiveCopyingDataset(
            n_tokens=ds_cfg.n_tokens,
            sequence_noise_length=ds_cfg.sequence_noise_length,
            sequence_target_length=ds_cfg.sequence_target_length,
            batches_per_epoch=ds_cfg.batches_per_epoch,
            batch_size=cfg.training_batch_size,
        )
        validation_dataset = SelectiveCopyingDataset.from_dataset(train_dataset)
        test_dataset = SelectiveCopyingDataset.from_dataset(train_dataset)

    else:
        raise NotImplementedError(f"Dataset '{ds_cfg.name}' is not implemented yet.")

    # Seed loader shuffling and worker-local dataset RNG streams independently per split.
    loader_generators = (
        [SeedManager.make_data_loader_generator(cfg.data_seed + offset) for offset in range(3)]
        if cfg.data_seed is not None
        else [SeedManager.make_data_loader_generator(None) for _ in range(3)]
    )

    # Initialize PyTorch DataLoaders for all splits
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training_batch_size,
        num_workers=cfg.num_workers,
        shuffle=ds_cfg.shuffle,
        pin_memory=True,
        worker_init_fn=seed_data_worker,
        generator=loader_generators[0],
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=cfg.validation_batch_size,
        num_workers=cfg.num_workers,
        shuffle=ds_cfg.shuffle,
        pin_memory=True,
        worker_init_fn=seed_data_worker,
        generator=loader_generators[1],
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.test_batch_size,
        num_workers=cfg.num_workers,
        shuffle=ds_cfg.shuffle,
        pin_memory=True,
        worker_init_fn=seed_data_worker,
        generator=loader_generators[2],
    )

    return train_loader, validation_loader, test_loader
