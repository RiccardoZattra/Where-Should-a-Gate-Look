"""Centralized, independent random streams for data and transformer code."""

import os
import random
import secrets

import numpy as np
import torch


class SeedManager:
    """Singleton owning the dataset RNG stream and transformer seed setup."""

    _instance = None

    def __deepcopy__(self, memo):
        return self

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._dataset_torch_generator = torch.Generator()
            cls._instance._dataset_numpy_generator = np.random.default_rng()
            cls._instance._dataset_python_generator = random.Random()

            # 1. Obbliga PyTorch a usare algoritmi deterministici
            torch.use_deterministic_algorithms(True)

            # 2. Obbliga cuBLAS a essere deterministico (richiesto da PyTorch >= 1.8)
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

            # 3. Disabilita la ricerca euristica di cuDNN e forza algoritmi deterministici
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        return cls._instance

    def seed_dataset(self, seed: int | None) -> None:
        """Seed only the private generators used to create synthetic data."""
        self.dataset_seed = seed
        if seed is None:
            self._dataset_torch_generator = torch.Generator()
            self._dataset_torch_generator.manual_seed(secrets.randbits(63))
            self._dataset_numpy_generator = np.random.default_rng()
            self._dataset_python_generator = random.Random()
            return

        self._dataset_torch_generator.manual_seed(seed)
        self._dataset_numpy_generator = np.random.default_rng(seed)
        self._dataset_python_generator.seed(seed)

    def seed_transformer(self, seed: int | None) -> None:
        """Seed PyTorch immediately before transformer parameters are created."""
        if seed is None:
            seed = secrets.randbits(63)
        self.transformer_seed = seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def dataset_rand(self, *size, **kwargs) -> torch.Tensor:
        """Sample uniform values from the private dataset PyTorch generator."""
        return torch.rand(*size, generator=self._dataset_torch_generator, **kwargs)

    def dataset_randn(self, *size, **kwargs) -> torch.Tensor:
        """Sample normal values from the private dataset PyTorch generator."""
        return torch.randn(*size, generator=self._dataset_torch_generator, **kwargs)

    def dataset_randint(self, a: int, b: int) -> int:
        """Sample an integer from the private dataset Python generator."""
        return self._dataset_python_generator.randint(a, b)

    def dataset_choice(self, population):
        """Choose one item using the private dataset Python generator."""
        return self._dataset_python_generator.choice(population)

    def dataset_choices(self, population, k: int) -> list:
        """Sample items with replacement using the private dataset Python generator."""
        return self._dataset_python_generator.choices(population, k=k)

    def dataset_sample(self, population, k: int) -> list:
        """Sample items without replacement using the private dataset Python generator."""
        return self._dataset_python_generator.sample(population, k)

    def dataset_numpy_randn(self, *size) -> np.ndarray:
        """Sample normal values from the private dataset NumPy generator."""
        return self._dataset_numpy_generator.standard_normal(size)

    @staticmethod
    def dataset_sklearn_random_state(seed: int) -> np.random.RandomState:
        """Create a scikit-learn-compatible random state for dataset splitting."""
        return np.random.RandomState(seed)

    def dataset_multivariate_normal(self, covariance: torch.Tensor) -> torch.Tensor:
        """Sample a zero-mean Gaussian using the private dataset generator."""
        chol = torch.linalg.cholesky(covariance)
        standard_normal = self.dataset_randn(
            covariance.shape[0], dtype=covariance.dtype, device=covariance.device
        )
        return chol @ standard_normal

    @staticmethod
    def make_data_loader_generator(seed: int | None) -> torch.Generator:
        """Create an independent generator for deterministic worker seeding."""
        generator = torch.Generator()
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)
        return generator

    def transformer_randn(self, *size, **kwargs) -> torch.Tensor:
        """Sample from PyTorch's transformer stream seeded by ``seed_transformer``."""
        return torch.randn(*size, **kwargs)

    def transformer_rand(self, *size, **kwargs) -> torch.Tensor:
        """Sample uniform values from the transformer stream seeded by ``seed_transformer``."""
        return torch.rand(*size, **kwargs)

    def get_rng_states(self) -> dict:
        """Return the current state of all random-number generators.

        Returns:
          dict: Dictionary containing the global PyTorch, NumPy, and dataset-specific RNG states.
        """
        states = {
            "torch_global": torch.get_rng_state(),
            "numpy_global": np.random.get_state(),
            "numpy_dataset": self._dataset_numpy_generator.bit_generator.state,
            "dataset_torch": self._dataset_torch_generator.get_state(),
            "dataset_numpy": self._dataset_numpy_generator.bit_generator.state,
            "python_dataset": self._dataset_python_generator.getstate(),
        }
        if torch.cuda.is_available():
            states["cuda_global"] = torch.cuda.get_rng_state_all()
        return states

    def set_rng_states(self, states: dict) -> None:
        """Restore the saved state of all random-number generators.

        Parameters:
          states : dict
            Dictionary of RNG states returned by get_rng_states.

        Returns:
          None: This method restores the state of the global and dataset generators in place.
        """
        torch.set_rng_state(states["torch_global"])
        np.random.set_state(states["numpy_global"])
        self._dataset_numpy_generator.bit_generator.state = states["numpy_dataset"]
        self._dataset_torch_generator.set_state(states["dataset_torch"])
        self._dataset_python_generator.setstate(states["python_dataset"])
        self._dataset_numpy_generator.bit_generator.state = states["dataset_numpy"]
        if "cuda_global" in states and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(states["cuda_global"])


def seed_data_worker(worker_id: int) -> None:
    """Initialize a worker's private data stream from its DataLoader seed."""
    del worker_id
    SeedManager().seed_dataset(torch.initial_seed() % (2**32))
