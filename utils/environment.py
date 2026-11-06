"""Utility functions related to the environment/device running the training."""

import logging
import os
from functools import lru_cache

import torch

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_torch_device() -> torch.device:
    """Get Torch device based on CUDA availability.
    Use LRU Cache so result works like a simpleton.
    """
    if torch.cuda.is_available():
        cuda_v = torch.version.cuda
        cudnn_v = torch.backends.cudnn.version()
        gpu_n = torch.cuda.device_count()
        logger.info(f"Device: Using CUDA ({cuda_v}) | cudnn ({cudnn_v}) | GPUs: {gpu_n}")
        return torch.device("cuda")

    elif torch.backends.mps.is_available():
        logger.info("Device: Using MPS")
        return torch.device("mps")
    else:
        logger.info("Device: Using CPU")
        return torch.device("cpu")


def get_job_id() -> str:
    """Get the Job ID from the Slurm cluster.
    If the environment variable is not set (i.e. running locally), the default
    ID is `DATA`.
    """
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        logger.info("Training on local machine.")
        return "DATA"

    logger.info(f"Training on Slurm cluster. Job ID: {job_id}")
    return job_id
