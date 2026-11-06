"""Factory utilities for constructing neural network models from configuration."""

import logging

import torch
from torch import nn

from models.embeddings import (
    ScalePixelEmbeddind,
    ScalePixelEmbeddindFlatImages,
    TrainableEmbeddingOrthogonalInit,
    TrainableEmbeddingQuasiOrthogonalInit,
    TrainableEmbeddingRandomInit,
)
from models.gated_ssm import GatedSSM
from models.layer_smoothing_problem import LayerSmoothingProblem
from models.NN import NNComposition
from models.output_layers import EmbeddingDistanceDecoder, MLPClassifier
from models.mamba import Mamba

logger = logging.getLogger(__name__)


def construct_model(cfg, device: torch.device) -> nn.Module:
    """Return a complete neural network using the selected model configuration.

    Parameters:
      cfg : Any
        Configuration object describing model, dataset, and training settings.
      device : torch.device
        Device on which the model will be instantiated.

    Returns:
      nn.Module: Composed neural network instance.
    """
    logger.info("Constructing model")

    layers = [_build_layer(cfg=cfg, device=device) for _ in range(cfg.model.n_layer)]

    return NNComposition(
        embedding=_get_embedding(cfg=cfg, device=device),
        layers=layers,
        output_layer=_get_output_module(cfg=cfg, device=device),
    )


def _get_embedding(cfg, device: torch.device) -> nn.Embedding:
    """Select the proper embedding layer based on the dataset or task.

    Parameters:
      cfg : Any
        Configuration object containing model and dataset settings.
      device : torch.device
        Device on which the embedding layer should operate.

    Returns:
      nn.Embedding: The selected embedding layer.
    """
    emb_name = cfg.model.embedding

    logger.info(f"Padding index: {cfg.dataset.padding_idx}")

    # Add 1 to the vocab size if a padding index is specified, to account for the reserved
    # token used for padding. If padding_idx is None, no padding token is needed and use the
    # original vocab size.
    vocab_size = cfg.dataset.vocab_size + (0 if cfg.dataset.padding_idx is None else 1)

    if emb_name == "trainable_random_init":
        return TrainableEmbeddingRandomInit(
            vocab_size, cfg.model.d_model, device=device, padding_idx=cfg.dataset.padding_idx
        )

    elif emb_name == "trainable_ortho_init":
        return TrainableEmbeddingOrthogonalInit(
            vocab_size, cfg.model.d_model, device=device, padding_idx=cfg.dataset.padding_idx
        )
    elif emb_name == "trainable_quasi_ortho_init":
        return TrainableEmbeddingQuasiOrthogonalInit(
            vocab_size, cfg.model.d_model, device=device, padding_idx=cfg.dataset.padding_idx
        )
    elif emb_name == "scale_pixel_embedding_flat_imgs":
        return ScalePixelEmbeddindFlatImages()

    elif emb_name == "scale_pixel_embedding":
        return ScalePixelEmbeddind()

    else:
        raise NotImplementedError(f"Embedding '{emb_name}' not implemented yet.")


def _build_layer(cfg, device: torch.device) -> nn.Module:
    """Build a single instance of the model's inner layers.

    This is typically a Mamba block, but it can also be an isolated SSM module or
    a smoothing-problem-inspired layer.

    Parameters:
      cfg : Any
        Configuration object containing model settings.
      device : torch.device
        Device on which the layer should be created.

    Returns:
      nn.Module: A single layer instance matching the configured layer type.
    """
    if cfg.model.layer_name == "only_ssm":
        return _select_ssm_module(cfg=cfg, device=device)
    elif (
        cfg.model.layer_name == "layer_inspired_by_the_smoothing_problem"
        or cfg.model.layer_name == "layer_inspired_by_the_smoothing_problem_for_flat_imgs"
    ):
        return _get_layer_smoothing_problem(cfg=cfg, device=device)

    else:
        raise ImportError("Model not implemented")


def _get_layer_smoothing_problem(cfg, device: torch.device):
    """Return a smoothing-problem-inspired layer based on the configuration.

    Parameters:
      cfg : Any
        Configuration object defining the smoothing-problem layer type.
      device : torch.device
        Device on which the returned layer should operate.

    Returns:
      nn.Module: Smoothing-problem-inspired layer.
    """
    if cfg.model.layer_name == "layer_inspired_by_the_smoothing_problem":
        return LayerSmoothingProblem(
            _select_ssm_module(cfg=cfg, device=device),
            _select_ssm_module(cfg=cfg, device=device),
            _select_ssm_module(cfg=cfg, device=device),
            _select_ssm_module(cfg=cfg, device=device),
            device=device,
        )
    else:
        raise ImportError("This type of smoothing problem inspired layer is not implemented")


def _select_ssm_module(cfg, device: torch.device) -> nn.Module:
    """Select the SSM module implementation based on configuration.

    Parameters:
      cfg : Any
        Configuration object containing the SSM module selection.
      device : torch.device
        Device for the SSM module.

    Returns:
      nn.Module: Instantiated SSM module.
    """
    module_name = cfg.model.ssm_module

    kwargs = {
        "input_dim": cfg.model.d_inner,
        "output_dim": cfg.model.d_inner,
        "state_dim": cfg.model.d_state,
        "device": device,
    }

    if module_name == "gated":
        gating = cfg.model.gating
        positive_transition = cfg.model.positive_transition
        gate_bias = cfg.model.gate_bias
        logger.info(
            f"Gated SSM: gating={gating}, positive_transition={positive_transition}, "
            f"gate_bias={gate_bias}"
        )
        return GatedSSM(
            **kwargs,
            gating=gating,
            positive_transition=positive_transition,
            gate_bias=gate_bias,
        )

    if module_name == "S6":

        return Mamba(
            **kwargs,
            d_conv=4,
            expand=1,
            dtype=torch.float
        )

    raise NotImplementedError(
        f"SSM module '{module_name}' not implemented. Expected 'gated' or 'S6'."
    )


def _get_output_module(cfg, device: torch.device) -> nn.Module:
    """Select the proper output module based on the dataset or task.

    Parameters:
      cfg : Any
        Configuration object with output layer and dataset settings.
      device : torch.device
        Device on which the output module should operate.

    Returns:
      nn.Module: The configured output module.
    """
    out_layer_name = cfg.model.out_layer

    if out_layer_name == "embedding_distance_decoder":
        return EmbeddingDistanceDecoder(cfg.dataset.sequence_target_length)

    elif out_layer_name == "mlp_classifier":
        return MLPClassifier(
            cfg.model.input_dim_classifier,
            cfg.model.hidden_dim_classifier,
            cfg.model.output_dim_classifier,
            cfg.model.out_bias1_classifier,
            cfg.model.out_bias2_classifier,
            device=device,
        )

    else:
        raise NotImplementedError(f"Output Module '{out_layer_name}' not implemented yet.")
