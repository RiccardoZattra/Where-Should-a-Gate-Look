"""Output Modules to obtain predictions and logits"""

import logging

import torch
from torch import nn

from monitor.trackable_module import TrackableModule

log = logging.getLogger(__name__)


class EmbeddingDistanceDecoder(TrackableModule):
    """
    Decoder based on distance between given vector and embeddings

    This module interprets model outputs by computing the squared Euclidean
    distance between each input vector and all embedding vectors. These distances
    are converted into probabilities using a Softmin, and then mapped to logits.

    This effectively behaves like a nearest-neighbor decoder in embedding space,
    where predictions correspond to the closest embedding vectors.
    """

    def __init__(self, seq_target_len: int):
        super().__init__()

        # Number of final sequence positions to return
        self.seq_target_len = seq_target_len
        # Softmin over vocabulary dimension (converts distances → probabilities)
        self.softmin = nn.Softmin(dim=-1)

    # NOTE: The embedding is taken as an argument since we need to access its
    # weights
    def forward(self, u, embedding: nn.Embedding):
        """
        Parameters
            u: Hidden states [batch_size, seq_len, embedding_dim]
            embedding: Embedding layer (used to access embedding weights)

        Returns:
            predictions: [B, seq_target_len]
            logits: [B, vocab_size, seq_target_len]
        """
        # To try eventually
        # logits = torch.matmul(u, embedding.weight.T)


        # Reshape the output to prepare for distance computation [B, L, 1, D]
        outy = torch.unsqueeze(u, -2)

        # Expand embedding weights → [1, 1, vocab_size, D]
        emb_weights = torch.unsqueeze(torch.unsqueeze(embedding.weight, 0), 0)

        # Compute squared Euclidean distance → [B, L, vocab_size]
        tmp_distance = torch.sum((outy - emb_weights) ** 2, -1, keepdim=False)

        # Convert distances to probabilities (smaller distance → higher prob)
        sm = self.softmin(tmp_distance)

        # Convert probabilities to logits
        logits = torch.special.logit(sm, eps=1e-6)

        # Keep only last seq_target_len positions
        preds = logits[:, -self.seq_target_len :, :]

        # Argmax prediction over vocabulary
        out_list = torch.argmax(logits, dim=-1)
        out_list = out_list[:, -self.seq_target_len :]

        # Rearrange logits for CrossEntropy: [B, vocab, seq_len]
        return out_list, preds.permute(0, 2, 1)


class MLPClassifier(TrackableModule):
    """
    Multi-layer perceptron (MLP) classifier.

    This module applies a two-layer feedforward network with a GELU activation
    to map input features directly to output classes.

    Unlike sequence-based classifiers, this module assumes the input is already
    a fixed-size representation (no pooling is performed).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        out_bias1: bool,
        out_bias2: bool,
        device: torch.device,
    ):
        super().__init__()

        # Two-layer feedforward classifier
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, out_bias1, device=device),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim, out_bias2, device=device),
        )

    def forward(self, u, embedding: nn.Embedding):
        """
        Parameters
            u: Input features [B, D]
            embedding: Unused (kept for interface consistency)

        Returns:
            predictions: [B]
            logits: [B, output_dim]
        """

        logits = self.classifier(u)

        # Predicted class indices
        predictions = torch.argmax(logits, dim=-1)

        return predictions, logits
