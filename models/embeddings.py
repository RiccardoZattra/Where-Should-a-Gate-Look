"""Custom embedding layer implementations.

This module provides embedding layers for trainable token embeddings and
identity-based pixel scaling embeddings used across the model definitions.
"""

import torch
from torch import nn

from monitor.trackable_module import TrackableModule
from utils.SeedManager import SeedManager
import math

class TrainableEmbeddingRandomInit(TrackableModule):
    """Trainable embedding layer with default random initialization.

    This is a thin wrapper around nn.Embedding that enables tracking via
    TrackableModule while preserving standard trainable embedding behavior.

    Parameters:
      vocab_size : int
        Number of embedding tokens.
      embedding_dim : int
        Dimension of each embedding vector.
      device : torch.device
        Device where the embedding parameters are allocated.
      padding_idx : int
        Index reserved for padding tokens.

    Attributes:
      Myembedding : nn.Embedding
        Underlying embedding layer.
    """

    def __init__(
        self, vocab_size: int, embedding_dim: int, device: torch.device, padding_idx: int
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.device = device
        self.Myembedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=padding_idx, max_norm=None, device=device
        )

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Perform embedding lookup.

        Parameters:
          u : torch.Tensor
            Tensor of token indices.

        Returns:
          torch.Tensor: Corresponding embedding vectors.
        """
        return self.Myembedding(u)


class TrainableEmbeddingOrthogonalInit(TrackableModule):
    """Trainable embedding layer with orthogonal initialization.

    Embedding weights are initialized using a QR decomposition to approximate
    orthogonality (useful for stability and representation diversity).
    Padding index (0) is reserved and its embedding is frozen (not updated).

    Parameters
      vocab_size : int
        Number of embedding tokens.
      embedding_dim : int
        Dimension of each embedding vector.
      device : torch.device
        Device where the embedding parameters are allocated.
      padding_idx : int
        Index reserved for padding tokens.

    Attributes:
      Myembedding : nn.Embedding
        Underlying embedding layer.
    """

    def __init__(
        self, vocab_size: int, embedding_dim: int, device: torch.device, padding_idx: int
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.device = device
        self.Myembedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=padding_idx, max_norm=None, device=device
        )

        # Orthogonal initialization
        seed_manager = SeedManager()
        A = seed_manager.transformer_rand(self.embedding_dim, self.vocab_size).to(device)
        Q, _ = torch.linalg.qr(A, mode="reduced")

        if self.embedding_dim >= self.vocab_size:
            self.Myembedding.weight.data = (Q.T).clone()
        else:
            remaining_rows = seed_manager.transformer_rand(
                self.vocab_size - self.embedding_dim, self.embedding_dim
            ).to(device)
            emb_tot = torch.cat([Q, remaining_rows], dim=0)
            self.Myembedding.weight.data = emb_tot.clone()

        self.Myembedding.weight.requires_grad = True

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Perform embedding lookup.

        Parameters:
          u : torch.Tensor
            Tensor of token indices.

        Returns:
          torch.Tensor: Corresponding embedding vectors.
        """
        return self.Myembedding(u)

class TrainableEmbeddingQuasiOrthogonalInit(TrackableModule):
    """Trainable embedding layer with quasi orthogonal initialization
        based on Johnson-Lindenstrauss Initialization.

    Parameters
      vocab_size : int
        Number of embedding tokens.
      embedding_dim : int
        Dimension of each embedding vector.
      device : torch.device
        Device where the embedding parameters are allocated.
      padding_idx : int
        Index reserved for padding tokens.

    Attributes:
      Myembedding : nn.Embedding
        Underlying embedding layer.
    """

    def __init__(
        self, vocab_size: int, embedding_dim: int, device: torch.device, padding_idx: int
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.device = device
        
        self.Myembedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=padding_idx, max_norm=None, device=device
        )

        # Johnson-Lindenstrauss Initialization (Gaussian Random Projection)
        seed_manager = SeedManager()
        
        # 1. Generate the entire matrix at once.
        # Assuming transformer_rand generates a standard normal distribution N(0, 1).
        A = seed_manager.transformer_rand(self.vocab_size, self.embedding_dim).to(device)
        
        # 2. JL Scaling: divide by sqrt(embedding_dim). 
        # This ensures that the expected L2 norm of each vector is roughly 1,
        # and that the dot product between any two vectors is very close to 0.
        jl_embeddings = A / math.sqrt(self.embedding_dim)
        
        # 3. Handle padding_idx
        # Since we are overwriting the weights, we must manually zero out 
        # the padding vector to avoid breaking nn.Embedding's default behavior.
        if padding_idx is not None:
            with torch.no_grad():
                jl_embeddings[padding_idx] = 0.0

        # Assign weights
        self.Myembedding.weight.data = jl_embeddings.clone()
        self.Myembedding.weight.requires_grad = True

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Perform embedding lookup.

        Parameters:
          u : torch.Tensor
            Tensor of token indices.

        Returns:
          torch.Tensor: Corresponding embedding vectors.
        """
        return self.Myembedding(u)


class ScalePixelEmbeddindFlatImages(TrackableModule):
    """Identity embedding layer for flattened image inputs.

    This layer scales pixel values and reshapes input tensors to ensure the
    output has a feature dimension of size one for interface consistency.

    Attributes:
      Myembedding : nn.Identity
        Identity layer used for passthrough behavior.
    """

    def __init__(self):
        super().__init__()
        self.Myembedding = nn.Identity()

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Forward pass that scales pixels and adds a feature dimension.

        Parameters:
          u : torch.Tensor
            Input tensor containing pixel values.

        Returns:
          torch.Tensor: Scaled tensor with an added feature dimension of size one [...,1] this is done
        for interface compatibility to obtain [B,L,D=1]
        """
        return (u / 225.0).unsqueeze(2)


class ScalePixelEmbeddind(TrackableModule):
    """Identity embedding layer for pixel-scaled inputs.

    This layer scales pixel values but otherwise preserves the input tensor
    shape for compatibility with embedding-style interfaces.

    Attributes:
      Myembedding : nn.Identity
        Identity layer used for passthrough behavior.
    """

    def __init__(self):
        super().__init__()
        self.Myembedding = nn.Identity()

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Forward pass that scales pixel values.

        Parameters:
          u : torch.Tensor
            Input tensor containing pixel values.

        Returns:
          torch.Tensor: Tensor with the same shape as the input and scaled pixel values.
        """
        return u / 225.0
