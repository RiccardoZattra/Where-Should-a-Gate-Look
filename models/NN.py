"""Complete Neural Network composition module."""

from torch import nn

from monitor.trackable_module import TrackableModule


class NNComposition(TrackableModule):
    """Complete neural network model composition.

    Combines embedding, a cascade of recurrent layers, and an output module
    into a unified model pipeline.
    input -> Embedding -> Cascade of layers -> output module -> predictions, logits

    Attributes:
      embedding : nn.Embedding
        Embedding layer.
      layers : nn.ModuleList
        List of layers; one isolated SSM per layer in the experiments reported here.
      output : nn.Module
        Output module for final predictions.
    """

    def __init__(
        self, embedding: nn.Embedding, layers: list[nn.Module], output_layer: nn.Module
    ):
        super().__init__()

        self.embedding = embedding
        self.layers = nn.ModuleList(layers)
        self.output = output_layer

    def forward(self, u):
        """Process input through the embedding and the cascade of layers.

        Applies embedding lookup, passes through the layers in order, then
        applies the output layer for final predictions.

        Parameters:
          u : torch.Tensor
            Input token indices of shape (batch_size, sequence_length).

        Returns:
          Predictions and logits from the output layer.
        """
        u = self.embedding(u)

        for layer in self.layers:
            u = layer(u)

        # Pass the embedding layer instance, in case the output requires it
        return self.output(u, self.embedding.Myembedding)
