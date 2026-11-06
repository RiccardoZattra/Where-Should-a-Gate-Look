import torch
from torch import nn

from monitor.trackable_module import TrackableModule


class LayerSmoothingProblem(TrackableModule):
    """Four-SSM layer with multiple sequence transformations.

    Applies separate SSMs to four transformations of the input: original,
    transposed, flipped, and transposed+flipped. Concatenates the final
    outputs from each SSM.

    A PyTorch module that applies four state-space models (SSMs) to different
    transformations of the input sequence and concatenates their final outputs.

    This module processes the input tensor in four different ways:
    1. Original sequence
    2. Transposed sequence (swap feature and sequence dimensions)
    3. Reversed sequence (flip along sequence dimension)
    4. Transposed + reversed sequence

    Each transformed input is passed through a separate SSM module. The last
    time-step output from each SSM is extracted and concatenated to form the
    final output representation.




    Parameters:
    ssm1 : nn.Module
        SSM applied to the original input.
    ssm2 : nn.Module
        SSM applied to the transposed input.
    ssm3 : nn.Module
        SSM applied to the flipped input.
    ssm4 : nn.Module
        SSM applied to the transposed and flipped input.
    device : torch.device
        Device on which the model operates.

    Attributes:
      ssm1 : nn.Module
        SSM applied to the original input.
      ssm2 : nn.Module
        SSM applied to the transposed input.
      ssm3 : nn.Module
        SSM applied to the flipped input.
      ssm4 : nn.Module
        SSM applied to the transposed and flipped input.

    Shape:
        - Input: (batch_size, sequence_length, feature_dim)
        - Output: (batch_size, 4 * hidden_dim)
    """

    def __init__(
        self,
        ssm1: nn.Module,
        ssm2: nn.Module,
        ssm3: nn.Module,
        ssm4: nn.Module,
        device: torch.device,
    ):
        super().__init__()
        self.ssm1 = ssm1
        self.ssm2 = ssm2
        self.ssm3 = ssm3
        self.ssm4 = ssm4

    def forward(self, u):
        """Forward pass of the LayerSmoothingProblem module.

        Applies four different transformations to the input sequence and processes
        each variant with a separate state-space model (SSM). The final output is
        obtained by concatenating the last time-step representations from each SSM.

        The following transformations are applied to the input:
            1. Original input: u
            2. Transposed input: u.transpose(1, 2)
            3. Flipped input: torch.flip(u, dims=[1])
            4. Transposed + flipped input:
            torch.flip(u.transpose(1, 2), dims=[1])

        Each transformed tensor is passed through a corresponding SSM:
            out1 = ssm1(u)
            out2 = ssm2(u_transpose)
            out3 = ssm3(u_flip)
            out4 = ssm4(u_transpose_flip)

        From each output, only the last time-step is selected:
            out[:, -1, :]

        The resulting tensors are concatenated along the feature dimension.

        Parameters:
          u : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, feature_dim).

        Returns:
          torch.Tensor: Concatenated output of shape (batch_size, 4*hidden_dim).

        Notes:
            Assumes all SSM outputs have shape (batch_size, sequence_length, hidden_dim)
            The flip operation is applied along dimension 1 (sequence dimension)
            After transpose, dimension semantics change:
            what was feature_dim becomes sequence_length and vice versa
            All SSM outputs must have matching hidden_dim for concatenation
        """
        u_flip = torch.flip(u, dims=[1])
        u_transpose = u.transpose(1, 2)
        u_transpose_flip = torch.flip(u_transpose, dims=[1])

        out1 = self.ssm1(u)
        out2 = self.ssm2(u_transpose)
        out3 = self.ssm3(u_flip)
        out4 = self.ssm4(u_transpose_flip)

        out = torch.cat((out1[:, -1, :], out2[:, -1, :], out3[:, -1, :], out4[:, -1, :]), dim=1)

        return out
