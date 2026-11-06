import copy

import torch

from utils.SeedManager import SeedManager


class SelectiveCopyingDataset(torch.utils.data.Dataset):
    """Selective Copying Dataset.

    A variant of the Copying task, it is a synthetic benchmark for sequence models.
    A small number of "content" tokens are placed at random positions inside a long
    sequence of "blank" (noise) tokens, followed by a region of "marker" tokens.
    The model must reproduce the content tokens, in their original order, in
    correspondence with the markers, while ignoring the blanks.
    Sequences are generated dynamically. Each sequence has the structure:

        [ blanks with M content tokens scattered at random positions ] [ M markers ]

    and the target is the sequence of the M content tokens in the order in which
    they appear. The total input length is "noise_length + 2 * memorize_length".

    Args:
        n_tokens (int): Size of the content alphabet.
        sequence_noise_length (int): Number of blank (noise) positions among which the
            content tokens are scattered.
        sequence_target_length (int): Number of content tokens to copy (and hence the
            length of the target and of the marker region).
        batches_per_epoch (int): Number of batches per epoch.
        batch_size (int): Batch size for each epoch.

    Attributes:
        A (int): Content alphabet size. Content tokens take values
            in the range [1, n_tokens].
        seq_noise_length (int): Number of blank positions.
        seq_target_len (int): Number of tokens to memorize.
        blank_token (int): Token id reserved for blanks/noise (0).
        marker_token (int): Token id reserved for markers ("n_tokens + 1").
        vocab_size (int): Effective vocabulary size ("n_tokens + 2").
        batches_per_epoch (int): Number of batches per epoch.
        batch_sz (int): Batch size for each epoch.

    Methods:
        from_dataset(dataset): Create a new SelectiveCopyingDataset instance with the same
        configuration as an existing dataset.
        __len__(): Returns the nominal number of samples (batches_per_epoch * batch_size).
        __getitem__(index): Dynamically generate one (sequence, target) pair.
    """

    @classmethod
    def from_dataset(cls, dataset):
        """Create a new SelectiveCopyingDataset instance with the same configuration
        as an existing dataset.

        Args:
            dataset (SelectiveCopyingDataset): Instance to copy configuration from.

        Returns:
            SelectiveCopyingDataset: A new dataset istance with the same configuration.
        """
        return copy.deepcopy(dataset)

    def __init__(
        self,
        n_tokens,
        sequence_noise_length,
        sequence_target_length,
        batches_per_epoch,
        batch_size,
    ):
        super().__init__()
        self.A = n_tokens
        self.seq_noise_length = sequence_noise_length
        self.seq_target_len = sequence_target_length
        self.batches_per_epoch = batches_per_epoch
        self.batch_sz = batch_size

        # Token scheme: content in [1, A], blank = 0, marker = A + 1.
        self.blank_token = 0
        self.marker_token = self.A + 1
        self.vocab_size = self.A + 2

        # Safety checks
        if self.A < 1:
            raise ValueError(f"n_tokens={self.A} must be at least 1.")
        if self.seq_target_len < 1:
            raise ValueError(f"sequence_target_length={self.seq_target_len} must be at least 1.")
        if self.seq_noise_length < 0:
            raise ValueError(f"sequence_noise_length={self.seq_noise_length} must be non-negative.")

    def __len__(self):
        """Return the nominal number of samples in the dataset."""
        return self.batches_per_epoch * self.batch_sz

    def __getitem__(self, _) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate a single (sequence, target) sample from the dataset.

        Args:
            _ : Ignored (index not used since data is generated dynamically).

        Returns:
            tuple: A tuple containing:
                - seq (torch.Tensor): Input sequence of length
                  "sequence_noise_length + 2 * sequence_target_length": a scattered region of sequence_target_len content
                  tokens at random sorted positions among blanks, followed by sequence_target_len markers.
                - target (torch.Tensor): Target sequence, i.e. the sequence_target_len content tokens in order of
                  appearance.
        """
        # Content tokens to copy: sequence_target_len values drawn uniformly from [1, A].
        tokens = [
            SeedManager().dataset_randint(1, self.A) for _ in range(self.seq_target_len)
        ]

        # Scatter them at sequence_target_len random sorted positions among (sequence_noise_length + sequence_target_len) slots.
        total = self.seq_noise_length + self.seq_target_len
        positions = sorted(SeedManager().dataset_sample(range(total), self.seq_target_len))

        data = [self.blank_token] * total
        for pos, tok in zip(positions, tokens):
            data[pos] = tok

        # Marker region: M markers signalling where the model must output.
        markers = [self.marker_token] * self.seq_target_len

        seq = data + markers
        target = tokens  # content tokens, in order of appearance (positions sorted)

        return torch.tensor(seq), torch.tensor(target)
