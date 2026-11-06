import copy

import torch

from utils.SeedManager import SeedManager


class Seq2SeqDataset(torch.utils.data.Dataset):
    """Sequence to Sequence Dataset.

    The dataset is designed to test selectivity and memory in sequence models.
    It generates synthetic sequences with specific patterns, to be used in sequence-to-sequence
    learning tasks, such as induction head or extended induction head tasks.
    Sequences are dynamically generated such that a "trigger" sequence appears twice: once
    in the middle of the sequence, followed by the "target" sequence, and once at the end.
    The rest of the sequence is filled with random tokens from the vocabulary, ensuring that
    no triggers are present in those segments.
    At the end of the sequence, there is a padding segment composed of a reserved index that is
    not part of the vocabulary. This ensures that the model only tries to predict the target sequence
    after having seen the last trigger in its entirety, and not before.
    The goal is to predict the "target" sequence following the first occurrence of the trigger.
    Depending on the 'trigger_pool_size' configuration, the trigger sequence can either be unique
    for each sample or drawn from a pre-generated pool.

    Parameters:
        vocabulary_size : int
            Vocabulary size.
        sequence_length : int
            Total length of each generated sequence.
        sequence_trigger_length : int
            Length of the trigger sequence.
        sequence_noise_length : int
            Length of the noise segment between trigger and target.
        sequence_target_length : int
            Length of the target sequence to be predicted.
        padding_idx : int | None
            Index used for padding tokens, or None if padding is disabled.
        batches_per_epoch : int
            Number of batches available per epoch.
        batch_size : int
            Number of samples per batch.
        ds_cfg : DictConfig
            Dataset configuration object.

    Attributes:
      V : int
        Vocabulary size including padding if used.
      L : int
        Total generated sequence length.
      seq_trigger_len : int
        Trigger sequence length.
      seq_noise_len : int
        Noise sequence length.
      seq_target_len : int
        Target sequence length.
      batches_per_epoch : int
        Number of batches per epoch.
      batch_sz : int
        Batch size.
      padding_idx : int | None
        Reserved index used for padding.
      valid_vocab : list[int]
        Vocabulary indices excluding the padding token.
      trigger_pool : list[list[int]] | None
        Pre-generated pool of triggers or None for dynamic triggers.
    """

    @classmethod
    def from_dataset(cls, dataset):
        """Create a new Seq2SeqDataset instance from an existing dataset.

        Parameters:
          dataset : Seq2SeqDataset
            The existing dataset instance to copy configuration from.

        Returns:
          Seq2SeqDataset: A new dataset instance with the same configuration.
        """

        return copy.deepcopy(dataset)

    def __init__(
        self,
        vocabulary_size,
        sequence_length,
        sequence_trigger_length,
        sequence_noise_length,
        sequence_target_length,
        padding_idx,
        batches_per_epoch,
        batch_size,
        ds_cfg,
    ):
        super().__init__()
        self.V = vocabulary_size
        self.L = sequence_length
        self.seq_trigger_len = sequence_trigger_length
        self.seq_noise_len = sequence_noise_length
        self.seq_target_len = sequence_target_length
        self.batches_per_epoch = batches_per_epoch
        self.batch_sz = batch_size
        self.padding_idx = padding_idx

        if self.padding_idx is None and self.seq_target_len > 1:
            raise ValueError(
                "Padding index must be specified when target sequence length is greater than 1."
            )

        if self.padding_idx is not None:
            self.V += 1  # Account for the padding index in the vocabulary size

        # Safety check: prevent negative lengths
        if 2 * self.seq_trigger_len + self.seq_target_len + self.seq_noise_len > self.L:
            raise ValueError("Sequence length L is too short to fit the triggers and target.")

        # Safety check: ensure vocabulary is large enough for the exclusion
        if self.V < 3:
            raise ValueError(
                f"Vocabulary size V={self.V} is too small.  "
                f"there must be at least 3 valid tokens: "
                f"one for the padding token, "
                f"one for the first trigger's token "
                f"and one for the rest of the sequence.",
            )

        if ds_cfg.trigger_pool_size < 0:
            raise ValueError("Trigger pool size must be non-negative.")

        # The available vocabulary excludes the padding token at index 'padding_idx'
        self.valid_vocab = [v for v in range(self.V) if v != self.padding_idx]

        # Depending on the 'trigger_pool_size' configuration, we either:
        # trigger_pool_size == 0: generate a completely unique trigger for each sample on the fly.
        # trigger_pool_size > 0: pre-generate a pool of triggers and sample from it for each
        # sequence. (If pool size is 1, this just repeatedly picks the same single trigger.)
        if ds_cfg.trigger_pool_size != 0:
            self.trigger_pool = []
            for _ in range(ds_cfg.trigger_pool_size):
                self.trigger_pool.append(
                  SeedManager().dataset_choices(self.valid_vocab, k=self.seq_trigger_len)
                )
        else:
            self.trigger_pool = None

    def _clean_segment(self, segment: list, trigger: list) -> list:
        """Scans a segment to remove accidental triggers by replacing
        a random token of the sequence with a non-trigger token.
        Parameters:
          segment : list
            Token sequence to be cleaned.
          trigger : list
            Trigger sequence to remove from the segment.

        Returns:
          list: Cleaned segment with accidental triggers removed.
        """
        assert len(trigger) == self.seq_trigger_len, "Trigger length does not match configuration."

        m = len(trigger)
        allowed_vocab = [v for v in self.valid_vocab if v not in trigger]
        cleaned = list(segment)

        for idx in range(len(cleaned) - m + 1):
            if cleaned[idx : idx + m] == trigger:
                target_idx = idx + SeedManager().dataset_randint(0, m - 1)
                cleaned[target_idx] = SeedManager().dataset_choice(allowed_vocab)

        return cleaned

    def __len__(self):
        """Total number of samples in dataset.

        Theoretically speaking the dataset is very large, so the length
        is calculated as the total number of batches per epoch * batch size.

        Returns:
          int: Total number of examples in the dataset.
        """
        return self.batches_per_epoch * self.batch_sz

    def __getitem__(self, _) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate single sample from the dataset.

        Every sample is generated depending on the configuration parameters: when the
        trigger pool size is 0, a unique trigger is generated for each sample; otherwise,
        triggers are sampled from a pre-generated pool.
        The method does not ensure that after a trigger is changed, the new trigger is different
        from the previous one but the probability of repeating the same trigger out of a
        pool of size N for k samples is (1/N)^k, which becomes very small for reasonably sized
        pools and sample sizes.
        The method ensures that the trigger sequence does not appear in the prefix, target,
        or inner segments by cleaning them after generation.
        The final sequence is assembled in the format:
        prefix + trigger + target + inner + trigger + padding.
        To do this it tries to match the trigger with the prefix, target and inner segments, and
        if it finds a match, it replaces a random token in the segment with a non-trigger token.
        Due to this cleaning process, the probability a token to be a non-trigger token is slighlty
        higher than to be a trigger token.
        The padding is needed so that the model only tries to predict the target sequence after having
        seen the last trigger in its entirety, and not before.

        Parameters:
            _ : int
                Ignored (index not used since data is generated dynamically).

        Returns:
          tuple[torch.Tensor, torch.Tensor]:
            seq : Input sequence with a specific pattern:
                prefix + trigger sequence + target sequences + inner + trigger sequence + padding.
                Prefix and inner sequences are composed of random tokens from the valid vocabulary.
                The padding is needed so that the model only tries to predict the target sequence after
                having seen the last trigger in its entirety, and not before.
            target : Target sequence tensor.
        """

        # 1. Randomize the trigger depending on the trigger pool size configuration
        # Depending on the trigger pool size configuration, we either generate a unique trigger
        # for each sample on the fly, or we sample it from the pre-generated pool.
        # It does not ensure that after a trigger is changed, the new trigger is different from the previous one.
        if self.trigger_pool is None:
            # Dynamically generate a completely unique trigger for THIS sequence
            trigger = SeedManager().dataset_choices(self.valid_vocab, k=self.seq_trigger_len)
        else:
            # Pick a trigger from the pre-generated pool
            # (If pool size is 1, this just repeatedly picks the same single trigger)
            trigger = SeedManager().dataset_choice(self.trigger_pool)

        # 2. Calculate remaining length and randomly split it
        remaining_len = (
            self.L - (2 * self.seq_trigger_len) - self.seq_target_len - self.seq_noise_len
        )
        prefix_len = SeedManager().dataset_randint(0, remaining_len)
        inter_len = remaining_len - prefix_len

        # 3. Generate the initial components using the FULL vocabulary
        prefix = SeedManager().dataset_choices(self.valid_vocab, k=prefix_len)
        noise = SeedManager().dataset_choices(self.valid_vocab, k=self.seq_noise_len)
        target = SeedManager().dataset_choices(self.valid_vocab, k=self.seq_target_len)
        inter = SeedManager().dataset_choices(self.valid_vocab, k=inter_len)
        padding = [self.padding_idx] * (len(target) - 1)

        # 4. Clean each segment individually
        prefix = self._clean_segment(prefix, trigger)
        noise = self._clean_segment(noise, trigger)
        target = self._clean_segment(target, trigger)
        inter = self._clean_segment(inter, trigger)

        # 5. Assemble the complete sequence
        seq = prefix + trigger + noise + target + inter + trigger + padding

        # Return as tensors (standard for PyTorch embedding/loss layers)
        return torch.tensor(seq), torch.tensor(target)
