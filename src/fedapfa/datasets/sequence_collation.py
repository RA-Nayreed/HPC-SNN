"""Variable-length batch padding for [batch, time, features] model inputs."""

from typing import NamedTuple

import torch


class EventBatch(NamedTuple):
    inputs: torch.Tensor
    labels: torch.Tensor
    lengths: torch.Tensor
    valid_mask: torch.Tensor


def collate_event_sequences(samples: list[tuple[torch.Tensor, int]]) -> EventBatch:
    if not samples:
        raise ValueError("cannot collate an empty batch")
    sequences, labels = zip(*samples, strict=True)
    if any(sequence.ndim != 2 or sequence.shape[0] < 1 for sequence in sequences):
        raise ValueError("every sequence must have shape [time, features] with at least one bin")
    features = sequences[0].shape[1]
    if any(sequence.shape[1] != features for sequence in sequences):
        raise ValueError("all sequences must use the same feature count")
    lengths = torch.tensor([sequence.shape[0] for sequence in sequences], dtype=torch.long)
    inputs = torch.zeros((len(sequences), int(lengths.max()), features), dtype=sequences[0].dtype)
    for index, sequence in enumerate(sequences):
        inputs[index, : sequence.shape[0]] = sequence
    valid_mask = torch.arange(inputs.shape[1]).unsqueeze(0) < lengths.unsqueeze(1)
    return EventBatch(inputs, torch.tensor(labels, dtype=torch.long), lengths, valid_mask)
