"""Configuration data structures."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    seed: int
    dataset: str
    model: str
