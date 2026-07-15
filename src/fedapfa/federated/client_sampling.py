"""Independent deterministic client-selection schedules."""

from __future__ import annotations

import torch


class ClientSelectionSchedule:
    def __init__(self, client_ids: list[str], seed: int):
        if len(client_ids) < 2 or len(client_ids) != len(set(client_ids)):
            raise ValueError("client identifiers must be unique and contain at least two clients")
        self.client_ids = list(client_ids)
        self.generator = torch.Generator().manual_seed(seed)
        self.next_round = 1

    def select(self, round_number: int, count: int) -> list[str]:
        if round_number != self.next_round:
            raise ValueError(f"expected communication round {self.next_round}, got {round_number}")
        if not 0 < count <= len(self.client_ids):
            raise ValueError("selected-client count is outside the client collection")
        permutation = torch.randperm(len(self.client_ids), generator=self.generator).tolist()
        selected = [self.client_ids[index] for index in permutation[:count]]
        if len(selected) != len(set(selected)):
            raise RuntimeError("client selection contains a duplicate")
        self.next_round += 1
        return selected

    def state_dict(self) -> dict:
        return {"generator_state": self.generator.get_state(), "next_round": self.next_round}

    def load_state_dict(self, state: dict) -> None:
        self.generator.set_state(state["generator_state"])
        self.next_round = int(state["next_round"])
