"""Attempt-aware execution intervals and timing reconciliation."""

from __future__ import annotations

import json
import math
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

from .clock import MonotonicClock, SystemMonotonicClock

INTERVAL_CATEGORIES = frozenset(
    {
        "client_training",
        "model_distribution",
        "result_collection",
        "aggregation",
        "validation",
        "checkpoint_writing",
        "communication_round",
        "training_execution",
        "idle_before",
        "idle_after",
    }
)


@dataclass(frozen=True)
class ClientIntervalIdentity:
    dataset: str
    experiment: str
    scientific_seed: int
    communication_round: int
    selected_position: int
    client_id: str
    training_seed: int
    execution_attempt: int
    gpu_uuid: str

    def key(self) -> tuple:
        return tuple(asdict(self).values())


@dataclass(frozen=True)
class IntervalRecord:
    interval_id: str
    category: str
    execution_attempt: int
    gpu_uuid: str
    start_ns: int
    end_ns: int
    wall_seconds: float
    accepted: bool
    exclusion_reason: str | None = None
    identity: dict | None = None
    data_wait_seconds: float | None = None
    cuda_event_seconds: float | None = None
    residual_host_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.category not in INTERVAL_CATEGORIES:
            raise ValueError(f"unsupported interval category: {self.category}")
        if self.end_ns < self.start_ns or self.wall_seconds < 0:
            raise ValueError("interval duration cannot be negative")
        expected = (self.end_ns - self.start_ns) / 1_000_000_000
        if not math.isclose(self.wall_seconds, expected, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("interval wall duration differs from monotonic timestamps")
        components = (self.data_wait_seconds, self.cuda_event_seconds, self.residual_host_seconds)
        if any(value is not None for value in components):
            if any(value is None or value < 0 for value in components):
                raise ValueError("client timing components must be complete and non-negative")
            if not math.isclose(sum(components), self.wall_seconds, rel_tol=0.0, abs_tol=2e-6):
                raise ValueError("client timing components do not reconcile with wall duration")

    def record(self) -> dict:
        return asdict(self)


class IntervalRecorder:
    """Append intervals promptly and reject duplicate accepted client identities."""

    def __init__(self, path: str | Path, clock: MonotonicClock | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.clock = clock or SystemMonotonicClock()
        self._lock = threading.Lock()
        self._sequence = 0
        self._accepted_client_keys: set[tuple] = set()
        self._last_client_end_ns: dict[int, int] = {}
        if self.path.is_file():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                value = json.loads(line)
                self._sequence += 1
                if value.get("category") == "client_training" and value.get("accepted"):
                    identity = ClientIntervalIdentity(**value["identity"])
                    self._accepted_client_keys.add(identity.key())
                    attempt = int(value["execution_attempt"])
                    end_ns = int(value["end_ns"])
                    self._last_client_end_ns[attempt] = max(
                        self._last_client_end_ns.get(attempt, end_ns), end_ns
                    )

    def _append(self, record: IntervalRecord) -> None:
        payload = json.dumps(record.record(), sort_keys=True, allow_nan=False) + "\n"
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())

    def record_client(
        self,
        identity: ClientIntervalIdentity,
        start_ns: int,
        end_ns: int,
        data_wait_seconds: float,
        cuda_event_seconds: float,
        accepted: bool = True,
        exclusion_reason: str | None = None,
    ) -> IntervalRecord:
        wall = (end_ns - start_ns) / 1_000_000_000
        residual = wall - data_wait_seconds - cuda_event_seconds
        if residual < -2e-6:
            raise ValueError("measured timing components exceed client wall duration")
        residual = max(0.0, residual)
        with self._lock:
            key = identity.key()
            if accepted and key in self._accepted_client_keys:
                raise ValueError("accepted client interval identity is duplicated")
            prior_end = self._last_client_end_ns.get(identity.execution_attempt)
            if accepted and prior_end is not None and start_ns < prior_end:
                raise ValueError("accepted client intervals overlap")
            self._sequence += 1
            interval_id = f"attempt-{identity.execution_attempt}-client-{self._sequence}"
            if accepted:
                self._accepted_client_keys.add(key)
                self._last_client_end_ns[identity.execution_attempt] = end_ns
        record = IntervalRecord(
            interval_id,
            "client_training",
            identity.execution_attempt,
            identity.gpu_uuid,
            start_ns,
            end_ns,
            wall,
            accepted,
            exclusion_reason,
            asdict(identity),
            data_wait_seconds,
            cuda_event_seconds,
            residual,
        )
        self._append(record)
        return record

    @contextmanager
    def interval(
        self, category: str, execution_attempt: int, gpu_uuid: str, identity: dict | None = None
    ) -> Iterator[None]:
        start_ns = self.clock.now_ns()
        accepted = False
        reason = None
        try:
            yield
            accepted = True
        except BaseException as error:
            reason = type(error).__name__
            raise
        finally:
            end_ns = self.clock.now_ns()
            with self._lock:
                self._sequence += 1
                interval_id = f"attempt-{execution_attempt}-interval-{self._sequence}"
            self._append(
                IntervalRecord(
                    interval_id,
                    category,
                    execution_attempt,
                    gpu_uuid,
                    start_ns,
                    end_ns,
                    (end_ns - start_ns) / 1_000_000_000,
                    accepted,
                    reason,
                    identity,
                )
            )
