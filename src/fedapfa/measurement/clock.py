"""Monotonic clocks and optional CUDA-event timing adapters."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

import torch


class MonotonicClock(Protocol):
    def now_ns(self) -> int: ...


class SystemMonotonicClock:
    def now_ns(self) -> int:
        return time.monotonic_ns()


@dataclass(frozen=True)
class TimingResult:
    start_ns: int
    end_ns: int
    cuda_seconds: float


class ClientTimingAdapter(Protocol):
    def start(self) -> None: ...

    def begin_device_work(self) -> object: ...

    def end_device_work(self, token: object) -> None: ...

    def finish(self) -> TimingResult: ...


class CpuTimingAdapter:
    """CPU adapter used to test timing semantics without claiming CUDA timing."""

    def __init__(self, clock: MonotonicClock | None = None) -> None:
        self.clock = clock or SystemMonotonicClock()
        self._start_ns: int | None = None
        self._work_ns = 0

    def start(self) -> None:
        if self._start_ns is not None:
            raise RuntimeError("timing adapter has already started")
        self._start_ns = self.clock.now_ns()

    def begin_device_work(self) -> object:
        return self.clock.now_ns()

    def end_device_work(self, token: object) -> None:
        ended = self.clock.now_ns()
        started = int(token)
        if ended < started:
            raise RuntimeError("timing clock moved backwards")
        self._work_ns += ended - started

    def finish(self) -> TimingResult:
        if self._start_ns is None:
            raise RuntimeError("timing adapter was not started")
        ended = self.clock.now_ns()
        if ended < self._start_ns:
            raise RuntimeError("timing clock moved backwards")
        return TimingResult(self._start_ns, ended, self._work_ns / 1_000_000_000)


class CudaTimingAdapter:
    """Collect CUDA event pairs on the active stream and synchronize once."""

    def __init__(self, device: torch.device, clock: MonotonicClock | None = None) -> None:
        if device.type != "cuda":
            raise ValueError("CUDA timing requires a CUDA device")
        self.device = device
        self.clock = clock or SystemMonotonicClock()
        self._start_ns: int | None = None
        self._pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []

    def start(self) -> None:
        if self._start_ns is not None:
            raise RuntimeError("timing adapter has already started")
        self._start_ns = self.clock.now_ns()

    def begin_device_work(self) -> object:
        event = torch.cuda.Event(enable_timing=True)
        event.record(torch.cuda.current_stream(self.device))
        return event

    def end_device_work(self, token: object) -> None:
        if not isinstance(token, torch.cuda.Event):
            raise TypeError("CUDA timing token is invalid")
        ended = torch.cuda.Event(enable_timing=True)
        ended.record(torch.cuda.current_stream(self.device))
        self._pairs.append((token, ended))

    def finish(self) -> TimingResult:
        if self._start_ns is None:
            raise RuntimeError("timing adapter was not started")
        torch.cuda.synchronize(self.device)
        ended_ns = self.clock.now_ns()
        milliseconds = sum(start.elapsed_time(end) for start, end in self._pairs)
        return TimingResult(self._start_ns, ended_ns, milliseconds / 1000.0)
