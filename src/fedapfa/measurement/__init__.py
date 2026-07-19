"""Resource measurement primitives for authoritative client execution."""

from .client_interval import ClientIntervalIdentity, IntervalRecord, IntervalRecorder
from .energy import EnergyEstimate, integrate_energy
from .power import DeviceSample, NvmlAdapter, NvmlProcessSampler, PowerSampler, ProcessPowerSampler

__all__ = [
    "ClientIntervalIdentity",
    "DeviceSample",
    "EnergyEstimate",
    "IntervalRecord",
    "IntervalRecorder",
    "NvmlAdapter",
    "NvmlProcessSampler",
    "PowerSampler",
    "ProcessPowerSampler",
    "integrate_energy",
]
