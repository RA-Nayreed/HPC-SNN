"""Resource measurement primitives for authoritative client execution."""

import importlib

_EXPORT_MODULES = {
    "ClientIntervalIdentity": ".client_interval",
    "IntervalRecord": ".client_interval",
    "IntervalRecorder": ".client_interval",
    "DeviceSample": ".power",
    "NvmlAdapter": ".power",
    "NvmlProcessSampler": ".power",
    "PowerSampler": ".power",
    "ProcessPowerSampler": ".power",
    "EnergyEstimate": ".energy",
    "integrate_energy": ".energy",
    "NodeNvmlProcessSampler": ".multi_gpu_energy",
    "NodeTelemetrySample": ".multi_gpu_energy",
    "integrate_physical_devices": ".multi_gpu_energy",
    "merge_node_telemetry": ".multi_gpu_energy",
    "read_node_telemetry": ".multi_gpu_energy",
    "validate_client_interval_nonoverlap": ".multi_gpu_energy",
    "validate_comparative_calibration": ".multi_gpu_energy",
}

__all__ = sorted(_EXPORT_MODULES)


def __getattr__(name: str):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(name)
    module = importlib.import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
