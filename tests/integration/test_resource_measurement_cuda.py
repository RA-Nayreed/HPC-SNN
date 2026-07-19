import importlib.util

import pytest
import torch

from fedapfa.measurement.clock import CudaTimingAdapter
from fedapfa.measurement.power import NvmlAdapter


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_events_and_nvml_sample_on_one_resolved_device():
    if torch.cuda.device_count() != 1:
        pytest.skip("test requires one visible CUDA device")
    if importlib.util.find_spec("pynvml") is None:
        pytest.skip("NVML binding is unavailable")
    device = torch.device("cuda", 0)
    value = getattr(torch.cuda.get_device_properties(device), "uuid", None)
    if value is None:
        pytest.skip("CUDA UUID is unavailable")
    uuid = value.decode() if isinstance(value, bytes) else str(value)
    if not uuid.startswith("GPU-"):
        uuid = f"GPU-{uuid}"
    adapter = NvmlAdapter(uuid, 1)
    try:
        sample = adapter.sample(1, "2026-01-01T00:00:00+00:00", 100)
        assert sample.gpu_uuid == uuid and sample.power_watts is not None
    finally:
        adapter.close()
    timing = CudaTimingAdapter(device)
    timing.start()
    token = timing.begin_device_work()
    torch.ones(1024, device=device).square_()
    timing.end_device_work(token)
    result = timing.finish()
    assert result.cuda_seconds >= 0
