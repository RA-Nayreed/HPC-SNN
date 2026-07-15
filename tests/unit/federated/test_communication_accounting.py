import torch

from fedapfa.federated.communication_accounting import communication_for_clients, model_payload_bytes


def test_logical_model_bytes_match_manual_tensor_calculation():
    state = {
        "weight": torch.zeros((2, 3), dtype=torch.float32),
        "counter": torch.zeros(2, dtype=torch.int64),
    }
    expected = 6 * 4 + 2 * 8
    assert model_payload_bytes(state) == expected
    communication = communication_for_clients(expected, 5)
    assert communication == {
        "download_bytes": expected * 5,
        "upload_bytes": expected * 5,
        "total_bytes": expected * 10,
    }
