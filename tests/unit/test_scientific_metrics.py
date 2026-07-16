import pytest
import torch

from fedapfa.federated.aggregation import state_difference_cosine_similarity
from fedapfa.metrics.classification import (
    confusion_matrix,
    macro_f1_from_confusion_matrix,
    per_class_accuracy,
)
from fedapfa.metrics.client_fairness import (
    PROXY_EXPLANATION,
    client_distribution_weighted_validation_accuracy,
    fairness_proxy_record,
)
from fedapfa.training.optimization import learning_rate_for_round, make_federated_optimizer


def test_classification_metrics_use_target_rows_and_zero_undefined_classes():
    matrix = confusion_matrix([0, 1, 1, 0], [0, 0, 1, 1], classes=3)
    assert matrix == [[1, 1, 0], [1, 1, 0], [0, 0, 0]]
    assert per_class_accuracy(matrix) == pytest.approx([0.5, 0.5, 0.0])
    assert macro_f1_from_confusion_matrix(matrix) == pytest.approx(1 / 3)


def test_client_distribution_weighted_validation_proxy_is_explicit_and_summarized():
    assert client_distribution_weighted_validation_accuracy([0.8, 0.4], {"0": 3, "1": 1}) == pytest.approx(0.7)
    artifact = {
        "clients": [
            {"client_id": "client_00", "class_counts": {"0": 3, "1": 1}},
            {"client_id": "client_01", "class_counts": {"0": 1, "1": 3}},
        ]
    }
    record = fairness_proxy_record([0.8, 0.4], artifact)
    assert record["definition"] == PROXY_EXPLANATION
    assert record["statistics"]["minimum"] == pytest.approx(0.5)
    assert record["statistics"]["maximum"] == pytest.approx(0.7)


def test_update_cosine_similarity_handles_alignment_opposition_and_zero_norm():
    server = {"weight": torch.tensor([0.0, 0.0])}
    aggregate = {"weight": torch.tensor([1.0, 0.0])}
    assert state_difference_cosine_similarity({"weight": torch.tensor([2.0, 0.0])}, server, aggregate) == 1.0
    assert state_difference_cosine_similarity({"weight": torch.tensor([-2.0, 0.0])}, server, aggregate) == -1.0
    assert state_difference_cosine_similarity(server, server, aggregate) == 0.0
    assert state_difference_cosine_similarity(aggregate, server, server) == 0.0


def test_sgd_momentum_and_round_boundary_learning_rate_reductions():
    settings = {
        "optimizer": "sgd",
        "learning_rate": 0.1,
        "momentum": 0.95,
        "weight_decay": 0.0,
        "learning_rate_reduction_rounds": [40, 60, 80],
        "learning_rate_reduction_factor": 5,
    }
    assert [learning_rate_for_round(settings, value) for value in (40, 41, 61, 81)] == pytest.approx(
        [0.1, 0.02, 0.004, 0.0008]
    )
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = make_federated_optimizer([parameter], settings, 41)
    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.param_groups[0]["momentum"] == pytest.approx(0.95)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.02)
