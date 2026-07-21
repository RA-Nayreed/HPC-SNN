"""Scientific scheduling types and frozen public identities."""

from __future__ import annotations

from dataclasses import asdict, dataclass


SCHEDULING_STRATEGIES = (
    "round_robin",
    "example_count_longest_processing_time",
    "event_structure_longest_processing_time",
)

EVENT_STRUCTURE_FEATURES = (
    "example_count",
    "local_batch_count",
    "total_raw_input_events",
    "mean_sequence_length",
    "median_sequence_length",
    "maximum_sequence_length",
    "total_valid_time_bins",
    "estimated_padded_time_bins",
    "padding_fraction",
    "event_density",
)

ASSIGNMENT_TIE_BREAKING_VERSION = "cost_position_client_then_load_count_rank_v1"


@dataclass(frozen=True)
class ScheduledClient:
    """One already-selected client assigned to one global process rank."""

    selected_position: int
    client_id: str
    process_rank: int
    cost_source: str
    cost: float
    features: dict[str, float] | None = None

    def record(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SchedulingPlan:
    """Deterministic round assignment and pre-execution measurement record."""

    strategy: str
    assignments: list[ScheduledClient]
    predicted_process_loads: dict[str, float]
    feature_availability: dict[str, bool]
    model_artifact_path: str
    model_sha256: str
    model_provenance_identity: str
    model_name: str
    static_feature_lookup_seconds: float
    invariant_feature_extraction_seconds: float
    seed_dependent_feature_seconds: float
    model_prediction_seconds: float
    sorting_and_assignment_seconds: float
    scheduler_seconds_before_broadcast: float
    metadata_serialized_bytes: int
    privacy_metadata: list[dict]
    assignment_broadcast_seconds: float = 0.0
    total_scheduler_seconds: float = 0.0
    tie_breaking_version: str = ASSIGNMENT_TIE_BREAKING_VERSION

    def with_broadcast(self, duration: float) -> SchedulingPlan:
        if duration < 0:
            raise ValueError("assignment broadcast duration cannot be negative")
        return SchedulingPlan(
            **{
                **self.__dict__,
                "assignment_broadcast_seconds": duration,
                "total_scheduler_seconds": self.scheduler_seconds_before_broadcast + duration,
            }
        )

    def record(self) -> dict:
        value = asdict(self)
        value["assignments"] = [assignment.record() for assignment in self.assignments]
        return value
