"""Example-count longest-processing-time assignment."""

from .assignment import assign_selected_clients


def example_count_longest_processing_time(
    selected_client_ids: list[str], process_count: int, example_counts: dict[str, int]
):
    return assign_selected_clients(
        selected_client_ids,
        process_count,
        "example_count_longest_processing_time",
        example_counts,
        cost_source="training_example_count",
    )
