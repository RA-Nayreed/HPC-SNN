"""Selected-position round-robin assignment."""

from .assignment import assign_selected_clients


def round_robin(selected_client_ids: list[str], process_count: int):
    return assign_selected_clients(
        selected_client_ids,
        process_count,
        "round_robin",
        cost_source="selected_position",
    )
