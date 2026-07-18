"""Single-node synchronous distributed FedAvg execution."""

from .assignment_broadcast import ClientAssignment, assign_clients
from .process_context import ProcessContext

__all__ = ["ClientAssignment", "ProcessContext", "assign_clients"]
