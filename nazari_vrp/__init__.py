"""Paper-aligned VRP solver based on Nazari et al. (NeurIPS 2018)."""

from nazari_vrp.data import VRPBatch, VRPConfig, capacity_for_size, generate_batch
from nazari_vrp.insertion import (
    InsertionResult,
    actions_to_priority_order,
    best_insertion_construct,
    insertion_delta,
    learning_based_insertion,
    route_load,
    solution_cost,
)
from nazari_vrp.model import NazariVRPModel
from nazari_vrp.solver import (
    DecodeResult,
    beam_search_decode,
    rollout,
    routes_from_actions,
    validate_actions,
)

__all__ = [
    "DecodeResult",
    "InsertionResult",
    "NazariVRPModel",
    "VRPBatch",
    "VRPConfig",
    "actions_to_priority_order",
    "beam_search_decode",
    "best_insertion_construct",
    "capacity_for_size",
    "generate_batch",
    "insertion_delta",
    "learning_based_insertion",
    "route_load",
    "rollout",
    "routes_from_actions",
    "solution_cost",
    "validate_actions",
]
