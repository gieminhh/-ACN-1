from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch

from nazari_vrp.data import VRPBatch
from nazari_vrp.model import NazariVRPModel
from nazari_vrp.solver import DecodeResult, beam_search_decode, rollout


Route = list[int]


@dataclass
class InsertionResult:
    priority_orders: list[list[int]]
    routes: list[list[Route]]
    cost: torch.Tensor
    reward: torch.Tensor
    log_likelihood: torch.Tensor
    entropy: torch.Tensor
    value: torch.Tensor


def actions_to_priority_order(actions: Iterable[int], num_customers: int) -> list[int]:
    """Convert a decoded route sequence into a customer insertion priority."""
    order: list[int] = []
    seen: set[int] = set()

    for raw_action in actions:
        action = int(raw_action)
        if action == 0:
            continue
        if 1 <= action <= num_customers and action not in seen:
            order.append(action)
            seen.add(action)

    for customer in range(1, num_customers + 1):
        if customer not in seen:
            order.append(customer)
            seen.add(customer)

    return order


def euclidean(coords: torch.Tensor, a: int, b: int) -> float:
    delta = coords[a] - coords[b]
    return float(torch.linalg.norm(delta).item())


def insertion_delta(coords: torch.Tensor, a: int, customer: int, b: int) -> float:
    """Extra distance when inserting customer between a and b."""
    return euclidean(coords, a, customer) + euclidean(coords, customer, b) - euclidean(coords, a, b)


def route_load(route: Sequence[int], demand: torch.Tensor) -> float:
    load = 0.0
    for node in route:
        if node > 0:
            load += float(demand[node - 1].item())
    return load


def route_cost(route: Sequence[int], coords: torch.Tensor) -> float:
    if len(route) < 2:
        return 0.0
    return sum(euclidean(coords, route[i], route[i + 1]) for i in range(len(route) - 1))


def solution_cost(routes: Sequence[Sequence[int]], coords: torch.Tensor) -> float:
    return sum(route_cost(route, coords) for route in routes)


def best_insertion_construct(
    priority_order: Sequence[int],
    coords: torch.Tensor,
    demand: torch.Tensor,
    capacity: float,
) -> list[Route]:
    """Build CVRP routes by best insertion.

    For each customer j, test every feasible edge a -> b in the current routes
    and insert j where Delta(a,j,b) is smallest. If no current route can accept
    j without exceeding vehicle capacity, open a new route [0, j, 0].
    """
    routes: list[Route] = []

    for customer in priority_order:
        demand_j = float(demand[customer - 1].item())
        best_route_idx: int | None = None
        best_position: int | None = None
        best_delta = float("inf")

        for route_idx, route in enumerate(routes):
            if route_load(route, demand) + demand_j > capacity + 1e-9:
                continue

            for position in range(1, len(route)):
                a = route[position - 1]
                b = route[position]
                delta = insertion_delta(coords, a, int(customer), b)
                if delta < best_delta:
                    best_delta = delta
                    best_route_idx = route_idx
                    best_position = position

        if best_route_idx is None or best_position is None:
            routes.append([0, int(customer), 0])
        else:
            routes[best_route_idx].insert(best_position, int(customer))

    return routes


def construct_from_decode(
    batch: VRPBatch,
    decoded: DecodeResult,
) -> InsertionResult:
    coords = batch.coords_with_depot().detach().cpu()
    demand = batch.demand.detach().cpu()
    capacity = batch.capacity.detach().cpu()

    all_orders: list[list[int]] = []
    all_routes: list[list[Route]] = []
    costs: list[float] = []

    for idx in range(batch.batch_size):
        order = actions_to_priority_order(
            decoded.actions[idx].detach().cpu().tolist(),
            batch.num_customers,
        )
        routes = best_insertion_construct(
            priority_order=order,
            coords=coords[idx],
            demand=demand[idx],
            capacity=float(capacity[idx].item()),
        )
        all_orders.append(order)
        all_routes.append(routes)
        costs.append(solution_cost(routes, coords[idx]))

    cost_tensor = torch.tensor(costs, dtype=torch.float32, device=batch.depot.device)
    return InsertionResult(
        priority_orders=all_orders,
        routes=all_routes,
        cost=cost_tensor,
        reward=-cost_tensor,
        log_likelihood=decoded.log_likelihood,
        entropy=decoded.entropy,
        value=decoded.value,
    )


def learning_based_insertion(
    model: NazariVRPModel,
    batch: VRPBatch,
    decode_type: str = "greedy",
    beam_width: int = 5,
) -> InsertionResult:
    """Use the learned policy to guide a best-insertion heuristic."""
    if decode_type == "beam":
        if batch.batch_size != 1:
            raise ValueError("beam guided insertion expects batch_size=1")
        decoded = beam_search_decode(model, batch, beam_width=beam_width)
    else:
        decoded = rollout(model, batch, decode_type=decode_type)

    return construct_from_decode(batch, decoded)
