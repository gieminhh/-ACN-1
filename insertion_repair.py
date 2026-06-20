"""
insertion_repair.py

Bước Insertion Repair cho CVRP.

Mục đích:
- Có nghiệm ban đầu từ Greedy / Sampling / Beam / Best Insertion.
- Thử lấy từng khách hàng ra và chèn lại vào vị trí khác.
- Chỉ nhận thay đổi nếu tổng cost giảm và không vi phạm tải trọng.

Tính chất:
    C_after <= C_before
vì thuật toán chỉ nhận thao tác làm cost giảm.
"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import List, Sequence, Tuple

Coord = Tuple[float, float]
Route = List[int]


def euclidean(coords: Sequence[Coord], a: int, b: int) -> float:
    ax, ay = coords[a]
    bx, by = coords[b]
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def route_cost(route: Sequence[int], coords: Sequence[Coord]) -> float:
    if len(route) < 2:
        return 0.0
    return sum(euclidean(coords, route[i], route[i + 1]) for i in range(len(route) - 1))


def solution_cost(routes: Sequence[Sequence[int]], coords: Sequence[Coord]) -> float:
    return sum(route_cost(route, coords) for route in routes)


def route_load(route: Sequence[int], demands: Sequence[float]) -> float:
    total = 0.0
    for node in route:
        node = int(node)
        if node != 0 and node < len(demands):
            total += float(demands[node])
    return total


def normalize_routes(routes: Sequence[Sequence[int]]) -> List[Route]:
    normalized: List[Route] = []
    for route in routes:
        r = [int(x) for x in route]
        if not r:
            continue
        if r[0] != 0:
            r = [0] + r
        if r[-1] != 0:
            r = r + [0]
        if len(r) > 2:
            normalized.append(r)
    return normalized


def remove_empty_routes(routes: List[Route]) -> List[Route]:
    return [r for r in routes if len(r) > 2]


def relocate_once(
    routes: Sequence[Sequence[int]],
    coords: Sequence[Coord],
    demands: Sequence[float],
    capacity: float,
):
    current = normalize_routes(routes)
    current_cost = solution_cost(current, coords)

    best_routes = deepcopy(current)
    best_cost = current_cost

    for src_r_idx, src_route in enumerate(current):
        for src_pos in range(1, len(src_route) - 1):
            customer = int(src_route[src_pos])

            routes_removed = deepcopy(current)
            routes_removed[src_r_idx].pop(src_pos)
            routes_removed = remove_empty_routes(routes_removed)

            # Thử chèn customer vào mọi vị trí của mọi route hiện có.
            for dst_r_idx, dst_route in enumerate(routes_removed):
                if route_load(dst_route, demands) + float(demands[customer]) > capacity + 1e-9:
                    continue

                for dst_pos in range(1, len(dst_route)):
                    candidate = deepcopy(routes_removed)
                    candidate[dst_r_idx].insert(dst_pos, customer)
                    candidate_cost = solution_cost(candidate, coords)

                    if candidate_cost + 1e-9 < best_cost:
                        best_cost = candidate_cost
                        best_routes = candidate

            # Thử tách customer thành route riêng, chỉ nhận nếu cost giảm.
            candidate = deepcopy(routes_removed)
            candidate.append([0, customer, 0])
            if route_load(candidate[-1], demands) <= capacity + 1e-9:
                candidate_cost = solution_cost(candidate, coords)
                if candidate_cost + 1e-9 < best_cost:
                    best_cost = candidate_cost
                    best_routes = candidate

    improved = best_cost + 1e-9 < current_cost
    return best_routes, improved, current_cost - best_cost


def insertion_repair(
    routes: Sequence[Sequence[int]],
    coords: Sequence[Coord],
    demands: Sequence[float],
    capacity: float,
    max_iter: int = 50,
):
    repaired = normalize_routes(routes)
    initial_cost = solution_cost(repaired, coords)

    history = []
    for it in range(1, max_iter + 1):
        new_routes, improved, improvement = relocate_once(
            repaired,
            coords,
            demands,
            capacity,
        )
        if not improved:
            break

        repaired = new_routes
        history.append(
            {
                "iter": it,
                "improvement": improvement,
                "cost": solution_cost(repaired, coords),
            }
        )

    final_cost = solution_cost(repaired, coords)
    return repaired, {
        "initial_cost": initial_cost,
        "final_cost": final_cost,
        "improvement": initial_cost - final_cost,
        "iterations": len(history),
        "not_worse": final_cost <= initial_cost + 1e-9,
        "history": history,
    }
