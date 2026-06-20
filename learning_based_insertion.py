"""
learning_based_insertion.py

Module bổ sung để đề tài đúng hơn với tên:
"Phương pháp heuristic chèn dựa trên học máy cho bài toán CVRP".

Ý tưởng:
1. Mô hình PPO + Attention sinh ra thứ tự ưu tiên khách hàng.
2. Thuật toán Best Insertion lấy từng khách hàng theo thứ tự đó.
3. Với mỗi khách hàng j, thử chèn j vào mọi vị trí khả thi trong các tuyến hiện có.
4. Chọn vị trí có chi phí tăng thêm nhỏ nhất:
       Delta(a, j, b) = c(a,j) + c(j,b) - c(a,b)
5. Không chèn nếu làm tuyến vượt tải Q.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Tuple

import torch


Coord = Tuple[float, float]
Route = List[int]


def euclidean(coords: Sequence[Coord], a: int, b: int) -> float:
    """Khoảng cách Euclid giữa hai node a và b."""
    ax, ay = coords[a]
    bx, by = coords[b]
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def route_cost(route: Sequence[int], coords: Sequence[Coord]) -> float:
    """Tổng chiều dài của một tuyến."""
    if len(route) < 2:
        return 0.0
    return sum(euclidean(coords, route[i], route[i + 1]) for i in range(len(route) - 1))


def solution_cost(routes: Sequence[Sequence[int]], coords: Sequence[Coord]) -> float:
    """Tổng cost của toàn bộ lời giải CVRP."""
    return sum(route_cost(route, coords) for route in routes)


def route_load(route: Sequence[int], demands: Sequence[float]) -> float:
    """Tổng demand trên một tuyến, bỏ depot 0."""
    total = 0.0
    for node in route:
        if node != 0 and node < len(demands):
            total += float(demands[node])
    return total


def insertion_delta(coords: Sequence[Coord], a: int, j: int, b: int) -> float:
    """
    Chi phí tăng thêm khi chèn khách hàng j vào giữa a và b.

    Delta(a,j,b) = c(a,j) + c(j,b) - c(a,b)
    """
    return euclidean(coords, a, j) + euclidean(coords, j, b) - euclidean(coords, a, b)


def actions_to_customer_order(actions: Iterable[int], num_customers: int) -> List[int]:
    """
    Chuyển chuỗi actions của mô hình thành thứ tự ưu tiên khách hàng.
    Bỏ depot 0 và bỏ node trùng.
    """
    order: List[int] = []
    seen = set()

    for action in actions:
        node = int(action)
        if node == 0:
            continue
        if 1 <= node <= num_customers and node not in seen:
            order.append(node)
            seen.add(node)

    # Dự phòng: nếu vì lý do nào đó actions thiếu khách hàng, thêm các khách còn thiếu vào cuối.
    for node in range(1, num_customers + 1):
        if node not in seen:
            order.append(node)
            seen.add(node)

    return order


def best_insertion_construct(
    customer_order: Sequence[int],
    coords: Sequence[Coord],
    demands: Sequence[float],
    capacity: float = 1.0,
) -> List[Route]:
    """
    Xây dựng lời giải CVRP bằng Best Insertion.

    Khác append-based:
    - Không chỉ thêm khách vào cuối route.
    - Với mỗi khách hàng, thử mọi vị trí giữa hai node liên tiếp trong các route.
    - Chọn vị trí có Delta nhỏ nhất và không vượt tải.

    Lưu ý:
    - Nếu chưa có route nào hoặc không route nào còn đủ tải, mở route mới [0, j, 0].
    """
    routes: List[Route] = []

    for customer in customer_order:
        demand_j = float(demands[customer]) if customer < len(demands) else 0.0

        best_route_idx = None
        best_pos = None
        best_delta = float("inf")

        # Thử chèn vào mọi vị trí của mọi route hiện có.
        for r_idx, route in enumerate(routes):
            current_load = route_load(route, demands)
            if current_load + demand_j > capacity + 1e-9:
                continue

            # route dạng [0, ..., 0], có thể chèn vào giữa route[pos-1] và route[pos]
            for pos in range(1, len(route)):
                a = route[pos - 1]
                b = route[pos]
                delta = insertion_delta(coords, a, customer, b)
                if delta < best_delta:
                    best_delta = delta
                    best_route_idx = r_idx
                    best_pos = pos

        if best_route_idx is None:
            # Không có vị trí khả thi trong route cũ -> mở route mới.
            routes.append([0, int(customer), 0])
        else:
            routes[best_route_idx].insert(best_pos, int(customer))

    return routes


def get_vehicle_capacity(td) -> float:
    """
    Lấy tải trọng xe từ TensorDict nếu có.
    Một số phiên bản RL4CO dùng capacity, một số dùng vehicle_capacity.
    Nếu không tìm thấy, dùng 1.0 vì demand thường đã được chuẩn hóa theo capacity.
    """
    for key in ("vehicle_capacity", "capacity"):
        try:
            if key in td.keys():
                value = td[key]
                if isinstance(value, torch.Tensor):
                    return float(value.flatten()[0].detach().cpu().item())
                return float(value)
        except Exception:
            pass
    return 1.0


def get_coords_demands_from_td(td):
    """
    Lấy tọa độ và demand từ TensorDict RL4CO.

    Hàm này cố gắng tương thích với cả hai dạng:
    - locs đã gồm depot.
    - depot nằm riêng và locs chỉ gồm khách hàng.
    """
    locs_tensor = td["locs"][0].detach().cpu()
    locs = [(float(x), float(y)) for x, y in locs_tensor.numpy()]

    # Nếu có depot riêng, ưu tiên ghép depot + locs khách hàng.
    try:
        if "depot" in td.keys():
            depot_tensor = td["depot"][0].detach().cpu()
            depot = (float(depot_tensor[0].item()), float(depot_tensor[1].item()))
            coords = [depot] + locs
        else:
            coords = locs
    except Exception:
        coords = locs

    demand_raw = td["demand"][0].detach().cpu().numpy()
    demands = [0.0] + [float(x) for x in demand_raw]

    # Nếu coords đang nhiều hơn demands đúng 1 depot là ổn.
    # Nếu locs trong repo cũ đã gồm depot, demands cũng đã thêm 0 nên vẫn khớp.
    return coords, demands


def model_actions_to_priority_order(model, td, num_customers: int, beam_width: int = 5) -> List[int]:
    """
    Dùng mô hình PPO + Attention để sinh thứ tự ưu tiên khách hàng.
    Ở đây dùng beam_search để lấy chuỗi hành động có chất lượng tốt hơn greedy.
    """
    with torch.no_grad():
        out = model(td.clone(), decode_type="beam_search", beam_width=beam_width)
    actions = out["actions"][0].detach().cpu().numpy().tolist()
    return actions_to_customer_order(actions, num_customers=num_customers)
