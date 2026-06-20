from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch.distributions import Categorical

from nazari_vrp.data import VRPBatch
from nazari_vrp.model import NazariVRPModel


@dataclass
class DecodeResult:
    actions: torch.Tensor
    cost: torch.Tensor
    reward: torch.Tensor
    log_likelihood: torch.Tensor
    entropy: torch.Tensor
    value: torch.Tensor


def build_action_mask(
    remaining: torch.Tensor,
    load: torch.Tensor,
    current_node: torch.Tensor,
    active: torch.Tensor,
) -> torch.Tensor:
    """Mask infeasible next destinations.

    Customers are feasible only if they still have demand and fit into the
    current remaining load. The depot is feasible after leaving it, and it is
    forced when no customer can be served from the current load.
    """
    mask = torch.zeros_like(remaining, dtype=torch.bool)
    feasible_customers = (remaining[:, 1:] > 1e-6) & (
        remaining[:, 1:] <= load[:, None] + 1e-6
    )
    has_feasible_customer = feasible_customers.any(dim=1)
    has_unserved_customer = remaining[:, 1:].sum(dim=1) > 1e-6
    at_depot = current_node == 0

    mask[:, 1:] = feasible_customers
    mask[:, 0] = (~at_depot) | (~has_feasible_customer & has_unserved_customer)

    mask[~active] = False
    mask[~active, 0] = True

    no_action = ~mask.any(dim=1)
    mask[no_action, 0] = True
    return mask


def rollout(
    model: NazariVRPModel,
    batch: VRPBatch,
    decode_type: str = "sampling",
    max_steps: int | None = None,
) -> DecodeResult:
    """Construct CVRP routes autoregressively."""
    if decode_type not in {"sampling", "greedy"}:
        raise ValueError("decode_type must be 'sampling' or 'greedy'")

    device = batch.depot.device
    batch_size = batch.batch_size
    max_steps = max_steps or (3 * batch.num_customers + 10)

    coords = batch.coords_with_depot()
    static_emb = model.encode_static(batch)
    remaining = batch.demand_with_depot().clone()
    capacity = batch.capacity
    load = capacity.clone()
    current_node = torch.zeros(batch_size, dtype=torch.long, device=device)
    hidden = model.initial_hidden(batch_size, device)
    value = model.baseline_value(static_emb, remaining, capacity)

    actions: list[torch.Tensor] = []
    log_likelihood = torch.zeros(batch_size, device=device)
    entropy = torch.zeros(batch_size, device=device)
    cost = torch.zeros(batch_size, device=device)

    for _ in range(max_steps):
        active = ~((remaining[:, 1:].sum(dim=1) <= 1e-6) & (current_node == 0))
        if not bool(active.any()):
            break

        mask = build_action_mask(remaining, load, current_node, active)
        logits, hidden = model.step_logits(
            static_emb=static_emb,
            remaining=remaining,
            load=load,
            capacity=capacity,
            current_node=current_node,
            hidden=hidden,
            action_mask=mask,
        )
        distribution = Categorical(logits=logits)

        if decode_type == "sampling":
            action = distribution.sample()
        else:
            action = logits.argmax(dim=1)
        action = torch.where(active, action, torch.zeros_like(action))

        log_likelihood = log_likelihood + distribution.log_prob(action) * active.float()
        entropy = entropy + distribution.entropy() * active.float()

        from_xy = coords.gather(
            1,
            current_node.view(batch_size, 1, 1).expand(-1, 1, 2),
        ).squeeze(1)
        to_xy = coords.gather(
            1,
            action.view(batch_size, 1, 1).expand(-1, 1, 2),
        ).squeeze(1)
        cost = cost + torch.linalg.norm(from_xy - to_xy, dim=1) * active.float()

        selected_demand = remaining.gather(1, action[:, None]).squeeze(1)
        is_depot = action == 0
        load = torch.where(is_depot, capacity, load - selected_demand)
        remaining = remaining.scatter(1, action[:, None], torch.zeros_like(action[:, None]).float())
        remaining[:, 0] = 0.0

        current_node = action
        actions.append(action)

    if actions:
        actions_tensor = torch.stack(actions, dim=1)
    else:
        actions_tensor = torch.zeros(batch_size, 0, dtype=torch.long, device=device)

    reward = -cost
    return DecodeResult(
        actions=actions_tensor,
        cost=cost,
        reward=reward,
        log_likelihood=log_likelihood,
        entropy=entropy,
        value=value,
    )


def _state_done(remaining: torch.Tensor, current_node: torch.Tensor) -> bool:
    return bool((remaining[:, 1:].sum() <= 1e-6) and (int(current_node.item()) == 0))


def beam_search_decode(
    model: NazariVRPModel,
    batch: VRPBatch,
    beam_width: int = 5,
    max_steps: int | None = None,
) -> DecodeResult:
    """Beam search for one CVRP instance.

    The beam is pruned by sequence log-probability, then the completed route
    with the shortest travel distance is returned, matching the paper's
    RL-BS evaluation idea.
    """
    if batch.batch_size != 1:
        raise ValueError("beam_search_decode currently expects batch_size=1")

    device = batch.depot.device
    max_steps = max_steps or (3 * batch.num_customers + 10)
    coords = batch.coords_with_depot()
    static_emb = model.encode_static(batch)
    capacity = batch.capacity

    start_state = {
        "actions": [],
        "remaining": batch.demand_with_depot().clone(),
        "load": capacity.clone(),
        "current": torch.zeros(1, dtype=torch.long, device=device),
        "hidden": model.initial_hidden(1, device),
        "score": 0.0,
        "cost": 0.0,
    }
    beams = [start_state]
    completed = []

    with torch.no_grad():
        for _ in range(max_steps):
            candidates = []
            for state in beams:
                if _state_done(state["remaining"], state["current"]):
                    completed.append(state)
                    candidates.append(state)
                    continue

                active = torch.tensor([True], device=device)
                mask = build_action_mask(
                    state["remaining"],
                    state["load"],
                    state["current"],
                    active,
                )
                logits, hidden = model.step_logits(
                    static_emb=static_emb,
                    remaining=state["remaining"],
                    load=state["load"],
                    capacity=capacity,
                    current_node=state["current"],
                    hidden=state["hidden"],
                    action_mask=mask,
                )
                log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)
                valid_count = int(mask.sum().item())
                top_k = min(max(1, beam_width), valid_count)
                top_scores, top_actions = torch.topk(log_probs, k=top_k)

                for log_prob, action in zip(top_scores, top_actions):
                    action = action.view(1)
                    from_xy = coords[:, int(state["current"].item()), :]
                    to_xy = coords[:, int(action.item()), :]
                    distance = float(torch.linalg.norm(from_xy - to_xy, dim=1).item())

                    selected_demand = state["remaining"].gather(1, action[:, None]).squeeze(1)
                    is_depot = int(action.item()) == 0
                    next_load = capacity.clone() if is_depot else state["load"] - selected_demand
                    next_remaining = state["remaining"].clone()
                    next_remaining[:, int(action.item())] = 0.0
                    next_remaining[:, 0] = 0.0

                    candidates.append(
                        {
                            "actions": [*state["actions"], int(action.item())],
                            "remaining": next_remaining,
                            "load": next_load,
                            "current": action,
                            "hidden": hidden.clone(),
                            "score": state["score"] + float(log_prob.item()),
                            "cost": state["cost"] + distance,
                        }
                    )

            candidates.sort(key=lambda item: item["score"], reverse=True)
            beams = candidates[:beam_width]
            if beams and all(_state_done(b["remaining"], b["current"]) for b in beams):
                completed.extend(beams)
                break

    if not completed:
        completed = beams

    best = min(completed, key=lambda item: item["cost"])
    actions = torch.tensor([best["actions"]], dtype=torch.long, device=device)
    cost = torch.tensor([best["cost"]], dtype=torch.float32, device=device)
    value = model.baseline_value(static_emb, batch.demand_with_depot(), capacity)
    return DecodeResult(
        actions=actions,
        cost=cost,
        reward=-cost,
        log_likelihood=torch.tensor([best["score"]], dtype=torch.float32, device=device),
        entropy=torch.zeros(1, device=device),
        value=value,
    )


def routes_from_actions(actions: Iterable[int]) -> list[list[int]]:
    routes: list[list[int]] = []
    route = [0]
    for raw_action in actions:
        action = int(raw_action)
        if action == 0:
            if len(route) > 1:
                route.append(0)
                routes.append(route)
                route = [0]
            continue
        route.append(action)

    if len(route) > 1:
        route.append(0)
        routes.append(route)
    return routes


def validate_actions(actions: Iterable[int], demand: torch.Tensor, capacity: float) -> tuple[bool, str]:
    visited = set()
    load = float(capacity)
    current_at_depot = True
    num_customers = int(demand.numel())

    for raw_action in actions:
        action = int(raw_action)
        if len(visited) == num_customers and action == 0:
            current_at_depot = True
            break

        if action == 0:
            load = float(capacity)
            current_at_depot = True
            continue

        current_at_depot = False
        if action < 1 or action > num_customers:
            return False, f"invalid customer id {action}"
        if action in visited:
            return False, f"customer {action} was visited more than once"

        need = float(demand[action - 1].item())
        if need > load + 1e-6:
            return False, f"capacity exceeded before customer {action}"

        load -= need
        visited.add(action)

    if len(visited) != num_customers:
        return False, "not all customers were served"
    if not current_at_depot:
        return False, "route does not end at depot"
    return True, "valid"
