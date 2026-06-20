import torch

from nazari_vrp import (
    NazariVRPModel,
    VRPConfig,
    beam_search_decode,
    best_insertion_construct,
    generate_batch,
    learning_based_insertion,
    rollout,
    solution_cost,
    validate_actions,
)


def test_greedy_rollout_returns_valid_routes():
    config = VRPConfig(num_customers=10)
    batch = generate_batch(4, config, seed=123)
    model = NazariVRPModel(embed_dim=32, hidden_dim=32)

    with torch.no_grad():
        result = rollout(model, batch, decode_type="greedy")

    assert result.actions.shape[0] == 4
    assert torch.isfinite(result.cost).all()

    for idx in range(batch.batch_size):
        valid, message = validate_actions(
            result.actions[idx].tolist(),
            batch.demand[idx],
            config.capacity,
        )
        assert valid, message


def test_beam_search_returns_valid_route():
    config = VRPConfig(num_customers=10)
    batch = generate_batch(1, config, seed=456)
    model = NazariVRPModel(embed_dim=32, hidden_dim=32)

    with torch.no_grad():
        result = beam_search_decode(model, batch, beam_width=3)

    valid, message = validate_actions(
        result.actions[0].tolist(),
        batch.demand[0],
        config.capacity,
    )
    assert valid, message
    assert torch.isfinite(result.cost).all()


def test_best_insertion_constructs_capacity_feasible_routes():
    config = VRPConfig(num_customers=10)
    batch = generate_batch(1, config, seed=789)
    coords = batch.coords_with_depot()[0]
    demand = batch.demand[0]
    priority_order = list(range(1, config.num_customers + 1))

    routes = best_insertion_construct(priority_order, coords, demand, config.capacity)
    actions = []
    for route in routes:
        actions.extend(route[1:])

    valid, message = validate_actions(actions, demand, config.capacity)
    assert valid, message
    assert solution_cost(routes, coords) > 0


def test_learning_based_insertion_returns_valid_routes():
    config = VRPConfig(num_customers=10)
    batch = generate_batch(2, config, seed=321)
    model = NazariVRPModel(embed_dim=32, hidden_dim=32)

    with torch.no_grad():
        result = learning_based_insertion(model, batch, decode_type="greedy")

    assert len(result.routes) == 2
    assert len(result.priority_orders[0]) == config.num_customers
    assert torch.isfinite(result.cost).all()

    for idx, routes in enumerate(result.routes):
        actions = []
        for route in routes:
            actions.extend(route[1:])
        valid, message = validate_actions(actions, batch.demand[idx], config.capacity)
        assert valid, message
