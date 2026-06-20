from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

from nazari_vrp import (
    NazariVRPModel,
    VRPConfig,
    generate_batch,
    learning_based_insertion,
    validate_actions,
)
from nazari_vrp.train import load_checkpoint


CHECKPOINT_PATH = Path("nazari_vrp_checkpoint.pt")


st.set_page_config(page_title="Learning-based Insertion for CVRP", layout="wide")


@st.cache_resource
def get_model(checkpoint_mtime: float | None) -> tuple[NazariVRPModel, str]:
    if CHECKPOINT_PATH.exists():
        model = load_checkpoint(CHECKPOINT_PATH, device="cpu")
        return model, f"Loaded checkpoint: {CHECKPOINT_PATH}"

    model = NazariVRPModel()
    model.eval()
    return model, "No checkpoint found. Showing an untrained model for code demo."


def plot_routes(batch, routes: list[list[int]]) -> go.Figure:
    coords = batch.coords_with_depot()[0].detach().cpu()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[float(coords[0, 0])],
            y=[float(coords[0, 1])],
            mode="markers+text",
            marker=dict(size=16, color="#ef4444", symbol="star"),
            text=["Depot"],
            textposition="top center",
            name="Depot",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=coords[1:, 0].tolist(),
            y=coords[1:, 1].tolist(),
            mode="markers+text",
            marker=dict(size=9, color="#2563eb"),
            text=[str(i) for i in range(1, coords.size(0))],
            textposition="top center",
            name="Customers",
        )
    )

    palette = [
        "#111827",
        "#16a34a",
        "#f97316",
        "#7c3aed",
        "#0891b2",
        "#be123c",
        "#4d7c0f",
    ]
    for idx, route in enumerate(routes, start=1):
        route_coords = coords[route]
        fig.add_trace(
            go.Scatter(
                x=route_coords[:, 0].tolist(),
                y=route_coords[:, 1].tolist(),
                mode="lines+markers",
                line=dict(width=3, color=palette[(idx - 1) % len(palette)]),
                marker=dict(size=6),
                name=f"Route {idx}",
            )
        )

    fig.update_layout(
        height=620,
        margin=dict(l=20, r=20, t=35, b=20),
        xaxis=dict(range=[-0.05, 1.05], title="x"),
        yaxis=dict(range=[-0.05, 1.05], title="y", scaleanchor="x", scaleratio=1),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def route_loads(routes: list[list[int]], demand: torch.Tensor) -> list[float]:
    loads = []
    for route in routes:
        total = 0.0
        for node in route:
            if node > 0:
                total += float(demand[node - 1].item())
        loads.append(total)
    return loads


st.title("Learning-based Insertion Heuristic for CVRP")
st.caption(
    "Nazari-style RNN + attention learns a customer priority order, then Best Insertion places each customer at the lowest extra-distance position."
)

with st.sidebar:
    st.header("Experiment")
    num_customers = st.selectbox("Customers", [10, 20, 50, 100], index=1)
    config = VRPConfig(num_customers=num_customers)
    seed = st.number_input("Seed", min_value=0, max_value=999999, value=42, step=1)
    decode_label = st.radio("Priority decoder", ["Greedy", "Beam Search"], horizontal=False)
    beam_width = st.slider("Beam width", min_value=2, max_value=20, value=5, step=1)

checkpoint_mtime = CHECKPOINT_PATH.stat().st_mtime if CHECKPOINT_PATH.exists() else None
model, model_status = get_model(checkpoint_mtime)

batch = generate_batch(1, config, seed=int(seed))
with torch.no_grad():
    result = learning_based_insertion(
        model,
        batch,
        decode_type="beam" if decode_label == "Beam Search" else "greedy",
        beam_width=beam_width,
    )

routes = result.routes[0]
priority_order = result.priority_orders[0]
actions_for_validation = []
for route in routes:
    actions_for_validation.extend(route[1:])
is_valid, valid_message = validate_actions(
    actions_for_validation,
    batch.demand[0].cpu(),
    config.capacity,
)
loads = route_loads(routes, batch.demand[0].cpu())

top = st.columns(4)
top[0].metric("Cost", f"{float(result.cost.item()):.4f}")
top[1].metric("Routes", len(routes))
top[2].metric("Capacity", f"{config.capacity:.0f}")
top[3].metric("Validity", "OK" if is_valid else "Fail")

st.info(model_status)
if not is_valid:
    st.error(valid_message)

st.plotly_chart(plot_routes(batch, routes), width="stretch")

table = pd.DataFrame(
    {
        "route": list(range(1, len(routes) + 1)),
        "load": loads,
        "path": [" -> ".join(map(str, route)) for route in routes],
    }
)
st.dataframe(table, width="stretch", hide_index=True)

st.subheader("Learned priority order")
st.code(" -> ".join(map(str, priority_order)), language="text")

with st.expander("Paper mapping"):
    st.markdown(
        """
- Static input: depot and customer coordinates in the unit square.
- Dynamic input: remaining customer demand and current vehicle load.
- Decoder: recurrent state updated from the previously selected node.
- Attention: scores every feasible destination at each step.
- Masking: visited customers, customers exceeding remaining load, and invalid depot repeats are blocked.
- Learning part: the Nazari-style policy produces a customer priority order.
- Insertion part: each customer is inserted into the feasible route position with the smallest Delta(a,j,b).
- Reward: negative total distance after insertion, so maximizing reward improves the insertion-guided solution.
- Inference: greedy-guided insertion and beam-guided insertion.
"""
    )
