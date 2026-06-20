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
        model.eval()
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


def dataset_table_from_batch(batch, map_name: str, seed_value: int) -> pd.DataFrame:
    """Convert the randomly generated CVRP instance into a readable table."""
    coords = batch.coords_with_depot()[0].detach().cpu()
    demands = batch.demand_with_depot()[0].detach().cpu()
    capacity = float(batch.capacity[0].item())

    rows = []
    for node_id in range(coords.size(0)):
        rows.append(
            {
                "MAP": map_name,
                "SEED": seed_value,
                "NODE": node_id,
                "TYPE": "Depot" if node_id == 0 else "Customer",
                "X": round(float(coords[node_id, 0].item()), 4),
                "Y": round(float(coords[node_id, 1].item()), 4),
                "DEMAND": round(float(demands[node_id].item()), 2),
                "CAPACITY_Q": capacity,
            }
        )
    return pd.DataFrame(rows)


def actions_from_routes(routes: list[list[int]]) -> list[int]:
    actions: list[int] = []
    for route in routes:
        # route has form [0, ..., 0]. Keep the ending 0 so validate_actions sees depot return.
        actions.extend(route[1:])
    return actions


def solve_one_map(model, config: VRPConfig, map_seed: int, decode_type: str, beam_width: int):
    batch = generate_batch(1, config, seed=int(map_seed))
    with torch.no_grad():
        result = learning_based_insertion(
            model,
            batch,
            decode_type=decode_type,
            beam_width=beam_width,
        )
    routes = result.routes[0]
    priority_order = result.priority_orders[0]
    actions_for_validation = actions_from_routes(routes)
    is_valid, valid_message = validate_actions(
        actions_for_validation,
        batch.demand[0].cpu(),
        config.capacity,
    )
    return batch, result, routes, priority_order, is_valid, valid_message


def route_table(routes: list[list[int]], demand: torch.Tensor) -> pd.DataFrame:
    loads = route_loads(routes, demand)
    return pd.DataFrame(
        {
            "ROUTE": list(range(1, len(routes) + 1)),
            "LOAD": loads,
            "PATH": [" -> ".join(map(str, route)) for route in routes],
        }
    )


def run_many_maps(
    model: NazariVRPModel,
    config: VRPConfig,
    base_seed: int,
    num_maps: int,
    beam_width: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result_rows: list[dict[str, object]] = []
    dataset_tables: list[pd.DataFrame] = []

    for map_idx in range(num_maps):
        map_name = f"MAP_{map_idx + 1:02d}"
        map_seed = int(base_seed) + map_idx

        map_batch = generate_batch(1, config, seed=map_seed)
        dataset_tables.append(dataset_table_from_batch(map_batch, map_name, map_seed))

        for method_name, decode_type in [
            ("Greedy-guided Insertion", "greedy"),
            ("Beam-guided Insertion", "beam"),
        ]:
            with torch.no_grad():
                result = learning_based_insertion(
                    model,
                    map_batch,
                    decode_type=decode_type,
                    beam_width=beam_width,
                )

            routes = result.routes[0]

            is_valid, valid_message = validate_actions(
                actions_from_routes(routes),
                map_batch.demand[0].cpu(),
                config.capacity,
            )

            result_rows.append(
                {
                    "MAP": map_name,
                    "SEED": map_seed,
                    "METHOD": method_name,
                    "COST": round(float(result.cost.item()), 4),
                    "ROUTES": len(routes),
                    "CAPACITY_Q": config.capacity,
                    "VALID": "OK" if is_valid else valid_message,
                    "PRIORITY_ORDER": " -> ".join(map(str, result.priority_orders[0])),
                    "ROUTE_PATHS": " | ".join(
                        " -> ".join(map(str, route)) for route in routes
                    ),
                }
            )

    results_df = pd.DataFrame(result_rows)
    datasets_df = pd.concat(dataset_tables, ignore_index=True)

    return results_df, datasets_df


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
    num_maps = st.slider("Number of MAPs", min_value=1, max_value=30, value=10, step=1)

checkpoint_mtime = CHECKPOINT_PATH.stat().st_mtime if CHECKPOINT_PATH.exists() else None
model, model_status = get_model(checkpoint_mtime)

current_decode_type = "beam" if decode_label == "Beam Search" else "greedy"
batch, result, routes, priority_order, is_valid, valid_message = solve_one_map(
    model=model,
    config=config,
    map_seed=int(seed),
    decode_type=current_decode_type,
    beam_width=beam_width,
)

st.info(model_status)
if not is_valid:
    st.error(valid_message)

st.subheader("Current MAP result")
top = st.columns(4)
top[0].metric("Cost", f"{float(result.cost.item()):.4f}")
top[1].metric("Routes", len(routes))
top[2].metric("Capacity", f"{config.capacity:.0f}")
top[3].metric("Validity", "OK" if is_valid else "Fail")

st.plotly_chart(plot_routes(batch, routes), width="stretch")
st.dataframe(route_table(routes, batch.demand[0].cpu()), width="stretch", hide_index=True)

st.subheader("Learned priority order")
st.code(" -> ".join(map(str, priority_order)), language="text")

with st.expander("Dữ liệu ngẫu nhiên của MAP hiện tại", expanded=True):
    current_dataset_df = dataset_table_from_batch(
        batch=batch,
        map_name=f"MAP_SEED_{int(seed)}",
        seed_value=int(seed),
    )
    st.dataframe(current_dataset_df, width="stretch", hide_index=True)
    st.download_button(
        "Tải dữ liệu MAP hiện tại (.csv)",
        data=current_dataset_df.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"dataset_map_seed_{int(seed)}.csv",
        mime="text/csv",
    )

st.divider()
st.subheader("Bảng so sánh nhiều MAP và dữ liệu đầu vào từng MAP")
st.caption(
    "Mỗi MAP được sinh ngẫu nhiên theo seed riêng: MAP_01 dùng seed gốc, MAP_02 dùng seed gốc + 1, ..."
)

if st.button("Chạy bảng nhiều MAP", type="primary"):
    with st.spinner("Đang chạy các MAP và tạo bảng dữ liệu đầu vào..."):
        results_df, datasets_df = run_many_maps(
            model=model,
            config=config,
            base_seed=int(seed),
            num_maps=int(num_maps),
            beam_width=beam_width,
        )

    st.markdown("### Kết quả các MAP")
    st.dataframe(results_df, width="stretch", hide_index=True)
    st.download_button(
        "Tải bảng kết quả các MAP (.csv)",
        data=results_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="cvrp_map_results.csv",
        mime="text/csv",
    )

    st.markdown("### Dữ liệu ngẫu nhiên của từng MAP")
    st.dataframe(datasets_df, width="stretch", hide_index=True)
    st.download_button(
        "Tải dữ liệu đầu vào từng MAP (.csv)",
        data=datasets_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="cvrp_map_datasets.csv",
        mime="text/csv",
    )

    with st.expander("Xem riêng từng MAP"):
        selected_map = st.selectbox("Chọn MAP", sorted(datasets_df["MAP"].unique()))
        st.dataframe(
            datasets_df[datasets_df["MAP"] == selected_map],
            width="stretch",
            hide_index=True,
        )

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
