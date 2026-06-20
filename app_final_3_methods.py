"""
app_final_3_methods.py

Bản cuối: so sánh 3 phương pháp, dùng CÙNG MỘT cách tính cost cho tất cả.

1. Greedy Decoder
2. Sampling Decoder
3. Proposed Hybrid Learning-based Insertion Heuristic

Sửa lỗi quan trọng:
- Không lấy cost từ reward của model để so sánh với cost tự tính.
- Tất cả cost đều được tính lại bằng solution_cost(routes, coords).
- Vì vậy nếu Proposed chọn nguồn "sampling" thì Proposed cost không thể tự nhiên lớn hơn Sampling do lệch cách tính.
"""

import os
import time
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch

from rl4co.envs import CVRPEnv
from rl4co.models.rl.ppo import PPO
from rl4co.models.zoo.am.policy import AttentionModelPolicy

from learning_based_insertion import (
    actions_to_customer_order,
    best_insertion_construct,
    get_coords_demands_from_td,
    get_vehicle_capacity,
    route_load,
    solution_cost,
)
from insertion_repair import insertion_repair


NUM_NODES = 20
MAP_COUNT = 10
DEFAULT_BEAM_WIDTH = 5
DEFAULT_SAMPLING_SAMPLES = 50
DEFAULT_TEMPERATURE = 0.8

CHECKPOINT_CANDIDATES = [
    "Bo_nao_AI_CVRP/ppo_attention_cvrp_best.ckpt",
    "Bo_nao_AI_CVRP/epoch=99-step=156800.ckpt",
    "ppo_attention_cvrp_best.ckpt",
    "ppo_attention_cvrp.ckpt",
    "epoch=99-step=156800.ckpt",
]

st.set_page_config(page_title="CVRP - Final 3 Methods", layout="wide")


def actions_to_routes(actions):
    routes = []
    current = [0]

    for action in actions:
        node = int(action)
        if node == 0:
            if len(current) > 1:
                current.append(0)
                routes.append(current)
                current = [0]
        else:
            current.append(node)

    if len(current) > 1:
        current.append(0)
        routes.append(current)

    return routes


def find_checkpoint():
    for path in CHECKPOINT_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


@st.cache_resource
def load_env_and_model():
    env = CVRPEnv(generator_params=dict(num_loc=NUM_NODES))
    policy = AttentionModelPolicy(env_name=env.name)

    ckpt = find_checkpoint()
    if ckpt:
        model = PPO.load_from_checkpoint(
            ckpt,
            env=env,
            policy=policy,
            map_location="cpu",
        )
        status = f"Đã load checkpoint: {ckpt}"
    else:
        model = PPO(env, policy)
        status = "Không tìm thấy checkpoint, đang dùng model chưa train"

    model.to("cpu")
    model.eval()
    return env, model, status


def generate_td(env, seed):
    torch.manual_seed(seed)
    return env.reset(batch_size=[1])


def flatten_best_actions(out):
    """
    Lấy chuỗi action tốt nhất từ output model.
    Cost KHÔNG lấy từ đây để so sánh, vì phải tính thống nhất lại bằng solution_cost.
    """
    reward = out["reward"].detach().cpu().reshape(-1)
    costs = -reward
    best_idx = int(torch.argmin(costs).item())

    actions_tensor = out["actions"].detach().cpu()

    if actions_tensor.ndim == 1:
        actions_2d = actions_tensor.unsqueeze(0)
    elif actions_tensor.ndim == 2:
        actions_2d = actions_tensor
    else:
        actions_2d = actions_tensor.reshape(-1, actions_tensor.shape[-1])

    if best_idx >= actions_2d.shape[0]:
        best_idx = 0

    return actions_2d[best_idx].numpy().tolist()


def decode_solution(model, td, decode_type, coords, beam_width, sampling_samples, temperature, seed):
    """
    Sinh routes và tính cost thống nhất bằng solution_cost(routes, coords).
    """
    torch.manual_seed(seed)

    start = time.time()
    with torch.no_grad():
        if decode_type == "greedy":
            out = model(td.clone(), decode_type="greedy")
        elif decode_type == "sampling":
            out = model(
                td.clone(),
                decode_type="sampling",
                samples=sampling_samples,
                temperature=temperature,
            )
        elif decode_type == "beam":
            out = model(
                td.clone(),
                decode_type="beam_search",
                beam_width=beam_width,
            )
        else:
            raise ValueError("decode_type không hợp lệ")

    runtime = time.time() - start
    actions = flatten_best_actions(out)
    routes = actions_to_routes(actions)
    cost = solution_cost(routes, coords)

    return {
        "name": decode_type,
        "cost": cost,
        "routes": routes,
        "runtime": runtime,
        "actions": actions,
    }


def repair_candidate(candidate, coords, demands, capacity):
    """
    Repair một nghiệm ứng viên.

    Quan trọng:
    - cost_before được tính lại bằng solution_cost.
    - cost_after cũng từ insertion_repair theo cùng cost.
    - Nếu repair không cải thiện thì final_cost <= initial_cost.
    """
    cost_before = solution_cost(candidate["routes"], coords)

    repaired_routes, info = insertion_repair(
        routes=candidate["routes"],
        coords=coords,
        demands=demands,
        capacity=capacity,
        max_iter=50,
    )

    return {
        "source": candidate["name"],
        "cost_before": cost_before,
        "cost": info["final_cost"],
        "improvement": cost_before - info["final_cost"],
        "routes": repaired_routes,
        "iterations": info["iterations"],
        "runtime": candidate["runtime"],
    }


def solve_map(env, model, map_id, beam_width, sampling_samples, temperature):
    seed = 42 + int(map_id)
    td = generate_td(env, seed)

    coords, demands = get_coords_demands_from_td(td)
    capacity = get_vehicle_capacity(td)

    greedy = decode_solution(
        model, td, "greedy", coords,
        beam_width, sampling_samples, temperature,
        seed + 1,
    )
    sampling = decode_solution(
        model, td, "sampling", coords,
        beam_width, sampling_samples, temperature,
        seed + 2,
    )

    # Beam chỉ dùng nội bộ cho Proposed, không hiển thị thành phương pháp thứ 4.
    beam = decode_solution(
        model, td, "beam", coords,
        beam_width, sampling_samples, temperature,
        seed + 3,
    )

    # Learning-guided Best Insertion: AI sinh thứ tự, heuristic chèn từng khách.
    with torch.no_grad():
        out_for_order = model(td.clone(), decode_type="beam_search", beam_width=beam_width)

    order_actions = flatten_best_actions(out_for_order)
    order = actions_to_customer_order(order_actions, num_customers=len(demands) - 1)

    guided_routes = best_insertion_construct(
        customer_order=order,
        coords=coords,
        demands=demands,
        capacity=capacity,
    )
    guided_cost = solution_cost(guided_routes, coords)

    guided = {
        "name": "guided_best_insertion",
        "cost": guided_cost,
        "routes": guided_routes,
        "runtime": 0.0,
        "actions": order,
    }

    # Proposed = nhiều nghiệm khởi tạo + insertion repair + chọn nghiệm có cost nhỏ nhất.
    start_proposed = time.time()
    candidates = [beam, guided]
    repaired_candidates = [repair_candidate(c, coords, demands, capacity) for c in candidates]
    proposed = min(repaired_candidates, key=lambda x: x["cost"])
    proposed_runtime = time.time() - start_proposed

    return {
        "map": f"MAP_{map_id:02d}",
        "seed": seed,
        "coords": coords,
        "demands": demands,
        "capacity": capacity,

        "greedy": {
            "cost": greedy["cost"],
            "runtime": greedy["runtime"],
            "vehicles": len(greedy["routes"]),
            "routes": greedy["routes"],
        },
        "sampling": {
            "cost": sampling["cost"],
            "runtime": sampling["runtime"],
            "vehicles": len(sampling["routes"]),
            "routes": sampling["routes"],
        },
        "proposed": {
            "cost": proposed["cost"],
            "runtime": proposed_runtime,
            "vehicles": len(proposed["routes"]),
            "routes": proposed["routes"],
            "source": proposed["source"],
            "cost_before": proposed["cost_before"],
            "improvement": proposed["improvement"],
            "repair_iterations": proposed["iterations"],
            "all_candidates": repaired_candidates,
        },
    }


def run_all(env, model, beam_width, sampling_samples, temperature):
    rows = []
    detail = {}
    progress = st.progress(0)

    for idx, map_id in enumerate(range(1, MAP_COUNT + 1), start=1):
        res = solve_map(env, model, map_id, beam_width, sampling_samples, temperature)
        detail[res["map"]] = res

        best_map = {
            "Greedy Decoder": res["greedy"]["cost"],
            "Sampling Decoder": res["sampling"]["cost"],
            "Proposed Hybrid Insertion": res["proposed"]["cost"],
        }
        best_alg = min(best_map.keys(), key=lambda k: best_map[k])

        rows.append(
            {
                "MAP": res["map"],

                "GREEDY_COST": res["greedy"]["cost"],
                "GREEDY_TIME_S": res["greedy"]["runtime"],
                "GREEDY_VEHICLES": res["greedy"]["vehicles"],

                "SAMPLING_COST": res["sampling"]["cost"],
                "SAMPLING_TIME_S": res["sampling"]["runtime"],
                "SAMPLING_VEHICLES": res["sampling"]["vehicles"],

                "PROPOSED_COST": res["proposed"]["cost"],
                "PROPOSED_TIME_S": res["proposed"]["runtime"],
                "PROPOSED_VEHICLES": res["proposed"]["vehicles"],
                "PROPOSED_SOURCE": res["proposed"]["source"],
                "PROPOSED_COST_BEFORE_REPAIR": res["proposed"]["cost_before"],
                "PROPOSED_IMPROVEMENT": res["proposed"]["improvement"],
                "PROPOSED_REPAIR_ITER": res["proposed"]["repair_iterations"],

                "BEST_ALGORITHM": best_alg,
            }
        )

        progress.progress(idx / MAP_COUNT)

    progress.empty()
    return pd.DataFrame(rows), detail


def plot_routes(coords, routes, title):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=[coords[i][0] for i in range(1, len(coords))],
            y=[coords[i][1] for i in range(1, len(coords))],
            mode="markers+text",
            text=[str(i) for i in range(1, len(coords))],
            textposition="top center",
            marker=dict(size=8),
            name="Customers",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[coords[0][0]],
            y=[coords[0][1]],
            mode="markers+text",
            text=["Depot"],
            textposition="top center",
            marker=dict(size=18, symbol="diamond"),
            name="Depot",
        )
    )

    for idx, route in enumerate(routes, start=1):
        fig.add_trace(
            go.Scatter(
                x=[coords[n][0] for n in route],
                y=[coords[n][1] for n in route],
                mode="lines+markers",
                name=f"Xe {idx}",
            )
        )

    fig.update_layout(
        title=title,
        height=680,
        template="plotly_white",
        xaxis_title="X",
        yaxis_title="Y",
    )
    return fig


env, model, ckpt_status = load_env_and_model()

st.title("CVRP - Final Comparison: 3 Methods")
st.caption("Greedy, Sampling và thuật toán đề xuất Hybrid Learning-based Insertion.")
st.info(ckpt_status)

with st.sidebar:
    st.header("Tham số")
    beam_width = st.slider("Beam width", 2, 50, DEFAULT_BEAM_WIDTH, 1)
    sampling_samples = st.slider("Sampling samples", 10, 200, DEFAULT_SAMPLING_SAMPLES, 10)
    temperature = st.slider("Sampling temperature", 0.1, 2.0, DEFAULT_TEMPERATURE, 0.1)

tab1, tab2, tab3 = st.tabs(["Bảng so sánh", "Bản đồ", "Giải thích"])

with tab1:
    if st.button("Chạy / cập nhật bảng") or "final3_df_v2" not in st.session_state:
        with st.spinner("Đang chạy thực nghiệm..."):
            df, detail = run_all(env, model, beam_width, sampling_samples, temperature)
            st.session_state["final3_df_v2"] = df
            st.session_state["final3_detail_v2"] = detail
            st.session_state["last_run_v2"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    df = st.session_state["final3_df_v2"]
    detail = st.session_state["final3_detail_v2"]

    st.caption(f"Lần chạy gần nhất: {st.session_state['last_run_v2']}")

    df_show = df.copy()
    for col in df_show.columns:
        if col.endswith("_COST") or col.endswith("_TIME_S") or col.endswith("_IMPROVEMENT"):
            df_show[col] = df_show[col].round(4)

    st.dataframe(df_show, use_container_width=True, height=450)

    avg_g = df["GREEDY_COST"].mean()
    avg_s = df["SAMPLING_COST"].mean()
    avg_p = df["PROPOSED_COST"].mean()

    cost_dict = {"Greedy": avg_g, "Sampling": avg_s, "Proposed": avg_p}
    best_avg = min(cost_dict, key=lambda x: cost_dict[x])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Greedy cost TB", f"{avg_g:.4f}")
    c2.metric("Sampling cost TB", f"{avg_s:.4f}")
    c3.metric("Proposed cost TB", f"{avg_p:.4f}")
    c4.metric("Best avg", best_avg)

    cost_long = df[["MAP", "GREEDY_COST", "SAMPLING_COST", "PROPOSED_COST"]].melt(
        id_vars="MAP",
        var_name="ALGORITHM",
        value_name="COST",
    )
    cost_long["ALGORITHM"] = cost_long["ALGORITHM"].replace(
        {
            "GREEDY_COST": "Greedy",
            "SAMPLING_COST": "Sampling",
            "PROPOSED_COST": "Proposed",
        }
    )

    fig = px.bar(
        cost_long,
        x="MAP",
        y="COST",
        color="ALGORITHM",
        barmode="group",
        text="COST",
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(height=520, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    if "final3_detail_v2" not in st.session_state:
        df, detail = run_all(env, model, beam_width, sampling_samples, temperature)
        st.session_state["final3_df_v2"] = df
        st.session_state["final3_detail_v2"] = detail

    detail = st.session_state["final3_detail_v2"]

    selected_map = st.selectbox("Chọn map", list(detail.keys()))
    method = st.selectbox(
        "Chọn phương pháp",
        ["Greedy Decoder", "Sampling Decoder", "Proposed Hybrid Insertion"],
    )

    res = detail[selected_map]
    if method == "Greedy Decoder":
        item = res["greedy"]
    elif method == "Sampling Decoder":
        item = res["sampling"]
    else:
        item = res["proposed"]

    st.plotly_chart(
        plot_routes(
            res["coords"],
            item["routes"],
            f"{method} - {selected_map} | cost={item['cost']:.4f}",
        ),
        use_container_width=True,
    )

    st.subheader("Thông tin route")
    st.write(f"Cost: {item['cost']:.4f}")
    st.write(f"Số xe: {item['vehicles']}")

    if method == "Proposed Hybrid Insertion":
        st.write(f"Nguồn nghiệm tốt nhất trong Proposed: {item['source']}")
        st.write(f"Cost trước repair: {item['cost_before']:.4f}")
        st.write(f"Mức cải thiện: {item['improvement']:.4f}")
        st.write(f"Số vòng repair: {item['repair_iterations']}")

        with st.expander("Ứng viên nội bộ của Proposed"):
            for cand in item["all_candidates"]:
                st.write(
                    f"{cand['source']}: before={cand['cost_before']:.4f}, "
                    f"after={cand['cost']:.4f}, improvement={cand['improvement']:.4f}"
                )

    for idx, route in enumerate(item["routes"], start=1):
        load = route_load(route, res["demands"])
        st.write(f"Xe {idx} | Load={load:.4f} | " + " → ".join(map(str, route)))

with tab3:
    st.markdown(
        """
### Bảng chỉ so sánh 3 phương pháp

1. Greedy Decoder  
2. Sampling Decoder  
3. Proposed Hybrid Learning-based Insertion  

Beam Search không còn là một cột so sánh riêng. Nó chỉ là một nguồn nghiệm khởi tạo nội bộ trong thuật toán đề xuất.

### Thuật toán đề xuất

Thuật toán đề xuất là một heuristic lai:

- PPO + Attention sinh các nghiệm ứng viên.
- Learning-based Best Insertion tạo nghiệm theo hướng chèn.
- Insertion Repair cải thiện nghiệm bằng phép chèn lại khách hàng.
- Cuối cùng chọn nghiệm có cost nhỏ nhất trong các ứng viên.

### Sửa lỗi so sánh cost

Tất cả phương pháp trong bảng đều được tính lại cost bằng cùng một hàm `solution_cost(routes, coords)`.
Nhờ vậy kết quả Greedy, Sampling và Proposed được so sánh trên cùng một thước đo.

### Không khẳng định tối ưu toàn cục

Thuật toán chỉ chọn nghiệm tốt nhất trong các ứng viên đã sinh và đã repair.  
Không chứng minh tối ưu toàn cục cho CVRP.
        """
    )
