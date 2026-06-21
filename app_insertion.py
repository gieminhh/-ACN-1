"""
app_insertion.py

Streamlit app dùng để chạy đúng hướng:
PPO + Attention sinh thứ tự ưu tiên khách hàng
+ Best Insertion Heuristic chèn khách hàng vào vị trí tốt nhất.

Chạy:
    streamlit run app_insertion.py
"""

import os
import time
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch

from learning_based_insertion import (
    best_insertion_construct,
    get_coords_demands_from_td,
    get_vehicle_capacity,
    model_actions_to_priority_order,
    route_load,
    solution_cost,
)
from rl4co.envs import CVRPEnv
from rl4co.models.rl.ppo import PPO
from rl4co.models.zoo.am.policy import AttentionModelPolicy

NUM_NODES = 20
MAP_COUNT = 10
DEFAULT_BEAM_WIDTH = 5
DEFAULT_SAMPLING_SAMPLES = 50
DEFAULT_TEMPERATURE = 0.8

CHECKPOINT_CANDIDATES = [
    "Bo_nao_AI_CVRP/ppo_attention_cvrp_best.ckpt",
    "ppo_attention_cvrp_best.ckpt",
    "ppo_attention_cvrp.ckpt",
    "epoch=99-step=156800.ckpt",
]

st.set_page_config(page_title="CVRP - Learning-based Insertion", layout="wide")


def find_checkpoint():
    for path in CHECKPOINT_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


@st.cache_resource
def load_env_and_model():
    """Nạp môi trường CVRP và model PPO + Attention cho app trung gian.

    File này so sánh decoder baseline với Best Insertion, còn bản đầy đủ cuối
    cùng nằm ở app_final_3_methods.py.
    """
    env = CVRPEnv(generator_params=dict(num_loc=NUM_NODES))
    policy = AttentionModelPolicy(env_name=env.name)
    checkpoint_path = find_checkpoint()

    if checkpoint_path is not None:
        model = PPO.load_from_checkpoint(
            checkpoint_path,
            env=env,
            policy=policy,
            map_location="cpu",
        )
        checkpoint_status = f"Đã load checkpoint: {checkpoint_path}"
    else:
        model = PPO(env, policy)
        checkpoint_status = "Không tìm thấy checkpoint, đang dùng model chưa train"

    model.to("cpu")
    model.eval()
    return env, model, checkpoint_status


def generate_td(env, seed: int):
    torch.manual_seed(seed)
    return env.reset(batch_size=[1])


def actions_to_routes(actions):
    """Chuyển actions decoder trực tiếp thành routes để hiển thị baseline Greedy/Sampling."""
    routes = []
    current_route = [0]

    for action in actions:
        node = int(action)
        if node == 0:
            if len(current_route) > 1:
                current_route.append(0)
                routes.append(current_route)
                current_route = [0]
        else:
            current_route.append(node)

    if len(current_route) > 1:
        current_route.append(0)
        routes.append(current_route)

    return routes


def get_cost_from_model_output(out):
    reward = out["reward"]
    if reward.numel() == 1:
        return -float(reward.item())
    return -float(reward.mean().item())


def solve_decoder_baseline(env, model, map_id, algorithm_key, sampling_samples=50, temperature=0.8):
    """Chạy Greedy hoặc Sampling như baseline cũ."""
    seed = 42 + int(map_id)
    td = generate_td(env, seed)

    start = time.time()
    with torch.no_grad():
        if algorithm_key == "greedy":
            out = model(td.clone(), decode_type="greedy")
        elif algorithm_key == "sampling":
            out = model(
                td.clone(),
                decode_type="sampling",
                samples=sampling_samples,
                temperature=temperature,
            )
        else:
            raise ValueError("algorithm_key phải là greedy hoặc sampling")

    runtime = time.time() - start
    actions = out["actions"][0].detach().cpu().numpy().tolist()
    routes = actions_to_routes(actions)
    coords, demands = get_coords_demands_from_td(td)

    return {
        "map": f"MAP_{int(map_id):02d}",
        "seed": seed,
        "algorithm_key": algorithm_key,
        "cost": get_cost_from_model_output(out),
        "runtime": runtime,
        "vehicles": len(routes),
        "routes": routes,
        "coords": coords,
        "demands": demands,
    }


def solve_learning_based_insertion(env, model, map_id, beam_width=5):
    """
    Thuật toán đề xuất đúng nghĩa chèn:
    1. PPO + Attention + Beam Search sinh thứ tự ưu tiên khách hàng.
    2. Best Insertion chèn từng khách hàng vào vị trí tăng cost nhỏ nhất.
    """
    # Hàm này là lõi của bản app_insertion:
    # model chỉ sinh thứ tự khách, còn Best Insertion mới dựng route cuối cùng.
    seed = 42 + int(map_id)
    td = generate_td(env, seed)
    coords, demands = get_coords_demands_from_td(td)
    capacity = get_vehicle_capacity(td)
    num_customers = len(demands) - 1

    start = time.time()
    priority_order = model_actions_to_priority_order(
        model=model,
        td=td,
        num_customers=num_customers,
        beam_width=beam_width,
    )

    routes = best_insertion_construct(
        customer_order=priority_order,
        coords=coords,
        demands=demands,
        capacity=capacity,
    )

    runtime = time.time() - start
    cost = solution_cost(routes, coords)

    return {
        "map": f"MAP_{int(map_id):02d}",
        "seed": seed,
        "algorithm_key": "insertion",
        "cost": cost,
        "runtime": runtime,
        "vehicles": len(routes),
        "routes": routes,
        "coords": coords,
        "demands": demands,
        "priority_order": priority_order,
        "capacity": capacity,
    }


def run_all_maps(env, model, beam_width=5, sampling_samples=50, temperature=0.8):
    """Chạy toàn bộ MAP và gom kết quả thành bảng so sánh."""
    rows = []
    detail = {}
    progress = st.progress(0)
    total_jobs = MAP_COUNT * 3
    done = 0

    for map_id in range(1, MAP_COUNT + 1):
        map_name = f"MAP_{map_id:02d}"
        detail[map_name] = {}

        greedy = solve_decoder_baseline(
            env, model, map_id, "greedy",
            sampling_samples=sampling_samples,
            temperature=temperature,
        )
        done += 1
        progress.progress(done / total_jobs)

        sampling = solve_decoder_baseline(
            env, model, map_id, "sampling",
            sampling_samples=sampling_samples,
            temperature=temperature,
        )
        done += 1
        progress.progress(done / total_jobs)

        insertion = solve_learning_based_insertion(
            env, model, map_id,
            beam_width=beam_width,
        )
        done += 1
        progress.progress(done / total_jobs)

        detail[map_name]["greedy"] = greedy
        detail[map_name]["sampling"] = sampling
        detail[map_name]["insertion"] = insertion

        costs = {
            "Greedy Decoder": greedy["cost"],
            "Sampling Decoder": sampling["cost"],
            "PPO + Attention + Best Insertion": insertion["cost"],
        }
        best_algorithm = min(costs.keys(), key=lambda k: costs[k])

        rows.append({
            "MAP": map_name,
            "GREEDY_COST": greedy["cost"],
            "GREEDY_TIME_S": greedy["runtime"],
            "GREEDY_VEHICLES": greedy["vehicles"],
            "SAMPLING_COST": sampling["cost"],
            "SAMPLING_TIME_S": sampling["runtime"],
            "SAMPLING_VEHICLES": sampling["vehicles"],
            "INSERTION_COST": insertion["cost"],
            "INSERTION_TIME_S": insertion["runtime"],
            "INSERTION_VEHICLES": insertion["vehicles"],
            "BEST_ALGORITHM": best_algorithm,
        })

    progress.empty()
    return pd.DataFrame(rows), detail


def metric_card(label, value):
    st.metric(label, value)


def plot_routes(coords, routes, title):
    fig = go.Figure()

    customer_x = [coords[i][0] for i in range(1, len(coords))]
    customer_y = [coords[i][1] for i in range(1, len(coords))]
    customer_text = [str(i) for i in range(1, len(coords))]

    fig.add_trace(go.Scatter(
        x=customer_x,
        y=customer_y,
        mode="markers+text",
        text=customer_text,
        textposition="top center",
        marker=dict(size=8),
        name="Customers",
    ))

    fig.add_trace(go.Scatter(
        x=[coords[0][0]],
        y=[coords[0][1]],
        mode="markers+text",
        text=["Depot"],
        textposition="top center",
        marker=dict(size=18, symbol="diamond"),
        name="Depot",
    ))

    for idx, route in enumerate(routes, start=1):
        xs = [coords[node][0] for node in route]
        ys = [coords[node][1] for node in route]
        fig.add_trace(go.Scatter(
            x=xs,
            y=ys,
            mode="lines+markers",
            name=f"Xe {idx}: {len(route) - 2} khách",
            line=dict(width=3),
            marker=dict(size=6),
        ))

    fig.update_layout(
        title=title,
        height=680,
        xaxis_title="X",
        yaxis_title="Y",
        template="plotly_white",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


env, model, checkpoint_status = load_env_and_model()

st.title("CVRP - Learning-based Insertion Heuristic")
st.caption("PPO + Attention sinh thứ tự ưu tiên khách hàng, sau đó Best Insertion chèn khách hàng vào vị trí tăng cost nhỏ nhất.")
st.info(checkpoint_status)

with st.sidebar:
    st.header("Tham số")
    beam_width = st.slider("Beam width dùng để sinh thứ tự ưu tiên", 2, 50, DEFAULT_BEAM_WIDTH, 1)
    sampling_samples = st.slider("Sampling samples", 10, 200, DEFAULT_SAMPLING_SAMPLES, 10)
    temperature = st.slider("Sampling temperature", 0.1, 2.0, DEFAULT_TEMPERATURE, 0.1)

tab1, tab2, tab3 = st.tabs(["Bảng so sánh", "Bản đồ lộ trình", "Giải thích thuật toán chèn"])

with tab1:
    run_clicked = st.button("Chạy / cập nhật bảng")
    if run_clicked or "comparison_df_insertion" not in st.session_state:
        with st.spinner("Đang chạy thực nghiệm..."):
            df, detail = run_all_maps(env, model, beam_width, sampling_samples, temperature)
            st.session_state["comparison_df_insertion"] = df
            st.session_state["detail_insertion"] = detail
            st.session_state["last_run_insertion"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    df = st.session_state["comparison_df_insertion"]
    detail = st.session_state["detail_insertion"]

    st.caption(f"Lần chạy gần nhất: {st.session_state.get('last_run_insertion', '')}")

    df_show = df.copy()
    for col in df_show.columns:
        if col.endswith("_COST") or col.endswith("_TIME_S"):
            df_show[col] = df_show[col].round(4)

    st.dataframe(df_show, use_container_width=True, height=450)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Greedy cost TB", f"{df['GREEDY_COST'].mean():.4f}")
    with c2:
        metric_card("Sampling cost TB", f"{df['SAMPLING_COST'].mean():.4f}")
    with c3:
        metric_card("Insertion cost TB", f"{df['INSERTION_COST'].mean():.4f}")
    with c4:
        best_avg = min(
            {
                "Greedy": df["GREEDY_COST"].mean(),
                "Sampling": df["SAMPLING_COST"].mean(),
                "Best Insertion": df["INSERTION_COST"].mean(),
            },
            key=lambda k: {
                "Greedy": df["GREEDY_COST"].mean(),
                "Sampling": df["SAMPLING_COST"].mean(),
                "Best Insertion": df["INSERTION_COST"].mean(),
            }[k],
        )
        metric_card("Best avg", best_avg)

    cost_long = df[["MAP", "GREEDY_COST", "SAMPLING_COST", "INSERTION_COST"]].melt(
        id_vars="MAP",
        var_name="ALGORITHM",
        value_name="COST",
    )
    cost_long["ALGORITHM"] = cost_long["ALGORITHM"].replace({
        "GREEDY_COST": "Greedy",
        "SAMPLING_COST": "Sampling",
        "INSERTION_COST": "PPO + Best Insertion",
    })

    fig_cost = px.bar(
        cost_long,
        x="MAP",
        y="COST",
        color="ALGORITHM",
        barmode="group",
        text="COST",
    )
    fig_cost.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig_cost.update_layout(height=520, template="plotly_white")
    st.plotly_chart(fig_cost, use_container_width=True)

with tab2:
    if "detail_insertion" not in st.session_state:
        with st.spinner("Đang tạo dữ liệu ban đầu..."):
            df, detail = run_all_maps(env, model, beam_width, sampling_samples, temperature)
            st.session_state["comparison_df_insertion"] = df
            st.session_state["detail_insertion"] = detail

    detail = st.session_state["detail_insertion"]

    col1, col2 = st.columns(2)
    with col1:
        selected_map = st.selectbox("Chọn map", list(detail.keys()))
    with col2:
        algorithm_label = st.selectbox(
            "Chọn thuật toán",
            ["Greedy Decoder", "Sampling Decoder", "PPO + Attention + Best Insertion"],
        )

    algorithm_map = {
        "Greedy Decoder": "greedy",
        "Sampling Decoder": "sampling",
        "PPO + Attention + Best Insertion": "insertion",
    }

    result = detail[selected_map][algorithm_map[algorithm_label]]
    title = f"{algorithm_label} - {selected_map} | cost={result['cost']:.4f} | vehicles={result['vehicles']}"

    st.plotly_chart(plot_routes(result["coords"], result["routes"], title), use_container_width=True)

    st.subheader("Routes")
    for idx, route in enumerate(result["routes"], start=1):
        load = route_load(route, result["demands"])
        st.write(f"Xe {idx} | Load={load:.4f} | " + " → ".join(map(str, route)))

    if algorithm_map[algorithm_label] == "insertion":
        st.subheader("Thứ tự ưu tiên do PPO + Attention + Beam sinh ra")
        st.write(result.get("priority_order", []))
        st.caption("Best Insertion không append bừa vào cuối tuyến; nó thử chèn từng khách hàng vào mọi vị trí khả thi và chọn vị trí có Delta nhỏ nhất.")

with tab3:
    st.markdown(
        """
### Thuật toán đề xuất trong file này

Thuật toán đề xuất gồm 2 bước:

**Bước 1: PPO + Attention sinh thứ tự ưu tiên khách hàng**

Mô hình đã huấn luyện sinh chuỗi hành động bằng Beam Search. Chuỗi này được chuyển thành thứ tự ưu tiên khách hàng, bỏ depot 0 và bỏ node trùng.

**Bước 2: Best Insertion Heuristic**

Với mỗi khách hàng `j` trong thứ tự ưu tiên, thuật toán thử chèn `j` vào mọi vị trí giữa hai node liên tiếp `(a,b)` trong các tuyến hiện có.

Công thức chi phí tăng thêm:

`Delta(a,j,b) = c(a,j) + c(j,b) - c(a,b)`

Thuật toán chọn vị trí có `Delta` nhỏ nhất và không làm tuyến vượt tải trọng `Q`.

Nếu không có tuyến nào chèn được, thuật toán mở tuyến mới `[0, j, 0]`.

### Điểm khác với code append cũ

Code append cũ chỉ làm:

`0 → A → B → C → 0`

File này làm đúng insertion hơn vì một khách hàng có thể được chèn vào giữa tuyến:

`0 → A → C → 0`

nếu chèn `B`, thuật toán thử:

`0 → B → A → C → 0`

`0 → A → B → C → 0`

`0 → A → C → B → 0`

rồi chọn vị trí tăng cost nhỏ nhất.
        """
    )
