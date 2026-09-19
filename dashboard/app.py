"""
dashboard/app.py
-----------------
Interactive Streamlit dashboard for the Quantum-Enhanced Adaptive
Urban Traffic Optimization prototype.

Run with:   streamlit run dashboard/app.py    (from the project root)

Two tabs:
    "Live Network"  -- build a network, pick a controller (quantum /
        fixed-time / greedy), step the simulation, fire dynamic events
        (congestion spike / accident / road closure), dispatch an
        ambulance through an emergency green corridor, and watch the
        graph + metrics update live.
    "Quantum vs Classical"  -- run all three controllers on identical
        conditions (same seed, same optional congestion spike / ambulance
        dispatch at the same step) and compare waiting time, queue
        length, throughput, fuel and CO2 side by side.
"""

import os
import sys

# Make `import src...` work no matter what directory `streamlit run` is
# launched from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")  # headless-safe backend, avoids GUI issues on servers/CI
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import streamlit as st

from src.classical_baseline import FixedTimeController, GreedyController  # noqa: F401 (used indirectly)
from src.emergency_corridor import EmergencyCorridor
from src.metrics import compare, compute_metrics
from src.simulation import TrafficSimulator
from src.traffic_network import TrafficNetwork

st.set_page_config(page_title="Quantum Traffic Optimizer", layout="wide")

CONTROLLER_LABELS = {
    "quantum": "Quantum (QAOA)",
    "fixed": "Classical -- Fixed-Time",
    "greedy": "Classical -- Greedy (adaptive, uncoordinated)",
}


# ----------------------------------------------------------------------
# Session state bootstrap
# ----------------------------------------------------------------------
def _ensure_state():
    if "network" not in st.session_state:
        st.session_state.network = TrafficNetwork(n_intersections=6, seed=42)
        st.session_state.corridor = EmergencyCorridor()
        st.session_state.sim = None
        st.session_state.controller_type = "quantum"
        st.session_state.history = []
        st.session_state.comparison = None


_ensure_state()


def _rebuild_network(n, seed):
    st.session_state.network = TrafficNetwork(n_intersections=n, seed=seed)
    st.session_state.corridor = EmergencyCorridor()
    st.session_state.sim = None
    st.session_state.history = []


def _ensure_sim(controller_type, decision_interval, qaoa_reps, qaoa_maxiter, coordination_weight, seed):
    """Create the simulator once per network, then just sync the tunable
    parameters onto it every rerun -- this way tweaking a slider doesn't
    throw away the accumulated history/queues, only a fresh network does."""
    if st.session_state.sim is None:
        st.session_state.sim = TrafficSimulator(
            st.session_state.network,
            controller_type=controller_type,
            decision_interval=decision_interval,
            qaoa_reps=qaoa_reps,
            qaoa_maxiter=qaoa_maxiter,
            coordination_weight=coordination_weight,
            seed=seed,
            corridor=st.session_state.corridor,
        )
    sim = st.session_state.sim
    sim.controller_type = controller_type
    sim.decision_interval = max(1, decision_interval)
    sim.qaoa_reps = qaoa_reps
    sim.qaoa_maxiter = qaoa_maxiter
    sim.coordination_weight = coordination_weight
    sim._fixed.cycle_length = decision_interval * 2
    return sim


# ----------------------------------------------------------------------
# Drawing helpers
# ----------------------------------------------------------------------
def draw_network(network: TrafficNetwork, corridor: EmergencyCorridor):
    g = network.graph
    pos = nx.get_node_attributes(g, "pos")
    if not pos:
        pos = nx.spring_layout(g, seed=1)

    fig, ax = plt.subplots(figsize=(5, 4))

    queue_totals = [sum(g.nodes[n]["queue"].values()) for n in g.nodes]
    vmax = max(queue_totals + [1])

    corridor_nodes = set(corridor.path) if corridor.is_active() else set()
    current_amb_node = corridor.current_node() if corridor.is_active() else None

    node_colors = queue_totals
    node_edgecolors = [
        "red" if n == current_amb_node else ("orange" if n in corridor_nodes else "black") for n in g.nodes
    ]
    node_linewidths = [3.0 if n == current_amb_node else (2.0 if n in corridor_nodes else 0.8) for n in g.nodes]

    edge_styles = []
    edge_colors = []
    for u, v, data in g.edges(data=True):
        on_corridor = corridor.is_active() and u in corridor.path and v in corridor.path
        edge_styles.append("dashed" if data["status"] == "closed" else "solid")
        edge_colors.append("orange" if on_corridor else ("lightgrey" if data["status"] == "closed" else "grey"))

    nx.draw_networkx_edges(g, pos, ax=ax, edge_color=edge_colors, style=edge_styles, width=2)
    nodes = nx.draw_networkx_nodes(
        g,
        pos,
        ax=ax,
        node_color=node_colors,
        cmap="YlOrRd",
        vmin=0,
        vmax=vmax,
        node_size=700,
        edgecolors=node_edgecolors,
        linewidths=node_linewidths,
    )
    labels = {n: f"{n}\n{'NS' if g.nodes[n]['phase'] == 0 else 'EW'}\u25CF" for n in g.nodes}
    nx.draw_networkx_labels(g, pos, labels=labels, ax=ax, font_size=8)
    fig.colorbar(nodes, ax=ax, label="Total queue length", shrink=0.8)
    ax.set_title("Road network -- node color = congestion, ring = ambulance route")
    ax.axis("off")
    return fig


def node_state_table(network: TrafficNetwork) -> pd.DataFrame:
    rows = []
    for n in network.nodes():
        node = network.graph.nodes[n]
        rows.append(
            {
                "Intersection": n,
                "Phase (green)": "NS" if node["phase"] == 0 else "EW",
                "Queue NS": node["queue"]["NS"],
                "Queue EW": node["queue"]["EW"],
                "Density NS": round(node["density"]["NS"], 2),
                "Density EW": round(node["density"]["EW"], 2),
                "Capacity": node["capacity"],
                "Emergency lock": node.get("emergency_lock"),
            }
        )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Comparison runner
# ----------------------------------------------------------------------
def run_comparison(n, seed, steps, decision_interval, qaoa_reps, qaoa_maxiter, coordination_weight,
                    include_spike, include_ambulance, amb_start, amb_end):
    histories = {}
    for ctrl in ["quantum", "fixed", "greedy"]:
        net = TrafficNetwork(n_intersections=n, seed=seed)
        corridor = EmergencyCorridor()
        sim = TrafficSimulator(
            net,
            controller_type=ctrl,
            decision_interval=decision_interval,
            qaoa_reps=qaoa_reps,
            qaoa_maxiter=qaoa_maxiter,
            coordination_weight=coordination_weight,
            seed=seed,
            corridor=corridor,
        )
        for t in range(steps):
            if include_spike and t == max(1, steps // 4):
                for node in list(net.nodes())[: min(3, n)]:
                    net.trigger_congestion_spike(node, direction="EW", magnitude=0.5)
            if include_ambulance and t == max(1, steps // 3) and amb_start != amb_end:
                try:
                    corridor.activate(net, amb_start, amb_end, steps_per_node=2)
                except Exception:
                    pass
            sim.step()
        histories[ctrl] = sim.history
    return histories


# ----------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------
st.title("Quantum-Enhanced Adaptive Urban Traffic Optimization")
st.caption(
    "Hybrid quantum-classical traffic signal control: a NetworkX road network, a QUBO "
    "signal-timing model solved with QAOA on a local simulator, an emergency green corridor, "
    "and a side-by-side comparison against classical baselines."
)

tab_live, tab_compare = st.tabs(["Live Network", "Quantum vs Classical"])

# ============================= LIVE TAB ================================
with tab_live:
    with st.sidebar:
        st.header("1. Network")
        n_intersections = st.slider("Number of intersections", 4, 8, 6, key="n_int")
        seed = st.number_input("Random seed", 0, 9999, 42, key="net_seed")
        if st.button("Build / Reset Network"):
            _rebuild_network(n_intersections, seed)
            st.rerun()

        st.header("2. Controller")
        controller_type = st.selectbox(
            "Active controller", ["quantum", "fixed", "greedy"], format_func=lambda k: CONTROLLER_LABELS[k]
        )
        decision_interval = st.slider("Decision interval (steps)", 1, 6, 3)
        qaoa_reps = st.slider("QAOA layers (p)", 1, 4, 2)
        qaoa_maxiter = st.slider("Optimizer iterations", 20, 300, 80, step=20)
        coordination_weight = st.slider("Coordination weight (green-wave strength)", 0.0, 10.0, 2.0, step=0.5)

        st.header("3. Run")
        steps_to_run = st.slider("Steps to advance", 1, 20, 5)
        if st.button("Step Simulation", type="primary"):
            sim = _ensure_sim(controller_type, decision_interval, qaoa_reps, qaoa_maxiter, coordination_weight, seed)
            with st.spinner(f"Running {steps_to_run} step(s) with {CONTROLLER_LABELS[controller_type]}..."):
                new_hist = sim.run(steps_to_run)
            st.session_state.history.extend(new_hist)

        st.header("4. Dynamic events")
        nodes = st.session_state.network.nodes()
        event_node = st.selectbox("Target intersection", nodes, key="event_node")
        c1, c2 = st.columns(2)
        if c1.button("Congestion spike"):
            st.session_state.network.trigger_congestion_spike(event_node)
            st.toast(f"Congestion spike at intersection {event_node}")
        if c2.button("Accident"):
            st.session_state.network.trigger_accident(event_node)
            st.toast(f"Accident at intersection {event_node}")

        close_a, close_b = st.columns(2)
        node_a = close_a.selectbox("Road A", nodes, key="close_a")
        node_b = close_b.selectbox("Road B", nodes, key="close_b")
        c3, c4 = st.columns(2)
        if c3.button("Close road A-B"):
            st.session_state.network.trigger_road_closure(node_a, node_b)
            st.toast(f"Closed road {node_a}-{node_b}")
        if c4.button("Reopen all roads"):
            for u, v in st.session_state.network.graph.edges():
                st.session_state.network.reopen_road(u, v)
            st.toast("All roads reopened")

        st.header("5. Emergency vehicle")
        amb_start = st.selectbox("Ambulance start", nodes, key="amb_start")
        amb_end = st.selectbox("Ambulance destination", nodes, key="amb_end", index=len(nodes) - 1)
        steps_per_node = st.slider("Ambulance speed (steps/intersection)", 1, 5, 2)
        c5, c6 = st.columns(2)
        if c5.button("Dispatch ambulance"):
            try:
                path = st.session_state.corridor.activate(
                    st.session_state.network, amb_start, amb_end, steps_per_node=steps_per_node
                )
                st.toast(f"Green corridor activated: {' -> '.join(map(str, path))}")
            except Exception as e:  # e.g. no open path due to closures
                st.error(f"Could not route ambulance: {e}")
        if c6.button("Cancel ambulance"):
            st.session_state.corridor.clear(st.session_state.network)

    left, right = st.columns([3, 2])
    with left:
        st.pyplot(draw_network(st.session_state.network, st.session_state.corridor))
        if st.session_state.corridor.is_active():
            st.info(
                f"Ambulance en route: node {st.session_state.corridor.current_node()} -- "
                f"~{st.session_state.corridor.eta_steps_remaining()} steps to destination"
            )
    with right:
        st.subheader("Intersection states")
        st.dataframe(node_state_table(st.session_state.network), hide_index=True, use_container_width=True)

        if st.session_state.history:
            last = st.session_state.history[-1]
            if last.get("qaoa"):
                st.caption(f"Last QAOA solve -- backend: `{last['qaoa']['backend']}`, energy: {last['qaoa']['energy']:.3f}")

    st.subheader("Live metrics")
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)[["t", "total_queue", "discharged"]]
        m1, m2, m3 = st.columns(3)
        metrics_now = compute_metrics(st.session_state.history)
        m1.metric("Avg queue length", f"{metrics_now['avg_queue_length']:.1f} veh")
        m2.metric("Throughput (total)", f"{metrics_now['throughput']} veh")
        m3.metric("CO2 (total run)", f"{metrics_now['co2_kg']:.2f} kg")
        st.line_chart(df.set_index("t")[["total_queue", "discharged"]])
    else:
        st.caption("Click **Step Simulation** in the sidebar to start generating traffic.")

# =========================== COMPARISON TAB =============================
with tab_compare:
    st.subheader("Quantum (QAOA) vs classical baselines -- identical conditions")
    st.caption(
        "Runs three independent simulations (Quantum / Fixed-Time / Greedy) from the same "
        "seed and, optionally, the same congestion spike and ambulance dispatch at the same "
        "step -- so any difference in the results is attributable to the *controller*, not "
        "to random noise."
    )
    with st.form("comparison_form"):
        cc1, cc2, cc3 = st.columns(3)
        cmp_n = cc1.slider("Intersections", 4, 8, 6, key="cmp_n")
        cmp_seed = cc2.number_input("Seed", 0, 9999, 42, key="cmp_seed")
        cmp_steps = cc3.slider("Steps to simulate", 10, 60, 30, key="cmp_steps")

        cc4, cc5, cc6 = st.columns(3)
        cmp_decision_interval = cc4.slider("Decision interval", 1, 6, 3, key="cmp_di")
        cmp_reps = cc5.slider("QAOA layers (p)", 1, 4, 2, key="cmp_reps")
        cmp_maxiter = cc6.slider("Optimizer iterations", 20, 300, 80, step=20, key="cmp_maxiter")
        cmp_coord = st.slider("Coordination weight", 0.0, 10.0, 2.0, step=0.5, key="cmp_coord")

        cc7, cc8 = st.columns(2)
        include_spike = cc7.checkbox("Include a congestion spike", value=True)
        include_ambulance = cc8.checkbox("Include an ambulance dispatch", value=True)
        nodes_for_cmp = list(range(cmp_n))
        cc9, cc10 = st.columns(2)
        cmp_amb_start = cc9.selectbox("Ambulance start", nodes_for_cmp, key="cmp_amb_start")
        cmp_amb_end = cc10.selectbox("Ambulance end", nodes_for_cmp, index=len(nodes_for_cmp) - 1, key="cmp_amb_end")

        submitted = st.form_submit_button("Run comparison", type="primary")

    if submitted:
        with st.spinner("Running all three controllers..."):
            histories = run_comparison(
                cmp_n, cmp_seed, cmp_steps, cmp_decision_interval, cmp_reps, cmp_maxiter, cmp_coord,
                include_spike, include_ambulance, cmp_amb_start, cmp_amb_end,
            )
        st.session_state.comparison = compare(histories)

    if st.session_state.comparison:
        metrics_df = pd.DataFrame(st.session_state.comparison).T
        metrics_df.index = [CONTROLLER_LABELS[k] for k in metrics_df.index]
        st.dataframe(
            metrics_df.rename(
                columns={
                    "avg_waiting_time": "Avg waiting time (veh)",
                    "avg_queue_length": "Avg queue length (veh)",
                    "throughput": "Throughput (veh)",
                    "fuel_liters": "Fuel (L)",
                    "co2_kg": "CO2 (kg)",
                }
            ).round(2),
            use_container_width=True,
        )

        bar1, bar2 = st.columns(2)
        with bar1:
            st.bar_chart(metrics_df[["avg_waiting_time"]].rename(columns={"avg_waiting_time": "Avg waiting time"}))
        with bar2:
            st.bar_chart(metrics_df[["throughput"]])
        bar3, bar4 = st.columns(2)
        with bar3:
            st.bar_chart(metrics_df[["fuel_liters"]].rename(columns={"fuel_liters": "Fuel (L)"}))
        with bar4:
            st.bar_chart(metrics_df[["co2_kg"]].rename(columns={"co2_kg": "CO2 (kg)"}))

        q = st.session_state.comparison["quantum"]
        f = st.session_state.comparison["fixed"]
        if f["avg_waiting_time"] > 0:
            improvement = 100 * (f["avg_waiting_time"] - q["avg_waiting_time"]) / f["avg_waiting_time"]
            st.success(
                f"Quantum-optimized signals changed average waiting time by "
                f"{improvement:+.1f}% versus fixed-time signals on this run."
            )
    else:
        st.caption("Configure a scenario above and click **Run comparison**.")
