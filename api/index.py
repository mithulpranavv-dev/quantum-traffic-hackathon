from __future__ import annotations

from flask import Flask, jsonify, request
import numpy as np

from src.emergency_corridor import EmergencyCorridor
from src.metrics import compare, compute_metrics
from src.simulation import TrafficSimulator
from src.traffic_network import TrafficNetwork

app = Flask(__name__)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _run_controller(options: dict, controller: str) -> dict:
    intersections = int(options.get("intersections", 6))
    steps = min(max(int(options.get("steps", 12)), 1), 40)
    seed = int(options.get("seed", 42))
    decision_interval = min(max(int(options.get("decision_interval", 3)), 1), 6)
    coordination_weight = min(max(float(options.get("coordination_weight", 2.0)), 0), 10)
    network = TrafficNetwork(n_intersections=intersections, seed=seed)
    corridor = EmergencyCorridor()
    simulator = TrafficSimulator(
        network,
        controller_type=controller,
        decision_interval=decision_interval,
        qaoa_reps=1,
        qaoa_maxiter=20,
        coordination_weight=coordination_weight,
        seed=seed,
        corridor=corridor,
    )

    if options.get("spike"):
        network.trigger_congestion_spike(0, direction="EW", magnitude=0.5)
    if options.get("ambulance") and intersections > 1:
        corridor.activate(network, 0, intersections - 1, steps_per_node=2)

    history = simulator.run(steps)
    nodes = []
    for node in network.nodes():
        state = network.graph.nodes[node]
        nodes.append(
            {
                "id": node,
                "phase": state["phase"],
                "queue_ns": state["queue"]["NS"],
                "queue_ew": state["queue"]["EW"],
                "density_ns": round(state["density"]["NS"], 3),
                "density_ew": round(state["density"]["EW"], 3),
                "emergency_lock": state.get("emergency_lock"),
                "position": state.get("pos"),
            }
        )

    return {
        "controller": controller,
        "metrics": compute_metrics(history),
        "history": history,
        "nodes": nodes,
        "edges": [
            {"from": u, "to": v, "status": data["status"]}
            for u, v, data in network.graph.edges(data=True)
        ],
        "emergency_path": corridor.path,
    }


@app.get("/api/health")
def health():
    return jsonify({"ok": True, "service": "quantum-traffic-api"})


@app.post("/api/simulate")
def simulate():
    options = request.get_json(silent=True) or {}
    controller = options.get("controller", "quantum")
    if controller not in {"quantum", "fixed", "greedy"}:
        return jsonify({"error": "controller must be quantum, fixed, or greedy"}), 400

    try:
        return jsonify(_json_safe(_run_controller(options, controller)))
    except Exception as error:
        return jsonify({"error": str(error)}), 400


@app.post("/api/compare")
def comparison():
    options = request.get_json(silent=True) or {}
    try:
        results = {
            controller: _run_controller(options, controller)
            for controller in ("quantum", "fixed", "greedy")
        }
        return jsonify(
            _json_safe(
            {
                "results": {
                    controller: result["metrics"] for controller, result in results.items()
                },
                "histories": {
                    controller: result["history"] for controller, result in results.items()
                },
            }
            )
        )
    except Exception as error:
        return jsonify({"error": str(error)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
