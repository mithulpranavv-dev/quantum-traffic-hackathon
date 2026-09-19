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
    intersections = min(max(int(options.get("intersections", 6)), 4), 8)
    steps = min(max(int(options.get("steps", 12)), 1), 40)
    seed = int(options.get("seed", 42))
    decision_interval = min(max(int(options.get("decision_interval", 3)), 1), 6)
    coordination_weight = min(max(float(options.get("coordination_weight", 2.0)), 0), 10)
    qaoa_reps = min(max(int(options.get("qaoa_reps", 1)), 1), 4)
    qaoa_maxiter = min(max(int(options.get("qaoa_maxiter", 20)), 20), 300)
    event_step = min(max(int(options.get("event_step", max(1, steps // 4))), 0), steps - 1)
    network = TrafficNetwork(n_intersections=intersections, seed=seed)
    corridor = EmergencyCorridor()
    simulator = TrafficSimulator(
        network,
        controller_type=controller,
        decision_interval=decision_interval,
        qaoa_reps=qaoa_reps,
        qaoa_maxiter=qaoa_maxiter,
        coordination_weight=coordination_weight,
        seed=seed,
        corridor=corridor,
    )

    event_log = []
    for step in range(steps):
        if step == event_step:
            if options.get("spike"):
                event_log.append(network.trigger_congestion_spike(0, direction="EW", magnitude=0.5))
            accident_node = options.get("accident_node")
            if options.get("accident") and accident_node is not None:
                event_log.append(network.trigger_accident(min(max(int(accident_node), 0), intersections - 1)))
            closure = options.get("closure") or {}
            if options.get("road_closure") and closure:
                closure_from = min(max(int(closure["from"]), 0), intersections - 1)
                closure_to = min(max(int(closure["to"]), 0), intersections - 1)
                event_log.append(network.trigger_road_closure(closure_from, closure_to))
            if options.get("ambulance") and intersections > 1:
                start = min(max(int(options.get("ambulance_start", 0)), 0), intersections - 1)
                end = min(max(int(options.get("ambulance_end", intersections - 1)), 0), intersections - 1)
                try:
                    path = corridor.activate(
                        network,
                        start,
                        end,
                        steps_per_node=int(options.get("steps_per_node", 2)),
                    )
                    event_log.append({"type": "ambulance", "start": start, "end": end, "path": path})
                except Exception as error:
                    event_log.append({"type": "ambulance_error", "error": str(error)})
        simulator.step()

    history = simulator.history
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
        "events": event_log,
        "qaoa_backend": next(
            (record["qaoa"]["backend"] for record in reversed(history) if record.get("qaoa")),
            None,
        ),
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
