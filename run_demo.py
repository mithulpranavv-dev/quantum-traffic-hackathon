#!/usr/bin/env python3
"""
run_demo.py
-----------
A quick, dependency-light command-line smoke test / demo of the whole
pipeline -- useful for verifying the install works *before* you fire up
the Streamlit dashboard, and for a terminal-only walkthrough if a judge
asks "does this actually run?".

Usage:
    python run_demo.py
    python run_demo.py --intersections 6 --steps 30 --seed 42
"""

import argparse

from src.traffic_network import TrafficNetwork
from src.simulation import TrafficSimulator
from src.emergency_corridor import EmergencyCorridor
from src.metrics import compare
from src.qaoa_solver import QISKIT_AVAILABLE


def main():
    parser = argparse.ArgumentParser(description="Quantum traffic optimization demo")
    parser.add_argument("--intersections", type=int, default=6)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--qaoa-reps", type=int, default=2)
    parser.add_argument("--qaoa-maxiter", type=int, default=80)
    parser.add_argument("--coordination-weight", type=float, default=2.0)
    parser.add_argument("--decision-interval", type=int, default=3)
    parser.add_argument("--with-events", action="store_true", help="fire a congestion spike, an accident, "
                                                                     "and dispatch an ambulance during the run")
    args = parser.parse_args()

    print(f"Qiskit available: {QISKIT_AVAILABLE}  "
          f"(quantum solves will run on {'qiskit' if QISKIT_AVAILABLE else 'the built-in NumPy fallback'})\n")

    histories = {}
    for controller in ["quantum", "fixed", "greedy"]:
        print(f"--- running controller: {controller} ---")
        net = TrafficNetwork(n_intersections=args.intersections, seed=args.seed)
        corridor = EmergencyCorridor()
        sim = TrafficSimulator(
            net,
            controller_type=controller,
            decision_interval=args.decision_interval,
            qaoa_reps=args.qaoa_reps,
            qaoa_maxiter=args.qaoa_maxiter,
            coordination_weight=args.coordination_weight,
            seed=args.seed,
            corridor=corridor,
        )

        for t in range(args.steps):
            if args.with_events:
                if t == max(1, args.steps // 5):
                    ev = net.trigger_congestion_spike(0, direction="EW")
                    print(f"  t={t}: {ev}")
                if t == max(2, args.steps // 4):
                    ev = net.trigger_accident(min(2, args.intersections - 1))
                    print(f"  t={t}: {ev}")
                if t == max(3, args.steps // 3):
                    end_node = args.intersections - 1
                    path = corridor.activate(net, 0, end_node, steps_per_node=2)
                    print(f"  t={t}: ambulance dispatched, route={path}")
            sim.step()

        histories[controller] = sim.history
        print(f"  finished {args.steps} steps, final total queue = {net.total_queue()}\n")

    print("=== Metrics comparison ===")
    results = compare(histories)
    header = f"{'controller':<10} {'avg_wait':>10} {'avg_queue':>10} {'throughput':>11} {'fuel_L':>8} {'co2_kg':>8}"
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        print(
            f"{name:<10} {m['avg_waiting_time']:>10.2f} {m['avg_queue_length']:>10.2f} "
            f"{m['throughput']:>11} {m['fuel_liters']:>8.2f} {m['co2_kg']:>8.2f}"
        )


if __name__ == "__main__":
    main()
