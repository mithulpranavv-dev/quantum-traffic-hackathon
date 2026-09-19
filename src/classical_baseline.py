"""
classical_baseline.py
----------------------
Two classical controllers used as baselines to compare against the
quantum-optimized signal plan:

1. FixedTimeController
   The traditional approach still used at most real-world
   intersections: each phase runs for a fixed duration, then switches,
   regardless of how much traffic is actually waiting. No knowledge of
   density/queue is used at all.

2. GreedyController
   A simple "adaptive but not coordinated" rule-based controller: each
   intersection independently looks at its own NS vs EW pressure
   (same congestion signal QAOA uses) and greedily picks whichever
   direction is more congested. This is what many "smart" traffic
   lights already do -- locally reactive, but with no attempt to
   coordinate with neighbours (no green wave), which is exactly the
   gap the QUBO's coordination term is designed to close.

Both controllers expose the same interface:
    decide(network) -> Dict[int, int]      # {node: phase}
so simulation.py can swap between quantum / fixed / greedy without any
other code changes.
"""

from __future__ import annotations

from typing import Dict

from src.traffic_network import TrafficNetwork


class FixedTimeController:
    def __init__(self, cycle_length: int = 6):
        """cycle_length: number of simulation steps each phase holds
        before switching."""
        self.cycle_length = cycle_length
        self._step_count = 0

    def decide(self, network: TrafficNetwork) -> Dict[int, int]:
        phase = 0 if (self._step_count // self.cycle_length) % 2 == 0 else 1
        self._step_count += 1
        return {n: phase for n in network.nodes()}


class GreedyController:
    """Locally-adaptive, but uncoordinated: every intersection reacts
    only to its own queues, with no regard for its neighbours."""

    def decide(self, network: TrafficNetwork) -> Dict[int, int]:
        decisions = {}
        for n in network.nodes():
            node = network.graph.nodes[n]
            pressure_ns = node["density"]["NS"] * (1 + node["queue"]["NS"])
            pressure_ew = node["density"]["EW"] * (1 + node["queue"]["EW"])
            decisions[n] = 1 if pressure_ew > pressure_ns else 0
        return decisions
