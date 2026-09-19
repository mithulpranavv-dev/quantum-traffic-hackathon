"""
emergency_corridor.py
----------------------
Implements the "Emergency Green Corridor" requirement: given a
start/end intersection for an ambulance, bias every signal along its
route so it stays green for as long as the ambulance needs it, then
automatically restores normal operation once it has passed.

How it plugs into the rest of the system
    1. `activate()` finds the shortest open-road path from start to end
       (networkx shortest_path on the network's routing view, which
       already excludes any closed roads -- so it re-routes cleanly
       around an active road-closure event).
    2. For every intersection on that path, it works out which phase
       the ambulance needs (based on the axis of the road it's
       travelling on through that node) and records it as an
       `emergency_lock` on the node.
    3. Every simulation step, `simulation.py` passes these locked
       targets into `build_qubo(..., emergency_targets=...)`, which
       adds the large penalty term (3) described in
       qubo_formulation.py -- so whichever controller is driving the
       network (quantum OR classical) is forced to keep that phase
       green regardless of what its own objective would otherwise
       prefer.
    4. `advance()` is called once per step to move the ambulance along
       the path at a configurable speed (in "nodes per step"); once it
       reaches the last node, `is_active()` becomes False and
       `active_targets()` returns {} again, so normal operation (no
       bias) resumes immediately -- no manual "restore" call needed.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import networkx as nx

from src.traffic_network import TrafficNetwork


class EmergencyCorridor:
    def __init__(self):
        self.path: List[int] = []
        self.progress_steps: int = 0
        self.steps_per_node: int = 2  # how many sim steps the ambulance spends "at" each node
        self.start: Optional[int] = None
        self.end: Optional[int] = None

    def activate(self, network: TrafficNetwork, start: int, end: int, steps_per_node: int = 2) -> List[int]:
        """Start a new emergency run. Returns the path found (for the
        dashboard to draw), or raises if no open route exists."""
        routing_graph = network.routing_view()
        self.path = nx.shortest_path(routing_graph, start, end)
        self.start, self.end = start, end
        self.progress_steps = 0
        self.steps_per_node = max(1, steps_per_node)
        self._apply_locks(network)
        return self.path

    def is_active(self) -> bool:
        return len(self.path) > 0

    def current_node(self) -> Optional[int]:
        if not self.path:
            return None
        node_idx = min(self.progress_steps // self.steps_per_node, len(self.path) - 1)
        return self.path[node_idx]

    def _required_phase(self, network: TrafficNetwork, node: int) -> int:
        axis = network.graph.nodes[node]["axis"]
        return 0 if axis == "NS" else 1

    def _apply_locks(self, network: TrafficNetwork) -> None:
        for node in self.path:
            network.graph.nodes[node]["emergency_lock"] = self._required_phase(network, node)

    def active_targets(self, network: TrafficNetwork) -> Dict[int, int]:
        """Nodes the ambulance is currently at or approaching -- these
        get the big QUBO penalty. We keep a small look-ahead window (the
        current node plus the next one) so the light ahead is already
        green by the time the ambulance arrives."""
        if not self.path:
            return {}
        node_idx = min(self.progress_steps // self.steps_per_node, len(self.path) - 1)
        window = self.path[node_idx : node_idx + 2]
        return {n: self._required_phase(network, n) for n in window}

    def advance(self, network: TrafficNetwork) -> None:
        """Move the ambulance forward by one simulation step. Clears the
        lock on nodes it has fully passed, and clears everything once it
        reaches the destination."""
        if not self.path:
            return
        self.progress_steps += 1
        node_idx = self.progress_steps // self.steps_per_node
        # release the lock on any node strictly behind the ambulance
        for node in self.path[: max(node_idx - 1, 0)]:
            if network.graph.nodes[node]["emergency_lock"] is not None:
                network.graph.nodes[node]["emergency_lock"] = None

        if node_idx >= len(self.path):
            self.clear(network)

    def clear(self, network: TrafficNetwork) -> None:
        for node in self.path:
            network.graph.nodes[node]["emergency_lock"] = None
        self.path = []
        self.progress_steps = 0
        self.start, self.end = None, None

    def eta_steps_remaining(self) -> int:
        if not self.path:
            return 0
        total_steps = len(self.path) * self.steps_per_node
        return max(0, total_steps - self.progress_steps)
