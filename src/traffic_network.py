"""
traffic_network.py
-------------------
Models a small urban road network as a NetworkX graph.

Each intersection (node) has two competing traffic directions, labelled
'NS' (north-south) and 'EW' (east-west). Every intersection can only let
ONE of those two directions flow at a time -- that is the signal's
"phase":

    phase 0  ->  NS green / EW red
    phase 1  ->  EW green / NS red

This binary phase-per-intersection is exactly what makes the signal
timing problem a natural fit for a QUBO / Ising formulation later on
(see qubo_formulation.py): x_i in {0, 1} for every intersection i.

Each node stores:
    density   : dict {'NS': float, 'EW': float}   in [0, 1] -- how busy
                each direction currently is (0 = empty, 1 = gridlocked)
    queue     : dict {'NS': int,   'EW': int}      -- vehicles waiting
    capacity  : int   -- max vehicles that can be discharged per step
                per direction while green
    phase     : int (0 or 1) -- current signal phase
    axis      : 'NS' or 'EW' -- which axis this node sits on for the
                purposes of the emergency corridor (see
                emergency_corridor.py); assigned from its main
                through-edge.

Each edge stores:
    axis      : 'NS' or 'EW' -- which of the two competing directions
                this road segment belongs to
    length_m  : float  -- rough length in metres (synthetic)
    capacity  : int    -- vehicles/step the road segment can carry
    status    : 'open' or 'closed'  -- can be closed by a "road closure"
                event
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import networkx as nx


class TrafficNetwork:
    """A small, fully-connected grid-ish network of signalised intersections."""

    def __init__(self, n_intersections: int = 6, seed: int = 42):
        if not (4 <= n_intersections <= 8):
            raise ValueError("n_intersections must be between 4 and 8 for this prototype")

        self.n = n_intersections
        self.rng = random.Random(seed)
        self.graph = nx.Graph()
        self._build_graph()
        self._init_node_state()
        self._active_events: List[dict] = []  # log of applied events, for the dashboard

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def _build_graph(self) -> None:
        """Build a connected, roughly grid-like layout for n intersections.

        We lay nodes out on a near-square grid (e.g. 6 -> 3x2) and connect
        grid neighbours. This keeps the topology realistic (every
        intersection has 2-4 neighbours, like real city blocks) while
        staying tiny enough for a QAOA circuit to simulate instantly.
        """
        import math

        cols = math.ceil(math.sqrt(self.n))
        rows = math.ceil(self.n / cols)

        positions = {}
        idx = 0
        for r in range(rows):
            for c in range(cols):
                if idx >= self.n:
                    break
                positions[idx] = (c, r)
                self.graph.add_node(idx, pos=(c, r))
                idx += 1

        # connect grid neighbours (right and down)
        for i, (ci, ri) in positions.items():
            for j, (cj, rj) in positions.items():
                if i >= j:
                    continue
                if (abs(ci - cj) == 1 and ri == rj) or (abs(ri - rj) == 1 and ci == cj):
                    axis = "EW" if ri == rj else "NS"
                    self.graph.add_edge(
                        i,
                        j,
                        axis=axis,
                        length_m=self.rng.randint(150, 500),
                        capacity=self.rng.randint(8, 14),
                        status="open",
                    )

        # Safety net: if the grid trimming left anything disconnected
        # (can happen for odd n), stitch components together.
        if not nx.is_connected(self.graph):
            comps = list(nx.connected_components(self.graph))
            for k in range(len(comps) - 1):
                a = next(iter(comps[k]))
                b = next(iter(comps[k + 1]))
                self.graph.add_edge(a, b, axis="NS", length_m=300, capacity=10, status="open")

    def _init_node_state(self) -> None:
        for node in self.graph.nodes:
            # a node's "axis" for corridor purposes = axis of its majority edges
            edge_axes = [self.graph.edges[e]["axis"] for e in self.graph.edges(node)]
            main_axis = max(set(edge_axes), key=edge_axes.count) if edge_axes else "NS"

            self.graph.nodes[node].update(
                density={"NS": self.rng.uniform(0.1, 0.5), "EW": self.rng.uniform(0.1, 0.5)},
                queue={"NS": self.rng.randint(0, 5), "EW": self.rng.randint(0, 5)},
                capacity=self.rng.randint(6, 10),
                phase=self.rng.randint(0, 1),
                axis=main_axis,
                emergency_lock=None,  # set to required phase while an ambulance is passing
            )

    # ------------------------------------------------------------------
    # State access helpers
    # ------------------------------------------------------------------
    def nodes(self) -> List[int]:
        return list(self.graph.nodes)

    def get_state_snapshot(self) -> Dict[int, dict]:
        """Deep-ish copy of every node's current state, for logging/dashboard."""
        return {n: dict(self.graph.nodes[n]) for n in self.graph.nodes}

    def set_phase(self, node: int, phase: int) -> None:
        """Set a node's signal phase, respecting an active emergency lock."""
        lock = self.graph.nodes[node].get("emergency_lock")
        self.graph.nodes[node]["phase"] = lock if lock is not None else phase

    def total_queue(self) -> int:
        return sum(sum(self.graph.nodes[n]["queue"].values()) for n in self.graph.nodes)

    def open_neighbors(self, node: int):
        """Neighbours reachable via an open (non-closed) road."""
        for nbr in self.graph.neighbors(node):
            if self.graph.edges[node, nbr]["status"] == "open":
                yield nbr

    # ------------------------------------------------------------------
    # Dynamic events (the "Dynamic event handling" requirement)
    # ------------------------------------------------------------------
    def trigger_congestion_spike(self, node: int, direction: Optional[str] = None, magnitude: float = 0.4) -> dict:
        """Sharply raise density/queue at one intersection, simulating a
        sudden surge (e.g. a nearby event letting out)."""
        directions = [direction] if direction else ["NS", "EW"]
        for d in directions:
            self.graph.nodes[node]["density"][d] = min(1.0, self.graph.nodes[node]["density"][d] + magnitude)
            self.graph.nodes[node]["queue"][d] += self.rng.randint(6, 12)
        event = {"type": "congestion_spike", "node": node, "direction": direction, "magnitude": magnitude}
        self._active_events.append(event)
        return event

    def trigger_accident(self, node: int, direction: Optional[str] = None, capacity_factor: float = 0.3) -> dict:
        """An accident at an intersection reduces its discharge capacity
        for the affected direction (fewer vehicles can get through per
        green interval) until cleared."""
        directions = [direction] if direction else ["NS", "EW"]
        self.graph.nodes[node].setdefault("capacity_factor", {})
        for d in directions:
            self.graph.nodes[node]["capacity_factor"][d] = capacity_factor
        event = {"type": "accident", "node": node, "direction": direction, "capacity_factor": capacity_factor}
        self._active_events.append(event)
        return event

    def clear_accident(self, node: int) -> None:
        self.graph.nodes[node]["capacity_factor"] = {}

    def trigger_road_closure(self, node_a: int, node_b: int) -> dict:
        """Close the road segment between two intersections. Routing
        (e.g. the ambulance path finder) will automatically avoid it."""
        if self.graph.has_edge(node_a, node_b):
            self.graph.edges[node_a, node_b]["status"] = "closed"
        event = {"type": "road_closure", "edge": (node_a, node_b)}
        self._active_events.append(event)
        return event

    def reopen_road(self, node_a: int, node_b: int) -> None:
        if self.graph.has_edge(node_a, node_b):
            self.graph.edges[node_a, node_b]["status"] = "open"

    def routing_view(self) -> nx.Graph:
        """A view of the graph containing only currently-open edges, for
        pathfinding (used by the emergency corridor and by throughput
        calculations)."""
        open_edges = [
            (u, v) for u, v, d in self.graph.edges(data=True) if d["status"] == "open"
        ]
        g = nx.Graph()
        g.add_nodes_from(self.graph.nodes(data=True))
        g.add_edges_from((u, v, self.graph.edges[u, v]) for u, v in open_edges)
        return g

    def log(self) -> List[dict]:
        return list(self._active_events)
