"""
simulation.py
-------------
A lightweight, dependency-free discrete-time traffic simulator (no
SUMO). Each call to `TrafficSimulator.step()` advances the network by
one time step:

    1. ARRIVALS   -- synthetic traffic is generated: each direction at
                      each intersection gets a Poisson-distributed
                      number of new vehicles, with rate proportional to
                      that direction's current density.
    2. DECISION    -- every `decision_interval` steps (or immediately
                      if an emergency corridor is active), the active
                      controller (quantum / fixed-time / greedy) is
                      asked for a phase for every intersection. The
                      network then applies those phases via
                      `network.set_phase()`, which automatically
                      overrides with any active emergency lock.
    3. DISCHARGE   -- the green direction at each intersection releases
                      up to its (possibly accident-reduced) capacity
                      worth of vehicles from its queue; the red
                      direction's queue only grows.
    4. FEEDBACK    -- each direction's `density` is recomputed from its
                      new queue length, so the next decision cycle
                      reacts to what actually happened (closed feedback
                      loop -- this is what makes the signals "adaptive"
                      rather than open-loop).
    5. EMERGENCY   -- if a corridor is active, it is advanced by one
                      step (moves the ambulance further along its
                      route, releasing locks behind it).

`run(steps)` just calls `step()` in a loop and collects one dict of
results per step, which metrics.py and the dashboard both consume.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from src.traffic_network import TrafficNetwork
from src.qubo_formulation import build_qubo
from src.qaoa_solver import solve_qaoa
from src.classical_baseline import FixedTimeController, GreedyController
from src.emergency_corridor import EmergencyCorridor

ARRIVAL_RATE_SCALE = 3.0  # max expected new vehicles/step at density=1.0
DENSITY_SMOOTHING_CAPACITY_MULTIPLIER = 5  # queue level considered "fully congested"


class TrafficSimulator:
    def __init__(
        self,
        network: TrafficNetwork,
        controller_type: str = "quantum",
        decision_interval: int = 3,
        qaoa_reps: int = 2,
        qaoa_maxiter: int = 80,
        coordination_weight: float = 1.0,
        seed: int = 0,
        corridor: Optional[EmergencyCorridor] = None,
    ):
        assert controller_type in ("quantum", "fixed", "greedy")
        self.network = network
        self.controller_type = controller_type
        self.decision_interval = max(1, decision_interval)
        self.qaoa_reps = qaoa_reps
        self.qaoa_maxiter = qaoa_maxiter
        self.coordination_weight = coordination_weight
        self.rng = np.random.default_rng(seed)

        self._fixed = FixedTimeController(cycle_length=decision_interval * 2)
        self._greedy = GreedyController()
        self.corridor = corridor if corridor is not None else EmergencyCorridor()

        self._t = 0
        self._last_qaoa_result: Optional[dict] = None
        self.history: List[dict] = []

    # ------------------------------------------------------------------
    def _generate_arrivals(self) -> None:
        for n in self.network.nodes():
            node = self.network.graph.nodes[n]
            for d in ("NS", "EW"):
                rate = node["density"][d] * ARRIVAL_RATE_SCALE
                arrivals = int(self.rng.poisson(rate))
                node["queue"][d] += arrivals

    def _decide_phases(self) -> Dict[int, int]:
        emergency_targets = self.corridor.active_targets(self.network) if self.corridor.is_active() else {}
        force_now = bool(emergency_targets)

        if self.controller_type == "quantum":
            if self._t % self.decision_interval == 0 or force_now:
                h, Q, offset = build_qubo(
                    self.network,
                    coordination_weight=self.coordination_weight,
                    emergency_targets=emergency_targets,
                )
                self._last_qaoa_result = solve_qaoa(
                    h, Q, offset, reps=self.qaoa_reps, maxiter=self.qaoa_maxiter, seed=self._t + 1
                )
            decisions = self._last_qaoa_result["x"] if self._last_qaoa_result else {
                n: self.network.graph.nodes[n]["phase"] for n in self.network.nodes()
            }
        elif self.controller_type == "fixed":
            decisions = self._fixed.decide(self.network)
        else:  # greedy
            decisions = self._greedy.decide(self.network)

        return decisions

    def _discharge_and_feedback(self) -> int:
        total_discharged = 0
        for n in self.network.nodes():
            node = self.network.graph.nodes[n]
            green_dir = "NS" if node["phase"] == 0 else "EW"
            red_dir = "EW" if green_dir == "NS" else "NS"

            factor = node.get("capacity_factor", {}).get(green_dir, 1.0)
            effective_capacity = max(1, int(node["capacity"] * factor))

            discharge = min(node["queue"][green_dir], effective_capacity)
            node["queue"][green_dir] -= discharge
            total_discharged += discharge

            # closed-loop feedback: density reflects the *current* queue,
            # so next decision cycle reacts to what actually happened
            cap_ref = max(node["capacity"], 1) * DENSITY_SMOOTHING_CAPACITY_MULTIPLIER
            node["density"][green_dir] = min(1.0, node["queue"][green_dir] / cap_ref)
            node["density"][red_dir] = min(1.0, node["queue"][red_dir] / cap_ref)

        return total_discharged

    # ------------------------------------------------------------------
    def step(self) -> dict:
        self._generate_arrivals()

        decisions = self._decide_phases()
        for n, phase in decisions.items():
            self.network.set_phase(n, phase)

        discharged = self._discharge_and_feedback()

        if self.corridor.is_active():
            self.corridor.advance(self.network)

        record = {
            "t": self._t,
            "total_queue": self.network.total_queue(),
            "discharged": discharged,
            "phases": {n: self.network.graph.nodes[n]["phase"] for n in self.network.nodes()},
            "queues": {n: dict(self.network.graph.nodes[n]["queue"]) for n in self.network.nodes()},
            "densities": {n: dict(self.network.graph.nodes[n]["density"]) for n in self.network.nodes()},
            "emergency_active": self.corridor.is_active(),
            "emergency_node": self.corridor.current_node() if self.corridor.is_active() else None,
            "qaoa": (
                {"backend": self._last_qaoa_result["backend"], "energy": self._last_qaoa_result["energy"]}
                if self.controller_type == "quantum" and self._last_qaoa_result
                else None
            ),
        }
        self.history.append(record)
        self._t += 1
        return record

    def run(self, steps: int) -> List[dict]:
        for _ in range(steps):
            self.step()
        return self.history
