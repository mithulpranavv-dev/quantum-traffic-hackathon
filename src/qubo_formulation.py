"""
qubo_formulation.py
--------------------
Turns the *current* traffic state of a TrafficNetwork into a QUBO
(Quadratic Unconstrained Binary Optimization) problem.

Decision variables
    x_i in {0, 1} for every intersection i
    x_i = 0  ->  keep/switch to phase 0 (NS green)
    x_i = 1  ->  keep/switch to phase 1 (EW green)

Objective (minimise):

    E(x) = sum_i  h_i * x_i                     (1) local congestion term
         + sum_{(i,j) in edges}  J_ij * (x_i - x_j)^2   (2) coordination term
         + sum_i  L_i * (x_i - target_i)^2       (3) emergency-corridor lock

(1) Local congestion term
    h_i = capacity-normalised (EW density - NS density) for node i.
    If EW is more congested than NS, h_i is negative, so the optimizer
    is rewarded for setting x_i = 1 (EW green) -- and vice-versa. This
    is the direct "adaptive signal" piece: greener light for whichever
    direction currently needs it more.

(2) Coordination term
    (x_i - x_j)^2 = x_i + x_j - 2*x_i*x_j for binary x. It equals 0 when
    x_i == x_j and 1 otherwise. Minimising it rewards neighbouring
    intersections for choosing the SAME phase, which is what produces a
    "green wave" -- consecutive intersections open in the same
    direction so a platoon of cars doesn't hit a red at every block.
    J_ij is weighted by how congested the shared edge is, so we only
    pay to coordinate intersections that actually have traffic to
    coordinate.

(3) Emergency-corridor lock
    While an ambulance is travelling through the network
    (emergency_corridor.py), intersections along its route get a large
    penalty L_i for disagreeing with the phase the ambulance needs
    (target_i), which overwhelms (1) and (2) and forces the optimizer
    to give it a green corridor.

The QUBO is returned as:
    h       : dict {i: float}                  linear coefficients
    Q       : dict {(i, j): float}   (i < j)    quadratic coefficients
    offset  : float                             constant term (doesn't
                                                  affect the argmin, kept
                                                  for completeness)

This dict form is solver-agnostic: qaoa_solver.py converts it to an
Ising Hamiltonian for QAOA, and classical_baseline.py / a brute-force
check can evaluate it directly.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from src.traffic_network import TrafficNetwork

EMERGENCY_PENALTY = 25.0  # large enough to dominate normal congestion terms


def build_qubo(
    network: TrafficNetwork,
    coordination_weight: float = 1.0,
    emergency_targets: Optional[Dict[int, int]] = None,
) -> Tuple[Dict[int, float], Dict[Tuple[int, int], float], float]:
    """Build (h, Q, offset) for the current state of `network`.

    Parameters
    ----------
    coordination_weight : scales term (2) relative to term (1). Higher
        values favour city-wide green waves over hyper-local reaction.
    emergency_targets : {node: required_phase} for an active ambulance
        route (see emergency_corridor.py). None if no emergency is active.
    """
    h: Dict[int, float] = {n: 0.0 for n in network.nodes()}
    Q: Dict[Tuple[int, int], float] = {}
    offset = 0.0
    emergency_targets = emergency_targets or {}

    # ---- (1) local congestion term ----
    for n in network.nodes():
        node = network.graph.nodes[n]
        if node.get("isolated", False):
            continue
        # Congestion pressure = density weighted by queue length, so a
        # long queue in a currently-low-density direction still counts.
        pressure_ns = node["density"]["NS"] * (1 + node["queue"]["NS"])
        pressure_ew = node["density"]["EW"] * (1 + node["queue"]["EW"])
        # h_i * x_i is minimised; x_i=1 means EW green, so we need h_i
        # to be NEGATIVE (a reward) when EW is the more congested
        # direction, i.e. h_i = pressure_NS - pressure_EW.
        h[n] += pressure_ns - pressure_ew

    # ---- (2) coordination term: (x_i - x_j)^2 = x_i + x_j - 2 x_i x_j ----
    for u, v, data in network.graph.edges(data=True):
        if (
            data["status"] != "open"
            or network.graph.nodes[u].get("isolated", False)
            or network.graph.nodes[v].get("isolated", False)
        ):
            continue  # a closed road has nothing to coordinate
        edge_load = data.get("capacity", 10)
        edge_congestion = (
            network.graph.nodes[u]["density"][data["axis"]]
            + network.graph.nodes[v]["density"][data["axis"]]
        ) / 2.0
        j_uv = coordination_weight * edge_congestion * (10.0 / max(edge_load, 1))

        h[u] += j_uv
        h[v] += j_uv
        key = (u, v) if u < v else (v, u)
        Q[key] = Q.get(key, 0.0) + (-2.0 * j_uv)

    # ---- (3) emergency-corridor lock: L * (x_i - target_i)^2 ----
    # For binary x_i, target_i: (x_i - t)^2 = x_i - 2*t*x_i + t^2
    #   t=0 -> x_i        (penalises x_i=1)
    #   t=1 -> -x_i + 1   (penalises x_i=0)
    for n, target in emergency_targets.items():
        if target == 0:
            h[n] += EMERGENCY_PENALTY
        else:
            h[n] += -EMERGENCY_PENALTY
            offset += EMERGENCY_PENALTY

    return h, Q, offset


def evaluate_qubo(h: Dict[int, float], Q: Dict[Tuple[int, int], float], offset: float, x: Dict[int, int]) -> float:
    """Evaluate E(x) directly -- used to score/verify any candidate
    solution (QAOA sample, brute force, or classical heuristic) on the
    exact same objective."""
    energy = offset
    for i, hi in h.items():
        energy += hi * x[i]
    for (i, j), qij in Q.items():
        energy += qij * x[i] * x[j]
    return energy


def brute_force_solve(h: Dict[int, float], Q: Dict[Tuple[int, int], float], offset: float):
    """Exact solver by exhaustive enumeration. Only practical for the
    small n (<=8) used in this prototype -- used as ground truth to
    sanity-check the QAOA result during a demo."""
    nodes = sorted(h.keys())
    n = len(nodes)
    best_x, best_e = None, float("inf")
    for bits in range(2 ** n):
        x = {nodes[k]: (bits >> k) & 1 for k in range(n)}
        e = evaluate_qubo(h, Q, offset, x)
        if e < best_e:
            best_e, best_x = e, x
    return best_x, best_e
