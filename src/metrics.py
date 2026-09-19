"""
metrics.py
----------
Turns raw per-step simulation history into the comparison metrics the
dashboard shows: average waiting time, average queue length,
throughput, fuel consumption, and CO2 emissions.

All formulas here are DELIBERATELY simple, linear, and fully
documented -- this is a hackathon prototype, not a calibrated
transportation model. The point is that the SAME formula is applied to
every controller (quantum, fixed-time, greedy), so the *relative*
comparison between them is meaningful even though the absolute numbers
are illustrative.

Definitions
-----------
Average waiting time (veh-steps per vehicle-step, an approximation of
    Little's Law: average number waiting ~ average time each vehicle
    waits, when arrival rate is roughly constant):
        avg_waiting_time = mean over steps of (total queue length across
                            all intersections & both directions)

Average queue length:
        avg_queue_length = same series as above, reported as vehicles

Throughput:
        throughput = total vehicles discharged (green + moved through
                     an intersection) over the whole run

Fuel consumption (litres) -- two components:
        idle component  = IDLE_FUEL_RATE_L_PER_VEH_STEP * (vehicle-steps
                           spent waiting in a queue)
        moving component = MOVE_FUEL_RATE_L_PER_VEH * (vehicles that
                            successfully moved through, i.e. throughput)
    IDLE_FUEL_RATE_L_PER_VEH_STEP models fuel burned by an idling
    engine per vehicle per simulation step; MOVE_FUEL_RATE_L_PER_VEH
    models the (larger) fuel cost of actually accelerating through an
    intersection. Both are rough, commonly-cited-order-of-magnitude
    constants for a hackathon demo, not a validated emissions model.

CO2 emissions (kg):
        co2_kg = fuel_consumption_l * CO2_PER_LITRE_PETROL_KG
    using the standard ~2.31 kg CO2 per litre of petrol combusted.
"""

from __future__ import annotations

from typing import Dict, List

# --- documented, adjustable constants -------------------------------
IDLE_FUEL_RATE_L_PER_VEH_STEP = 0.0025   # litres burned per waiting vehicle per step
MOVE_FUEL_RATE_L_PER_VEH = 0.02          # litres burned per vehicle that clears an intersection
CO2_PER_LITRE_PETROL_KG = 2.31           # kg CO2 per litre of petrol (standard conversion factor)


def compute_metrics(history: List[dict]) -> Dict[str, float]:
    """`history` is a list of per-step dicts produced by
    simulation.TrafficSimulator.run(), each containing at least:
        total_queue   : int   (sum of all queues that step)
        discharged    : int   (vehicles that moved through this step)
    """
    if not history:
        return {
            "avg_waiting_time": 0.0,
            "avg_queue_length": 0.0,
            "throughput": 0,
            "fuel_liters": 0.0,
            "co2_kg": 0.0,
        }

    total_queue_series = [step["total_queue"] for step in history]
    discharged_series = [step["discharged"] for step in history]

    avg_queue_length = sum(total_queue_series) / len(total_queue_series)
    throughput = sum(discharged_series)

    # vehicle-steps spent waiting = sum of queue length at every step
    # (a vehicle sitting in a 5-long queue for 3 steps contributes 15
    # vehicle-steps of idling)
    idle_vehicle_steps = sum(total_queue_series)

    fuel_liters = (
        IDLE_FUEL_RATE_L_PER_VEH_STEP * idle_vehicle_steps
        + MOVE_FUEL_RATE_L_PER_VEH * throughput
    )
    co2_kg = fuel_liters * CO2_PER_LITRE_PETROL_KG

    return {
        "avg_waiting_time": avg_queue_length,  # Little's-law proxy, see module docstring
        "avg_queue_length": avg_queue_length,
        "throughput": throughput,
        "fuel_liters": fuel_liters,
        "co2_kg": co2_kg,
    }


def compare(histories: Dict[str, List[dict]]) -> Dict[str, Dict[str, float]]:
    """histories: {controller_name: history_list} -> {controller_name: metrics}"""
    return {name: compute_metrics(h) for name, h in histories.items()}
