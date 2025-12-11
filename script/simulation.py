#!/usr/bin/env python3
"""
Main simulation loop for two-body gravitational simulation.

This module provides the SimulationResult dataclass and run_simulation function.

Usage:
    from script.simulation import run_simulation, SimulationResult
"""

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Optional

from script.units import Mass, Time, TwoBodyState
from script.physics import (
    total_momentum,
    angular_momentum,
    total_energy,
)
from script.integrator import rk4_step, check_stability, STABILITY_THRESHOLD


@dataclass
class SimulationResult:
    """Container for simulation results."""
    metadata: dict
    times: NDArray[np.float64]
    positions: NDArray[np.float64]
    velocities: NDArray[np.float64]
    momentum: NDArray[np.float64]
    angular_momentum: NDArray[np.float64]
    energy: NDArray[np.float64]
    collision: bool


def run_simulation(
    initial_state: TwoBodyState,
    m1: Mass,
    m2: Mass,
    dt: Time,
    t_max: Time,
    stability_threshold: Optional[float] = None
) -> SimulationResult:
    """
    Run the two-body gravitational simulation.

    Args:
        initial_state: Initial two-body state
        m1: Mass of body 1
        m2: Mass of body 2
        dt: Time step
        t_max: Maximum simulation time
        stability_threshold: Custom stability threshold (m/s²), or None for default

    Returns:
        SimulationResult containing all recorded data
    """
    threshold = stability_threshold if stability_threshold is not None else STABILITY_THRESHOLD

    dt_val = float(dt)
    t_max_val = float(t_max)
    n_steps = int(t_max_val / dt_val) + 1

    # Pre-allocate arrays
    times = np.zeros(n_steps)
    positions = np.zeros((n_steps, 2, 2))
    velocities = np.zeros((n_steps, 2, 2))
    momentum_arr = np.zeros((n_steps, 2))
    angular_momentum_arr = np.zeros((n_steps, 3))  # 3D vector
    energy_arr = np.zeros(n_steps)

    # Initialize
    state = initial_state.copy()
    collision = False
    actual_steps = n_steps

    for i in range(n_steps):
        t = i * dt_val
        times[i] = t
        positions[i] = state.array[:, 0]
        velocities[i] = state.array[:, 1]

        # Compute conserved quantities
        p = total_momentum(state, m1, m2)
        L = angular_momentum(state, m1, m2)
        E = total_energy(state, m1, m2)

        momentum_arr[i] = p.array
        angular_momentum_arr[i] = L.array
        energy_arr[i] = float(E)

        # Check stability before next step
        if not check_stability(state, m1, m2, threshold):
            collision = True
            actual_steps = i + 1
            break

        # Advance state (except on last iteration)
        if i < n_steps - 1:
            state = rk4_step(state, dt, m1, m2)

    # Trim arrays if collision occurred
    if collision:
        times = times[:actual_steps]
        positions = positions[:actual_steps]
        velocities = velocities[:actual_steps]
        momentum_arr = momentum_arr[:actual_steps]
        angular_momentum_arr = angular_momentum_arr[:actual_steps]
        energy_arr = energy_arr[:actual_steps]

    metadata = {
        'm1': float(m1),
        'm2': float(m2),
        'dt': dt_val,
        't_max': t_max_val,
        'n_steps': actual_steps,
        'stability_threshold': threshold,
    }

    return SimulationResult(
        metadata=metadata,
        times=times,
        positions=positions,
        velocities=velocities,
        momentum=momentum_arr,
        angular_momentum=angular_momentum_arr,
        energy=energy_arr,
        collision=collision
    )


def main() -> None:
    """Module test."""
    print("Simulation module loaded successfully.")


if __name__ == '__main__':
    main()
