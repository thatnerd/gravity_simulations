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

from script.physics import (
    total_momentum,
    angular_momentum,
    total_energy,
)
from script.integrator import rk4_step, check_stability


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
    initial_state: NDArray[np.float64],
    m1: float,
    m2: float,
    dt: float = 0.01,
    t_max: float = 100.0,
    stability_threshold: Optional[float] = None
) -> SimulationResult:
    """
    Run the two-body gravitational simulation.

    Args:
        initial_state: Initial state array of shape (2, 2, 2)
        m1: Mass of body 1 (kg)
        m2: Mass of body 2 (kg)
        dt: Time step (s)
        t_max: Maximum simulation time (s)
        stability_threshold: Custom stability threshold (m/s²), or None for default

    Returns:
        SimulationResult containing all recorded data
    """
    from script.integrator import STABILITY_THRESHOLD
    threshold = stability_threshold if stability_threshold is not None else STABILITY_THRESHOLD

    n_steps = int(t_max / dt) + 1

    # Pre-allocate arrays
    times = np.zeros(n_steps)
    positions = np.zeros((n_steps, 2, 2))
    velocities = np.zeros((n_steps, 2, 2))
    momentum_arr = np.zeros((n_steps, 2))
    angular_momentum_arr = np.zeros(n_steps)
    energy_arr = np.zeros(n_steps)

    # Initialize
    state = initial_state.copy()
    collision = False
    actual_steps = n_steps

    for i in range(n_steps):
        t = i * dt
        times[i] = t
        positions[i] = state[:, 0]
        velocities[i] = state[:, 1]
        momentum_arr[i] = total_momentum(state, m1, m2)
        angular_momentum_arr[i] = angular_momentum(state, m1, m2)
        energy_arr[i] = total_energy(state, m1, m2)

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
        'm1': m1,
        'm2': m2,
        'dt': dt,
        't_max': t_max,
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
