#!/usr/bin/env python3
"""
Main simulation loop for gravitational simulation.

This module provides the SimulationResult dataclass and run_simulation function.

Usage:
    from script.simulation import run_simulation, SimulationResult
"""

from itertools import combinations
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Optional

from script.units import Mass, Time, Body
from script.physics import (
    momentum,
    angular_momentum,
    kinetic_energy,
    potential_energy,
)
from script.integrator import rk4_step, check_stability, STABILITY_THRESHOLD


@dataclass
class SimulationResult:
    """Container for simulation results."""
    metadata: dict
    times: NDArray[np.float64]
    positions: NDArray[np.float64]  # (n_steps, n_bodies, 2)
    velocities: NDArray[np.float64]  # (n_steps, n_bodies, 2)
    momentum: NDArray[np.float64]  # (n_steps, 2) - total
    angular_momentum: NDArray[np.float64]  # (n_steps, 3) - total
    energy: NDArray[np.float64]  # (n_steps,) - total
    collision: bool


def _total_momentum(bodies: list[Body]) -> NDArray[np.float64]:
    """Compute total momentum of all bodies."""
    total = np.zeros(2, dtype=np.float64)
    for b in bodies:
        total += momentum(b).array
    return total


def _total_angular_momentum(bodies: list[Body]) -> NDArray[np.float64]:
    """Compute total angular momentum of all bodies."""
    total = np.zeros(3, dtype=np.float64)
    for b in bodies:
        total += angular_momentum(b).array
    return total


def _total_energy(bodies: list[Body]) -> float:
    """Compute total energy (kinetic + potential) of all bodies."""
    # Kinetic energy
    ke = sum(float(kinetic_energy(b)) for b in bodies)

    # Potential energy (sum over pairs)
    pe = 0.0
    for i, j in combinations(range(len(bodies)), 2):
        pe += float(potential_energy(bodies[i], bodies[j]))

    return ke + pe


def run_simulation(
    bodies: list[Body],
    dt: Time,
    t_max: Time,
    stability_threshold: Optional[float] = None
) -> SimulationResult:
    """
    Run the gravitational simulation.

    Args:
        bodies: List of Body objects (initial state)
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
    n_bodies = len(bodies)

    # Pre-allocate arrays
    times = np.zeros(n_steps)
    positions = np.zeros((n_steps, n_bodies, 2))
    velocities = np.zeros((n_steps, n_bodies, 2))
    momentum_arr = np.zeros((n_steps, 2))
    angular_momentum_arr = np.zeros((n_steps, 3))
    energy_arr = np.zeros(n_steps)

    # Initialize
    current_bodies = bodies
    collision = False
    actual_steps = n_steps

    for i in range(n_steps):
        t = i * dt_val
        times[i] = t

        # Record positions and velocities
        for j, b in enumerate(current_bodies):
            positions[i, j] = b.position.array
            velocities[i, j] = b.velocity.array

        # Compute conserved quantities
        momentum_arr[i] = _total_momentum(current_bodies)
        angular_momentum_arr[i] = _total_angular_momentum(current_bodies)
        energy_arr[i] = _total_energy(current_bodies)

        # Check stability before next step
        if not check_stability(current_bodies, threshold):
            collision = True
            actual_steps = i + 1
            break

        # Advance state (except on last iteration)
        if i < n_steps - 1:
            current_bodies = rk4_step(current_bodies, dt)

    # Trim arrays if collision occurred
    if collision:
        times = times[:actual_steps]
        positions = positions[:actual_steps]
        velocities = velocities[:actual_steps]
        momentum_arr = momentum_arr[:actual_steps]
        angular_momentum_arr = angular_momentum_arr[:actual_steps]
        energy_arr = energy_arr[:actual_steps]

    # Extract masses for metadata
    masses = [float(b.mass) for b in bodies]

    metadata = {
        'masses': masses,
        'm1': masses[0] if len(masses) > 0 else None,
        'm2': masses[1] if len(masses) > 1 else None,
        'n_bodies': n_bodies,
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
