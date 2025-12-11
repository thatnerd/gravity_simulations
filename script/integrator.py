#!/usr/bin/env python3
"""
Numerical integration for two-body gravitational simulation.

This module provides RK4 integration and stability checking.

Usage:
    from script.integrator import rk4_step, check_stability
"""

import numpy as np
from numpy.typing import NDArray

from script.physics import compute_accelerations

# Acceleration threshold for collision detection (m/s²)
STABILITY_THRESHOLD: float = 1e12


def derivatives(
    state: NDArray[np.float64],
    m1: float,
    m2: float
) -> NDArray[np.float64]:
    """
    Compute time derivatives of the state vector.

    For each body:
        d(position)/dt = velocity
        d(velocity)/dt = acceleration

    Args:
        state: State array of shape (2, 2, 2) - [body, pos/vel, x/y]
        m1: Mass of body 1 (kg)
        m2: Mass of body 2 (kg)

    Returns:
        Derivative array of same shape as state
    """
    derivs = np.zeros_like(state)

    # Position derivatives are velocities
    derivs[0, 0] = state[0, 1]
    derivs[1, 0] = state[1, 1]

    # Velocity derivatives are accelerations
    a1, a2 = compute_accelerations(state, m1, m2)
    derivs[0, 1] = a1
    derivs[1, 1] = a2

    return derivs


def rk4_step(
    state: NDArray[np.float64],
    dt: float,
    m1: float,
    m2: float
) -> NDArray[np.float64]:
    """
    Perform one RK4 integration step.

    Args:
        state: Current state array of shape (2, 2, 2)
        dt: Time step (s)
        m1: Mass of body 1 (kg)
        m2: Mass of body 2 (kg)

    Returns:
        New state after time dt
    """
    k1 = derivatives(state, m1, m2)
    k2 = derivatives(state + 0.5 * dt * k1, m1, m2)
    k3 = derivatives(state + 0.5 * dt * k2, m1, m2)
    k4 = derivatives(state + dt * k3, m1, m2)

    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def check_stability(
    state: NDArray[np.float64],
    m1: float,
    m2: float,
    threshold: float = STABILITY_THRESHOLD
) -> bool:
    """
    Check if the simulation is numerically stable.

    Returns False if acceleration magnitude exceeds threshold,
    indicating bodies are too close (collision).

    Args:
        state: State array of shape (2, 2, 2)
        m1: Mass of body 1 (kg)
        m2: Mass of body 2 (kg)
        threshold: Maximum allowed acceleration (m/s²)

    Returns:
        True if stable, False if collision detected
    """
    a1, a2 = compute_accelerations(state, m1, m2)

    a1_mag = np.linalg.norm(a1)
    a2_mag = np.linalg.norm(a2)

    return a1_mag < threshold and a2_mag < threshold


def main() -> None:
    """Module test."""
    print("Integrator module loaded successfully.")
    print(f"Stability threshold: {STABILITY_THRESHOLD:.2e} m/s²")


if __name__ == '__main__':
    main()
