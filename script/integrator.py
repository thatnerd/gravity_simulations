#!/usr/bin/env python3
"""
Numerical integration for two-body gravitational simulation.

This module provides RK4 integration and stability checking.
Uses typed physical quantities at the API boundary but raw numpy
arrays internally for performance.

Usage:
    from script.integrator import rk4_step, check_stability
"""

import numpy as np
from numpy.typing import NDArray

from script.units import Mass, Time, TwoBodyState, Acceleration

# Acceleration threshold for collision detection (m/s²)
STABILITY_THRESHOLD: float = 1e12


def _compute_accelerations_fast(
    state: NDArray[np.float64],
    m1: float,
    m2: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Fast acceleration computation using raw arrays.

    Internal function for use in tight loops.

    Args:
        state: State array of shape (2, 2, 2) - [body, pos/vel, x/y]
        m1: Mass of body 1 (kg)
        m2: Mass of body 2 (kg)

    Returns:
        Tuple of (a1, a2) acceleration arrays
    """
    from script.physics import G

    r1 = state[0, 0]
    r2 = state[1, 0]

    r_vec = r2 - r1
    r_mag = np.linalg.norm(r_vec)
    r_hat = r_vec / r_mag
    force_mag = G * m1 * m2 / (r_mag ** 2)

    f1 = force_mag * r_hat
    a1 = f1 / m1
    a2 = -f1 / m2  # Newton's third law

    return a1, a2


def _derivatives_fast(
    state: NDArray[np.float64],
    m1: float,
    m2: float
) -> NDArray[np.float64]:
    """
    Fast derivative computation using raw arrays.

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
    a1, a2 = _compute_accelerations_fast(state, m1, m2)
    derivs[0, 1] = a1
    derivs[1, 1] = a2

    return derivs


def _rk4_step_fast(
    state: NDArray[np.float64],
    dt: float,
    m1: float,
    m2: float
) -> NDArray[np.float64]:
    """
    Fast RK4 step using raw arrays.

    Internal function for use in tight loops.

    Args:
        state: Current state array of shape (2, 2, 2)
        dt: Time step (s)
        m1: Mass of body 1 (kg)
        m2: Mass of body 2 (kg)

    Returns:
        New state after time dt
    """
    k1 = _derivatives_fast(state, m1, m2)
    k2 = _derivatives_fast(state + 0.5 * dt * k1, m1, m2)
    k3 = _derivatives_fast(state + 0.5 * dt * k2, m1, m2)
    k4 = _derivatives_fast(state + dt * k3, m1, m2)

    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def rk4_step(
    state: TwoBodyState,
    dt: Time,
    m1: Mass,
    m2: Mass
) -> TwoBodyState:
    """
    Perform one RK4 integration step.

    Args:
        state: Current two-body state
        dt: Time step
        m1: Mass of body 1
        m2: Mass of body 2

    Returns:
        New state after time dt
    """
    new_array = _rk4_step_fast(state.array, float(dt), float(m1), float(m2))
    return TwoBodyState.from_array(new_array)


def compute_accelerations(
    state: TwoBodyState,
    m1: Mass,
    m2: Mass
) -> tuple[Acceleration, Acceleration]:
    """
    Compute typed acceleration vectors for both bodies.

    Args:
        state: Two-body state containing positions and velocities
        m1: Mass of body 1
        m2: Mass of body 2

    Returns:
        Tuple of (a1, a2) typed acceleration vectors
    """
    a1_arr, a2_arr = _compute_accelerations_fast(state.array, float(m1), float(m2))
    return Acceleration(a1_arr), Acceleration(a2_arr)


def check_stability(
    state: TwoBodyState,
    m1: Mass,
    m2: Mass,
    threshold: float = STABILITY_THRESHOLD
) -> bool:
    """
    Check if the simulation is numerically stable.

    Returns False if acceleration magnitude exceeds threshold,
    indicating bodies are too close (collision).

    Args:
        state: Two-body state
        m1: Mass of body 1
        m2: Mass of body 2
        threshold: Maximum allowed acceleration (m/s²)

    Returns:
        True if stable, False if collision detected
    """
    a1_arr, a2_arr = _compute_accelerations_fast(state.array, float(m1), float(m2))

    a1_mag = np.linalg.norm(a1_arr)
    a2_mag = np.linalg.norm(a2_arr)

    return a1_mag < threshold and a2_mag < threshold


def main() -> None:
    """Module test."""
    print("Integrator module loaded successfully.")
    print(f"Stability threshold: {STABILITY_THRESHOLD:.2e} m/s²")


if __name__ == '__main__':
    main()
