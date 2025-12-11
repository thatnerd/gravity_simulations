#!/usr/bin/env python3
"""
Physics calculations for two-body gravitational simulation.

This module provides functions for computing gravitational forces,
accelerations, and conserved quantities (energy, momentum, angular momentum).

Usage:
    from script.physics import gravitational_force, total_energy
"""

import numpy as np
from numpy.typing import NDArray

# Gravitational constant in SI units: m³/(kg·s²)
G: float = 6.67430e-11


def gravitational_force(
    m1: float,
    m2: float,
    r1: NDArray[np.float64],
    r2: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Compute gravitational force on body 1 due to body 2.

    Args:
        m1: Mass of body 1 (kg)
        m2: Mass of body 2 (kg)
        r1: Position vector of body 1 (m)
        r2: Position vector of body 2 (m)

    Returns:
        Force vector on body 1 (N)
    """
    r_vec = r2 - r1
    r_mag = np.linalg.norm(r_vec)
    r_hat = r_vec / r_mag
    force_mag = G * m1 * m2 / (r_mag ** 2)
    return force_mag * r_hat


def compute_accelerations(
    state: NDArray[np.float64],
    m1: float,
    m2: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute acceleration vectors for both bodies.

    Args:
        state: State array of shape (2, 2, 2) - [body, pos/vel, x/y]
        m1: Mass of body 1 (kg)
        m2: Mass of body 2 (kg)

    Returns:
        Tuple of (a1, a2) acceleration vectors
    """
    r1 = state[0, 0]
    r2 = state[1, 0]

    f1 = gravitational_force(m1, m2, r1, r2)
    a1 = f1 / m1
    a2 = -f1 / m2  # Newton's third law

    return a1, a2


def total_momentum(
    state: NDArray[np.float64],
    m1: float,
    m2: float
) -> NDArray[np.float64]:
    """
    Compute total momentum of the two-body system.

    Args:
        state: State array of shape (2, 2, 2) - [body, pos/vel, x/y]
        m1: Mass of body 1 (kg)
        m2: Mass of body 2 (kg)

    Returns:
        Total momentum vector (kg·m/s)
    """
    v1 = state[0, 1]
    v2 = state[1, 1]
    return m1 * v1 + m2 * v2


def angular_momentum(
    state: NDArray[np.float64],
    m1: float,
    m2: float
) -> float:
    """
    Compute total angular momentum of the system about the origin.

    In 2D, angular momentum is a scalar (z-component of r × p).

    Args:
        state: State array of shape (2, 2, 2) - [body, pos/vel, x/y]
        m1: Mass of body 1 (kg)
        m2: Mass of body 2 (kg)

    Returns:
        Total angular momentum (kg·m²/s), positive for counter-clockwise
    """
    r1, v1 = state[0, 0], state[0, 1]
    r2, v2 = state[1, 0], state[1, 1]

    # L = r × p = r × (m*v) = m * (r × v)
    # In 2D: r × v = rx*vy - ry*vx
    L1 = m1 * (r1[0] * v1[1] - r1[1] * v1[0])
    L2 = m2 * (r2[0] * v2[1] - r2[1] * v2[0])

    return L1 + L2


def kinetic_energy(
    state: NDArray[np.float64],
    m1: float,
    m2: float
) -> float:
    """
    Compute total kinetic energy of the system.

    Args:
        state: State array of shape (2, 2, 2) - [body, pos/vel, x/y]
        m1: Mass of body 1 (kg)
        m2: Mass of body 2 (kg)

    Returns:
        Total kinetic energy (J)
    """
    v1 = state[0, 1]
    v2 = state[1, 1]

    KE1 = 0.5 * m1 * np.dot(v1, v1)
    KE2 = 0.5 * m2 * np.dot(v2, v2)

    return KE1 + KE2


def potential_energy(
    state: NDArray[np.float64],
    m1: float,
    m2: float
) -> float:
    """
    Compute gravitational potential energy of the system.

    Args:
        state: State array of shape (2, 2, 2) - [body, pos/vel, x/y]
        m1: Mass of body 1 (kg)
        m2: Mass of body 2 (kg)

    Returns:
        Gravitational potential energy (J), always negative
    """
    r1 = state[0, 0]
    r2 = state[1, 0]
    r = np.linalg.norm(r2 - r1)

    return -G * m1 * m2 / r


def total_energy(
    state: NDArray[np.float64],
    m1: float,
    m2: float
) -> float:
    """
    Compute total mechanical energy of the system.

    Args:
        state: State array of shape (2, 2, 2) - [body, pos/vel, x/y]
        m1: Mass of body 1 (kg)
        m2: Mass of body 2 (kg)

    Returns:
        Total energy (J)
    """
    return kinetic_energy(state, m1, m2) + potential_energy(state, m1, m2)


def main() -> None:
    """Module test."""
    print("Physics module loaded successfully.")
    print(f"G = {G} m³/(kg·s²)")


if __name__ == '__main__':
    main()
