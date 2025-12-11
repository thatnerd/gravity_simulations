#!/usr/bin/env python3
"""
Physics calculations for two-body gravitational simulation.

This module provides functions for computing gravitational forces,
accelerations, and conserved quantities (energy, momentum, angular momentum).

All functions use typed physical quantities from the units module.

Usage:
    from script.physics import gravitational_force, total_energy
"""

import numpy as np
from numpy.typing import NDArray

from script.units import (
    Mass, Energy, AngularMomentum as AngularMomentumType,
    Position, Velocity, Acceleration, Force, Momentum,
    TwoBodyState,
)

# Gravitational constant in SI units: m³/(kg·s²)
G: float = 6.6743e-11


def gravitational_force(
    m1: Mass,
    m2: Mass,
    r1: Position,
    r2: Position
) -> Force:
    """
    Compute gravitational force on body 1 due to body 2.

    Args:
        m1: Mass of body 1
        m2: Mass of body 2
        r1: Position of body 1
        r2: Position of body 2

    Returns:
        Force on body 1 (directed toward body 2)
    """
    r_vec = r2.array - r1.array
    r_mag = np.linalg.norm(r_vec)
    r_hat = r_vec / r_mag
    force_mag = G * float(m1) * float(m2) / (r_mag ** 2)
    return Force(force_mag * r_hat)


def compute_accelerations(
    state: TwoBodyState,
    m1: Mass,
    m2: Mass
) -> tuple[Acceleration, Acceleration]:
    """
    Compute acceleration vectors for both bodies.

    Args:
        state: Two-body state containing positions and velocities
        m1: Mass of body 1
        m2: Mass of body 2

    Returns:
        Tuple of (a1, a2) typed acceleration vectors
    """
    r1 = state.position(0)
    r2 = state.position(1)

    f1 = gravitational_force(m1, m2, r1, r2)
    a1 = f1 / m1
    a2 = -f1 / m2  # Newton's third law

    return a1, a2


def total_momentum(
    state: TwoBodyState,
    m1: Mass,
    m2: Mass
) -> Momentum:
    """
    Compute total momentum of the two-body system.

    Args:
        state: Two-body state containing positions and velocities
        m1: Mass of body 1
        m2: Mass of body 2

    Returns:
        Total momentum vector
    """
    v1 = state.velocity(0)
    v2 = state.velocity(1)
    return m1 * v1 + m2 * v2


def angular_momentum(
    state: TwoBodyState,
    m1: Mass,
    m2: Mass
) -> AngularMomentumType:
    """
    Compute total angular momentum of the system about the origin.

    In 2D, angular momentum is a scalar (z-component of r × p).

    Args:
        state: Two-body state containing positions and velocities
        m1: Mass of body 1
        m2: Mass of body 2

    Returns:
        Total angular momentum (positive for counter-clockwise)
    """
    r1, v1 = state.position(0), state.velocity(0)
    r2, v2 = state.position(1), state.velocity(1)

    # L = r × p = r × (m*v) = m * (r × v)
    # In 2D: r × v = rx*vy - ry*vx
    L1 = float(m1) * (r1.x * v1.y - r1.y * v1.x)
    L2 = float(m2) * (r2.x * v2.y - r2.y * v2.x)

    return AngularMomentumType(L1 + L2)


def kinetic_energy(
    state: TwoBodyState,
    m1: Mass,
    m2: Mass
) -> Energy:
    """
    Compute total kinetic energy of the system.

    Args:
        state: Two-body state containing positions and velocities
        m1: Mass of body 1
        m2: Mass of body 2

    Returns:
        Total kinetic energy
    """
    v1 = state.velocity(0)
    v2 = state.velocity(1)

    KE1 = 0.5 * float(m1) * np.dot(v1.array, v1.array)
    KE2 = 0.5 * float(m2) * np.dot(v2.array, v2.array)

    return Energy(KE1 + KE2)


def potential_energy(
    state: TwoBodyState,
    m1: Mass,
    m2: Mass
) -> Energy:
    """
    Compute gravitational potential energy of the system.

    Args:
        state: Two-body state containing positions and velocities
        m1: Mass of body 1
        m2: Mass of body 2

    Returns:
        Gravitational potential energy (always negative)
    """
    r1 = state.position(0)
    r2 = state.position(1)
    r = np.linalg.norm(r2.array - r1.array)

    return Energy(-G * float(m1) * float(m2) / r)


def total_energy(
    state: TwoBodyState,
    m1: Mass,
    m2: Mass
) -> Energy:
    """
    Compute total mechanical energy of the system.

    Args:
        state: Two-body state containing positions and velocities
        m1: Mass of body 1
        m2: Mass of body 2

    Returns:
        Total energy (kinetic + potential)
    """
    return kinetic_energy(state, m1, m2) + potential_energy(state, m1, m2)


def main() -> None:
    """Module test."""
    print("Physics module loaded successfully.")
    print(f"G = {G} m³/(kg·s²)")


if __name__ == '__main__':
    main()
