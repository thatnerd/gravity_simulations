#!/usr/bin/env python3
"""
Physics calculations for gravitational simulation.

This module provides functions for computing gravitational forces,
accelerations, and conserved quantities (energy, momentum, angular momentum).

Functions operate on single Body objects or pairs of Body objects.

Usage:
    from script.physics import gravitational_force, kinetic_energy
"""

import numpy as np

from script.units import (
    Mass, Energy, AngularMomentum as AngularMomentumType,
    Position, Velocity, Acceleration, Force, Momentum,
    Body,
)

# Gravitational constant in SI units: m³/(kg·s²)
G: float = 6.6743e-11


def gravitational_force(b1: Body, b2: Body) -> Force:
    """
    Compute gravitational force on body 1 due to body 2.

    Args:
        b1: Body experiencing the force
        b2: Body exerting the force

    Returns:
        Force on b1 (directed toward b2)
    """
    r_vec = b2.position.array - b1.position.array
    r_mag = np.linalg.norm(r_vec)
    r_hat = r_vec / r_mag
    force_mag = G * float(b1.mass) * float(b2.mass) / (r_mag ** 2)
    return Force(force_mag * r_hat)


def momentum(body: Body) -> Momentum:
    """
    Compute momentum of a single body.

    Args:
        body: Body to compute momentum for

    Returns:
        Momentum vector (p = m*v)
    """
    return body.mass * body.velocity


def angular_momentum(body: Body) -> AngularMomentumType:
    """
    Compute angular momentum of a single body about the origin.

    L = r × p where p = m*v (momentum).

    Args:
        body: Body to compute angular momentum for

    Returns:
        Angular momentum vector (z-component nonzero for 2D motion in xy-plane)
    """
    p = momentum(body)
    return body.position.cross(p)


def kinetic_energy(body: Body) -> Energy:
    """
    Compute kinetic energy of a single body.

    Args:
        body: Body to compute kinetic energy for

    Returns:
        Kinetic energy (KE = 0.5 * m * v²)
    """
    v = body.velocity
    return Energy(0.5 * float(body.mass) * np.dot(v.array, v.array))


def potential_energy(b1: Body, b2: Body) -> Energy:
    """
    Compute gravitational potential energy between two bodies.

    Args:
        b1: First body
        b2: Second body

    Returns:
        Gravitational potential energy (always negative)
    """
    r = np.linalg.norm(b2.position.array - b1.position.array)
    return Energy(-G * float(b1.mass) * float(b2.mass) / r)


def main() -> None:
    """Module test."""
    print("Physics module loaded successfully.")
    print(f"G = {G} m³/(kg·s²)")


if __name__ == '__main__':
    main()
