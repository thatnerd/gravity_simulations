#!/usr/bin/env python3
"""
Numerical integration for gravitational simulation.

This module provides RK4 integration and stability checking.
Works with lists of Body objects for n-body support.

Usage:
    from script.integrator import rk4_step, check_stability
"""

from itertools import combinations
import numpy as np
from numpy.typing import NDArray

from script.units import Mass, Time, Body, Position, Velocity, Acceleration
from script.physics import G, gravitational_force

# Acceleration threshold for collision detection (m/s²)
STABILITY_THRESHOLD: float = 1e12


def compute_accelerations(bodies: list[Body]) -> list[Acceleration]:
    """
    Compute acceleration vectors for all bodies.

    Uses Newton's third law to compute each pair interaction only once.

    Args:
        bodies: List of Body objects

    Returns:
        List of Acceleration vectors, one per body
    """
    n = len(bodies)
    accels = [np.zeros(2, dtype=np.float64) for _ in range(n)]

    for i, j in combinations(range(n), 2):
        # Force on body i from body j
        f = gravitational_force(bodies[i], bodies[j])

        # a = F/m, and Newton's third law
        accels[i] += f.array / float(bodies[i].mass)
        accels[j] -= f.array / float(bodies[j].mass)

    return [Acceleration(a) for a in accels]


def _derivatives(bodies: list[Body]) -> tuple[list[NDArray], list[NDArray]]:
    """
    Compute derivatives for RK4 integration.

    For each body:
        d(position)/dt = velocity
        d(velocity)/dt = acceleration

    Args:
        bodies: List of Body objects

    Returns:
        Tuple of (position_derivatives, velocity_derivatives) as numpy arrays
    """
    accels = compute_accelerations(bodies)

    pos_derivs = [b.velocity.array for b in bodies]
    vel_derivs = [a.array for a in accels]

    return pos_derivs, vel_derivs


def rk4_step(bodies: list[Body], dt: Time) -> list[Body]:
    """
    Perform one RK4 integration step.

    Args:
        bodies: List of Body objects
        dt: Time step

    Returns:
        New list of Body objects after time dt
    """
    dt_val = float(dt)
    n = len(bodies)

    # Extract current state
    positions = [b.position.array.copy() for b in bodies]
    velocities = [b.velocity.array.copy() for b in bodies]
    masses = [b.mass for b in bodies]

    def make_bodies(pos_list: list[NDArray], vel_list: list[NDArray]) -> list[Body]:
        return [
            Body(masses[i], Position(pos_list[i]), Velocity(vel_list[i]))
            for i in range(n)
        ]

    # k1
    dp1, dv1 = _derivatives(bodies)

    # k2
    pos_k2 = [positions[i] + 0.5 * dt_val * dp1[i] for i in range(n)]
    vel_k2 = [velocities[i] + 0.5 * dt_val * dv1[i] for i in range(n)]
    dp2, dv2 = _derivatives(make_bodies(pos_k2, vel_k2))

    # k3
    pos_k3 = [positions[i] + 0.5 * dt_val * dp2[i] for i in range(n)]
    vel_k3 = [velocities[i] + 0.5 * dt_val * dv2[i] for i in range(n)]
    dp3, dv3 = _derivatives(make_bodies(pos_k3, vel_k3))

    # k4
    pos_k4 = [positions[i] + dt_val * dp3[i] for i in range(n)]
    vel_k4 = [velocities[i] + dt_val * dv3[i] for i in range(n)]
    dp4, dv4 = _derivatives(make_bodies(pos_k4, vel_k4))

    # Combine
    new_bodies = []
    for i in range(n):
        new_pos = positions[i] + (dt_val / 6.0) * (dp1[i] + 2*dp2[i] + 2*dp3[i] + dp4[i])
        new_vel = velocities[i] + (dt_val / 6.0) * (dv1[i] + 2*dv2[i] + 2*dv3[i] + dv4[i])
        new_bodies.append(Body(masses[i], Position(new_pos), Velocity(new_vel)))

    return new_bodies


def check_stability(
    bodies: list[Body],
    threshold: float = STABILITY_THRESHOLD
) -> bool:
    """
    Check if the simulation is numerically stable.

    Returns False if any acceleration magnitude exceeds threshold,
    indicating bodies are too close (collision).

    Args:
        bodies: List of Body objects
        threshold: Maximum allowed acceleration (m/s²)

    Returns:
        True if stable, False if collision detected
    """
    accels = compute_accelerations(bodies)
    for a in accels:
        if a.magnitude() >= threshold:
            return False
    return True


def main() -> None:
    """Module test."""
    print("Integrator module loaded successfully.")
    print(f"Stability threshold: {STABILITY_THRESHOLD:.2e} m/s²")


if __name__ == '__main__':
    main()
