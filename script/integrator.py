#!/usr/bin/env python3
"""
Numerical integration for gravitational simulation.

This module provides symplectic (Yoshida) and RK4 integration methods.
Works with lists of Body objects for n-body support.

Usage:
    from script.integrator import integrate_step, check_stability
"""

from itertools import combinations
from typing import Callable, Literal
import numpy as np
from numpy.typing import NDArray

from script.units import Mass, Time, Body, Position, Velocity, Acceleration
from script.physics import G, gravitational_force

# Acceleration threshold for collision detection (m/s²)
STABILITY_THRESHOLD: float = 1e12

# Integrator type
IntegratorType = Literal['yoshida', 'rk4']

# ============================================================
# Yoshida 4th Order Symplectic Integrator Coefficients
# ============================================================
# Reference: Yoshida, H. (1990). "Construction of higher order
# symplectic integrators". Physics Letters A, 150, 262-268.

_CBRT_2 = 2.0 ** (1.0 / 3.0)
_W0 = -_CBRT_2 / (2.0 - _CBRT_2)
_W1 = 1.0 / (2.0 - _CBRT_2)

# Position update coefficients (drift)
_C1 = _W1 / 2.0
_C2 = (_W0 + _W1) / 2.0
_C3 = _C2
_C4 = _C1

# Velocity update coefficients (kick)
_D1 = _W1
_D2 = _W0
_D3 = _W1
# _D4 = 0 (no final kick needed)


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


# ============================================================
# Yoshida Symplectic Integrator
# ============================================================

def _drift(bodies: list[Body], dt: float) -> list[Body]:
    """
    Position drift: x += v * dt

    Args:
        bodies: List of Body objects
        dt: Time step (already scaled by coefficient)

    Returns:
        New list of Body objects with updated positions
    """
    return [
        Body(
            b.mass,
            Position(b.position.array + b.velocity.array * dt),
            b.velocity
        )
        for b in bodies
    ]


def _kick(bodies: list[Body], dt: float) -> list[Body]:
    """
    Velocity kick: v += a(x) * dt

    Args:
        bodies: List of Body objects
        dt: Time step (already scaled by coefficient)

    Returns:
        New list of Body objects with updated velocities
    """
    accels = compute_accelerations(bodies)
    return [
        Body(
            b.mass,
            b.position,
            Velocity(b.velocity.array + accels[i].array * dt)
        )
        for i, b in enumerate(bodies)
    ]


def yoshida_step(bodies: list[Body], dt: Time) -> list[Body]:
    """
    Perform one Yoshida 4th order symplectic integration step.

    The Yoshida integrator is symplectic, meaning it preserves phase-space
    volume and exhibits bounded energy oscillations rather than secular drift.
    This makes it ideal for long-term orbital simulations.

    Args:
        bodies: List of Body objects
        dt: Time step

    Returns:
        New list of Body objects after time dt
    """
    dt_val = float(dt)

    # Yoshida 4th order: alternating drift (position) and kick (velocity)
    # using carefully chosen coefficients for 4th order accuracy

    # Step 1: drift by c1*dt
    bodies = _drift(bodies, _C1 * dt_val)
    # Step 2: kick by d1*dt
    bodies = _kick(bodies, _D1 * dt_val)
    # Step 3: drift by c2*dt
    bodies = _drift(bodies, _C2 * dt_val)
    # Step 4: kick by d2*dt
    bodies = _kick(bodies, _D2 * dt_val)
    # Step 5: drift by c3*dt
    bodies = _drift(bodies, _C3 * dt_val)
    # Step 6: kick by d3*dt
    bodies = _kick(bodies, _D3 * dt_val)
    # Step 7: drift by c4*dt
    bodies = _drift(bodies, _C4 * dt_val)
    # (d4 = 0, so no final kick)

    return bodies


# ============================================================
# RK4 Integrator
# ============================================================

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

    RK4 is a general-purpose 4th order method with good local accuracy,
    but it is not symplectic and will exhibit energy drift over long
    simulations.

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


# ============================================================
# Integrator Dispatcher
# ============================================================

_INTEGRATORS: dict[str, Callable[[list[Body], Time], list[Body]]] = {
    'yoshida': yoshida_step,
    'rk4': rk4_step,
}


def integrate_step(
    bodies: list[Body],
    dt: Time,
    method: IntegratorType = 'yoshida'
) -> list[Body]:
    """
    Perform one integration step using the specified method.

    Args:
        bodies: List of Body objects
        dt: Time step
        method: Integration method ('yoshida' or 'rk4')

    Returns:
        New list of Body objects after time dt

    Raises:
        ValueError: If method is not recognized
    """
    if method not in _INTEGRATORS:
        raise ValueError(f"Unknown integrator: {method}. Choose from: {list(_INTEGRATORS.keys())}")
    return _INTEGRATORS[method](bodies, dt)


# ============================================================
# Stability Check
# ============================================================

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
    print(f"Available integrators: {list(_INTEGRATORS.keys())}")
    print(f"Default integrator: yoshida")
    print(f"\nYoshida coefficients:")
    print(f"  w0 = {_W0:.10f}")
    print(f"  w1 = {_W1:.10f}")
    print(f"  c1 = c4 = {_C1:.10f}")
    print(f"  c2 = c3 = {_C2:.10f}")
    print(f"  d1 = d3 = {_D1:.10f}")
    print(f"  d2 = {_D2:.10f}")


if __name__ == '__main__':
    main()
