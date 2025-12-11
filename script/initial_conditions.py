#!/usr/bin/env python3
"""
Initial condition generation for two-body gravitational simulation.

This module provides functions to generate stable orbital configurations
with zero total momentum.

Usage:
    from script.initial_conditions import generate_circular_orbit
"""

import numpy as np
from numpy.typing import NDArray
import random

from script.physics import G
from script.units import Mass, Time, TwoBodyState, position, velocity


def generate_random_masses(
    ratio_max: float = 10.0,
    m_min: float = 1e11,
    m_max: float = 1e13
) -> tuple[Mass, Mass]:
    """
    Generate two random masses with ratio less than ratio_max.

    Args:
        ratio_max: Maximum allowed ratio between masses
        m_min: Minimum mass (kg)
        m_max: Maximum mass (kg)

    Returns:
        Tuple of (m1, m2) Mass objects
    """
    # Generate first mass uniformly in log space
    log_m1 = random.uniform(np.log10(m_min), np.log10(m_max))
    m1_val = 10 ** log_m1

    # Generate second mass within ratio constraint
    log_ratio = random.uniform(-np.log10(ratio_max), np.log10(ratio_max))
    m2_val = m1_val * (10 ** log_ratio)

    # Clamp to valid range
    m2_val = max(m_min, min(m_max, m2_val))

    # Round to 5 significant figures (matching precision of G)
    m1_val = float(f"{m1_val:.4e}")
    m2_val = float(f"{m2_val:.4e}")

    return Mass(m1_val), Mass(m2_val)


def compute_orbital_parameters(
    m1: Mass,
    m2: Mass,
    period: Time
) -> tuple[float, float, float]:
    """
    Compute orbital parameters for circular orbit with given period.

    Uses Kepler's third law to find the separation distance,
    then computes velocities for circular orbit.

    Args:
        m1: Mass of body 1
        m2: Mass of body 2
        period: Orbital period

    Returns:
        Tuple of (separation, v1_magnitude, v2_magnitude)
        where separation is distance between bodies (m)
        and v1, v2 are orbital speeds (m/s)
    """
    M = float(m1) + float(m2)
    T = float(period)
    omega = 2 * np.pi / T

    # Kepler's third law: T² = (4π²/GM) * r³
    # Solving for r: r = (GM * T² / 4π²)^(1/3)
    r = (G * M * T**2 / (4 * np.pi**2)) ** (1/3)

    # Distance from center of mass
    r1 = r * float(m2) / M
    r2 = r * float(m1) / M

    # Orbital velocities (v = ω * r)
    v1 = omega * r1
    v2 = omega * r2

    return r, v1, v2


def generate_circular_orbit(
    m1: Mass,
    m2: Mass,
    period: Time
) -> TwoBodyState:
    """
    Generate initial state for circular orbit with zero total momentum.

    Bodies are placed on the x-axis, moving in the y-direction.
    Body 1 is at positive x with positive y velocity.
    Body 2 is at negative x with negative y velocity.
    Center of mass is at origin.

    Args:
        m1: Mass of body 1
        m2: Mass of body 2
        period: Orbital period

    Returns:
        TwoBodyState for the initial configuration
    """
    M = float(m1) + float(m2)
    r, v1_mag, v2_mag = compute_orbital_parameters(m1, m2, period)

    # Distances from center of mass
    r1 = r * float(m2) / M
    r2 = r * float(m1) / M

    # Create typed positions and velocities
    pos1 = position(r1, 0.0)
    vel1 = velocity(0.0, v1_mag)
    pos2 = position(-r2, 0.0)
    vel2 = velocity(0.0, -v2_mag)

    return TwoBodyState.from_bodies(pos1, vel1, pos2, vel2)


def generate_random_period(
    min_period: float = 1.0,
    max_period: float = 4.0
) -> Time:
    """
    Generate random orbital period in given range.

    Args:
        min_period: Minimum period (s)
        max_period: Maximum period (s)

    Returns:
        Random period as Time object
    """
    return Time(random.uniform(min_period, max_period))


def main() -> None:
    """Module test."""
    print("Initial conditions module loaded successfully.")

    m1, m2 = generate_random_masses()
    period = generate_random_period()

    print(f"Random masses: m1 = {float(m1):.3e} kg, m2 = {float(m2):.3e} kg")
    print(f"Mass ratio: {max(float(m1), float(m2)) / min(float(m1), float(m2)):.2f}")
    print(f"Random period: {float(period):.3f} s")

    state = generate_circular_orbit(m1, m2, period)
    print(f"Initial state shape: {state.array.shape}")


if __name__ == '__main__':
    main()
