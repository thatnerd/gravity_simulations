#!/usr/bin/env python3
"""
Initial condition generation for gravitational simulation.

This module provides functions to generate stable orbital configurations
with zero total momentum.

Usage:
    from script.initial_conditions import generate_elliptical_orbit
"""

import numpy as np
from numpy.typing import NDArray
import random

from script.physics import G
from script.units import Mass, Time, Body, Position, Velocity, position, velocity


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


def compute_semi_major_axis(m1: Mass, m2: Mass, period: Time) -> float:
    """
    Compute semi-major axis from period using Kepler's third law.

    Args:
        m1: Mass of body 1
        m2: Mass of body 2
        period: Orbital period

    Returns:
        Semi-major axis in meters
    """
    M = float(m1) + float(m2)
    T = float(period)
    # Kepler's third law: T² = 4π²a³/(GM)
    return (G * M * T**2 / (4 * np.pi**2)) ** (1/3)


def generate_elliptical_orbit(
    m1: Mass,
    m2: Mass,
    period: Time,
    eccentricity: float = 0.0,
    true_anomaly: float = 0.0
) -> list[Body]:
    """
    Generate initial state for elliptical orbit with zero total momentum.

    The orbit is defined by the semi-major axis (derived from period),
    eccentricity, and starting true anomaly. Center of mass is at origin.

    Args:
        m1: Mass of body 1
        m2: Mass of body 2
        period: Orbital period
        eccentricity: Orbital eccentricity (0 = circular, <1 = elliptical)
        true_anomaly: Starting angle from periapsis in radians

    Returns:
        List of two Body objects for the initial configuration
    """
    if not 0.0 <= eccentricity < 1.0:
        raise ValueError(f"Eccentricity must be in [0, 1), got {eccentricity}")

    M = float(m1) + float(m2)
    m1_f = float(m1)
    m2_f = float(m2)
    e = eccentricity
    theta = true_anomaly

    # Semi-major axis from Kepler's third law
    a = compute_semi_major_axis(m1, m2, period)

    # Semi-latus rectum
    p = a * (1 - e**2)

    # Distance between bodies at this angle
    r = p / (1 + e * np.cos(theta))

    # Position: body 1 at angle theta from periapsis direction
    # In the center-of-mass frame, distances are split by mass ratio
    r1 = r * m2_f / M  # body 1's distance from CoM
    r2 = r * m1_f / M  # body 2's distance from CoM

    # Position vectors (body 1 at angle theta, body 2 opposite)
    pos1_x = r1 * np.cos(theta)
    pos1_y = r1 * np.sin(theta)
    pos2_x = -r2 * np.cos(theta)
    pos2_y = -r2 * np.sin(theta)

    # Velocity calculation using vis-viva equation and angular momentum
    # Speed from vis-viva: v² = GM(2/r - 1/a)
    v_total = np.sqrt(G * M * (2/r - 1/a))

    # Flight path angle: angle between velocity and perpendicular to radius
    # gamma = atan2(e*sin(theta), 1 + e*cos(theta))
    gamma = np.arctan2(e * np.sin(theta), 1 + e * np.cos(theta))

    # Velocity direction: perpendicular to radius (theta + 90°) plus flight path angle
    # For counterclockwise orbit
    vel_angle = theta + np.pi/2 - gamma

    # Total velocity components
    v_total_x = v_total * np.cos(vel_angle)
    v_total_y = v_total * np.sin(vel_angle)

    # Split velocity by mass ratio (momentum conservation: m1*v1 + m2*v2 = 0)
    # v1 = (m2/M) * v_total, v2 = -(m1/M) * v_total
    vel1_x = (m2_f / M) * v_total_x
    vel1_y = (m2_f / M) * v_total_y
    vel2_x = -(m1_f / M) * v_total_x
    vel2_y = -(m1_f / M) * v_total_y

    pos1 = position(pos1_x, pos1_y)
    vel1 = velocity(vel1_x, vel1_y)
    pos2 = position(pos2_x, pos2_y)
    vel2 = velocity(vel2_x, vel2_y)

    return [Body(m1, pos1, vel1), Body(m2, pos2, vel2)]


def generate_random_eccentricity(max_e: float = 0.8) -> float:
    """
    Generate random eccentricity in [0, max_e).

    Args:
        max_e: Maximum eccentricity (must be < 1)

    Returns:
        Random eccentricity value
    """
    return random.uniform(0.0, max_e)


def generate_random_true_anomaly() -> float:
    """
    Generate random true anomaly in [0, 2π).

    Returns:
        Random angle in radians
    """
    return random.uniform(0.0, 2 * np.pi)


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
    eccentricity = generate_random_eccentricity()
    true_anomaly = generate_random_true_anomaly()

    print(f"Random masses: m1 = {float(m1):.3e} kg, m2 = {float(m2):.3e} kg")
    print(f"Mass ratio: {max(float(m1), float(m2)) / min(float(m1), float(m2)):.2f}")
    print(f"Random period: {float(period):.3f} s")
    print(f"Eccentricity: {eccentricity:.3f}")
    print(f"True anomaly: {np.degrees(true_anomaly):.1f}°")

    bodies = generate_elliptical_orbit(m1, m2, period, eccentricity, true_anomaly)
    print(f"Generated {len(bodies)} bodies")


if __name__ == '__main__':
    main()
