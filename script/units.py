#!/usr/bin/env python3
"""
Physical unit types for two-body gravitational simulation.

This module provides type-safe wrappers for physical quantities, ensuring
that operations respect dimensional analysis (e.g., Force / Mass = Acceleration).

Usage:
    from script.units import Mass, Position, Velocity, Force, Energy
    from script.units import mass, position, velocity  # factory functions
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Union
import numpy as np
from numpy.typing import NDArray


# ============================================================
# Base Types
# ============================================================

@dataclass(frozen=True, slots=True)
class Scalar:
    """Base class for scalar physical quantities."""
    value: float

    def __float__(self) -> float:
        # Ensure we return a pure Python float, not numpy.float64
        return float(self.value)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.value})"


@dataclass(frozen=True, slots=True)
class Vector2D:
    """Base class for 2D vector physical quantities."""
    _data: NDArray[np.float64]

    @property
    def x(self) -> float:
        return float(self._data[0])

    @property
    def y(self) -> float:
        return float(self._data[1])

    @property
    def array(self) -> NDArray[np.float64]:
        """Return underlying numpy array for computation."""
        return self._data

    def magnitude(self) -> float:
        """Return the magnitude (length) of this vector."""
        return float(np.linalg.norm(self._data))

    def to_3d(self) -> Vector3D:
        """Convert to 3D vector with z=0."""
        return Vector3D(np.array([self.x, self.y, 0.0], dtype=np.float64))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}([{self.x}, {self.y}])"


@dataclass(frozen=True, slots=True)
class Vector3D:
    """Base class for 3D vector physical quantities."""
    _data: NDArray[np.float64]

    @property
    def x(self) -> float:
        return float(self._data[0])

    @property
    def y(self) -> float:
        return float(self._data[1])

    @property
    def z(self) -> float:
        return float(self._data[2])

    @property
    def array(self) -> NDArray[np.float64]:
        """Return underlying numpy array for computation."""
        return self._data

    def magnitude(self) -> float:
        """Return the magnitude (length) of this vector."""
        return float(np.linalg.norm(self._data))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}([{self.x}, {self.y}, {self.z}])"


# ============================================================
# Scalar Types
# ============================================================

@dataclass(frozen=True, slots=True)
class Mass(Scalar):
    """Mass in kilograms (kg)."""

    def __mul__(self, other: Union[Velocity, float, int]) -> Union[Momentum, Mass]:
        """Mass * Velocity = Momentum, Mass * scalar = Mass"""
        if isinstance(other, Velocity):
            return Momentum(self.value * other._data)
        if isinstance(other, (int, float)):
            return Mass(self.value * other)
        return NotImplemented

    def __rmul__(self, other: Union[float, int]) -> Mass:
        """scalar * Mass = Mass"""
        if isinstance(other, (int, float)):
            return Mass(other * self.value)
        return NotImplemented


@dataclass(frozen=True, slots=True)
class Time(Scalar):
    """Time in seconds (s)."""
    pass


@dataclass(frozen=True, slots=True)
class Energy(Scalar):
    """Energy in joules (J)."""

    def __add__(self, other: Energy) -> Energy:
        if isinstance(other, Energy):
            return Energy(self.value + other.value)
        return NotImplemented

    def __radd__(self, other: Energy) -> Energy:
        if isinstance(other, Energy):
            return Energy(self.value + other.value)
        return NotImplemented

    def __sub__(self, other: Energy) -> Energy:
        if isinstance(other, Energy):
            return Energy(self.value - other.value)
        return NotImplemented

    def __neg__(self) -> Energy:
        return Energy(-self.value)


@dataclass(frozen=True, slots=True)
class AngularMomentum(Vector3D):
    """Angular momentum vector in kg·m²/s."""

    def __add__(self, other: AngularMomentum) -> AngularMomentum:
        if isinstance(other, AngularMomentum):
            return AngularMomentum(self._data + other._data)
        return NotImplemented

    def __radd__(self, other: AngularMomentum) -> AngularMomentum:
        if isinstance(other, AngularMomentum):
            return AngularMomentum(self._data + other._data)
        return NotImplemented

    def __neg__(self) -> AngularMomentum:
        return AngularMomentum(-self._data)


# ============================================================
# Vector Types
# ============================================================

@dataclass(frozen=True, slots=True)
class Position(Vector2D):
    """Position vector in meters (m)."""

    def __sub__(self, other: Position) -> Position:
        """Position - Position = Position (displacement)"""
        if isinstance(other, Position):
            return Position(self._data - other._data)
        return NotImplemented

    def __add__(self, other: Position) -> Position:
        if isinstance(other, Position):
            return Position(self._data + other._data)
        return NotImplemented

    def __neg__(self) -> Position:
        return Position(-self._data)

    def __mul__(self, scalar: Union[float, int]) -> Position:
        if isinstance(scalar, (int, float)):
            return Position(scalar * self._data)
        return NotImplemented

    def __rmul__(self, scalar: Union[float, int]) -> Position:
        if isinstance(scalar, (int, float)):
            return Position(scalar * self._data)
        return NotImplemented

    def __truediv__(self, scalar: Union[float, int]) -> Position:
        if isinstance(scalar, (int, float)):
            return Position(self._data / scalar)
        return NotImplemented

    def dot(self, other: Force) -> Energy:
        """Position · Force = Work (Energy)."""
        if isinstance(other, Force):
            return Energy(float(np.dot(self._data, other._data)))
        return NotImplemented

    def cross(self, other: Momentum) -> AngularMomentum:
        """Position × Momentum = Angular Momentum."""
        if isinstance(other, Momentum):
            # 2D cross product gives z-component only
            result = np.cross(
                np.array([self.x, self.y, 0.0]),
                np.array([other.x, other.y, 0.0])
            )
            return AngularMomentum(result)
        return NotImplemented


@dataclass(frozen=True, slots=True)
class Velocity(Vector2D):
    """Velocity vector in meters per second (m/s)."""

    def __add__(self, other: Velocity) -> Velocity:
        if isinstance(other, Velocity):
            return Velocity(self._data + other._data)
        return NotImplemented

    def __radd__(self, other: Velocity) -> Velocity:
        if isinstance(other, Velocity):
            return Velocity(self._data + other._data)
        return NotImplemented

    def __sub__(self, other: Velocity) -> Velocity:
        if isinstance(other, Velocity):
            return Velocity(self._data - other._data)
        return NotImplemented

    def __neg__(self) -> Velocity:
        return Velocity(-self._data)

    def __mul__(self, scalar: Union[float, int, Time]) -> Union[Velocity, Position]:
        """Velocity * scalar = Velocity, Velocity * Time = Position"""
        if isinstance(scalar, Time):
            return Position(self._data * scalar.value)
        if isinstance(scalar, (int, float)):
            return Velocity(scalar * self._data)
        return NotImplemented

    def __rmul__(self, scalar: Union[float, int]) -> Velocity:
        if isinstance(scalar, (int, float)):
            return Velocity(scalar * self._data)
        return NotImplemented


@dataclass(frozen=True, slots=True)
class Acceleration(Vector2D):
    """Acceleration vector in meters per second squared (m/s²)."""

    def __add__(self, other: Acceleration) -> Acceleration:
        if isinstance(other, Acceleration):
            return Acceleration(self._data + other._data)
        return NotImplemented

    def __radd__(self, other: Acceleration) -> Acceleration:
        if isinstance(other, Acceleration):
            return Acceleration(self._data + other._data)
        return NotImplemented

    def __neg__(self) -> Acceleration:
        return Acceleration(-self._data)

    def __mul__(self, scalar: Union[float, int, Time]) -> Union[Acceleration, Velocity]:
        """Acceleration * scalar = Acceleration, Acceleration * Time = Velocity"""
        if isinstance(scalar, Time):
            return Velocity(self._data * scalar.value)
        if isinstance(scalar, (int, float)):
            return Acceleration(scalar * self._data)
        return NotImplemented

    def __rmul__(self, scalar: Union[float, int]) -> Acceleration:
        if isinstance(scalar, (int, float)):
            return Acceleration(scalar * self._data)
        return NotImplemented


@dataclass(frozen=True, slots=True)
class Force(Vector2D):
    """Force vector in newtons (N)."""

    def __add__(self, other: Force) -> Force:
        if isinstance(other, Force):
            return Force(self._data + other._data)
        return NotImplemented

    def __radd__(self, other: Force) -> Force:
        if isinstance(other, Force):
            return Force(self._data + other._data)
        return NotImplemented

    def __neg__(self) -> Force:
        return Force(-self._data)

    def __truediv__(self, other: Mass) -> Acceleration:
        """Force / Mass = Acceleration"""
        if isinstance(other, Mass):
            return Acceleration(self._data / other.value)
        return NotImplemented


@dataclass(frozen=True, slots=True)
class Momentum(Vector2D):
    """Momentum vector in kg·m/s."""

    def __add__(self, other: Momentum) -> Momentum:
        if isinstance(other, Momentum):
            return Momentum(self._data + other._data)
        return NotImplemented

    def __radd__(self, other: Momentum) -> Momentum:
        if isinstance(other, Momentum):
            return Momentum(self._data + other._data)
        return NotImplemented

    def __neg__(self) -> Momentum:
        return Momentum(-self._data)


# ============================================================
# Body Container
# ============================================================

@dataclass(frozen=True, slots=True)
class Body:
    """
    Immutable body with mass, position, and velocity.

    This is the fundamental unit for gravitational simulations.
    """
    mass: Mass
    position: Position
    velocity: Velocity

    def with_position(self, new_position: Position) -> Body:
        """Return a new Body with updated position."""
        return Body(self.mass, new_position, self.velocity)

    def with_velocity(self, new_velocity: Velocity) -> Body:
        """Return a new Body with updated velocity."""
        return Body(self.mass, self.position, new_velocity)

    def with_state(self, new_position: Position, new_velocity: Velocity) -> Body:
        """Return a new Body with updated position and velocity."""
        return Body(self.mass, new_position, new_velocity)


# ============================================================
# Factory Functions
# ============================================================

def mass(value: float) -> Mass:
    """Create a Mass from a float value in kg."""
    return Mass(value)


def time(value: float) -> Time:
    """Create a Time from a float value in seconds."""
    return Time(value)


def energy(value: float) -> Energy:
    """Create an Energy from a float value in joules."""
    return Energy(value)


def angular_momentum(x: float, y: float, z: float) -> AngularMomentum:
    """Create an AngularMomentum vector from x, y, z components in kg·m²/s."""
    return AngularMomentum(np.array([x, y, z], dtype=np.float64))


def angular_momentum_z(z: float) -> AngularMomentum:
    """Create an AngularMomentum with only z-component (for 2D simulations)."""
    return AngularMomentum(np.array([0.0, 0.0, z], dtype=np.float64))


def position(x: float, y: float) -> Position:
    """Create a Position from x, y coordinates in meters."""
    return Position(np.array([x, y], dtype=np.float64))


def velocity(vx: float, vy: float) -> Velocity:
    """Create a Velocity from vx, vy components in m/s."""
    return Velocity(np.array([vx, vy], dtype=np.float64))


def acceleration(ax: float, ay: float) -> Acceleration:
    """Create an Acceleration from ax, ay components in m/s²."""
    return Acceleration(np.array([ax, ay], dtype=np.float64))


def force(fx: float, fy: float) -> Force:
    """Create a Force from fx, fy components in N."""
    return Force(np.array([fx, fy], dtype=np.float64))


def momentum(px: float, py: float) -> Momentum:
    """Create a Momentum from px, py components in kg·m/s."""
    return Momentum(np.array([px, py], dtype=np.float64))


def body(m: float, x: float, y: float, vx: float, vy: float) -> Body:
    """Create a Body from mass and position/velocity components."""
    return Body(Mass(m), position(x, y), velocity(vx, vy))


def main() -> None:
    """Module test."""
    print("Units module loaded successfully.")
    print(f"Mass(1e12) = {Mass(1e12)}")
    print(f"position(1.0, 2.0) = {position(1.0, 2.0)}")


if __name__ == '__main__':
    main()
