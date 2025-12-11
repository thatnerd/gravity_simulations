#!/usr/bin/env python3
"""
Unit tests for the physics unit types module.

Usage:
    python3 -m pytest test/test_units.py -v
"""

import numpy as np
import pytest

from script.units import (
    Scalar, Vector2D, Vector3D,
    Mass, Time, Energy, AngularMomentum,
    Position, Velocity, Acceleration, Force, Momentum,
    TwoBodyState,
    mass, time, energy, angular_momentum,
    position, velocity, acceleration, force, momentum,
)


class TestScalarBase:
    """Tests for base Scalar class."""

    def test_float_conversion(self) -> None:
        s = Scalar(42.5)
        assert float(s) == 42.5

    def test_repr(self) -> None:
        s = Scalar(1.0)
        assert "Scalar" in repr(s)
        assert "1.0" in repr(s)


class TestMass:
    """Tests for Mass scalar type."""

    def test_creation(self) -> None:
        m = Mass(1e12)
        assert float(m) == 1e12

    def test_factory_function(self) -> None:
        m = mass(5e10)
        assert isinstance(m, Mass)
        assert float(m) == 5e10

    def test_mass_times_velocity_gives_momentum(self) -> None:
        m = Mass(2.0)
        v = Velocity(np.array([3.0, 4.0]))
        p = m * v
        assert isinstance(p, Momentum)
        np.testing.assert_array_equal(p.array, np.array([6.0, 8.0]))

    def test_mass_times_scalar(self) -> None:
        m = Mass(1e12)
        result = m * 2.0
        assert isinstance(result, Mass)
        assert float(result) == 2e12

    def test_scalar_times_mass(self) -> None:
        m = Mass(1e12)
        result = 3.0 * m
        assert isinstance(result, Mass)
        assert float(result) == 3e12


class TestTime:
    """Tests for Time scalar type."""

    def test_creation(self) -> None:
        t = Time(10.0)
        assert float(t) == 10.0

    def test_factory_function(self) -> None:
        t = time(5.0)
        assert isinstance(t, Time)
        assert float(t) == 5.0


class TestEnergy:
    """Tests for Energy scalar type."""

    def test_addition(self) -> None:
        e1 = Energy(100.0)
        e2 = Energy(50.0)
        result = e1 + e2
        assert isinstance(result, Energy)
        assert float(result) == 150.0

    def test_subtraction(self) -> None:
        e1 = Energy(100.0)
        e2 = Energy(30.0)
        result = e1 - e2
        assert isinstance(result, Energy)
        assert float(result) == 70.0

    def test_negation(self) -> None:
        e = Energy(100.0)
        result = -e
        assert isinstance(result, Energy)
        assert float(result) == -100.0


class TestAngularMomentum:
    """Tests for AngularMomentum vector type."""

    def test_addition(self) -> None:
        L1 = AngularMomentum(np.array([0.0, 0.0, 100.0]))
        L2 = AngularMomentum(np.array([0.0, 0.0, 50.0]))
        result = L1 + L2
        assert isinstance(result, AngularMomentum)
        assert result.z == 150.0

    def test_properties(self) -> None:
        L = AngularMomentum(np.array([1.0, 2.0, 3.0]))
        assert L.x == 1.0
        assert L.y == 2.0
        assert L.z == 3.0

    def test_negation(self) -> None:
        L = AngularMomentum(np.array([1.0, 2.0, 3.0]))
        neg = -L
        assert neg.x == -1.0
        assert neg.y == -2.0
        assert neg.z == -3.0


class TestVector2DBase:
    """Tests for base Vector2D class."""

    def test_properties(self) -> None:
        v = Vector2D(np.array([3.0, 4.0]))
        assert v.x == 3.0
        assert v.y == 4.0

    def test_array_property(self) -> None:
        arr = np.array([1.0, 2.0])
        v = Vector2D(arr)
        np.testing.assert_array_equal(v.array, arr)

    def test_magnitude(self) -> None:
        v = Vector2D(np.array([3.0, 4.0]))
        assert v.magnitude() == 5.0

    def test_repr(self) -> None:
        v = Vector2D(np.array([1.0, 2.0]))
        assert "Vector2D" in repr(v)

    def test_to_3d(self) -> None:
        """Test conversion from 2D to 3D vector with z=0."""
        v = Vector2D(np.array([3.0, 4.0]))
        v3d = v.to_3d()
        assert isinstance(v3d, Vector3D)
        assert v3d.x == 3.0
        assert v3d.y == 4.0
        assert v3d.z == 0.0


class TestVector3DBase:
    """Tests for base Vector3D class."""

    def test_properties(self) -> None:
        v = Vector3D(np.array([1.0, 2.0, 3.0]))
        assert v.x == 1.0
        assert v.y == 2.0
        assert v.z == 3.0

    def test_array_property(self) -> None:
        arr = np.array([1.0, 2.0, 3.0])
        v = Vector3D(arr)
        np.testing.assert_array_equal(v.array, arr)

    def test_magnitude(self) -> None:
        v = Vector3D(np.array([0.0, 3.0, 4.0]))
        assert v.magnitude() == 5.0

    def test_repr(self) -> None:
        v = Vector3D(np.array([1.0, 2.0, 3.0]))
        assert "Vector3D" in repr(v)


class TestPosition:
    """Tests for Position vector type."""

    def test_factory_function(self) -> None:
        p = position(1.0, 2.0)
        assert isinstance(p, Position)
        assert p.x == 1.0
        assert p.y == 2.0

    def test_subtraction(self) -> None:
        p1 = position(4.0, 6.0)
        p2 = position(1.0, 2.0)
        displacement = p1 - p2
        assert isinstance(displacement, Position)
        np.testing.assert_array_equal(displacement.array, np.array([3.0, 4.0]))

    def test_addition(self) -> None:
        p1 = position(1.0, 2.0)
        p2 = position(3.0, 4.0)
        result = p1 + p2
        assert isinstance(result, Position)
        np.testing.assert_array_equal(result.array, np.array([4.0, 6.0]))

    def test_negation(self) -> None:
        p = position(1.0, -2.0)
        result = -p
        assert isinstance(result, Position)
        np.testing.assert_array_equal(result.array, np.array([-1.0, 2.0]))

    def test_scalar_multiplication(self) -> None:
        p = position(2.0, 3.0)
        result = p * 2.0
        assert isinstance(result, Position)
        np.testing.assert_array_equal(result.array, np.array([4.0, 6.0]))

    def test_scalar_rmul(self) -> None:
        p = position(2.0, 3.0)
        result = 3.0 * p
        assert isinstance(result, Position)
        np.testing.assert_array_equal(result.array, np.array([6.0, 9.0]))

    def test_division(self) -> None:
        p = position(4.0, 6.0)
        result = p / 2.0
        assert isinstance(result, Position)
        np.testing.assert_array_equal(result.array, np.array([2.0, 3.0]))

    def test_cross_with_momentum(self) -> None:
        """Test Position × Momentum = Angular Momentum (L = r × p)."""
        r = position(3.0, 0.0)  # Position along x-axis
        p = momentum(0.0, 4.0)  # Momentum along y-axis
        L = r.cross(p)
        assert isinstance(L, AngularMomentum)
        # Cross product: (3,0,0) × (0,4,0) = (0,0,12)
        assert L.x == 0.0
        assert L.y == 0.0
        assert L.z == 12.0

    def test_cross_negative_angular_momentum(self) -> None:
        """Test cross product for clockwise motion (negative angular momentum)."""
        r = position(3.0, 0.0)  # Position along x-axis
        p = momentum(0.0, -4.0)  # Momentum along negative y-axis
        L = r.cross(p)
        # Cross product: (3,0,0) × (0,-4,0) = (0,0,-12)
        assert L.z == -12.0

    def test_dot_with_force(self) -> None:
        """Test Position · Force = Work (Energy)."""
        d = position(3.0, 4.0)  # Displacement
        f = force(2.0, 1.0)  # Force
        W = d.dot(f)
        assert isinstance(W, Energy)
        # Dot product: 3*2 + 4*1 = 10
        assert float(W) == 10.0

    def test_dot_perpendicular_no_work(self) -> None:
        """Test that perpendicular force does no work."""
        d = position(1.0, 0.0)  # Displacement along x
        f = force(0.0, 5.0)  # Force along y (perpendicular)
        W = d.dot(f)
        assert float(W) == 0.0


class TestVelocity:
    """Tests for Velocity vector type."""

    def test_factory_function(self) -> None:
        v = velocity(1.0, 2.0)
        assert isinstance(v, Velocity)
        assert v.x == 1.0
        assert v.y == 2.0

    def test_addition(self) -> None:
        v1 = velocity(1.0, 2.0)
        v2 = velocity(3.0, 4.0)
        result = v1 + v2
        assert isinstance(result, Velocity)
        np.testing.assert_array_equal(result.array, np.array([4.0, 6.0]))

    def test_subtraction(self) -> None:
        v1 = velocity(5.0, 6.0)
        v2 = velocity(1.0, 2.0)
        result = v1 - v2
        assert isinstance(result, Velocity)
        np.testing.assert_array_equal(result.array, np.array([4.0, 4.0]))

    def test_velocity_times_time_gives_position(self) -> None:
        v = velocity(3.0, 4.0)
        t = Time(2.0)
        displacement = v * t
        assert isinstance(displacement, Position)
        np.testing.assert_array_equal(displacement.array, np.array([6.0, 8.0]))

    def test_scalar_multiplication(self) -> None:
        v = velocity(2.0, 3.0)
        result = v * 2.0
        assert isinstance(result, Velocity)
        np.testing.assert_array_equal(result.array, np.array([4.0, 6.0]))


class TestAcceleration:
    """Tests for Acceleration vector type."""

    def test_factory_function(self) -> None:
        a = acceleration(1.0, 2.0)
        assert isinstance(a, Acceleration)
        assert a.x == 1.0
        assert a.y == 2.0

    def test_addition(self) -> None:
        a1 = acceleration(1.0, 2.0)
        a2 = acceleration(3.0, 4.0)
        result = a1 + a2
        assert isinstance(result, Acceleration)
        np.testing.assert_array_equal(result.array, np.array([4.0, 6.0]))

    def test_acceleration_times_time_gives_velocity(self) -> None:
        a = acceleration(3.0, 4.0)
        t = Time(2.0)
        delta_v = a * t
        assert isinstance(delta_v, Velocity)
        np.testing.assert_array_equal(delta_v.array, np.array([6.0, 8.0]))

    def test_negation(self) -> None:
        a = acceleration(1.0, -2.0)
        result = -a
        assert isinstance(result, Acceleration)
        np.testing.assert_array_equal(result.array, np.array([-1.0, 2.0]))


class TestForce:
    """Tests for Force vector type."""

    def test_factory_function(self) -> None:
        f = force(10.0, 20.0)
        assert isinstance(f, Force)
        assert f.x == 10.0
        assert f.y == 20.0

    def test_force_divided_by_mass_gives_acceleration(self) -> None:
        f = Force(np.array([10.0, 0.0]))
        m = Mass(2.0)
        a = f / m
        assert isinstance(a, Acceleration)
        np.testing.assert_array_equal(a.array, np.array([5.0, 0.0]))

    def test_addition(self) -> None:
        f1 = force(1.0, 2.0)
        f2 = force(3.0, 4.0)
        result = f1 + f2
        assert isinstance(result, Force)
        np.testing.assert_array_equal(result.array, np.array([4.0, 6.0]))

    def test_negation(self) -> None:
        f = force(5.0, -3.0)
        result = -f
        assert isinstance(result, Force)
        np.testing.assert_array_equal(result.array, np.array([-5.0, 3.0]))


class TestMomentum:
    """Tests for Momentum vector type."""

    def test_factory_function(self) -> None:
        p = momentum(10.0, 20.0)
        assert isinstance(p, Momentum)
        assert p.x == 10.0
        assert p.y == 20.0

    def test_addition(self) -> None:
        p1 = momentum(1.0, 2.0)
        p2 = momentum(3.0, 4.0)
        result = p1 + p2
        assert isinstance(result, Momentum)
        np.testing.assert_array_equal(result.array, np.array([4.0, 6.0]))

    def test_negation(self) -> None:
        p = momentum(5.0, -3.0)
        result = -p
        assert isinstance(result, Momentum)
        np.testing.assert_array_equal(result.array, np.array([-5.0, 3.0]))


class TestTwoBodyState:
    """Tests for TwoBodyState container."""

    def test_from_bodies(self) -> None:
        p1 = position(1.0, 0.0)
        v1 = velocity(0.0, 1.0)
        p2 = position(-1.0, 0.0)
        v2 = velocity(0.0, -1.0)

        state = TwoBodyState.from_bodies(p1, v1, p2, v2)

        np.testing.assert_array_equal(state.position(0).array, p1.array)
        np.testing.assert_array_equal(state.velocity(0).array, v1.array)
        np.testing.assert_array_equal(state.position(1).array, p2.array)
        np.testing.assert_array_equal(state.velocity(1).array, v2.array)

    def test_from_array(self) -> None:
        data = np.array([
            [[1.0, 0.0], [0.0, 1.0]],
            [[-1.0, 0.0], [0.0, -1.0]]
        ])
        state = TwoBodyState.from_array(data)

        assert state.position(0).x == 1.0
        assert state.velocity(1).y == -1.0

    def test_array_property(self) -> None:
        data = np.zeros((2, 2, 2))
        state = TwoBodyState.from_array(data)
        np.testing.assert_array_equal(state.array, data)

    def test_copy(self) -> None:
        p1 = position(1.0, 0.0)
        v1 = velocity(0.0, 1.0)
        p2 = position(-1.0, 0.0)
        v2 = velocity(0.0, -1.0)

        state = TwoBodyState.from_bodies(p1, v1, p2, v2)
        state_copy = state.copy()

        # Modify original
        state._data[0, 0, 0] = 999.0

        # Copy should be unchanged
        assert state_copy.position(0).x == 1.0

    def test_position_returns_copy(self) -> None:
        """Ensure position() returns a copy, not a view."""
        data = np.array([
            [[1.0, 0.0], [0.0, 1.0]],
            [[-1.0, 0.0], [0.0, -1.0]]
        ])
        state = TwoBodyState.from_array(data)

        p = state.position(0)
        # Position is frozen, but verify array independence
        assert p.x == 1.0


class TestPhysicalEquations:
    """Integration tests for physical equation correctness."""

    def test_newtons_second_law(self) -> None:
        """F = ma, so F/m = a."""
        f = force(10.0, 20.0)
        m = mass(2.0)
        a = f / m
        assert isinstance(a, Acceleration)
        assert a.x == 5.0
        assert a.y == 10.0

    def test_momentum_equation(self) -> None:
        """p = mv."""
        m = mass(3.0)
        v = velocity(4.0, 5.0)
        p = m * v
        assert isinstance(p, Momentum)
        assert p.x == 12.0
        assert p.y == 15.0

    def test_kinematic_equation(self) -> None:
        """displacement = velocity * time."""
        v = velocity(10.0, 0.0)
        t = time(5.0)
        displacement = v * t
        assert isinstance(displacement, Position)
        assert displacement.x == 50.0
        assert displacement.y == 0.0

    def test_acceleration_velocity_relation(self) -> None:
        """delta_v = acceleration * time."""
        a = acceleration(2.0, 3.0)
        t = time(4.0)
        delta_v = a * t
        assert isinstance(delta_v, Velocity)
        assert delta_v.x == 8.0
        assert delta_v.y == 12.0


def main() -> None:
    """Run tests."""
    pytest.main([__file__, '-v'])


if __name__ == '__main__':
    main()
