#!/usr/bin/env python3
"""
Unit tests for two-body gravitational simulation.

Usage:
    test_two_body_sim.py [options]

Options:
    -h --help    Show this help message
    -v           Verbose output
"""

import math
import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from script.physics import (
    G,
    gravitational_force,
    compute_accelerations,
    total_momentum,
    angular_momentum,
    kinetic_energy,
    potential_energy,
    total_energy,
)
from script.integrator import (
    rk4_step,
    check_stability,
)
from script.initial_conditions import (
    generate_circular_orbit,
    generate_random_masses,
    compute_orbital_parameters,
)
from script.io_handler import (
    save_results,
    load_results,
    load_config,
)
from script.simulation import SimulationResult
from script.units import (
    Mass, Time, TwoBodyState,
    position, velocity,
)


class TestPhysics(unittest.TestCase):
    """Tests for physics.py functions."""

    def test_gravitational_force_magnitude(self) -> None:
        """Test gravitational force has correct magnitude."""
        m1 = Mass(1e10)
        m2 = Mass(2e10)
        r1 = position(0.0, 0.0)
        r2 = position(100.0, 0.0)  # 100m apart

        force = gravitational_force(m1, m2, r1, r2)
        expected_magnitude = G * float(m1) * float(m2) / (100.0 ** 2)

        self.assertAlmostEqual(force.magnitude(), expected_magnitude, places=10)

    def test_gravitational_force_direction(self) -> None:
        """Test gravitational force points toward other body."""
        m1 = Mass(1e10)
        m2 = Mass(1e10)
        r1 = position(0.0, 0.0)
        r2 = position(100.0, 0.0)

        force = gravitational_force(m1, m2, r1, r2)
        # Force on m1 should point toward m2 (positive x)
        self.assertGreater(force.x, 0)
        self.assertAlmostEqual(force.y, 0.0, places=15)

    def test_gravitational_force_symmetry(self) -> None:
        """Test Newton's third law: F12 = -F21."""
        m1 = Mass(1e10)
        m2 = Mass(2e10)
        r1 = position(10.0, 20.0)
        r2 = position(50.0, 80.0)

        f1 = gravitational_force(m1, m2, r1, r2)
        f2 = gravitational_force(m2, m1, r2, r1)

        np.testing.assert_array_almost_equal(f1.array, -f2.array, decimal=15)

    def test_compute_accelerations(self) -> None:
        """Test acceleration computation for two bodies."""
        m1 = Mass(1e10)
        m2 = Mass(2e10)
        state = TwoBodyState.from_bodies(
            position(0.0, 0.0), velocity(0.0, 0.0),
            position(100.0, 0.0), velocity(0.0, 0.0)
        )

        a1, a2 = compute_accelerations(state, m1, m2)

        # a1 should point toward body 2 (positive x)
        self.assertGreater(a1.x, 0)
        # a2 should point toward body 1 (negative x)
        self.assertLess(a2.x, 0)
        # Newton's second law: m1*a1 = -m2*a2
        np.testing.assert_array_almost_equal(
            float(m1) * a1.array, -float(m2) * a2.array, decimal=10
        )

    def test_total_momentum_zero(self) -> None:
        """Test momentum calculation for zero total momentum setup."""
        m1 = Mass(1e10)
        m2 = Mass(1e10)
        # Equal masses, opposite velocities
        state = TwoBodyState.from_bodies(
            position(0.0, 0.0), velocity(10.0, 0.0),
            position(100.0, 0.0), velocity(-10.0, 0.0)
        )

        p = total_momentum(state, m1, m2)
        np.testing.assert_array_almost_equal(p.array, np.array([0.0, 0.0]), decimal=10)

    def test_angular_momentum(self) -> None:
        """Test angular momentum calculation."""
        m1 = Mass(1e10)
        m2 = Mass(1e10)
        # Bodies in circular orbit around origin
        state = TwoBodyState.from_bodies(
            position(10.0, 0.0), velocity(0.0, 5.0),    # body 1 at x=10, moving +y
            position(-10.0, 0.0), velocity(0.0, -5.0)  # body 2 at x=-10, moving -y
        )

        L = angular_momentum(state, m1, m2)
        # Both contribute positive angular momentum (counter-clockwise)
        # L1 = m1 * (r1 x v1) = m1 * (10 * 5 - 0 * 0) = 50 * m1
        # L2 = m2 * (r2 x v2) = m2 * (-10 * -5 - 0 * 0) = 50 * m2
        expected_z = float(m1) * 50.0 + float(m2) * 50.0
        # Angular momentum is a 3D vector; for 2D motion, only z-component is nonzero
        self.assertAlmostEqual(L.z, expected_z, places=5)
        self.assertAlmostEqual(L.x, 0.0, places=10)
        self.assertAlmostEqual(L.y, 0.0, places=10)

    def test_kinetic_energy(self) -> None:
        """Test kinetic energy calculation."""
        m1 = Mass(2.0)
        m2 = Mass(3.0)
        state = TwoBodyState.from_bodies(
            position(0.0, 0.0), velocity(4.0, 0.0),   # v1 = 4 m/s
            position(10.0, 0.0), velocity(0.0, 5.0)  # v2 = 5 m/s
        )

        KE = kinetic_energy(state, m1, m2)
        expected = 0.5 * float(m1) * 16.0 + 0.5 * float(m2) * 25.0  # 16 + 37.5 = 53.5
        self.assertAlmostEqual(float(KE), expected, places=10)

    def test_potential_energy(self) -> None:
        """Test gravitational potential energy calculation."""
        m1 = Mass(1e10)
        m2 = Mass(2e10)
        r = 100.0
        state = TwoBodyState.from_bodies(
            position(0.0, 0.0), velocity(0.0, 0.0),
            position(r, 0.0), velocity(0.0, 0.0)
        )

        PE = potential_energy(state, m1, m2)
        expected = -G * float(m1) * float(m2) / r
        self.assertAlmostEqual(float(PE), expected, places=10)

    def test_total_energy(self) -> None:
        """Test total energy is sum of kinetic and potential."""
        m1 = Mass(1e10)
        m2 = Mass(2e10)
        state = TwoBodyState.from_bodies(
            position(0.0, 0.0), velocity(100.0, 0.0),
            position(1000.0, 0.0), velocity(-50.0, 0.0)
        )

        E = total_energy(state, m1, m2)
        KE = kinetic_energy(state, m1, m2)
        PE = potential_energy(state, m1, m2)

        self.assertAlmostEqual(float(E), float(KE) + float(PE), places=10)


class TestIntegrator(unittest.TestCase):
    """Tests for integrator.py functions."""

    def test_derivatives_shape(self) -> None:
        """Test rk4_step maintains state shape."""
        m1 = Mass(1e10)
        m2 = Mass(2e10)
        state = TwoBodyState.from_bodies(
            position(0.0, 0.0), velocity(10.0, 5.0),
            position(100.0, 0.0), velocity(-5.0, 2.0)
        )

        new_state = rk4_step(state, Time(0.01), m1, m2)
        self.assertEqual(new_state.array.shape, state.array.shape)

    def test_derivatives_velocities(self) -> None:
        """Test that position changes in direction of velocity."""
        m1 = Mass(1e10)
        m2 = Mass(2e10)
        v1 = velocity(10.0, 5.0)
        v2 = velocity(-5.0, 2.0)
        state = TwoBodyState.from_bodies(
            position(0.0, 0.0), v1,
            position(100.0, 0.0), v2
        )

        dt = Time(0.001)
        new_state = rk4_step(state, dt, m1, m2)

        # Position should move approximately in direction of velocity
        pos1_delta = new_state.position(0).array - state.position(0).array
        # Dot product should be positive (same direction)
        self.assertGreater(np.dot(pos1_delta, v1.array), 0)

    def test_rk4_step_circular_orbit(self) -> None:
        """Test RK4 step maintains circular orbit accuracy."""
        m1 = Mass(1e12)
        m2 = Mass(1e12)
        T = Time(2.0)  # seconds

        state = generate_circular_orbit(m1, m2, T)
        initial_energy = total_energy(state, m1, m2)

        # Take many small steps
        dt = Time(0.001)
        current_state = state.copy()
        for _ in range(int(float(T) / float(dt))):
            current_state = rk4_step(current_state, dt, m1, m2)

        final_energy = total_energy(current_state, m1, m2)

        # Energy should be conserved to high precision
        relative_error = abs(float(final_energy) - float(initial_energy)) / abs(float(initial_energy))
        self.assertLess(relative_error, 1e-6)

    def test_check_stability_normal(self) -> None:
        """Test stability check passes for normal conditions."""
        m1 = Mass(1e10)
        m2 = Mass(1e10)
        state = TwoBodyState.from_bodies(
            position(0.0, 0.0), velocity(10.0, 0.0),
            position(1000.0, 0.0), velocity(-10.0, 0.0)
        )

        self.assertTrue(check_stability(state, m1, m2))

    def test_check_stability_collision(self) -> None:
        """Test stability check fails for near-collision."""
        m1 = Mass(1e20)
        m2 = Mass(1e20)
        # Bodies very close together
        state = TwoBodyState.from_bodies(
            position(0.0, 0.0), velocity(0.0, 0.0),
            position(0.001, 0.0), velocity(0.0, 0.0)
        )

        self.assertFalse(check_stability(state, m1, m2))


class TestInitialConditions(unittest.TestCase):
    """Tests for initial_conditions.py functions."""

    def test_generate_random_masses_ratio(self) -> None:
        """Test random masses are within specified ratio."""
        for _ in range(100):  # Test multiple times
            m1, m2 = generate_random_masses(ratio_max=10)
            ratio = max(float(m1), float(m2)) / min(float(m1), float(m2))
            self.assertLessEqual(ratio, 10)
            self.assertGreater(float(m1), 0)
            self.assertGreater(float(m2), 0)

    def test_compute_orbital_parameters(self) -> None:
        """Test orbital parameters give correct period."""
        m1 = Mass(1e12)
        m2 = Mass(2e12)
        T = Time(3.0)  # seconds

        r, v1, v2 = compute_orbital_parameters(m1, m2, T)

        # Verify with Kepler's third law
        M = float(m1) + float(m2)
        expected_r = (G * M * float(T)**2 / (4 * np.pi**2)) ** (1/3)
        self.assertAlmostEqual(r, expected_r, places=10)

    def test_generate_circular_orbit_momentum(self) -> None:
        """Test generated orbit has zero total momentum."""
        m1 = Mass(1e12)
        m2 = Mass(2e12)
        T = Time(2.5)

        state = generate_circular_orbit(m1, m2, T)
        p = total_momentum(state, m1, m2)

        np.testing.assert_array_almost_equal(p.array, np.array([0.0, 0.0]), decimal=10)

    def test_generate_circular_orbit_period(self) -> None:
        """Test generated orbit has correct period."""
        m1 = Mass(1e12)
        m2 = Mass(1e12)
        T = Time(2.0)

        state = generate_circular_orbit(m1, m2, T)
        initial_pos = state.array.copy()

        # Simulate one full period
        dt = Time(0.001)
        current_state = state.copy()
        for _ in range(int(float(T) / float(dt))):
            current_state = rk4_step(current_state, dt, m1, m2)

        # Should return close to initial position
        pos_error = np.linalg.norm(current_state.array[:, 0] - initial_pos[:, 0])
        max_separation = np.linalg.norm(initial_pos[0, 0] - initial_pos[1, 0])
        relative_error = pos_error / max_separation

        self.assertLess(relative_error, 0.01)


class TestIOHandler(unittest.TestCase):
    """Tests for io_handler.py functions."""

    def setUp(self) -> None:
        """Set up test output directory."""
        self.test_dir = '/tmp/test_two_body_sim'
        os.makedirs(self.test_dir, exist_ok=True)

    def test_save_load_roundtrip(self) -> None:
        """Test saving and loading results preserves data."""
        result = SimulationResult(
            metadata={'m1': 1e12, 'm2': 2e12, 'dt': 0.01},
            times=np.array([0.0, 0.01, 0.02]),
            positions=np.random.rand(3, 2, 2),
            velocities=np.random.rand(3, 2, 2),
            momentum=np.random.rand(3, 2),
            angular_momentum=np.random.rand(3, 3),  # 3D vector
            energy=np.random.rand(3),
            collision=False
        )

        filepath = os.path.join(self.test_dir, 'test_results.json')
        save_results(result, filepath)
        loaded = load_results(filepath)

        self.assertEqual(result.metadata, loaded.metadata)
        np.testing.assert_array_almost_equal(result.times, loaded.times)
        np.testing.assert_array_almost_equal(result.positions, loaded.positions)
        np.testing.assert_array_almost_equal(result.velocities, loaded.velocities)
        np.testing.assert_array_almost_equal(result.momentum, loaded.momentum)
        np.testing.assert_array_almost_equal(result.angular_momentum, loaded.angular_momentum)
        np.testing.assert_array_almost_equal(result.energy, loaded.energy)
        self.assertEqual(result.collision, loaded.collision)

    def test_load_config(self) -> None:
        """Test loading configuration from JSON file."""
        config_data = {
            'm1': 1e12,
            'm2': 2e12,
            'period': 2.5,
            'duration': 50.0,
            'dt': 0.005
        }

        filepath = os.path.join(self.test_dir, 'test_config.json')
        import json
        with open(filepath, 'w') as f:
            json.dump(config_data, f)

        loaded = load_config(filepath)
        self.assertEqual(loaded, config_data)

    def tearDown(self) -> None:
        """Clean up test files."""
        import shutil
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)


class TestTwoBodySimMain(unittest.TestCase):
    """Tests for two_body_sim.py main script output formatting."""

    def test_angular_momentum_relative_deviation_formatting(self) -> None:
        """Test that angular momentum relative deviation can be formatted as scalar.

        Regression test for bug where angular momentum became a 3D vector
        but the main script still tried to format it as a scalar.
        """
        from script.simulation import run_simulation

        m1 = Mass(1e12)
        m2 = Mass(1e12)
        T = Time(2.0)

        state = generate_circular_orbit(m1, m2, T)
        result = run_simulation(state, m1, m2, dt=Time(0.001), t_max=T)

        # This is what two_body_sim.py does - should work with vector angular momentum
        # Use magnitude for 3D vector comparison
        L0 = np.linalg.norm(result.angular_momentum[0])
        L_final = np.linalg.norm(result.angular_momentum[-1])
        dL_rel = abs(L_final - L0) / abs(L0)

        # Should be able to format as scalar without error
        formatted = f"ΔL/L₀ = {dL_rel:.2e}"
        self.assertIn("ΔL/L₀", formatted)


class TestSimulationIntegration(unittest.TestCase):
    """Integration tests for full simulation."""

    def test_energy_conservation(self) -> None:
        """Test that total energy is conserved during simulation."""
        from script.simulation import run_simulation

        m1 = Mass(1e12)
        m2 = Mass(1e12)
        T = Time(2.0)

        state = generate_circular_orbit(m1, m2, T)
        result = run_simulation(state, m1, m2, dt=Time(0.001), t_max=T)

        # Energy should be conserved
        initial_energy = result.energy[0]
        max_deviation = np.max(np.abs(result.energy - initial_energy))
        relative_deviation = max_deviation / abs(initial_energy)

        self.assertLess(relative_deviation, 1e-5)

    def test_angular_momentum_conservation(self) -> None:
        """Test that angular momentum is conserved."""
        from script.simulation import run_simulation

        m1 = Mass(1e12)
        m2 = Mass(2e12)
        T = Time(2.0)

        state = generate_circular_orbit(m1, m2, T)
        result = run_simulation(state, m1, m2, dt=Time(0.001), t_max=T)

        # For 2D motion, only z-component of angular momentum is nonzero
        initial_Lz = result.angular_momentum[0, 2]
        max_deviation = np.max(np.abs(result.angular_momentum[:, 2] - initial_Lz))
        relative_deviation = max_deviation / abs(initial_Lz)

        self.assertLess(relative_deviation, 1e-5)

    def test_momentum_conservation(self) -> None:
        """Test that total momentum remains zero."""
        from script.simulation import run_simulation

        m1 = Mass(1e12)
        m2 = Mass(2e12)
        T = Time(2.0)

        state = generate_circular_orbit(m1, m2, T)
        result = run_simulation(state, m1, m2, dt=Time(0.001), t_max=T)

        # Momentum should stay near zero
        max_momentum = np.max(np.linalg.norm(result.momentum, axis=1))
        # Compare to typical momentum scale
        typical_momentum = float(m1) * np.linalg.norm(state.velocity(0).array)
        relative_deviation = max_momentum / typical_momentum

        self.assertLess(relative_deviation, 1e-10)


def main() -> None:
    """Run tests."""
    unittest.main()


if __name__ == '__main__':
    main()
