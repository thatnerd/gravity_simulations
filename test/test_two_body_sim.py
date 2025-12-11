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
    derivatives,
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


class TestPhysics(unittest.TestCase):
    """Tests for physics.py functions."""

    def test_gravitational_force_magnitude(self) -> None:
        """Test gravitational force has correct magnitude."""
        m1 = 1e10  # kg
        m2 = 2e10  # kg
        r1 = np.array([0.0, 0.0])
        r2 = np.array([100.0, 0.0])  # 100m apart

        force = gravitational_force(m1, m2, r1, r2)
        expected_magnitude = G * m1 * m2 / (100.0 ** 2)

        self.assertAlmostEqual(np.linalg.norm(force), expected_magnitude, places=10)

    def test_gravitational_force_direction(self) -> None:
        """Test gravitational force points toward other body."""
        m1 = 1e10
        m2 = 1e10
        r1 = np.array([0.0, 0.0])
        r2 = np.array([100.0, 0.0])

        force = gravitational_force(m1, m2, r1, r2)
        # Force on m1 should point toward m2 (positive x)
        self.assertGreater(force[0], 0)
        self.assertAlmostEqual(force[1], 0.0, places=15)

    def test_gravitational_force_symmetry(self) -> None:
        """Test Newton's third law: F12 = -F21."""
        m1 = 1e10
        m2 = 2e10
        r1 = np.array([10.0, 20.0])
        r2 = np.array([50.0, 80.0])

        f1 = gravitational_force(m1, m2, r1, r2)
        f2 = gravitational_force(m2, m1, r2, r1)

        np.testing.assert_array_almost_equal(f1, -f2, decimal=15)

    def test_compute_accelerations(self) -> None:
        """Test acceleration computation for two bodies."""
        m1 = 1e10
        m2 = 2e10
        # state shape: (2, 2, 2) - [body, pos/vel, x/y]
        state = np.array([
            [[0.0, 0.0], [0.0, 0.0]],   # body 1 at origin, stationary
            [[100.0, 0.0], [0.0, 0.0]]  # body 2 at x=100, stationary
        ])

        a1, a2 = compute_accelerations(state, m1, m2)

        # a1 should point toward body 2 (positive x)
        self.assertGreater(a1[0], 0)
        # a2 should point toward body 1 (negative x)
        self.assertLess(a2[0], 0)
        # Newton's second law: m1*a1 = -m2*a2
        np.testing.assert_array_almost_equal(m1 * a1, -m2 * a2, decimal=10)

    def test_total_momentum_zero(self) -> None:
        """Test momentum calculation for zero total momentum setup."""
        m1 = 1e10
        m2 = 1e10
        # Equal masses, opposite velocities
        state = np.array([
            [[0.0, 0.0], [10.0, 0.0]],
            [[100.0, 0.0], [-10.0, 0.0]]
        ])

        p = total_momentum(state, m1, m2)
        np.testing.assert_array_almost_equal(p, np.array([0.0, 0.0]), decimal=10)

    def test_angular_momentum(self) -> None:
        """Test angular momentum calculation."""
        m1 = 1e10
        m2 = 1e10
        # Bodies in circular orbit around origin
        state = np.array([
            [[10.0, 0.0], [0.0, 5.0]],    # body 1 at x=10, moving +y
            [[-10.0, 0.0], [0.0, -5.0]]   # body 2 at x=-10, moving -y
        ])

        L = angular_momentum(state, m1, m2)
        # Both contribute positive angular momentum (counter-clockwise)
        # L1 = m1 * (r1 x v1) = m1 * (10 * 5 - 0 * 0) = 50 * m1
        # L2 = m2 * (r2 x v2) = m2 * (-10 * -5 - 0 * 0) = 50 * m2
        expected = m1 * 50.0 + m2 * 50.0
        self.assertAlmostEqual(L, expected, places=5)

    def test_kinetic_energy(self) -> None:
        """Test kinetic energy calculation."""
        m1 = 2.0
        m2 = 3.0
        state = np.array([
            [[0.0, 0.0], [4.0, 0.0]],   # v1 = 4 m/s
            [[10.0, 0.0], [0.0, 5.0]]   # v2 = 5 m/s
        ])

        KE = kinetic_energy(state, m1, m2)
        expected = 0.5 * m1 * 16.0 + 0.5 * m2 * 25.0  # 16 + 37.5 = 53.5
        self.assertAlmostEqual(KE, expected, places=10)

    def test_potential_energy(self) -> None:
        """Test gravitational potential energy calculation."""
        m1 = 1e10
        m2 = 2e10
        r = 100.0
        state = np.array([
            [[0.0, 0.0], [0.0, 0.0]],
            [[r, 0.0], [0.0, 0.0]]
        ])

        PE = potential_energy(state, m1, m2)
        expected = -G * m1 * m2 / r
        self.assertAlmostEqual(PE, expected, places=10)

    def test_total_energy(self) -> None:
        """Test total energy is sum of kinetic and potential."""
        m1 = 1e10
        m2 = 2e10
        state = np.array([
            [[0.0, 0.0], [100.0, 0.0]],
            [[1000.0, 0.0], [-50.0, 0.0]]
        ])

        E = total_energy(state, m1, m2)
        KE = kinetic_energy(state, m1, m2)
        PE = potential_energy(state, m1, m2)

        self.assertAlmostEqual(E, KE + PE, places=10)


class TestIntegrator(unittest.TestCase):
    """Tests for integrator.py functions."""

    def test_derivatives_shape(self) -> None:
        """Test derivatives returns correct shape."""
        m1 = 1e10
        m2 = 2e10
        state = np.array([
            [[0.0, 0.0], [10.0, 5.0]],
            [[100.0, 0.0], [-5.0, 2.0]]
        ])

        derivs = derivatives(state, m1, m2)
        self.assertEqual(derivs.shape, state.shape)

    def test_derivatives_velocities(self) -> None:
        """Test that position derivatives equal velocities."""
        m1 = 1e10
        m2 = 2e10
        v1 = np.array([10.0, 5.0])
        v2 = np.array([-5.0, 2.0])
        state = np.array([
            [[0.0, 0.0], v1],
            [[100.0, 0.0], v2]
        ])

        derivs = derivatives(state, m1, m2)
        # d(position)/dt = velocity
        np.testing.assert_array_almost_equal(derivs[0, 0], v1, decimal=15)
        np.testing.assert_array_almost_equal(derivs[1, 0], v2, decimal=15)

    def test_rk4_step_circular_orbit(self) -> None:
        """Test RK4 step maintains circular orbit accuracy."""
        # Set up a simple circular orbit
        m1 = 1e12
        m2 = 1e12
        M = m1 + m2

        # Desired period
        T = 2.0  # seconds
        omega = 2 * np.pi / T

        # Separation from Kepler's third law
        r = (G * M * T**2 / (4 * np.pi**2)) ** (1/3)
        r1 = r * m2 / M
        r2 = r * m1 / M

        v1 = omega * r1
        v2 = omega * r2

        state = np.array([
            [[r1, 0.0], [0.0, v1]],
            [[-r2, 0.0], [0.0, -v2]]
        ])

        initial_energy = total_energy(state, m1, m2)

        # Take many small steps
        dt = 0.001
        current_state = state.copy()
        for _ in range(int(T / dt)):
            current_state = rk4_step(current_state, dt, m1, m2)

        final_energy = total_energy(current_state, m1, m2)

        # Energy should be conserved to high precision
        relative_error = abs(final_energy - initial_energy) / abs(initial_energy)
        self.assertLess(relative_error, 1e-6)

    def test_check_stability_normal(self) -> None:
        """Test stability check passes for normal conditions."""
        m1 = 1e10
        m2 = 1e10
        state = np.array([
            [[0.0, 0.0], [10.0, 0.0]],
            [[1000.0, 0.0], [-10.0, 0.0]]
        ])

        self.assertTrue(check_stability(state, m1, m2))

    def test_check_stability_collision(self) -> None:
        """Test stability check fails for near-collision."""
        m1 = 1e20
        m2 = 1e20
        # Bodies very close together
        state = np.array([
            [[0.0, 0.0], [0.0, 0.0]],
            [[0.001, 0.0], [0.0, 0.0]]
        ])

        self.assertFalse(check_stability(state, m1, m2))


class TestInitialConditions(unittest.TestCase):
    """Tests for initial_conditions.py functions."""

    def test_generate_random_masses_ratio(self) -> None:
        """Test random masses are within specified ratio."""
        for _ in range(100):  # Test multiple times
            m1, m2 = generate_random_masses(ratio_max=10)
            ratio = max(m1, m2) / min(m1, m2)
            self.assertLessEqual(ratio, 10)
            self.assertGreater(m1, 0)
            self.assertGreater(m2, 0)

    def test_compute_orbital_parameters(self) -> None:
        """Test orbital parameters give correct period."""
        m1 = 1e12
        m2 = 2e12
        T = 3.0  # seconds

        r, v1, v2 = compute_orbital_parameters(m1, m2, T)

        # Verify with Kepler's third law
        M = m1 + m2
        expected_r = (G * M * T**2 / (4 * np.pi**2)) ** (1/3)
        self.assertAlmostEqual(r, expected_r, places=10)

    def test_generate_circular_orbit_momentum(self) -> None:
        """Test generated orbit has zero total momentum."""
        m1 = 1e12
        m2 = 2e12
        T = 2.5

        state = generate_circular_orbit(m1, m2, T)
        p = total_momentum(state, m1, m2)

        np.testing.assert_array_almost_equal(p, np.array([0.0, 0.0]), decimal=10)

    def test_generate_circular_orbit_period(self) -> None:
        """Test generated orbit has correct period."""
        m1 = 1e12
        m2 = 1e12
        T = 2.0

        state = generate_circular_orbit(m1, m2, T)
        initial_pos = state.copy()

        # Simulate one full period
        dt = 0.001
        current_state = state.copy()
        for _ in range(int(T / dt)):
            current_state = rk4_step(current_state, dt, m1, m2)

        # Should return close to initial position
        pos_error = np.linalg.norm(current_state[:, 0] - initial_pos[:, 0])
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
            angular_momentum=np.random.rand(3),
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


class TestSimulationIntegration(unittest.TestCase):
    """Integration tests for full simulation."""

    def test_energy_conservation(self) -> None:
        """Test that total energy is conserved during simulation."""
        from script.simulation import run_simulation

        m1 = 1e12
        m2 = 1e12
        T = 2.0

        state = generate_circular_orbit(m1, m2, T)
        result = run_simulation(state, m1, m2, dt=0.001, t_max=T)

        # Energy should be conserved
        initial_energy = result.energy[0]
        max_deviation = np.max(np.abs(result.energy - initial_energy))
        relative_deviation = max_deviation / abs(initial_energy)

        self.assertLess(relative_deviation, 1e-5)

    def test_angular_momentum_conservation(self) -> None:
        """Test that angular momentum is conserved."""
        from script.simulation import run_simulation

        m1 = 1e12
        m2 = 2e12
        T = 2.0

        state = generate_circular_orbit(m1, m2, T)
        result = run_simulation(state, m1, m2, dt=0.001, t_max=T)

        initial_L = result.angular_momentum[0]
        max_deviation = np.max(np.abs(result.angular_momentum - initial_L))
        relative_deviation = max_deviation / abs(initial_L)

        self.assertLess(relative_deviation, 1e-5)

    def test_momentum_conservation(self) -> None:
        """Test that total momentum remains zero."""
        from script.simulation import run_simulation

        m1 = 1e12
        m2 = 2e12
        T = 2.0

        state = generate_circular_orbit(m1, m2, T)
        result = run_simulation(state, m1, m2, dt=0.001, t_max=T)

        # Momentum should stay near zero
        max_momentum = np.max(np.linalg.norm(result.momentum, axis=1))
        # Compare to typical momentum scale
        typical_momentum = m1 * np.linalg.norm(state[0, 1])
        relative_deviation = max_momentum / typical_momentum

        self.assertLess(relative_deviation, 1e-10)


def main() -> None:
    """Run tests."""
    unittest.main()


if __name__ == '__main__':
    main()
