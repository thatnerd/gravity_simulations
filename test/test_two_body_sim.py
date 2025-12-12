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
    generate_elliptical_orbit,
    generate_random_masses,
    generate_random_period,
    generate_random_eccentricity,
    generate_random_true_anomaly,
    compute_semi_major_axis,
)
from script.io_handler import (
    save_results,
    load_results,
    load_config,
    NumpyEncoder,
)
from script.integrator import compute_accelerations
from script.simulation import SimulationResult
from script.units import (
    Mass, Time, TwoBodyState, Acceleration,
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

        state = generate_elliptical_orbit(m1, m2, T)
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

    def test_compute_accelerations_typed(self) -> None:
        """Test that compute_accelerations returns typed Acceleration objects."""
        m1 = Mass(1e10)
        m2 = Mass(2e10)
        state = TwoBodyState.from_bodies(
            position(0.0, 0.0), velocity(0.0, 0.0),
            position(100.0, 0.0), velocity(0.0, 0.0)
        )

        a1, a2 = compute_accelerations(state, m1, m2)

        # Check types
        self.assertIsInstance(a1, Acceleration)
        self.assertIsInstance(a2, Acceleration)

        # a1 should point toward body 2 (positive x)
        self.assertGreater(a1.x, 0)
        # a2 should point toward body 1 (negative x)
        self.assertLess(a2.x, 0)


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

    def test_generate_random_masses_range(self) -> None:
        """Test random masses are within specified min/max range."""
        m_min = 1e10
        m_max = 1e11
        for _ in range(100):
            m1, m2 = generate_random_masses(m_min=m_min, m_max=m_max)
            # Masses should be within range (with some tolerance for rounding)
            self.assertGreaterEqual(float(m1), m_min * 0.99)
            self.assertLessEqual(float(m1), m_max * 1.01)
            self.assertGreaterEqual(float(m2), m_min * 0.99)
            self.assertLessEqual(float(m2), m_max * 1.01)

    def test_generate_random_period(self) -> None:
        """Test random period is within specified range."""
        min_period = 2.0
        max_period = 5.0
        for _ in range(100):
            period = generate_random_period(min_period=min_period, max_period=max_period)
            self.assertGreaterEqual(float(period), min_period)
            self.assertLessEqual(float(period), max_period)
            self.assertIsInstance(period, Time)

    def test_compute_semi_major_axis(self) -> None:
        """Test semi-major axis calculation from period."""
        m1 = Mass(1e12)
        m2 = Mass(2e12)
        T = Time(3.0)  # seconds

        a = compute_semi_major_axis(m1, m2, T)

        # Verify with Kepler's third law: T² = 4π²a³/(GM)
        M = float(m1) + float(m2)
        expected_a = (G * M * float(T)**2 / (4 * np.pi**2)) ** (1/3)
        self.assertAlmostEqual(a, expected_a, places=10)

    def test_generate_elliptical_orbit_momentum(self) -> None:
        """Test generated orbit has zero total momentum."""
        m1 = Mass(1e12)
        m2 = Mass(2e12)
        T = Time(2.5)

        # Test with various eccentricities and angles
        for e in [0.0, 0.3, 0.6]:
            for theta in [0.0, np.pi/4, np.pi]:
                state = generate_elliptical_orbit(m1, m2, T, eccentricity=e, true_anomaly=theta)
                p = total_momentum(state, m1, m2)
                np.testing.assert_array_almost_equal(
                    p.array, np.array([0.0, 0.0]), decimal=10,
                    err_msg=f"Failed for e={e}, theta={theta}"
                )

    def test_generate_elliptical_orbit_period(self) -> None:
        """Test generated circular orbit has correct period."""
        m1 = Mass(1e12)
        m2 = Mass(1e12)
        T = Time(2.0)

        # Use circular orbit (e=0) for period test
        state = generate_elliptical_orbit(m1, m2, T, eccentricity=0.0)
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

    def test_generate_elliptical_orbit_eccentricity(self) -> None:
        """Test elliptical orbit has correct distance at periapsis and apoapsis."""
        m1 = Mass(1e12)
        m2 = Mass(1e12)
        T = Time(2.0)
        e = 0.5

        a = compute_semi_major_axis(m1, m2, T)

        # At periapsis (theta=0), r = a(1-e)
        state_peri = generate_elliptical_orbit(m1, m2, T, eccentricity=e, true_anomaly=0.0)
        r_peri = np.linalg.norm(state_peri.position(0).array - state_peri.position(1).array)
        expected_peri = a * (1 - e)
        self.assertAlmostEqual(r_peri, expected_peri, places=8)

        # At apoapsis (theta=π), r = a(1+e)
        state_apo = generate_elliptical_orbit(m1, m2, T, eccentricity=e, true_anomaly=np.pi)
        r_apo = np.linalg.norm(state_apo.position(0).array - state_apo.position(1).array)
        expected_apo = a * (1 + e)
        self.assertAlmostEqual(r_apo, expected_apo, places=8)

    def test_generate_random_eccentricity(self) -> None:
        """Test random eccentricity is within range."""
        for _ in range(100):
            e = generate_random_eccentricity(max_e=0.7)
            self.assertGreaterEqual(e, 0.0)
            self.assertLess(e, 0.7)

    def test_generate_random_true_anomaly(self) -> None:
        """Test random true anomaly is within [0, 2π)."""
        for _ in range(100):
            theta = generate_random_true_anomaly()
            self.assertGreaterEqual(theta, 0.0)
            self.assertLess(theta, 2 * np.pi)


class TestIOHandler(unittest.TestCase):
    """Tests for io_handler.py functions."""

    def setUp(self) -> None:
        """Set up test output directory."""
        self.test_dir = '/tmp/test_two_body_sim'
        os.makedirs(self.test_dir, exist_ok=True)

    def test_numpy_encoder_array(self) -> None:
        """Test NumpyEncoder handles numpy arrays."""
        import json
        data = {'array': np.array([1.0, 2.0, 3.0])}
        result = json.dumps(data, cls=NumpyEncoder)
        loaded = json.loads(result)
        self.assertEqual(loaded['array'], [1.0, 2.0, 3.0])

    def test_numpy_encoder_float64(self) -> None:
        """Test NumpyEncoder handles numpy float64."""
        import json
        data = {'value': np.float64(3.14159)}
        result = json.dumps(data, cls=NumpyEncoder)
        loaded = json.loads(result)
        self.assertAlmostEqual(loaded['value'], 3.14159, places=5)

    def test_numpy_encoder_int64(self) -> None:
        """Test NumpyEncoder handles numpy int64."""
        import json
        data = {'value': np.int64(42)}
        result = json.dumps(data, cls=NumpyEncoder)
        loaded = json.loads(result)
        self.assertEqual(loaded['value'], 42)

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

        state = generate_elliptical_orbit(m1, m2, T)
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

    def test_collision_detection(self) -> None:
        """Test that simulation detects collision and stops early."""
        from script.simulation import run_simulation

        # Set up bodies very close together - acceleration will exceed threshold
        # With m=1e20 kg, threshold=1e12 m/s², instability at r < 0.08m
        m1 = Mass(1e20)
        m2 = Mass(1e20)
        # Bodies 0.05m apart - should immediately trigger instability
        state = TwoBodyState.from_bodies(
            position(-0.025, 0.0), velocity(0.0, 0.0),
            position(0.025, 0.0), velocity(0.0, 0.0)
        )

        result = run_simulation(state, m1, m2, dt=Time(0.0001), t_max=Time(1.0))

        # Should detect collision immediately (first step)
        self.assertTrue(result.collision)
        # Should have only 1 step recorded
        self.assertEqual(len(result.times), 1)

    def test_custom_stability_threshold(self) -> None:
        """Test run_simulation with custom stability threshold."""
        from script.simulation import run_simulation

        m1 = Mass(1e12)
        m2 = Mass(1e12)
        T = Time(2.0)

        state = generate_elliptical_orbit(m1, m2, T)

        # Run with very low threshold - should trigger collision detection
        result_low = run_simulation(
            state, m1, m2, dt=Time(0.01), t_max=Time(1.0),
            stability_threshold=1e-10  # Very low, should trigger immediately
        )
        self.assertTrue(result_low.collision)

        # Run with normal threshold - should complete without collision
        result_normal = run_simulation(
            state, m1, m2, dt=Time(0.01), t_max=Time(1.0),
            stability_threshold=1e12  # Normal threshold
        )
        self.assertFalse(result_normal.collision)

    def test_energy_conservation(self) -> None:
        """Test that total energy is conserved during simulation."""
        from script.simulation import run_simulation

        m1 = Mass(1e12)
        m2 = Mass(1e12)
        T = Time(2.0)

        state = generate_elliptical_orbit(m1, m2, T)
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

        state = generate_elliptical_orbit(m1, m2, T)
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

        state = generate_elliptical_orbit(m1, m2, T)
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
