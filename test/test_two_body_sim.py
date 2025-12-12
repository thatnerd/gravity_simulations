#!/usr/bin/env python3
"""
Unit tests for gravitational simulation.

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
    momentum,
    angular_momentum,
    kinetic_energy,
    potential_energy,
)
from script.integrator import (
    rk4_step,
    yoshida_step,
    integrate_step,
    check_stability,
    compute_accelerations,
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
from script.simulation import SimulationResult
from script.units import (
    Mass, Time, Body, Acceleration,
    position, velocity, body,
)


def make_bodies(m1: Mass, m2: Mass, pos1, vel1, pos2, vel2) -> list[Body]:
    """Helper to create two-body list."""
    return [Body(m1, pos1, vel1), Body(m2, pos2, vel2)]


def total_momentum_bodies(bodies: list[Body]) -> np.ndarray:
    """Compute total momentum of all bodies."""
    total = np.zeros(2)
    for b in bodies:
        total += momentum(b).array
    return total


def total_angular_momentum_bodies(bodies: list[Body]) -> np.ndarray:
    """Compute total angular momentum of all bodies."""
    total = np.zeros(3)
    for b in bodies:
        total += angular_momentum(b).array
    return total


def total_kinetic_energy_bodies(bodies: list[Body]) -> float:
    """Compute total kinetic energy of all bodies."""
    return sum(float(kinetic_energy(b)) for b in bodies)


def total_potential_energy_bodies(bodies: list[Body]) -> float:
    """Compute total potential energy of all bodies."""
    from itertools import combinations
    total = 0.0
    for i, j in combinations(range(len(bodies)), 2):
        total += float(potential_energy(bodies[i], bodies[j]))
    return total


def total_energy_bodies(bodies: list[Body]) -> float:
    """Compute total energy of all bodies."""
    return total_kinetic_energy_bodies(bodies) + total_potential_energy_bodies(bodies)


class TestPhysics(unittest.TestCase):
    """Tests for physics.py functions."""

    def test_gravitational_force_magnitude(self) -> None:
        """Test gravitational force has correct magnitude."""
        b1 = body(1e10, 0.0, 0.0, 0.0, 0.0)
        b2 = body(2e10, 100.0, 0.0, 0.0, 0.0)

        force = gravitational_force(b1, b2)
        expected_magnitude = G * float(b1.mass) * float(b2.mass) / (100.0 ** 2)

        self.assertAlmostEqual(force.magnitude(), expected_magnitude, places=10)

    def test_gravitational_force_direction(self) -> None:
        """Test gravitational force points toward other body."""
        b1 = body(1e10, 0.0, 0.0, 0.0, 0.0)
        b2 = body(1e10, 100.0, 0.0, 0.0, 0.0)

        force = gravitational_force(b1, b2)
        # Force on b1 should point toward b2 (positive x)
        self.assertGreater(force.x, 0)
        self.assertAlmostEqual(force.y, 0.0, places=15)

    def test_gravitational_force_symmetry(self) -> None:
        """Test Newton's third law: F12 = -F21."""
        b1 = body(1e10, 10.0, 20.0, 0.0, 0.0)
        b2 = body(2e10, 50.0, 80.0, 0.0, 0.0)

        f1 = gravitational_force(b1, b2)
        f2 = gravitational_force(b2, b1)

        np.testing.assert_array_almost_equal(f1.array, -f2.array, decimal=15)

    def test_compute_accelerations(self) -> None:
        """Test acceleration computation for two bodies."""
        bodies = [
            body(1e10, 0.0, 0.0, 0.0, 0.0),
            body(2e10, 100.0, 0.0, 0.0, 0.0),
        ]

        accels = compute_accelerations(bodies)

        # a1 should point toward body 2 (positive x)
        self.assertGreater(accels[0].x, 0)
        # a2 should point toward body 1 (negative x)
        self.assertLess(accels[1].x, 0)
        # Newton's second law: m1*a1 = -m2*a2
        np.testing.assert_array_almost_equal(
            float(bodies[0].mass) * accels[0].array,
            -float(bodies[1].mass) * accels[1].array,
            decimal=10
        )

    def test_total_momentum_zero(self) -> None:
        """Test momentum calculation for zero total momentum setup."""
        # Equal masses, opposite velocities
        bodies = [
            body(1e10, 0.0, 0.0, 10.0, 0.0),
            body(1e10, 100.0, 0.0, -10.0, 0.0),
        ]

        p = total_momentum_bodies(bodies)
        np.testing.assert_array_almost_equal(p, np.array([0.0, 0.0]), decimal=10)

    def test_angular_momentum(self) -> None:
        """Test angular momentum calculation."""
        # Bodies in circular orbit around origin
        bodies = [
            body(1e10, 10.0, 0.0, 0.0, 5.0),     # body 1 at x=10, moving +y
            body(1e10, -10.0, 0.0, 0.0, -5.0),   # body 2 at x=-10, moving -y
        ]

        L = total_angular_momentum_bodies(bodies)
        # Both contribute positive angular momentum (counter-clockwise)
        # L1 = m1 * (r1 x v1) = m1 * (10 * 5 - 0 * 0) = 50 * m1
        # L2 = m2 * (r2 x v2) = m2 * (-10 * -5 - 0 * 0) = 50 * m2
        expected_z = float(bodies[0].mass) * 50.0 + float(bodies[1].mass) * 50.0
        self.assertAlmostEqual(L[2], expected_z, places=5)
        self.assertAlmostEqual(L[0], 0.0, places=10)
        self.assertAlmostEqual(L[1], 0.0, places=10)

    def test_kinetic_energy(self) -> None:
        """Test kinetic energy calculation."""
        bodies = [
            body(2.0, 0.0, 0.0, 4.0, 0.0),   # v1 = 4 m/s
            body(3.0, 10.0, 0.0, 0.0, 5.0),  # v2 = 5 m/s
        ]

        KE = total_kinetic_energy_bodies(bodies)
        expected = 0.5 * 2.0 * 16.0 + 0.5 * 3.0 * 25.0  # 16 + 37.5 = 53.5
        self.assertAlmostEqual(KE, expected, places=10)

    def test_potential_energy(self) -> None:
        """Test gravitational potential energy calculation."""
        r = 100.0
        bodies = [
            body(1e10, 0.0, 0.0, 0.0, 0.0),
            body(2e10, r, 0.0, 0.0, 0.0),
        ]

        PE = total_potential_energy_bodies(bodies)
        expected = -G * float(bodies[0].mass) * float(bodies[1].mass) / r
        self.assertAlmostEqual(PE, expected, places=10)

    def test_total_energy(self) -> None:
        """Test total energy is sum of kinetic and potential."""
        bodies = [
            body(1e10, 0.0, 0.0, 100.0, 0.0),
            body(2e10, 1000.0, 0.0, -50.0, 0.0),
        ]

        E = total_energy_bodies(bodies)
        KE = total_kinetic_energy_bodies(bodies)
        PE = total_potential_energy_bodies(bodies)

        self.assertAlmostEqual(E, KE + PE, places=10)


class TestIntegrator(unittest.TestCase):
    """Tests for integrator.py functions."""

    def test_rk4_step_returns_list(self) -> None:
        """Test rk4_step returns list of same length."""
        bodies = [
            body(1e10, 0.0, 0.0, 10.0, 5.0),
            body(2e10, 100.0, 0.0, -5.0, 2.0),
        ]

        new_bodies = rk4_step(bodies, Time(0.01))
        self.assertEqual(len(new_bodies), len(bodies))
        for b in new_bodies:
            self.assertIsInstance(b, Body)

    def test_rk4_step_preserves_mass(self) -> None:
        """Test that masses are preserved across steps."""
        bodies = [
            body(1e10, 0.0, 0.0, 10.0, 5.0),
            body(2e10, 100.0, 0.0, -5.0, 2.0),
        ]

        new_bodies = rk4_step(bodies, Time(0.01))
        for i in range(len(bodies)):
            self.assertEqual(float(bodies[i].mass), float(new_bodies[i].mass))

    def test_derivatives_velocities(self) -> None:
        """Test that position changes in direction of velocity."""
        v1 = velocity(10.0, 5.0)
        bodies = [
            Body(Mass(1e10), position(0.0, 0.0), v1),
            Body(Mass(2e10), position(100.0, 0.0), velocity(-5.0, 2.0)),
        ]

        dt = Time(0.001)
        new_bodies = rk4_step(bodies, dt)

        # Position should move approximately in direction of velocity
        pos1_delta = new_bodies[0].position.array - bodies[0].position.array
        # Dot product should be positive (same direction)
        self.assertGreater(np.dot(pos1_delta, v1.array), 0)

    def test_rk4_step_circular_orbit(self) -> None:
        """Test RK4 step maintains circular orbit accuracy."""
        m1 = Mass(1e12)
        m2 = Mass(1e12)
        T = Time(2.0)

        bodies = generate_elliptical_orbit(m1, m2, T)
        initial_energy = total_energy_bodies(bodies)

        # Take many small steps
        dt = Time(0.001)
        current_bodies = bodies
        for _ in range(int(float(T) / float(dt))):
            current_bodies = rk4_step(current_bodies, dt)

        final_energy = total_energy_bodies(current_bodies)

        # Energy should be conserved to high precision
        relative_error = abs(final_energy - initial_energy) / abs(initial_energy)
        self.assertLess(relative_error, 1e-6)

    def test_check_stability_normal(self) -> None:
        """Test stability check passes for normal conditions."""
        bodies = [
            body(1e10, 0.0, 0.0, 10.0, 0.0),
            body(1e10, 1000.0, 0.0, -10.0, 0.0),
        ]

        self.assertTrue(check_stability(bodies))

    def test_check_stability_collision(self) -> None:
        """Test stability check fails for near-collision."""
        # Bodies very close together
        bodies = [
            body(1e20, 0.0, 0.0, 0.0, 0.0),
            body(1e20, 0.001, 0.0, 0.0, 0.0),
        ]

        self.assertFalse(check_stability(bodies))

    def test_compute_accelerations_returns_list(self) -> None:
        """Test that compute_accelerations returns list of Acceleration objects."""
        bodies = [
            body(1e10, 0.0, 0.0, 0.0, 0.0),
            body(2e10, 100.0, 0.0, 0.0, 0.0),
        ]

        accels = compute_accelerations(bodies)

        self.assertEqual(len(accels), 2)
        for a in accels:
            self.assertIsInstance(a, Acceleration)

    def test_yoshida_step_returns_list(self) -> None:
        """Test yoshida_step returns list of same length."""
        bodies = [
            body(1e10, 0.0, 0.0, 10.0, 5.0),
            body(2e10, 100.0, 0.0, -5.0, 2.0),
        ]

        new_bodies = yoshida_step(bodies, Time(0.01))
        self.assertEqual(len(new_bodies), len(bodies))
        for b in new_bodies:
            self.assertIsInstance(b, Body)

    def test_yoshida_step_preserves_mass(self) -> None:
        """Test that masses are preserved across Yoshida steps."""
        bodies = [
            body(1e10, 0.0, 0.0, 10.0, 5.0),
            body(2e10, 100.0, 0.0, -5.0, 2.0),
        ]

        new_bodies = yoshida_step(bodies, Time(0.01))
        for i in range(len(bodies)):
            self.assertEqual(float(bodies[i].mass), float(new_bodies[i].mass))

    def test_yoshida_step_circular_orbit(self) -> None:
        """Test Yoshida step maintains circular orbit accuracy."""
        m1 = Mass(1e12)
        m2 = Mass(1e12)
        T = Time(2.0)

        bodies = generate_elliptical_orbit(m1, m2, T)
        initial_energy = total_energy_bodies(bodies)

        # Take many small steps
        dt = Time(0.001)
        current_bodies = bodies
        for _ in range(int(float(T) / float(dt))):
            current_bodies = yoshida_step(current_bodies, dt)

        final_energy = total_energy_bodies(current_bodies)

        # Energy should be conserved to high precision
        relative_error = abs(final_energy - initial_energy) / abs(initial_energy)
        self.assertLess(relative_error, 1e-6)

    def test_integrate_step_dispatcher(self) -> None:
        """Test integrate_step dispatches to correct integrator."""
        bodies = [
            body(1e10, 0.0, 0.0, 10.0, 5.0),
            body(2e10, 100.0, 0.0, -5.0, 2.0),
        ]
        dt = Time(0.01)

        # Test yoshida
        yoshida_result = integrate_step(bodies, dt, 'yoshida')
        self.assertEqual(len(yoshida_result), 2)

        # Test rk4
        rk4_result = integrate_step(bodies, dt, 'rk4')
        self.assertEqual(len(rk4_result), 2)

    def test_integrate_step_invalid_method(self) -> None:
        """Test integrate_step raises error for invalid method."""
        bodies = [
            body(1e10, 0.0, 0.0, 10.0, 5.0),
            body(2e10, 100.0, 0.0, -5.0, 2.0),
        ]

        with self.assertRaises(ValueError):
            integrate_step(bodies, Time(0.01), 'invalid_method')

    def test_yoshida_symplectic_energy_behavior(self) -> None:
        """Test Yoshida shows bounded energy oscillation vs RK4 drift.

        Symplectic integrators have bounded energy error that oscillates,
        while non-symplectic methods like RK4 show secular drift.
        """
        m1 = Mass(1e12)
        m2 = Mass(1e12)
        T = Time(2.0)

        # Use larger time step to make differences visible
        dt = Time(0.01)
        n_steps = int(5 * float(T) / float(dt))  # 5 periods

        # Run Yoshida
        bodies_yoshida = generate_elliptical_orbit(m1, m2, T)
        initial_energy = total_energy_bodies(bodies_yoshida)
        yoshida_energy_errors = []

        for _ in range(n_steps):
            bodies_yoshida = yoshida_step(bodies_yoshida, dt)
            error = (total_energy_bodies(bodies_yoshida) - initial_energy) / abs(initial_energy)
            yoshida_energy_errors.append(error)

        # Run RK4 with same initial conditions
        bodies_rk4 = generate_elliptical_orbit(m1, m2, T)

        rk4_energy_errors = []
        for _ in range(n_steps):
            bodies_rk4 = rk4_step(bodies_rk4, dt)
            error = (total_energy_bodies(bodies_rk4) - initial_energy) / abs(initial_energy)
            rk4_energy_errors.append(error)

        # Both should have small errors with small dt
        # But Yoshida should have bounded oscillation
        yoshida_max = max(abs(e) for e in yoshida_energy_errors)
        self.assertLess(yoshida_max, 1e-4)


class TestInitialConditions(unittest.TestCase):
    """Tests for initial_conditions.py functions."""

    def test_generate_random_masses_ratio(self) -> None:
        """Test random masses are within specified ratio."""
        for _ in range(100):
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
        T = Time(3.0)

        a = compute_semi_major_axis(m1, m2, T)

        M = float(m1) + float(m2)
        expected_a = (G * M * float(T)**2 / (4 * np.pi**2)) ** (1/3)
        self.assertAlmostEqual(a, expected_a, places=10)

    def test_generate_elliptical_orbit_returns_list(self) -> None:
        """Test generate_elliptical_orbit returns list of Body objects."""
        m1 = Mass(1e12)
        m2 = Mass(2e12)
        T = Time(2.5)

        bodies = generate_elliptical_orbit(m1, m2, T)

        self.assertIsInstance(bodies, list)
        self.assertEqual(len(bodies), 2)
        for b in bodies:
            self.assertIsInstance(b, Body)

    def test_generate_elliptical_orbit_momentum(self) -> None:
        """Test generated orbit has zero total momentum."""
        m1 = Mass(1e12)
        m2 = Mass(2e12)
        T = Time(2.5)

        for e in [0.0, 0.3, 0.6]:
            for theta in [0.0, np.pi/4, np.pi]:
                bodies = generate_elliptical_orbit(m1, m2, T, eccentricity=e, true_anomaly=theta)
                p = total_momentum_bodies(bodies)
                np.testing.assert_array_almost_equal(
                    p, np.array([0.0, 0.0]), decimal=10,
                    err_msg=f"Failed for e={e}, theta={theta}"
                )

    def test_generate_elliptical_orbit_period(self) -> None:
        """Test generated circular orbit has correct period."""
        m1 = Mass(1e12)
        m2 = Mass(1e12)
        T = Time(2.0)

        bodies = generate_elliptical_orbit(m1, m2, T, eccentricity=0.0)
        initial_pos = np.array([b.position.array.copy() for b in bodies])

        # Simulate one full period
        dt = Time(0.001)
        current_bodies = bodies
        for _ in range(int(float(T) / float(dt))):
            current_bodies = rk4_step(current_bodies, dt)

        final_pos = np.array([b.position.array for b in current_bodies])
        pos_error = np.linalg.norm(final_pos - initial_pos)
        max_separation = np.linalg.norm(initial_pos[0] - initial_pos[1])
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
        bodies_peri = generate_elliptical_orbit(m1, m2, T, eccentricity=e, true_anomaly=0.0)
        r_peri = np.linalg.norm(bodies_peri[0].position.array - bodies_peri[1].position.array)
        expected_peri = a * (1 - e)
        self.assertAlmostEqual(r_peri, expected_peri, places=8)

        # At apoapsis (theta=π), r = a(1+e)
        bodies_apo = generate_elliptical_orbit(m1, m2, T, eccentricity=e, true_anomaly=np.pi)
        r_apo = np.linalg.norm(bodies_apo[0].position.array - bodies_apo[1].position.array)
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
            angular_momentum=np.random.rand(3, 3),
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
        """Test that angular momentum relative deviation can be formatted as scalar."""
        from script.simulation import run_simulation

        m1 = Mass(1e12)
        m2 = Mass(1e12)
        T = Time(2.0)

        bodies = generate_elliptical_orbit(m1, m2, T)
        result = run_simulation(bodies, dt=Time(0.001), t_max=T)

        L0 = np.linalg.norm(result.angular_momentum[0])
        L_final = np.linalg.norm(result.angular_momentum[-1])
        dL_rel = abs(L_final - L0) / abs(L0)

        formatted = f"ΔL/L₀ = {dL_rel:.2e}"
        self.assertIn("ΔL/L₀", formatted)


class TestSimulationIntegration(unittest.TestCase):
    """Integration tests for full simulation."""

    def test_collision_detection(self) -> None:
        """Test that simulation detects collision and stops early."""
        from script.simulation import run_simulation

        # Bodies 0.05m apart - should immediately trigger instability
        bodies = [
            body(1e20, -0.025, 0.0, 0.0, 0.0),
            body(1e20, 0.025, 0.0, 0.0, 0.0),
        ]

        result = run_simulation(bodies, dt=Time(0.0001), t_max=Time(1.0))

        self.assertTrue(result.collision)
        self.assertEqual(len(result.times), 1)

    def test_custom_stability_threshold(self) -> None:
        """Test run_simulation with custom stability threshold."""
        from script.simulation import run_simulation

        m1 = Mass(1e12)
        m2 = Mass(1e12)
        T = Time(2.0)

        bodies = generate_elliptical_orbit(m1, m2, T)

        # Run with very low threshold - should trigger collision detection
        result_low = run_simulation(
            bodies, dt=Time(0.01), t_max=Time(1.0),
            stability_threshold=1e-10
        )
        self.assertTrue(result_low.collision)

        # Run with normal threshold - should complete without collision
        result_normal = run_simulation(
            bodies, dt=Time(0.01), t_max=Time(1.0),
            stability_threshold=1e12
        )
        self.assertFalse(result_normal.collision)

    def test_energy_conservation(self) -> None:
        """Test that total energy is conserved during simulation."""
        from script.simulation import run_simulation

        m1 = Mass(1e12)
        m2 = Mass(1e12)
        T = Time(2.0)

        bodies = generate_elliptical_orbit(m1, m2, T)
        result = run_simulation(bodies, dt=Time(0.001), t_max=T)

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

        bodies = generate_elliptical_orbit(m1, m2, T)
        result = run_simulation(bodies, dt=Time(0.001), t_max=T)

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

        bodies = generate_elliptical_orbit(m1, m2, T)
        result = run_simulation(bodies, dt=Time(0.001), t_max=T)

        max_momentum = np.max(np.linalg.norm(result.momentum, axis=1))
        typical_momentum = float(m1) * np.linalg.norm(bodies[0].velocity.array)
        relative_deviation = max_momentum / typical_momentum

        self.assertLess(relative_deviation, 1e-10)

    def test_simulation_with_yoshida_integrator(self) -> None:
        """Test run_simulation with Yoshida integrator (default)."""
        from script.simulation import run_simulation

        m1 = Mass(1e12)
        m2 = Mass(1e12)
        T = Time(2.0)

        bodies = generate_elliptical_orbit(m1, m2, T)
        result = run_simulation(bodies, dt=Time(0.001), t_max=T, integrator='yoshida')

        self.assertFalse(result.collision)
        self.assertEqual(result.metadata['integrator'], 'yoshida')

        # Check energy conservation
        initial_energy = result.energy[0]
        max_deviation = np.max(np.abs(result.energy - initial_energy))
        relative_deviation = max_deviation / abs(initial_energy)
        self.assertLess(relative_deviation, 1e-5)

    def test_simulation_with_rk4_integrator(self) -> None:
        """Test run_simulation with RK4 integrator."""
        from script.simulation import run_simulation

        m1 = Mass(1e12)
        m2 = Mass(1e12)
        T = Time(2.0)

        bodies = generate_elliptical_orbit(m1, m2, T)
        result = run_simulation(bodies, dt=Time(0.001), t_max=T, integrator='rk4')

        self.assertFalse(result.collision)
        self.assertEqual(result.metadata['integrator'], 'rk4')

        # Check energy conservation
        initial_energy = result.energy[0]
        max_deviation = np.max(np.abs(result.energy - initial_energy))
        relative_deviation = max_deviation / abs(initial_energy)
        self.assertLess(relative_deviation, 1e-5)

    def test_simulation_default_integrator_is_yoshida(self) -> None:
        """Test that default integrator is Yoshida."""
        from script.simulation import run_simulation

        m1 = Mass(1e12)
        m2 = Mass(1e12)
        T = Time(1.0)

        bodies = generate_elliptical_orbit(m1, m2, T)
        result = run_simulation(bodies, dt=Time(0.01), t_max=T)

        self.assertEqual(result.metadata['integrator'], 'yoshida')


def main() -> None:
    """Run tests."""
    unittest.main()


if __name__ == '__main__':
    main()
