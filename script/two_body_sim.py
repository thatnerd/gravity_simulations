#!/usr/bin/env python3
"""
Two-Body Gravitational Simulation

Simulates two gravitationally interacting bodies in 2D using RK4 integration.
Records positions, velocities, momentum, angular momentum, and energy to JSON.
Optionally displays an animated visualization.

Usage:
    two_body_sim.py [options]
    two_body_sim.py --config <config_file>
    two_body_sim.py --playback <results_file> [--speed <multiplier>] [--save-video <file>]
    two_body_sim.py (-h | --help)

Options:
    -h --help                   Show this help message
    --config <config_file>      Load parameters from JSON config file
    --m1 <mass1>                Mass of body 1 in kg [default: random]
    --m2 <mass2>                Mass of body 2 in kg [default: random]
    --period <seconds>          Orbital period in seconds [default: random]
    --eccentricity <e>          Orbital eccentricity 0-1 [default: random]
    --angle <degrees>           Starting angle in degrees [default: random]
    --duration <seconds>        Total simulation time [default: 100]
    --dt <seconds>              Time step [default: 0.01]
    --integrator <method>       Integration method: yoshida or rk4 [default: yoshida]
    --output <file>             Output JSON file [default: output/simulation.json]
    --no-animate                Skip animation, only save data
    --speed <multiplier>        Playback speed multiplier [default: 1.0]
    --save-video <file>         Save animation to video file (mp4 or gif)
    --playback <results_file>   Play back existing results file
    --seed <seed>               Random seed for reproducibility

Examples:
    # Run with random parameters (elliptical orbit)
    python3 script/two_body_sim.py

    # Run with specific orbit parameters
    python3 script/two_body_sim.py --m1 1e12 --m2 2e12 --period 2.5 --eccentricity 0.5 --angle 45

    # Circular orbit (eccentricity = 0)
    python3 script/two_body_sim.py --eccentricity 0

    # Use RK4 integrator instead of Yoshida (default)
    python3 script/two_body_sim.py --integrator rk4

    # Run from config file
    python3 script/two_body_sim.py --config config/my_config.json

    # Run without animation (headless)
    python3 script/two_body_sim.py --no-animate --output output/my_sim.json

    # Play back saved results at 10x speed
    python3 script/two_body_sim.py --playback output/my_sim.json --speed 10

    # Save animation as video
    python3 script/two_body_sim.py --playback output/my_sim.json --save-video output/sim.mp4
"""

import sys
import os
import random

import numpy as np
from docopt import docopt

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from script.physics import G
from script.units import Mass, Time
from script.initial_conditions import (
    generate_elliptical_orbit,
    generate_random_masses,
    generate_random_period,
    generate_random_eccentricity,
    generate_random_true_anomaly,
)
from script.simulation import run_simulation
from script.io_handler import save_results, load_results, load_config
from script.visualizer import animate_simulation


def parse_mass(value: str) -> float | None:
    """Parse mass value, returning None for 'random'."""
    if value.lower() == 'random':
        return None
    return float(value)


def parse_period(value: str) -> float | None:
    """Parse period value, returning None for 'random'."""
    if value.lower() == 'random':
        return None
    return float(value)


def parse_float_or_random(value: str) -> float | None:
    """Parse float value, returning None for 'random'."""
    if value.lower() == 'random':
        return None
    return float(value)


def main() -> None:
    """Main entry point."""
    args = docopt(__doc__)

    # Handle playback mode
    if args['--playback']:
        result = load_results(args['--playback'])
        print(f"Loaded simulation: {result.metadata['n_steps']} steps")
        print(f"Masses: m1 = {result.metadata['m1']:.3e} kg, m2 = {result.metadata['m2']:.3e} kg")
        print(f"Time range: 0 to {result.times[-1]:.2f} s")

        if result.collision:
            print("WARNING: Simulation ended due to collision!")

        speed = float(args['--speed'])
        save_path = args['--save-video']

        animate_simulation(result, speed_multiplier=speed, save_path=save_path)
        return

    # Handle config file
    if args['--config']:
        config = load_config(args['--config'])
        m1_val = config.get('m1')
        m2_val = config.get('m2')
        period_val = config.get('period')
        eccentricity_val = config.get('eccentricity')
        angle_val = config.get('angle')  # degrees in config
        duration = config.get('duration', 100.0)
        dt = config.get('dt', 0.01)
        integrator = config.get('integrator', 'yoshida')
        output = config.get('output', 'output/simulation.json')
        no_animate = config.get('no_animate', False)
        speed = config.get('speed', 1.0)
        save_video = config.get('save_video')
        seed = config.get('seed')
    else:
        # Parse command line arguments
        m1_val = parse_mass(args['--m1']) if args['--m1'] else None
        m2_val = parse_mass(args['--m2']) if args['--m2'] else None
        period_val = parse_period(args['--period']) if args['--period'] else None
        eccentricity_val = parse_float_or_random(args['--eccentricity']) if args['--eccentricity'] else None
        angle_val = parse_float_or_random(args['--angle']) if args['--angle'] else None  # degrees
        duration = float(args['--duration'])
        dt = float(args['--dt'])
        integrator = args['--integrator']
        output = args['--output']
        no_animate = args['--no-animate']
        speed = float(args['--speed'])
        save_video = args['--save-video']
        seed = int(args['--seed']) if args['--seed'] else None

    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
        print(f"Random seed: {seed}")

    # Generate random values if needed
    if m1_val is None or m2_val is None:
        m1_gen, m2_gen = generate_random_masses()
        m1 = Mass(m1_val) if m1_val is not None else m1_gen
        m2 = Mass(m2_val) if m2_val is not None else m2_gen
    else:
        m1 = Mass(m1_val)
        m2 = Mass(m2_val)

    if period_val is None:
        period = generate_random_period()
    else:
        period = Time(period_val)

    if eccentricity_val is None:
        eccentricity = generate_random_eccentricity()
    else:
        eccentricity = eccentricity_val

    if angle_val is None:
        true_anomaly = generate_random_true_anomaly()
    else:
        true_anomaly = np.radians(angle_val)  # Convert degrees to radians

    # Print simulation parameters
    m1_f = float(m1)
    m2_f = float(m2)
    period_f = float(period)

    print("=" * 60)
    print("Two-Body Gravitational Simulation")
    print("=" * 60)
    print(f"Mass 1:          {m1_f:.3e} kg")
    print(f"Mass 2:          {m2_f:.3e} kg")
    print(f"Mass ratio:      {max(m1_f, m2_f) / min(m1_f, m2_f):.2f}")
    print(f"Orbital period:  {period_f:.3f} s")
    print(f"Eccentricity:    {eccentricity:.3f}")
    print(f"Starting angle:  {np.degrees(true_anomaly):.1f}°")
    print(f"Duration:        {duration:.1f} s")
    print(f"Time step:       {dt} s")
    print(f"Integrator:      {integrator}")
    print(f"Expected steps:  {int(duration / dt) + 1}")
    print(f"Output file:     {output}")
    print("=" * 60)

    # Generate initial conditions
    print("\nGenerating initial conditions...")
    bodies = generate_elliptical_orbit(m1, m2, period, eccentricity, true_anomaly)

    r1 = bodies[0].position.array
    r2 = bodies[1].position.array
    separation = float(np.linalg.norm(r2 - r1))
    print(f"Initial separation: {separation:.3e} m")

    # Run simulation
    print("\nRunning simulation...")
    dt_time = Time(dt)
    t_max_time = Time(duration)
    result = run_simulation(bodies, dt=dt_time, t_max=t_max_time, integrator=integrator)

    # Add extra metadata
    result.metadata['period'] = period_f
    result.metadata['eccentricity'] = eccentricity
    result.metadata['true_anomaly'] = true_anomaly
    result.metadata['initial_separation'] = separation
    result.metadata['G'] = G

    print(f"Simulation complete: {result.metadata['n_steps']} steps")

    if result.collision:
        print(f"WARNING: Collision detected at t = {result.times[-1]:.3f} s!")

    # Check conservation
    E0 = result.energy[0]
    E_final = result.energy[-1]
    dE_rel = abs(E_final - E0) / abs(E0)
    print(f"Energy conservation: ΔE/E₀ = {dE_rel:.2e}")

    # Angular momentum is a 3D vector; use magnitude for conservation check
    L0 = np.linalg.norm(result.angular_momentum[0])
    L_final = np.linalg.norm(result.angular_momentum[-1])
    dL_rel = abs(L_final - L0) / abs(L0)
    print(f"Angular momentum conservation: ΔL/L₀ = {dL_rel:.2e}")

    # Save results
    print(f"\nSaving results to {output}...")
    save_results(result, output)
    print("Results saved.")

    # Animate if requested
    if not no_animate:
        print("\nLaunching visualization...")
        animate_simulation(result, speed_multiplier=speed, save_path=save_video)
    elif save_video:
        print(f"\nSaving video to {save_video}...")
        animate_simulation(result, speed_multiplier=speed, save_path=save_video)


if __name__ == '__main__':
    main()
