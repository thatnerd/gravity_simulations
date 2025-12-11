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
    --duration <seconds>        Total simulation time [default: 100]
    --dt <seconds>              Time step [default: 0.01]
    --output <file>             Output JSON file [default: output/simulation.json]
    --no-animate                Skip animation, only save data
    --speed <multiplier>        Playback speed multiplier [default: 1.0]
    --save-video <file>         Save animation to video file (mp4 or gif)
    --playback <results_file>   Play back existing results file
    --seed <seed>               Random seed for reproducibility

Examples:
    # Run with random parameters
    python3 script/two_body_sim.py

    # Run with specific masses and period
    python3 script/two_body_sim.py --m1 1e12 --m2 2e12 --period 2.5

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

from docopt import docopt

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from script.physics import G
from script.initial_conditions import (
    generate_circular_orbit,
    generate_random_masses,
    generate_random_period,
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
        m1 = config.get('m1')
        m2 = config.get('m2')
        period = config.get('period')
        duration = config.get('duration', 100.0)
        dt = config.get('dt', 0.01)
        output = config.get('output', 'output/simulation.json')
        no_animate = config.get('no_animate', False)
        speed = config.get('speed', 1.0)
        save_video = config.get('save_video')
        seed = config.get('seed')
    else:
        # Parse command line arguments
        m1 = parse_mass(args['--m1']) if args['--m1'] else None
        m2 = parse_mass(args['--m2']) if args['--m2'] else None
        period = parse_period(args['--period']) if args['--period'] else None
        duration = float(args['--duration'])
        dt = float(args['--dt'])
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
    if m1 is None or m2 is None:
        m1_gen, m2_gen = generate_random_masses()
        m1 = m1 if m1 is not None else m1_gen
        m2 = m2 if m2 is not None else m2_gen

    if period is None:
        period = generate_random_period()

    # Print simulation parameters
    print("=" * 60)
    print("Two-Body Gravitational Simulation")
    print("=" * 60)
    print(f"Mass 1:          {m1:.3e} kg")
    print(f"Mass 2:          {m2:.3e} kg")
    print(f"Mass ratio:      {max(m1, m2) / min(m1, m2):.2f}")
    print(f"Orbital period:  {period:.3f} s")
    print(f"Duration:        {duration:.1f} s")
    print(f"Time step:       {dt} s")
    print(f"Expected steps:  {int(duration / dt) + 1}")
    print(f"Output file:     {output}")
    print("=" * 60)

    # Generate initial conditions
    print("\nGenerating initial conditions...")
    initial_state = generate_circular_orbit(m1, m2, period)

    r1 = initial_state[0, 0]
    r2 = initial_state[1, 0]
    separation = float(((r2 - r1) ** 2).sum() ** 0.5)
    print(f"Initial separation: {separation:.3e} m")

    # Run simulation
    print("\nRunning simulation...")
    result = run_simulation(initial_state, m1, m2, dt=dt, t_max=duration)

    # Add extra metadata
    result.metadata['period'] = period
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

    L0 = result.angular_momentum[0]
    L_final = result.angular_momentum[-1]
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
