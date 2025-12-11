#!/usr/bin/env python3
"""
File I/O for two-body gravitational simulation.

This module handles saving/loading simulation results and configuration files.

Usage:
    from script.io_handler import save_results, load_results, load_config
"""

import json
import numpy as np
from numpy.typing import NDArray
from typing import Any
import os

from script.simulation import SimulationResult


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy arrays."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        return super().default(obj)


def save_results(result: SimulationResult, filepath: str) -> None:
    """
    Save simulation results to JSON file.

    Data is stored per-timestep for easy inspection of state at any point in time.

    Args:
        result: SimulationResult to save
        filepath: Output file path
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)

    # Build timestep records
    timesteps = []
    n_steps = len(result.times)
    for i in range(n_steps):
        timesteps.append({
            'time': result.times[i],
            'positions': {
                'body1': {'x': result.positions[i, 0, 0], 'y': result.positions[i, 0, 1]},
                'body2': {'x': result.positions[i, 1, 0], 'y': result.positions[i, 1, 1]},
            },
            'velocities': {
                'body1': {'x': result.velocities[i, 0, 0], 'y': result.velocities[i, 0, 1]},
                'body2': {'x': result.velocities[i, 1, 0], 'y': result.velocities[i, 1, 1]},
            },
            'momentum': {'x': result.momentum[i, 0], 'y': result.momentum[i, 1]},
            'angular_momentum': {
                'x': result.angular_momentum[i, 0],
                'y': result.angular_momentum[i, 1],
                'z': result.angular_momentum[i, 2],
            },
            'energy': result.energy[i],
        })

    data = {
        'metadata': result.metadata,
        'collision': result.collision,
        'timesteps': timesteps,
    }

    with open(filepath, 'w') as f:
        json.dump(data, f, cls=NumpyEncoder, indent=2)


def load_results(filepath: str) -> SimulationResult:
    """
    Load simulation results from JSON file.

    Args:
        filepath: Input file path

    Returns:
        SimulationResult loaded from file
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    timesteps = data['timesteps']
    n_steps = len(timesteps)

    times = np.zeros(n_steps)
    positions = np.zeros((n_steps, 2, 2))
    velocities = np.zeros((n_steps, 2, 2))
    momentum = np.zeros((n_steps, 2))
    angular_momentum = np.zeros((n_steps, 3))  # 3D vector
    energy = np.zeros(n_steps)

    for i, ts in enumerate(timesteps):
        times[i] = ts['time']
        positions[i, 0, 0] = ts['positions']['body1']['x']
        positions[i, 0, 1] = ts['positions']['body1']['y']
        positions[i, 1, 0] = ts['positions']['body2']['x']
        positions[i, 1, 1] = ts['positions']['body2']['y']
        velocities[i, 0, 0] = ts['velocities']['body1']['x']
        velocities[i, 0, 1] = ts['velocities']['body1']['y']
        velocities[i, 1, 0] = ts['velocities']['body2']['x']
        velocities[i, 1, 1] = ts['velocities']['body2']['y']
        momentum[i, 0] = ts['momentum']['x']
        momentum[i, 1] = ts['momentum']['y']
        angular_momentum[i, 0] = ts['angular_momentum']['x']
        angular_momentum[i, 1] = ts['angular_momentum']['y']
        angular_momentum[i, 2] = ts['angular_momentum']['z']
        energy[i] = ts['energy']

    return SimulationResult(
        metadata=data['metadata'],
        times=times,
        positions=positions,
        velocities=velocities,
        momentum=momentum,
        angular_momentum=angular_momentum,
        energy=energy,
        collision=data['collision'],
    )


def load_config(filepath: str) -> dict:
    """
    Load configuration from JSON file.

    Args:
        filepath: Configuration file path

    Returns:
        Configuration dictionary
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def main() -> None:
    """Module test."""
    print("I/O handler module loaded successfully.")


if __name__ == '__main__':
    main()
