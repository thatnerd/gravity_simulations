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

    Args:
        result: SimulationResult to save
        filepath: Output file path
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)

    data = {
        'metadata': result.metadata,
        'times': result.times,
        'positions': result.positions,
        'velocities': result.velocities,
        'momentum': result.momentum,
        'angular_momentum': result.angular_momentum,
        'energy': result.energy,
        'collision': result.collision,
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

    return SimulationResult(
        metadata=data['metadata'],
        times=np.array(data['times']),
        positions=np.array(data['positions']),
        velocities=np.array(data['velocities']),
        momentum=np.array(data['momentum']),
        angular_momentum=np.array(data['angular_momentum']),
        energy=np.array(data['energy']),
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
