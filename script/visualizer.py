#!/usr/bin/env python3
"""
Visualization for two-body gravitational simulation.

This module provides matplotlib-based animation of simulation results.

Usage:
    from script.visualizer import animate_simulation

    python3 script/visualizer.py --input output/simulation.json --speed 2.0
"""

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from typing import Optional
import argparse

from script.simulation import SimulationResult
from script.io_handler import load_results


# Color scheme
BODY1_COLOR = '#E63946'        # Red
BODY1_TRAIL_COLOR = '#E6394640'  # Red with alpha
BODY2_COLOR = '#457B9D'        # Blue
BODY2_TRAIL_COLOR = '#457B9D40'  # Blue with alpha
BACKGROUND_COLOR = '#1D3557'   # Dark blue
GRID_COLOR = '#A8DADC'         # Light blue-green


def setup_figure(
    positions: NDArray[np.float64],
    margin: float = 1.2
) -> tuple[plt.Figure, plt.Axes]:
    """
    Set up the matplotlib figure and axes.

    Args:
        positions: Position array of shape (N, 2, 2)
        margin: Margin factor for axis limits

    Returns:
        Tuple of (figure, axes)
    """
    # Find bounds
    all_positions = positions.reshape(-1, 2)
    x_min, x_max = all_positions[:, 0].min(), all_positions[:, 0].max()
    y_min, y_max = all_positions[:, 1].min(), all_positions[:, 1].max()

    # Make square and add margin
    x_range = x_max - x_min
    y_range = y_max - y_min
    max_range = max(x_range, y_range) * margin

    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2

    fig, ax = plt.subplots(figsize=(10, 10), facecolor=BACKGROUND_COLOR)
    ax.set_facecolor(BACKGROUND_COLOR)

    ax.set_xlim(x_center - max_range/2, x_center + max_range/2)
    ax.set_ylim(y_center - max_range/2, y_center + max_range/2)

    ax.set_aspect('equal')
    ax.grid(True, color=GRID_COLOR, alpha=0.3, linestyle='--')

    ax.set_xlabel('x (m)', color='white')
    ax.set_ylabel('y (m)', color='white')
    ax.tick_params(colors='white')

    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)

    return fig, ax


def animate_simulation(
    result: SimulationResult,
    speed_multiplier: float = 1.0,
    save_path: Optional[str] = None,
    body_size: float = 0.02
) -> Optional[animation.FuncAnimation]:
    """
    Create and display animation of simulation results.

    Args:
        result: SimulationResult to animate
        speed_multiplier: Playback speed multiplier (1.0 = real-time)
        save_path: If provided, save animation to this file
        body_size: Size of bodies as fraction of plot size

    Returns:
        FuncAnimation object, or None if saved to file
    """
    positions = result.positions
    times = result.times
    dt = result.metadata['dt']
    n_frames = len(times)

    fig, ax = setup_figure(positions)

    # Calculate body display sizes
    x_lim = ax.get_xlim()
    plot_size = x_lim[1] - x_lim[0]
    radius = plot_size * body_size

    # Create trail lines
    trail1, = ax.plot([], [], color=BODY1_TRAIL_COLOR, linewidth=1.5)
    trail2, = ax.plot([], [], color=BODY2_TRAIL_COLOR, linewidth=1.5)

    # Create body circles
    body1 = Circle((0, 0), radius, color=BODY1_COLOR, zorder=10)
    body2 = Circle((0, 0), radius, color=BODY2_COLOR, zorder=10)
    ax.add_patch(body1)
    ax.add_patch(body2)

    # Create center of mass marker
    com_marker, = ax.plot([], [], 'w+', markersize=10, markeredgewidth=2)

    # Time display
    time_text = ax.text(
        0.02, 0.98, '', transform=ax.transAxes,
        color='white', fontsize=12, verticalalignment='top',
        fontfamily='monospace'
    )

    # Energy display
    energy_text = ax.text(
        0.02, 0.92, '', transform=ax.transAxes,
        color='white', fontsize=10, verticalalignment='top',
        fontfamily='monospace'
    )

    # Title
    m1 = result.metadata['m1']
    m2 = result.metadata['m2']
    ax.set_title(
        f'Two-Body Gravitational Simulation\nm₁ = {m1:.2e} kg, m₂ = {m2:.2e} kg',
        color='white', fontsize=12
    )

    def init():
        """Initialize animation."""
        trail1.set_data([], [])
        trail2.set_data([], [])
        body1.center = (positions[0, 0, 0], positions[0, 0, 1])
        body2.center = (positions[0, 1, 0], positions[0, 1, 1])
        time_text.set_text('')
        energy_text.set_text('')
        com_marker.set_data([], [])
        return trail1, trail2, body1, body2, time_text, energy_text, com_marker

    def update(frame: int):
        """Update animation frame."""
        # Update trails (show all history up to current frame)
        trail1.set_data(positions[:frame+1, 0, 0], positions[:frame+1, 0, 1])
        trail2.set_data(positions[:frame+1, 1, 0], positions[:frame+1, 1, 1])

        # Update body positions
        body1.center = (positions[frame, 0, 0], positions[frame, 0, 1])
        body2.center = (positions[frame, 1, 0], positions[frame, 1, 1])

        # Update time display
        time_text.set_text(f't = {times[frame]:.2f} s')

        # Update energy display
        E = result.energy[frame]
        E0 = result.energy[0]
        dE = (E - E0) / abs(E0) * 100 if E0 != 0 else 0
        energy_text.set_text(f'E = {E:.4e} J\nΔE/E₀ = {dE:.6f}%')

        # Center of mass (should be at origin)
        M = m1 + m2
        com_x = (m1 * positions[frame, 0, 0] + m2 * positions[frame, 1, 0]) / M
        com_y = (m1 * positions[frame, 0, 1] + m2 * positions[frame, 1, 1]) / M
        com_marker.set_data([com_x], [com_y])

        return trail1, trail2, body1, body2, time_text, energy_text, com_marker

    # Calculate frame interval for desired playback speed
    # real_time_interval = dt * 1000 ms per frame
    # adjusted_interval = real_time_interval / speed_multiplier
    interval = max(1, int(dt * 1000 / speed_multiplier))

    anim = animation.FuncAnimation(
        fig, update, init_func=init,
        frames=n_frames, interval=interval, blit=True
    )

    if save_path:
        print(f"Saving animation to {save_path}...")
        if save_path.endswith('.gif'):
            writer = animation.PillowWriter(fps=int(1000/interval))
        else:
            writer = animation.FFMpegWriter(fps=int(1000/interval))
        anim.save(save_path, writer=writer)
        plt.close(fig)
        print("Animation saved.")
        return None
    else:
        plt.show()
        return anim


def main() -> None:
    """Run visualizer from command line."""
    parser = argparse.ArgumentParser(description='Visualize two-body simulation results')
    parser.add_argument('--input', '-i', required=True, help='Input JSON file')
    parser.add_argument('--speed', '-s', type=float, default=1.0, help='Playback speed multiplier')
    parser.add_argument('--save', help='Save animation to file (mp4 or gif)')
    parser.add_argument('--body-size', type=float, default=0.02, help='Body size as fraction of plot')

    args = parser.parse_args()

    result = load_results(args.input)
    print(f"Loaded simulation: {result.metadata['n_steps']} steps")
    print(f"Time range: 0 to {result.times[-1]:.2f} s")

    if result.collision:
        print("WARNING: Simulation ended due to collision!")

    animate_simulation(
        result,
        speed_multiplier=args.speed,
        save_path=args.save,
        body_size=args.body_size
    )


if __name__ == '__main__':
    main()
