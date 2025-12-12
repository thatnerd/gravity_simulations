## Instructions for Claude
- Assume that the user is an experienced Python programmer
- Use Python 3 for all code unless instructed otherwise
- Put script files in the script/ directory
- Put all test files in the test/ directory
- Begin each file with a shebang
- Use docopt in every file for clarity, and a main() function
- Use type hints for every function
- Prioritize performance optimizations
- Use test-driven development wherever possible:
  - Begin with an architecture planning document
  - Write tests for functions and stub out functions initially, and confirm that the tests fail
  - After writing functions, run relevant tests and confirm they run correctly
  - Run the full test suite after any changes

# Project: Two-Body Gravitational Simulation

## Project Description
A 2D gravitational simulation of two bodies orbiting their common center of mass. Supports Yoshida 4th-order symplectic integration (default) and RK4, with SI units. Records orbital data to JSON and provides matplotlib visualization with playback controls.

## Tech Stack
- Python 3
- NumPy (numerical computation)
- Matplotlib (visualization/animation)
- docopt (CLI argument parsing)
- pytest (testing)

## Dependencies to Install
```bash
pip install numpy matplotlib docopt pytest
```

## Code Conventions
- Type hints on all functions
- Docstrings with Args/Returns documentation
- State vector shape: `(2, 2, 2)` = `[body_index, position_or_velocity, x_or_y]`
- SI units throughout (kg, m, s, J)

## Project Structure
```
gravity_simulations/
├── script/
│   ├── __init__.py
│   ├── physics.py           # Force, energy, momentum calculations
│   ├── integrator.py        # Yoshida/RK4 steppers, stability check
│   ├── initial_conditions.py # Circular orbit generation
│   ├── simulation.py        # Main loop, SimulationResult dataclass
│   ├── io_handler.py        # JSON save/load
│   ├── visualizer.py        # Matplotlib animation
│   └── two_body_sim.py      # Main CLI entry point
├── test/
│   ├── test_two_body_sim.py # Unit & integration tests
│   └── (old test files)     # Legacy, can be ignored
├── config/
│   └── example_config.json  # Sample configuration
├── doc/
│   └── two_body_architecture.md # Design document
├── output/                  # Simulation output files (gitignored)
├── .gitignore
├── CLAUDE.md
└── README.md
```

## Usage
```bash
# Run with random parameters (uses Yoshida integrator by default)
python3 script/two_body_sim.py

# Run with specific masses and period
python3 script/two_body_sim.py --m1 1e12 --m2 2e12 --period 2.5

# Use RK4 integrator instead of Yoshida
python3 script/two_body_sim.py --integrator rk4

# Load from config file
python3 script/two_body_sim.py --config config/example_config.json

# Headless mode (no animation)
python3 script/two_body_sim.py --no-animate --output output/my_sim.json

# Playback saved results
python3 script/two_body_sim.py --playback output/simulation.json --speed 10

# Save animation to video
python3 script/two_body_sim.py --playback output/simulation.json --save-video output/sim.mp4
```

## Current Status
- All modules implemented and working
- All 46 tests pass
- Yoshida 4th-order symplectic integrator implemented as default
- RK4 available as alternative via `--integrator rk4`
- Elliptical orbits fully supported via `--eccentricity` and `--angle`
- Visualization untested (requires display or video export)

## Integrators
- **Yoshida (default)**: 4th-order symplectic integrator. Preserves phase-space volume, exhibits bounded energy oscillations rather than secular drift. Ideal for long-term orbital simulations.
- **RK4**: Classical 4th-order Runge-Kutta. Better local accuracy but exhibits linear energy drift over time.

## Known Issues
- Visualization requires either a display or video export (headless VM limitation)
- Old test files in test/ directory are from previous project version

## Future Plans
- Add 3-body simulation
- Add real-time energy/momentum plots alongside animation
- Performance optimization for longer simulations

