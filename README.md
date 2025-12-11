# Two-Body Gravitational Simulation

2D gravitational simulation of two bodies orbiting their common center of mass. Uses RK4 numerical integration with type-safe SI units.

## Install

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Run with random parameters
python3 script/two_body_sim.py

# Specify masses and orbital period
python3 script/two_body_sim.py --m1 1e12 --m2 2e12 --period 2.5

# Headless mode (no animation)
python3 script/two_body_sim.py --no-animate --output output/sim.json

# Playback saved results
python3 script/two_body_sim.py --playback output/sim.json --speed 10

# Save animation to video
python3 script/two_body_sim.py --playback output/sim.json --save-video output/sim.mp4
```

## Project Structure

```
script/
  units.py              # Type-safe physical quantities (Mass, Position, Velocity, etc.)
  physics.py            # Force, energy, momentum calculations
  integrator.py         # RK4 stepper, collision detection
  initial_conditions.py # Circular orbit generation
  simulation.py         # Main simulation loop
  io_handler.py         # JSON save/load
  visualizer.py         # Matplotlib animation
  two_body_sim.py       # CLI entry point
test/
  test_two_body_sim.py  # Integration tests
  test_units.py         # Unit type tests
```

## Tests

```bash
python3 -m pytest test/ -v
```

## Type System

Physical quantities are wrapped in type-safe classes that enforce dimensional correctness:

```python
from script.units import Mass, Time, position, velocity

m = Mass(1e12)           # kg
v = velocity(10.0, 0.0)  # m/s
p = m * v                # Returns Momentum, not just an array
```

Operations return appropriate types: `Force / Mass` returns `Acceleration`, `Position.cross(Momentum)` returns `AngularMomentum`, etc.
