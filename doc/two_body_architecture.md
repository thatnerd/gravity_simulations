# Two-Body Gravitational Simulation - Architecture Document

## Overview

This simulation models two gravitationally interacting bodies in 2D space using SI units.
The system uses RK4 numerical integration and records all physical quantities for analysis.

## Requirements

### Physical Model
- Two bodies with masses m1 and m2 (differing by factor < 10)
- Gravitational constant G = 6.67430e-11 m³/(kg·s²)
- Initial conditions: zero total momentum, stable orbit
- Orbital period: 1-4 seconds (configurable)
- Total simulation time: 100 seconds (default)
- Time step: 0.01 seconds (default)

### Collision Detection
- Monitor acceleration magnitude at each step
- Stop simulation if acceleration exceeds stability threshold
- Threshold: when |a| > 1e12 m/s² (numerical instability indicator)

### Output Data (JSON)
- Metadata: masses, initial conditions, parameters
- Time series: t, positions, velocities
- Conserved quantities: total momentum, angular momentum, total energy
- (Kinetic and potential energy derivable from positions/velocities)

### Visualization
- Fixed frame (center of mass at origin)
- Each mass has distinct color
- Path history in muted color
- Configurable playback speed

## Module Structure

### 1. `script/two_body_sim.py` - Main entry point
Handles CLI parsing with docopt, config file loading, orchestrates simulation.

### 2. `script/physics.py` - Physics engine
Core physics calculations:
- `gravitational_force(m1, m2, r1, r2)` - Newton's law of gravitation
- `compute_accelerations(state, m1, m2)` - acceleration vectors for both bodies
- `total_momentum(state, m1, m2)` - vector sum of momenta
- `angular_momentum(state, m1, m2)` - L = r × p for system
- `kinetic_energy(state, m1, m2)` - sum of 0.5*m*v²
- `potential_energy(state, m1, m2)` - -G*m1*m2/r
- `total_energy(state, m1, m2)` - KE + PE

### 3. `script/integrator.py` - Numerical integration
- `rk4_step(state, dt, m1, m2)` - single RK4 step for two-body system
- `derivatives(state, m1, m2)` - computes [velocities, accelerations]
- `check_stability(state, m1, m2, threshold)` - collision/instability detection

### 4. `script/initial_conditions.py` - Setup helpers
- `generate_circular_orbit(m1, m2, period)` - create stable circular orbit
- `generate_random_masses(ratio_max=10)` - random masses within ratio constraint
- `compute_orbital_parameters(m1, m2, period)` - derive r, v from period

### 5. `script/simulation.py` - Main simulation loop
- `run_simulation(initial_state, m1, m2, dt, t_max)` - main loop
- Returns: SimulationResult with all recorded data

### 6. `script/io_handler.py` - File I/O
- `save_results(result, filename)` - write JSON output
- `load_config(filename)` - read JSON config file
- `load_results(filename)` - read results for playback

### 7. `script/visualizer.py` - Matplotlib animation
- `animate_simulation(result, speed_multiplier)` - create animation
- `setup_figure()` - configure plot aesthetics
- `update_frame(frame_num)` - animation callback

## Data Structures

### State Vector
```python
# Shape: (2, 2, 2) - [body_index, position/velocity, x/y]
# Or flattened: (8,) for integrator
state = np.array([
    [[x1, y1], [vx1, vy1]],  # body 1: position, velocity
    [[x2, y2], [vx2, vy2]]   # body 2: position, velocity
])
```

### SimulationResult
```python
@dataclass
class SimulationResult:
    metadata: dict           # masses, G, dt, etc.
    times: np.ndarray       # shape (N,)
    positions: np.ndarray   # shape (N, 2, 2) - [step, body, xy]
    velocities: np.ndarray  # shape (N, 2, 2)
    momentum: np.ndarray    # shape (N, 2) - total momentum vector
    angular_momentum: np.ndarray  # shape (N,) - scalar (2D)
    energy: np.ndarray      # shape (N,) - total energy
    collision: bool         # whether simulation ended due to collision
```

## CLI Interface (docopt)

```
Two-Body Gravitational Simulation

Usage:
    two_body_sim.py [options]
    two_body_sim.py --config <config_file>
    two_body_sim.py (-h | --help)

Options:
    -h --help                   Show this help message
    --config <config_file>      Load parameters from JSON config file
    --m1 <mass1>               Mass of body 1 in kg [default: random]
    --m2 <mass2>               Mass of body 2 in kg [default: random]
    --period <seconds>         Orbital period in seconds [default: random 1-4]
    --duration <seconds>       Total simulation time [default: 100]
    --dt <seconds>             Time step [default: 0.01]
    --output <file>            Output JSON file [default: output/simulation.json]
    --no-animate               Skip animation, only save data
    --speed <multiplier>       Playback speed multiplier [default: 1.0]
    --save-video <file>        Save animation to video file
```

## Physical Derivations

### Circular Orbit Setup
For two bodies in circular orbit around their center of mass:
- Total mass: M = m1 + m2
- Reduced mass: μ = m1*m2/M
- Orbital angular frequency: ω = 2π/T
- Separation distance: r = (G*M*T²/(4π²))^(1/3)  [Kepler's 3rd law]
- Distance from COM: r1 = r*m2/M, r2 = r*m1/M
- Orbital velocities: v1 = ω*r1, v2 = ω*r2 (perpendicular to radius)
- Bodies move in opposite directions to ensure zero total momentum

### Conservation Laws
- Total momentum: p = m1*v1 + m2*v2 = 0 (by construction)
- Angular momentum: L = m1*(r1 × v1) + m2*(r2 × v2)
- Total energy: E = KE + PE = 0.5*m1*v1² + 0.5*m2*v2² - G*m1*m2/r

## Testing Strategy

### Unit Tests (`test/test_two_body_sim.py`)
1. Physics functions:
   - Gravitational force magnitude and direction
   - Energy conservation for known orbits
   - Momentum calculation
   - Angular momentum calculation

2. Integrator:
   - RK4 accuracy for simple harmonic oscillator (known solution)
   - RK4 accuracy for circular orbit (should return to start after period)
   - Stability detection triggers correctly

3. Initial conditions:
   - Generated orbit has correct period
   - Total momentum is zero
   - Masses within specified ratio

4. I/O:
   - Round-trip JSON save/load preserves data

### Integration Tests
1. Full simulation of circular orbit:
   - Energy conserved to within tolerance
   - Angular momentum conserved
   - Returns to initial position after one period

2. Collision detection:
   - Highly eccentric orbit triggers collision detection
