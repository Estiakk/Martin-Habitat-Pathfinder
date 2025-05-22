# Quick Start Guide: Martian Habitat Pathfinder

This quick start guide provides essential instructions for getting started with the Martian Habitat Pathfinder system.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-organization/martian-habitat-pathfinder.git
cd martian-habitat-pathfinder
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Set up the data directory:
```bash
mkdir -p data/models data/analytics data/integration data/validation data/ui
```

4. Create a basic configuration file:
```bash
# Create a basic config file
cat > data/config.json << EOF
{
  "environment": {
    "temperature_range": [-120, 20],
    "pressure_range": [600, 700],
    "dust_opacity_range": [0.1, 0.9],
    "solar_irradiance_range": [0, 600]
  },
  "habitat": {
    "initial_power": 100,
    "initial_water": 500,
    "initial_oxygen": 200,
    "initial_food": 300,
    "initial_spare_parts": 50
  },
  "simulation": {
    "max_steps": 500,
    "difficulty": "normal"
  }
}
EOF
```

## Running the Dashboard

The easiest way to get started is to run the dashboard:

```bash
cd path/to/martian-habitat-pathfinder
python ui/dashboard.py
```

This will start the dashboard server on http://127.0.0.1:8050/. Open this URL in your web browser.

## Basic Dashboard Controls

1. **Step Simulation**: Click to advance one time step
2. **Reset Simulation**: Click to reset to initial conditions
3. **Auto-Pilot**: Click to toggle automatic simulation

## Making Decisions

1. **View AI Recommendations**: See suggested resource allocations
2. **Manual Control**: Adjust power allocation, ISRU mode, and maintenance
3. **Apply Settings**: Click to apply your manual settings

## Monitoring Resources

1. **Resource Panel**: Track power, water, oxygen, and food levels
2. **Environment Panel**: Monitor Martian environmental conditions
3. **Subsystems Panel**: Check status of habitat subsystems

## Next Steps

For detailed instructions, refer to the comprehensive User Guide in the docs directory.
