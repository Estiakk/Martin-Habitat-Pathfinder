# Martian Habitat Pathfinder User Guide

## Introduction

Welcome to the Martian Habitat Pathfinder User Guide. This comprehensive guide will walk you through how to use the Martian Habitat Pathfinder system for AI-driven resource management in Mars habitats. The system combines sophisticated AI approaches with human oversight through an intuitive interface to optimize resource allocation and ensure habitat sustainability.

## System Requirements

- Python 3.11 or higher
- Required Python packages:
  - numpy
  - pandas
  - matplotlib
  - torch
  - dash
  - plotly
  - flask
  - scikit-learn
  - gym

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

4. Create configuration file:
```bash
# Example configuration
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

## System Architecture

The Martian Habitat Pathfinder system consists of the following components:

1. **Data Infrastructure**: Data acquisition, preprocessing, feature engineering, and fusion
2. **Simulation Environment**: Mars habitat simulation with environmental modeling
3. **AI Implementation**: Reinforcement learning models and predictive analytics
4. **Decision Integration**: Integration of AI components for robust decision-making
5. **Human-AI Interface**: Dashboard for monitoring and controlling habitat resources

## Using the Simulation Environment

The simulation environment provides a realistic model of a Mars habitat, including resource dynamics, environmental conditions, and system failures.

### Running a Basic Simulation

```python
from simulations.rl_environment import MarsHabitatRLEnvironment
from simulations.rl_formulation import MarsHabitatRLFormulation

# Create RL formulation
formulation = MarsHabitatRLFormulation("path/to/data")

# Create environment
env = formulation.create_environment(difficulty='normal')

# Reset environment
state = env.reset()

# Run simulation for 10 steps
for i in range(10):
    # Define action
    action = {
        'power_allocation': {
            'life_support': 4.0,
            'isru': 3.0,
            'thermal_control': 3.0
        },
        'isru_mode': 'both',
        'maintenance_target': None
    }
    
    # Take action
    next_state, reward, done, info = env.step(action)
    
    # Print state
    print(f"Step {i+1}:")
    print(f"  Time: Sol {next_state['time'][0]}, Hour {next_state['time'][1]}")
    print(f"  Power: {next_state['habitat']['power']:.2f}")
    print(f"  Water: {next_state['habitat']['water']:.2f}")
    print(f"  Oxygen: {next_state['habitat']['oxygen']:.2f}")
    print(f"  Food: {next_state['habitat']['food']:.2f}")
    print(f"  Reward: {reward:.2f}")
    print(f"  Done: {done}")
    
    # Update state
    state = next_state
    
    # Check if simulation is done
    if done:
        print("Simulation ended early")
        break
```

### Customizing the Environment

You can customize the environment by modifying the configuration file or by passing parameters to the `create_environment` method:

```python
# Create environment with custom difficulty
env = formulation.create_environment(difficulty='hard')

# Create environment with custom initial conditions
env = formulation.create_environment(
    initial_power=200,
    initial_water=1000,
    initial_oxygen=500,
    initial_food=800
)

# Create environment with custom environmental conditions
env = formulation.create_environment(
    dust_storm_probability=0.2,
    temperature_range=[-150, 0],
    solar_irradiance_range=[0, 400]
)
```

## Using the AI Models

The Martian Habitat Pathfinder system includes several AI models for resource management:

### Baseline Reinforcement Learning Models

```python
from models.baseline_rl import MarsHabitatBaselineRL

# Create baseline RL model
baseline_rl = MarsHabitatBaselineRL("path/to/data")

# Load pre-trained model
dqn_model = baseline_rl.load_model("dqn_isru_mode_final.pth")

# Use model to select action
state_array = baseline_rl.preprocess_state(state)
action = baseline_rl.select_action(dqn_model, state_array, "isru_mode")
env_action = baseline_rl.convert_action_to_env_action(action, "isru_mode")

# Take action in environment
next_state, reward, done, info = env.step(env_action)
```

### Advanced Reinforcement Learning Models

```python
from models.advanced_rl import MarsHabitatAdvancedRL

# Create advanced RL model
advanced_rl = MarsHabitatAdvancedRL("path/to/data")

# Use hierarchical RL
hrl_action = advanced_rl.select_hrl_action(state)

# Use multi-agent RL
marl_action = advanced_rl.select_marl_action(state)

# Take action in environment
next_state, reward, done, info = env.step(hrl_action)
```

### Predictive Analytics

```python
from analytics.predictive_analytics import MarsHabitatPredictiveAnalytics

# Create predictive analytics
analytics = MarsHabitatPredictiveAnalytics("path/to/data")

# Load models
forecasters, anomaly_detectors, optimizer = analytics.load_models()

# Run analytics pipeline
results = analytics.run_analytics_pipeline(state)

# Get forecasts
forecasts = results['forecasting']
print(f"Power forecast (24h): {forecasts['power']}")

# Get anomaly detection results
anomalies = results['anomaly_detection']
print(f"Water anomaly detected: {anomalies['water']['anomaly']}")

# Get resource optimization
optimization = results['optimization']
print(f"Recommended power allocation: {optimization['power_allocation']}")
```

### Decision Integration

```python
from models.decision_integration import DecisionIntegrationSystem

# Create decision integration system
integration = DecisionIntegrationSystem("path/to/data")

# Make integrated decision
decision = integration.make_decision(state)

# Take action in environment
next_state, reward, done, info = env.step(decision)
```

## Using the Human-AI Interface

The Human-AI Interface provides a comprehensive dashboard for monitoring and controlling habitat resources and systems.

### Starting the Dashboard

```bash
cd path/to/martian-habitat-pathfinder
python ui/dashboard.py
```

This will start the dashboard server on http://127.0.0.1:8050/. Open this URL in your web browser to access the dashboard.

### Dashboard Components

The dashboard consists of the following components:

1. **Header**: Dashboard title and control buttons
2. **Status Bar**: Current sol, hour, and habitat status
3. **Resources Panel**: Graph of resource levels and current resource levels
4. **Environment & Systems Panel**: Graphs of environmental conditions and subsystem status
5. **Decision Support Panel**: AI recommendations, manual controls, forecasting, and anomaly detection

### Using the Dashboard

#### Simulation Control

1. **Step Simulation**: Click the "Step Simulation" button to advance the simulation by one time step
2. **Reset Simulation**: Click the "Reset Simulation" button to reset the simulation to initial conditions
3. **Auto-Pilot**: Click the "Auto-Pilot" button to toggle automatic stepping of the simulation

#### Decision Making

1. **AI Recommendations**: View suggested resource allocations in the "Recommendations" tab
2. **Manual Control**: Adjust power allocation, ISRU mode, and maintenance target using the sliders and radio buttons
3. **Apply Manual Settings**: Click the "Apply Manual Settings" button to apply your manual settings to the simulation

#### Monitoring

1. **Resource Levels**: Monitor current and historical resource levels in the "Resources" panel
2. **Environmental Conditions**: Track changes in the Martian environment in the "Environment" panel
3. **Subsystem Status**: Monitor the operational status of habitat subsystems in the "Subsystems" panel

#### Forecasting and Anomaly Detection

1. **Resource Forecasts**: View predicted resource levels in the "Forecasting" tab
2. **Forecast Horizon**: Adjust the prediction time horizon using the slider
3. **Anomaly Detection**: Monitor for potential system anomalies in the "Anomaly Detection" tab

## Example Scenarios

### Scenario 1: Managing a Dust Storm

Dust storms on Mars can significantly reduce solar power generation. Here's how to manage a dust storm using the Martian Habitat Pathfinder system:

1. **Monitor Environmental Conditions**: Watch for increasing dust opacity in the "Environment" panel
2. **Check Forecasts**: View resource forecasts to see the predicted impact on power levels
3. **Adjust Power Allocation**: Reduce power to non-essential systems (e.g., ISRU) and prioritize life support
4. **Change ISRU Mode**: Set ISRU mode to "Off" or "Water" (which uses less power than oxygen production)
5. **Monitor Resource Levels**: Keep a close eye on power levels to ensure they don't drop too low

### Scenario 2: Long-Term Resource Planning

For long-term resource planning, follow these steps:

1. **Run Simulation in Auto-Pilot**: Let the simulation run for several sols to establish baseline resource dynamics
2. **Analyze Resource Trends**: Look for patterns in resource consumption and generation
3. **Check Forecasts**: Use the forecasting feature to predict future resource levels
4. **Adjust Allocation Strategy**: Based on forecasts, adjust your resource allocation strategy
5. **Monitor and Refine**: Continuously monitor resource levels and refine your strategy as needed

### Scenario 3: Handling System Failures

System failures can occur randomly in the simulation. Here's how to handle them:

1. **Monitor Subsystem Status**: Watch for changes in subsystem status in the "Subsystems" panel
2. **Check Anomaly Detection**: Look for alerts in the "Anomaly Detection" tab
3. **Prioritize Maintenance**: Set the maintenance target to the failed subsystem
4. **Adjust Power Allocation**: Allocate more power to the subsystem being repaired
5. **Monitor Repair Progress**: Watch the subsystem status to see when it returns to operational

## Troubleshooting

### Common Issues

1. **Dashboard Not Starting**:
   - Check that all required packages are installed
   - Ensure the port (default: 8050) is not in use
   - Check for error messages in the console

2. **Simulation Crashing**:
   - Check that the configuration file is properly formatted
   - Ensure all required directories exist
   - Look for error messages in the console

3. **Models Not Loading**:
   - Verify that model files exist in the correct location
   - Check that the model format matches what the code expects
   - Ensure you have the correct version of PyTorch installed

### Getting Help

If you encounter issues not covered in this guide, please:

1. Check the documentation in the `docs` directory
2. Look for error messages in the console
3. Contact the development team at support@martian-habitat-pathfinder.org

## Advanced Topics

### Training Your Own Models

You can train your own reinforcement learning models using the provided training scripts:

```bash
cd path/to/martian-habitat-pathfinder
python models/train_baseline_rl.py --model dqn --episodes 1000
```

### Extending the System

The Martian Habitat Pathfinder system is designed to be extensible. Here are some ways you can extend it:

1. **Add New Models**: Create new AI models in the `models` directory
2. **Add New Analytics**: Implement new analytics in the `analytics` directory
3. **Enhance the Simulation**: Add new features to the simulation environment
4. **Improve the Dashboard**: Add new visualizations or controls to the dashboard

### Integration with Other Systems

The Martian Habitat Pathfinder system can be integrated with other systems:

1. **Hardware-in-the-Loop**: Connect to physical hardware for testing
2. **Mission Control Systems**: Integrate with mission control software
3. **VR/AR Interfaces**: Connect to virtual or augmented reality interfaces

## Conclusion

The Martian Habitat Pathfinder system provides a comprehensive solution for AI-driven resource management in Mars habitats. By following this guide, you should be able to use all aspects of the system, from running simulations to interacting with the dashboard interface.

Remember that the key to successful habitat management is balancing resource generation and consumption while adapting to changing environmental conditions and system states. The AI components can provide valuable recommendations, but human judgment and expertise are still essential for optimal decision-making.

Happy exploring on Mars!
