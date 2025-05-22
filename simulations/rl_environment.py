# Reinforcement Learning Environment for Mars Habitat Resource Management

import os
import numpy as np
import pandas as pd
import gym
from gym import spaces
import matplotlib.pyplot as plt
import json
import sys

# Add simulation directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulations.simulation_environment import MarsHabitatSimulation
from simulations.martian_environment import MartianEnvironmentModel

class MarsHabitatRLEnvironment(gym.Env):
    """
    Reinforcement Learning environment for Mars habitat resource management:
    - Wraps the MarsHabitatSimulation as a gym environment
    - Defines observation and action spaces
    - Provides reward function for resource optimization
    - Supports different difficulty levels and scenarios
    """
    
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, data_dir, config=None, difficulty='normal'):
        """
        Initialize the RL environment
        
        Args:
            data_dir (str): Directory containing data and configuration files
            config (dict): Configuration parameters (optional)
            difficulty (str): Difficulty level ('easy', 'normal', 'hard')
        """
        super(MarsHabitatRLEnvironment, self).__init__()
        
        self.data_dir = data_dir
        self.sim_dir = os.path.join(data_dir, "simulation")
        os.makedirs(self.sim_dir, exist_ok=True)
        
        # Set difficulty level
        self.difficulty = difficulty
        
        # Adjust config based on difficulty
        self.config = self._adjust_config_for_difficulty(config)
        
        # Create simulation
        self.simulation = MarsHabitatSimulation(data_dir, self.config)
        
        # Define action space
        # Actions:
        # 1. Power allocation to subsystems (life_support, isru, thermal_control)
        # 2. ISRU mode (water, oxygen, both, off)
        # 3. Maintenance target (power_system, life_support, isru, thermal_control, none)
        
        # Power allocation: 3 continuous values between 0-10 kW
        # ISRU mode: 4 discrete options
        # Maintenance target: 5 discrete options
        
        self.action_space = spaces.Dict({
            'power_allocation': spaces.Box(
                low=np.array([0.0, 0.0, 0.0]),
                high=np.array([10.0, 10.0, 10.0]),
                dtype=np.float32
            ),
            'isru_mode': spaces.Discrete(4),  # 0: water, 1: oxygen, 2: both, 3: off
            'maintenance_target': spaces.Discrete(5)  # 0-3: subsystems, 4: none
        })
        
        # Define observation space
        # Observations include:
        # - Time (sol, hour)
        # - Environmental conditions (temperature, dust_opacity, solar_irradiance, etc.)
        # - Habitat state (power, water, oxygen, food, spare_parts, etc.)
        # - Subsystem states (status, efficiency, maintenance_needed, etc.)
        
        # We'll use a Dict space with nested Box spaces
        
        self.observation_space = spaces.Dict({
            'time': spaces.Box(
                low=np.array([0, 0]),
                high=np.array([1000, 24]),
                dtype=np.float32
            ),
            'environment': spaces.Box(
                low=np.array([-150.0, 0.0, 0.0, 0.0, 0.0]),
                high=np.array([50.0, 1000.0, 100.0, 1.0, 1000.0]),
                dtype=np.float32
            ),  # temperature, pressure, wind_speed, dust_opacity, solar_irradiance
            'habitat': spaces.Box(
                low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, -20.0, 0.0, 0.0, 0.0]),
                high=np.array([1000.0, 10000.0, 1000.0, 10000.0, 1000.0, 50.0, 120000.0, 100.0, 10.0]),
                dtype=np.float32
            ),  # power, water, oxygen, food, spare_parts, internal_temperature, internal_pressure, internal_humidity, co2_level
            'subsystems': spaces.Box(
                low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1000.0, 100.0, 100.0, 1.0, 1.0, 10.0, 10.0]),
                dtype=np.float32
            )  # status (4), maintenance_needed (4), battery_charge, power_generation, power_consumption, heating_power
        })
        
        # Initialize state
        self.state = None
        self.steps = 0
        self.max_steps = self.config.get('max_sol', 100) * 24  # hours
        
        # Visualization
        self.fig = None
        self.ax = None
        
        print(f"Mars Habitat RL Environment initialized with difficulty: {difficulty}")
    
    def _adjust_config_for_difficulty(self, config):
        """
        Adjust configuration based on difficulty level
        
        Args:
            config (dict): Base configuration
            
        Returns:
            dict: Adjusted configuration
        """
        # Start with default config if none provided
        if config is None:
            config = {}
        
        # Create a copy to avoid modifying the original
        adjusted_config = config.copy() if config else {}
        
        # Adjust parameters based on difficulty
        if self.difficulty == 'easy':
            # More resources, lower failure rates, less extreme conditions
            adjusted_config.update({
                'habitat': {
                    'initial_resources': {
                        'power': 150,  # kWh in batteries
                        'water': 1500,  # liters
                        'oxygen': 750,  # kg
                        'food': 1500,  # kg
                        'spare_parts': 150  # units
                    }
                },
                'maintenance': {
                    'failure_rates': {
                        'power_system': 0.0005,  # probability per hour
                        'life_support': 0.001,
                        'isru': 0.0015,
                        'thermal_control': 0.0005
                    }
                },
                'dust_storm_probability': 0.005  # probability per sol
            })
        
        elif self.difficulty == 'hard':
            # Fewer resources, higher failure rates, more extreme conditions
            adjusted_config.update({
                'habitat': {
                    'initial_resources': {
                        'power': 75,  # kWh in batteries
                        'water': 750,  # liters
                        'oxygen': 350,  # kg
                        'food': 750,  # kg
                        'spare_parts': 75  # units
                    }
                },
                'maintenance': {
                    'failure_rates': {
                        'power_system': 0.002,  # probability per hour
                        'life_support': 0.004,
                        'isru': 0.006,
                        'thermal_control': 0.002
                    }
                },
                'dust_storm_probability': 0.02  # probability per sol
            })
        
        # For 'normal' difficulty, use default simulation settings
        
        return adjusted_config
    
    def reset(self):
        """
        Reset environment to initial state
        
        Returns:
            dict: Initial observation
        """
        # Reset simulation
        observation = self.simulation.reset()
        
        # Convert observation to RL state
        self.state = self._process_observation(observation)
        
        # Reset step counter
        self.steps = 0
        
        return self.state
    
    def step(self, action):
        """
        Take a step in the environment
        
        Args:
            action: Action to take
            
        Returns:
            tuple: (observation, reward, done, info)
        """
        # Process action
        sim_action = self._process_action(action)
        
        # Take step in simulation
        observation, sim_reward, done, info = self.simulation.step(sim_action)
        
        # Convert observation to RL state
        self.state = self._process_observation(observation)
        
        # Calculate reward
        reward = self._calculate_reward(observation, sim_reward)
        
        # Increment step counter
        self.steps += 1
        
        # Check if maximum steps reached
        if self.steps >= self.max_steps:
            done = True
        
        return self.state, reward, done, info
    
    def _process_action(self, action):
        """
        Process RL action into simulation action
        
        Args:
            action: RL action
            
        Returns:
            dict: Simulation action
        """
        # Extract components from action
        power_allocation = action['power_allocation']
        isru_mode_idx = action['isru_mode']
        maintenance_target_idx = action['maintenance_target']
        
        # Map ISRU mode index to string
        isru_modes = ['water', 'oxygen', 'both', 'off']
        isru_mode = isru_modes[isru_mode_idx]
        
        # Map maintenance target index to string
        maintenance_targets = ['power_system', 'life_support', 'isru', 'thermal_control', None]
        maintenance_target = maintenance_targets[maintenance_target_idx]
        
        # Create simulation action
        sim_action = {
            'power_allocation': {
                'life_support': float(power_allocation[0]),
                'isru': float(power_allocation[1]),
                'thermal_control': float(power_allocation[2])
            },
            'isru_mode': isru_mode
        }
        
        # Add maintenance target if not None
        if maintenance_target:
            sim_action['maintenance_target'] = maintenance_target
        
        return sim_action
    
    def _process_observation(self, observation):
        """
        Process simulation observation into RL state
        
        Args:
            observation: Simulation observation
            
        Returns:
            dict: RL state
        """
        # Extract components from observation
        time = observation['time']
        environment = observation['environment']
        habitat = observation['habitat']
        subsystems = observation['subsystems']
        
        # Create time array
        time_array = np.array([
            time['sol'],
            time['hour']
        ], dtype=np.float32)
        
        # Create environment array
        env_array = np.array([
            environment['temperature'],
            environment['pressure'],
            environment['wind_speed'],
            environment['dust_opacity'],
            environment['solar_irradiance']
        ], dtype=np.float32)
        
        # Create habitat array
        habitat_array = np.array([
            habitat['power'],
            habitat['water'],
            habitat['oxygen'],
            habitat['food'],
            habitat['spare_parts'],
            habitat['internal_temperature'],
            habitat['internal_pressure'],
            habitat['internal_humidity'],
            habitat['co2_level']
        ], dtype=np.float32)
        
        # Create subsystems array
        # Convert status strings to numbers
        status_map = {'operational': 1.0, 'failed': 0.0, 'disabled': 0.5}
        
        subsystems_array = np.array([
            status_map[subsystems['power_system']['status']],
            status_map[subsystems['life_support']['status']],
            status_map[subsystems['isru']['status']],
            status_map[subsystems['thermal_control']['status']],
            float(subsystems['power_system']['maintenance_needed']),
            float(subsystems['life_support']['maintenance_needed']),
            float(subsystems['isru']['maintenance_needed']),
            float(subsystems['thermal_control']['maintenance_needed']),
            subsystems['power_system']['battery_charge'],
            subsystems['power_system']['power_generation'],
            subsystems['power_system']['power_consumption'],
            subsystems['thermal_control']['heating_power']
        ], dtype=np.float32)
        
        # Create RL state
        state = {
            'time': time_array,
            'environment': env_array,
            'habitat': habitat_array,
            'subsystems': subsystems_array
        }
        
        return state
    
    def _calculate_reward(self, observation, sim_reward):
        """
        Calculate reward based on observation and simulation reward
        
        Args:
            observation: Simulation observation
            sim_reward: Reward from simulation
            
        Returns:
            float: RL reward
        """
        # Start with simulation reward
        reward = sim_reward
        
        # Add additional reward components based on RL objectives
        
        # Resource efficiency reward
        # Reward for maintaining resources at optimal levels (not too low, not too high)
        habitat = observation['habitat']
        subsystems = observation['subsystems']
        
        # Power efficiency: reward for matching generation and consumption
        power_generation = subsystems['power_system']['power_generation']
        power_consumption = subsystems['power_system']['power_consumption']
        battery_charge = subsystems['power_system']['battery_charge']
        battery_capacity = self.config.get('power_system', {}).get('battery_capacity', 150)
        
        # Optimal battery level is 40-60% of capacity
        optimal_battery_min = 0.4 * battery_capacity
        optimal_battery_max = 0.6 * battery_capacity
        
        if optimal_battery_min <= battery_charge <= optimal_battery_max:
            # Battery in optimal range
            reward += 0.1
        elif battery_charge < 0.1 * battery_capacity:
            # Battery critically low
            reward -= 0.5
        
        # Power balance reward
        power_balance = abs(power_generation - power_consumption)
        power_balance_reward = 0.1 * (1.0 - min(1.0, power_balance / 5.0))
        reward += power_balance_reward
        
        # Resource management reward
        # Reward for keeping resources in optimal ranges
        
        # Water level (optimal: 30-70% of initial)
        initial_water = self.config.get('habitat', {}).get('initial_resources', {}).get('water', 1000)
        water_level = habitat['water'] / initial_water
        
        if 0.3 <= water_level <= 0.7:
            reward += 0.05
        elif water_level < 0.1:
            reward -= 0.3
        
        # Oxygen level (optimal: 30-70% of initial)
        initial_oxygen = self.config.get('habitat', {}).get('initial_resources', {}).get('oxygen', 500)
        oxygen_level = habitat['oxygen'] / initial_oxygen
        
        if 0.3 <= oxygen_level <= 0.7:
            reward += 0.05
        elif oxygen_level < 0.1:
            reward -= 0.5
        
        # Maintenance efficiency
        # Reward for preventive maintenance, penalty for failures
        for subsystem_name, subsystem in subsystems.items():
            if subsystem['status'] == 'failed':
                reward -= 0.2
        
        # Comfort reward
        # Reward for maintaining comfortable habitat conditions
        
        # Temperature comfort (20-24°C is ideal)
        temp = habitat['internal_temperature']
        if 20 <= temp <= 24:
            reward += 0.05
        elif temp < 15 or temp > 30:
            reward -= 0.1
        
        # CO2 level (below 0.5% is ideal)
        co2 = habitat['co2_level']
        if co2 < 0.5:
            reward += 0.05
        elif co2 > 1.0:
            reward -= 0.2
        
        return reward
    
    def render(self, mode='human'):
        """
        Render the environment
        
        Args:
            mode (str): Rendering mode
            
        Returns:
            object: Rendering result
        """
        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots(2, 2, figsize=(12, 8))
            plt.ion()
            plt.show()
        
        # Clear axes
        for a in self.ax.flat:
            a.clear()
        
        # Get current observation
        observation = self.simulation._get_observation()
        
        # Plot resource levels
        self.ax[0, 0].bar(['Power', 'Water', 'Oxygen', 'Food'], 
                         [observation['habitat']['power'], 
                          observation['habitat']['water'], 
                          observation['habitat']['oxygen'],
                          observation['habitat']['food']])
        self.ax[0, 0].set_title('Resource Levels')
        
        # Plot environmental conditions
        env = observation['environment']
        self.ax[0, 1].bar(['Temp (°C)', 'Solar (W/m²/10)', 'Dust Opacity x100'], 
                         [env['temperature'], 
                          env['solar_irradiance']/10, 
                          env['dust_opacity']*100])
        self.ax[0, 1].set_title('Environmental Conditions')
        
        # Plot power balance
        power_system = observation['subsystems']['power_system']
        self.ax[1, 0].bar(['Generation', 'Consumption', 'Battery/10'], 
                         [power_system['power_generation'], 
                          power_system['power_consumption'],
                          power_system['battery_charge']/10])
        self.ax[1, 0].set_title('Power Balance (kW)')
        
        # Plot system status
        subsystems = observation['subsystems']
        status_values = []
        for name in ['power_system', 'life_support', 'isru', 'thermal_control']:
            if subsystems[name]['status'] == 'operational':
                status_values.append(1)
            elif subsystems[name]['status'] == 'disabled':
                status_values.append(0.5)
            else:
                status_values.append(0)
        
        self.ax[1, 1].bar(['Power', 'Life Support', 'ISRU', 'Thermal'], status_values)
        self.ax[1, 1].set_title('System Status')
        self.ax[1, 1].set_ylim(0, 1.2)
        
        # Add time information
        plt.suptitle(f"Sol {observation['time']['sol']}, Hour {observation['time']['hour']:.1f}")
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)
        
        if mode == 'rgb_array':
            # Convert plot to RGB array
            canvas = plt.get_current_fig_manager().canvas
            canvas.draw()
            image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(canvas.get_width_height()[::-1] + (3,))
            return image
        else:
            return self.fig
    
    def close(self):
        """
        Close environment
        """
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
    
    def seed(self, seed=None):
        """
        Set random seed
        
        Args:
            seed (int): Random seed
            
        Returns:
            list: Seed used
        """
        np.random.seed(seed)
        return [seed]
    
    def get_scenario_config(self, scenario_name):
        """
        Get configuration for a specific scenario
        
        Args:
            scenario_name (str): Scenario name
            
        Returns:
            dict: Scenario configuration
        """
        # Define scenarios
        scenarios = {
            'nominal': {},  # Default configuration
            
            'dust_storm': {
                'dust_storm_probability': 0.1,  # High probability of dust storms
                'dust_storm_duration': {
                    'min': 3,  # sols
                    'max': 15  # sols
                }
            },
            
            'resource_scarcity': {
                'habitat': {
                    'initial_resources': {
                        'power': 50,  # kWh in batteries
                        'water': 500,  # liters
                        'oxygen': 200,  # kg
                        'food': 500,  # kg
                        'spare_parts': 30  # units
                    }
                }
            },
            
            'system_failures': {
                'maintenance': {
                    'failure_rates': {
                        'power_system': 0.005,  # probability per hour
                        'life_support': 0.008,
                        'isru': 0.01,
                        'thermal_control': 0.005
                    }
                }
            },
            
            'polar_location': {
                'location': {
                    'name': "Korolev Crater",
                    'latitude': 73.0,
                    'longitude': 195.0,
                    'elevation': -2000,  # meters
                }
            }
        }
        
        # Return requested scenario config
        if scenario_name in scenarios:
            return scenarios[scenario_name]
        else:
            print(f"Warning: Scenario '{scenario_name}' not found. Using default configuration.")
            return {}
    
    def load_scenario(self, scenario_name):
        """
        Load a specific scenario
        
        Args:
            scenario_name (str): Scenario name
            
        Returns:
            dict: Initial observation
        """
        # Get scenario configuration
        scenario_config = self.get_scenario_config(scenario_name)
        
        # Update configuration
        self.config.update(scenario_config)
        
        # Recreate simulation with new config
        self.simulation = MarsHabitatSimulation(self.data_dir, self.config)
        
        # Reset environment
        return self.reset()

# Example usage
if __name__ == "__main__":
    # Create RL environment
    env = MarsHabitatRLEnvironment("/home/ubuntu/martian_habitat_pathfinder/data")
    
    # Reset environment
    observation = env.reset()
    
    # Run a few steps with random actions
    for i in range(24):  # One sol
        # Random action
        action = {
            'power_allocation': env.action_space['power_allocation'].sample(),
            'isru_mode': env.action_space['isru_mode'].sample(),
            'maintenance_target': env.action_space['maintenance_target'].sample()
        }
        
        # Take step
        observation, reward, done, info = env.step(action)
        
        # Render
        env.render()
        
        if done:
            break
    
    # Close environment
    env.close()
