# Reinforcement Learning Problem Formulation for Mars Habitat Resource Management

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import sys

# Add simulation directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulations.rl_environment import MarsHabitatRLEnvironment

class MarsHabitatRLFormulation:
    """
    Reinforcement Learning problem formulation for Mars habitat resource management:
    - Defines the MDP (Markov Decision Process) components
    - Provides reward shaping strategies
    - Implements curriculum learning scenarios
    - Supports evaluation metrics and benchmarks
    """
    
    def __init__(self, data_dir):
        """
        Initialize the RL problem formulation
        
        Args:
            data_dir (str): Directory containing data and configuration files
        """
        self.data_dir = data_dir
        self.rl_dir = os.path.join(data_dir, "rl")
        os.makedirs(self.rl_dir, exist_ok=True)
        
        # Define MDP components
        self.mdp_components = {
            "state_space": {
                "time": ["sol", "hour"],
                "environment": ["temperature", "pressure", "wind_speed", "dust_opacity", "solar_irradiance"],
                "habitat": ["power", "water", "oxygen", "food", "spare_parts", "internal_temperature", 
                           "internal_pressure", "internal_humidity", "co2_level"],
                "subsystems": ["power_system_status", "life_support_status", "isru_status", "thermal_control_status",
                              "power_system_maintenance", "life_support_maintenance", "isru_maintenance", 
                              "thermal_control_maintenance", "battery_charge", "power_generation", 
                              "power_consumption", "heating_power"]
            },
            "action_space": {
                "power_allocation": ["life_support", "isru", "thermal_control"],
                "isru_mode": ["water", "oxygen", "both", "off"],
                "maintenance_target": ["power_system", "life_support", "isru", "thermal_control", "none"]
            },
            "reward_components": {
                "resource_levels": {
                    "description": "Reward for maintaining optimal resource levels",
                    "weight": 0.4
                },
                "system_health": {
                    "description": "Reward for maintaining operational systems",
                    "weight": 0.3
                },
                "comfort": {
                    "description": "Reward for maintaining comfortable habitat conditions",
                    "weight": 0.1
                },
                "efficiency": {
                    "description": "Reward for efficient resource utilization",
                    "weight": 0.2
                }
            },
            "transition_dynamics": {
                "description": "Transitions are determined by the simulation environment",
                "stochasticity_sources": ["equipment failures", "dust storms", "environmental variations"]
            }
        }
        
        # Define curriculum learning stages
        self.curriculum_stages = [
            {
                "name": "basic_operations",
                "description": "Learn basic habitat operations without failures",
                "difficulty": "easy",
                "duration_sols": 10,
                "failure_rates_multiplier": 0.0,
                "dust_storm_probability": 0.0
            },
            {
                "name": "resource_management",
                "description": "Learn to manage resources efficiently",
                "difficulty": "easy",
                "duration_sols": 20,
                "failure_rates_multiplier": 0.2,
                "dust_storm_probability": 0.005
            },
            {
                "name": "system_failures",
                "description": "Learn to handle system failures",
                "difficulty": "normal",
                "duration_sols": 30,
                "failure_rates_multiplier": 1.0,
                "dust_storm_probability": 0.01
            },
            {
                "name": "extreme_conditions",
                "description": "Learn to handle extreme conditions and multiple failures",
                "difficulty": "hard",
                "duration_sols": 50,
                "failure_rates_multiplier": 1.5,
                "dust_storm_probability": 0.02
            }
        ]
        
        # Define evaluation scenarios
        self.evaluation_scenarios = [
            {
                "name": "nominal_operations",
                "description": "Normal operations without extreme events",
                "duration_sols": 30,
                "config_overrides": {}
            },
            {
                "name": "dust_storm_season",
                "description": "Extended period with high dust storm probability",
                "duration_sols": 30,
                "config_overrides": {
                    "dust_storm_probability": 0.05
                }
            },
            {
                "name": "resource_scarcity",
                "description": "Limited initial resources",
                "duration_sols": 30,
                "config_overrides": {
                    "habitat": {
                        "initial_resources": {
                            "power": 50,
                            "water": 500,
                            "oxygen": 200,
                            "food": 500,
                            "spare_parts": 30
                        }
                    }
                }
            },
            {
                "name": "system_failures",
                "description": "High probability of system failures",
                "duration_sols": 30,
                "config_overrides": {
                    "maintenance": {
                        "failure_rates": {
                            "power_system": 0.005,
                            "life_support": 0.008,
                            "isru": 0.01,
                            "thermal_control": 0.005
                        }
                    }
                }
            },
            {
                "name": "polar_mission",
                "description": "Mission at polar location with extreme temperature variations",
                "duration_sols": 30,
                "config_overrides": {
                    "location": {
                        "name": "Korolev Crater",
                        "latitude": 73.0,
                        "longitude": 195.0,
                        "elevation": -2000
                    }
                }
            }
        ]
        
        # Define evaluation metrics
        self.evaluation_metrics = {
            "survival_rate": {
                "description": "Percentage of episodes where habitat remains operational for full duration",
                "higher_is_better": True
            },
            "resource_efficiency": {
                "description": "Average resource levels maintained throughout mission",
                "higher_is_better": True
            },
            "comfort_index": {
                "description": "Average comfort conditions maintained throughout mission",
                "higher_is_better": True
            },
            "maintenance_efficiency": {
                "description": "Ratio of preventive to emergency maintenance actions",
                "higher_is_better": True
            },
            "power_balance": {
                "description": "Average alignment between power generation and consumption",
                "higher_is_better": True
            }
        }
        
        print(f"Mars Habitat RL Problem Formulation initialized")
    
    def create_environment(self, difficulty='normal', scenario=None):
        """
        Create RL environment with specified difficulty and scenario
        
        Args:
            difficulty (str): Difficulty level ('easy', 'normal', 'hard')
            scenario (str): Scenario name (optional)
            
        Returns:
            MarsHabitatRLEnvironment: RL environment
        """
        # Create environment with specified difficulty
        env = MarsHabitatRLEnvironment(self.data_dir, None, difficulty)
        
        # Load scenario if specified
        if scenario:
            env.load_scenario(scenario)
        
        return env
    
    def get_reward_function(self, weights=None):
        """
        Get reward function with specified component weights
        
        Args:
            weights (dict): Component weights (optional)
            
        Returns:
            dict: Reward function specification
        """
        # Use default weights if not specified
        if weights is None:
            weights = {
                component: info["weight"] 
                for component, info in self.mdp_components["reward_components"].items()
            }
        
        # Create reward function specification
        reward_function = {
            "components": self.mdp_components["reward_components"],
            "weights": weights,
            "description": "Weighted sum of reward components"
        }
        
        return reward_function
    
    def get_curriculum(self):
        """
        Get curriculum learning stages
        
        Returns:
            list: Curriculum stages
        """
        return self.curriculum_stages
    
    def get_evaluation_scenarios(self):
        """
        Get evaluation scenarios
        
        Returns:
            list: Evaluation scenarios
        """
        return self.evaluation_scenarios
    
    def get_evaluation_metrics(self):
        """
        Get evaluation metrics
        
        Returns:
            dict: Evaluation metrics
        """
        return self.evaluation_metrics
    
    def save_formulation(self, file_path=None):
        """
        Save RL problem formulation to file
        
        Args:
            file_path (str): Path to save file (optional)
            
        Returns:
            str: Path to saved file
        """
        # Use default path if not specified
        if file_path is None:
            file_path = os.path.join(self.rl_dir, "rl_formulation.json")
        
        # Create formulation dictionary
        formulation = {
            "mdp_components": self.mdp_components,
            "curriculum_stages": self.curriculum_stages,
            "evaluation_scenarios": self.evaluation_scenarios,
            "evaluation_metrics": self.evaluation_metrics
        }
        
        # Save to file
        with open(file_path, 'w') as f:
            json.dump(formulation, f, indent=2)
        
        print(f"RL problem formulation saved to {file_path}")
        return file_path
    
    def visualize_reward_components(self, save_path=None):
        """
        Visualize reward components
        
        Args:
            save_path (str): Path to save visualization (optional)
            
        Returns:
            tuple: Figure and axes objects
        """
        # Extract reward components and weights
        components = list(self.mdp_components["reward_components"].keys())
        weights = [info["weight"] for info in self.mdp_components["reward_components"].values()]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create bar chart
        bars = ax.bar(components, weights)
        
        # Add labels and title
        ax.set_xlabel('Reward Components')
        ax.set_ylabel('Weight')
        ax.set_title('Reward Function Components and Weights')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom')
        
        # Set y-axis limit
        ax.set_ylim(0, max(weights) * 1.2)
        
        # Add grid
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"Reward components visualization saved to {save_path}")
        
        return fig, ax
    
    def visualize_curriculum(self, save_path=None):
        """
        Visualize curriculum learning stages
        
        Args:
            save_path (str): Path to save visualization (optional)
            
        Returns:
            tuple: Figure and axes objects
        """
        # Extract curriculum stages
        stages = [stage["name"] for stage in self.curriculum_stages]
        durations = [stage["duration_sols"] for stage in self.curriculum_stages]
        failure_rates = [stage["failure_rates_multiplier"] for stage in self.curriculum_stages]
        dust_storm_probs = [stage["dust_storm_probability"] * 100 for stage in self.curriculum_stages]  # Convert to percentage
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot durations
        ax1.bar(stages, durations, color='skyblue')
        ax1.set_ylabel('Duration (sols)')
        ax1.set_title('Curriculum Learning Stage Durations')
        
        # Add value labels on bars
        for i, v in enumerate(durations):
            ax1.text(i, v + 1, str(v), ha='center')
        
        # Plot failure rates and dust storm probabilities
        x = np.arange(len(stages))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, failure_rates, width, label='Failure Rate Multiplier')
        bars2 = ax2.bar(x + width/2, dust_storm_probs, width, label='Dust Storm Probability (%)')
        
        ax2.set_xlabel('Curriculum Stage')
        ax2.set_ylabel('Value')
        ax2.set_title('Failure Rates and Dust Storm Probabilities')
        ax2.set_xticks(x)
        ax2.set_xticklabels(stages)
        ax2.legend()
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                        f'{height:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"Curriculum visualization saved to {save_path}")
        
        return fig, ax1, ax2

# Example usage
if __name__ == "__main__":
    # Create RL problem formulation
    formulation = MarsHabitatRLFormulation("/home/ubuntu/martian_habitat_pathfinder/data")
    
    # Save formulation
    formulation.save_formulation()
    
    # Visualize reward components
    formulation.visualize_reward_components("/home/ubuntu/martian_habitat_pathfinder/simulations/reward_components.png")
    
    # Visualize curriculum
    formulation.visualize_curriculum("/home/ubuntu/martian_habitat_pathfinder/simulations/curriculum.png")
