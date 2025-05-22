"""
Integration module for connecting the Martian Habitat simulation environment
with the Ollama LLM integration.
"""

import os
import sys
import json
import logging
from typing import Dict, Any, List, Tuple, Optional

# Add parent directory to path to import ollama_integration
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.ollama_integration import MarsHabitatLLMAgent, MarsHabitatLLMRL

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("simulation_llm_bridge")

class SimulationLLMBridge:
    """
    Bridge class to connect the Martian Habitat simulation environment
    with the Ollama LLM integration.
    
    This class provides methods to:
    1. Initialize the LLM agent with the simulation environment
    2. Convert simulation state to LLM-compatible format
    3. Apply LLM-generated actions to the simulation
    4. Log and visualize LLM decision-making process
    """
    
    def __init__(self, data_dir: str, 
                 model_name: str = "llama2",
                 cache_dir: Optional[str] = None,
                 rl_agent_path: Optional[str] = None):
        """
        Initialize the simulation-LLM bridge.
        
        Args:
            data_dir: Directory containing data and configuration files
            model_name: Name of the Ollama model to use
            cache_dir: Directory to cache LLM responses
            rl_agent_path: Path to pre-trained RL agent (optional)
        """
        self.data_dir = data_dir
        self.model_name = model_name
        
        # Initialize LLM-RL integration
        self.llm_rl = MarsHabitatLLMRL(
            data_dir=data_dir,
            model_name=model_name,
            rl_agent_path=rl_agent_path,
            cache_dir=cache_dir
        )
        
        # Initialize decision history
        self.decision_history = []
        
        logger.info(f"Simulation-LLM bridge initialized with model {model_name}")
    
    def convert_simulation_state(self, sim_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert simulation state to LLM-compatible format.
        
        Args:
            sim_state: State from simulation environment
            
        Returns:
            Converted state in LLM-compatible format
        """
        # Extract components from simulation state
        # This conversion depends on the exact format of the simulation state
        # Adjust as needed based on your simulation implementation
        
        try:
            # Example conversion - adjust based on actual simulation state format
            llm_state = {
                "time": sim_state.get("time", [0, 0]),
                "environment": {
                    "temperature": sim_state.get("environment", {}).get("temperature", 0),
                    "pressure": sim_state.get("environment", {}).get("pressure", 0),
                    "dust_opacity": sim_state.get("environment", {}).get("dust_opacity", 0),
                    "solar_irradiance": sim_state.get("environment", {}).get("solar_irradiance", 0)
                },
                "habitat": {
                    "power": sim_state.get("resources", {}).get("power", 0),
                    "water": sim_state.get("resources", {}).get("water", 0),
                    "oxygen": sim_state.get("resources", {}).get("oxygen", 0),
                    "food": sim_state.get("resources", {}).get("food", 0),
                    "spare_parts": sim_state.get("resources", {}).get("spare_parts", 0)
                },
                "subsystems": {
                    "power_system": {
                        "status": sim_state.get("subsystems", {}).get("power_system", {}).get("status", "operational"),
                        "maintenance_needed": sim_state.get("subsystems", {}).get("power_system", {}).get("maintenance_needed", 0)
                    },
                    "life_support": {
                        "status": sim_state.get("subsystems", {}).get("life_support", {}).get("status", "operational"),
                        "maintenance_needed": sim_state.get("subsystems", {}).get("life_support", {}).get("maintenance_needed", 0)
                    },
                    "isru": {
                        "status": sim_state.get("subsystems", {}).get("isru", {}).get("status", "operational"),
                        "maintenance_needed": sim_state.get("subsystems", {}).get("isru", {}).get("maintenance_needed", 0)
                    },
                    "thermal_control": {
                        "status": sim_state.get("subsystems", {}).get("thermal_control", {}).get("status", "operational"),
                        "maintenance_needed": sim_state.get("subsystems", {}).get("thermal_control", {}).get("maintenance_needed", 0)
                    }
                }
            }
            
            return llm_state
        except Exception as e:
            logger.error(f"Error converting simulation state: {e}")
            # Return a minimal valid state as fallback
            return {
                "time": [0, 0],
                "environment": {
                    "temperature": -60,
                    "pressure": 650,
                    "dust_opacity": 0.3,
                    "solar_irradiance": 500
                },
                "habitat": {
                    "power": 100,
                    "water": 500,
                    "oxygen": 200,
                    "food": 300,
                    "spare_parts": 50
                },
                "subsystems": {
                    "power_system": {"status": "operational", "maintenance_needed": 0.1},
                    "life_support": {"status": "operational", "maintenance_needed": 0.1},
                    "isru": {"status": "operational", "maintenance_needed": 0.1},
                    "thermal_control": {"status": "operational", "maintenance_needed": 0.1}
                }
            }
    
    def convert_llm_action(self, llm_action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert LLM-generated action to simulation-compatible format.
        
        Args:
            llm_action: Action from LLM agent
            
        Returns:
            Converted action in simulation-compatible format
        """
        # Example conversion - adjust based on actual simulation action format
        try:
            sim_action = {
                "power_allocation": llm_action.get("power_allocation", {
                    "life_support": 4.0,
                    "isru": 3.0,
                    "thermal_control": 3.0
                }),
                "isru_mode": llm_action.get("isru_mode", "both"),
                "maintenance_target": llm_action.get("maintenance_target")
            }
            
            return sim_action
        except Exception as e:
            logger.error(f"Error converting LLM action: {e}")
            # Return a default action as fallback
            return {
                "power_allocation": {
                    "life_support": 4.0,
                    "isru": 3.0,
                    "thermal_control": 3.0
                },
                "isru_mode": "both",
                "maintenance_target": None
            }
    
    def get_action(self, sim_state: Dict[str, Any], 
                  mode: str = "llm",
                  with_explanation: bool = False) -> Dict[str, Any]:
        """
        Get action for the current simulation state.
        
        Args:
            sim_state: Current simulation state
            mode: Action selection mode (llm, rl, hybrid)
            with_explanation: Whether to include explanation
            
        Returns:
            Action dictionary, optionally with explanation
        """
        # Convert simulation state to LLM-compatible format
        llm_state = self.convert_simulation_state(sim_state)
        
        # Get action from LLM-RL integration
        llm_action = self.llm_rl.select_action(llm_state, mode=mode)
        
        # Convert LLM action to simulation-compatible format
        sim_action = self.convert_llm_action(llm_action)
        
        # Get explanation if requested
        explanation = None
        if with_explanation:
            explanation = self.llm_rl.explain_action(llm_state, llm_action)
        
        # Record decision in history
        self.decision_history.append({
            "time": llm_state["time"],
            "state": llm_state,
            "action": llm_action,
            "explanation": explanation
        })
        
        # Return action with explanation if requested
        if with_explanation:
            return {
                "action": sim_action,
                "explanation": explanation
            }
        else:
            return sim_action
    
    def generate_scenario(self, difficulty: str = "normal", 
                         scenario_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a scenario for the simulation.
        
        Args:
            difficulty: Scenario difficulty (easy, normal, hard)
            scenario_type: Type of scenario (dust_storm, solar_flare, etc.)
            
        Returns:
            Scenario dictionary in simulation-compatible format
        """
        # Generate scenario using LLM
        llm_scenario = self.llm_rl.llm_agent.generate_scenario(
            difficulty=difficulty,
            scenario_type=scenario_type
        )
        
        # Convert to simulation-compatible format
        # This conversion depends on the exact format expected by the simulation
        # Adjust as needed based on your simulation implementation
        sim_scenario = {
            "description": llm_scenario.get("description", ""),
            "environment": llm_scenario.get("environment", {}),
            "resources": llm_scenario.get("resources", {}),
            "subsystems": llm_scenario.get("subsystems", {}),
            "events": llm_scenario.get("events", [])
        }
        
        return sim_scenario
    
    def save_decision_history(self, output_path: Optional[str] = None) -> str:
        """
        Save decision history to file.
        
        Args:
            output_path: Path to save history (optional)
            
        Returns:
            Path where history was saved
        """
        if output_path is None:
            output_path = os.path.join(self.data_dir, "decision_history.json")
        
        with open(output_path, 'w') as f:
            json.dump(self.decision_history, f, indent=2)
        
        logger.info(f"Decision history saved to {output_path}")
        return output_path
    
    def answer_query(self, query: str, sim_state: Optional[Dict[str, Any]] = None) -> str:
        """
        Answer a natural language query about the simulation.
        
        Args:
            query: User query string
            sim_state: Current simulation state (optional)
            
        Returns:
            Response string
        """
        # Convert simulation state if provided
        llm_state = None
        if sim_state:
            llm_state = self.convert_simulation_state(sim_state)
        
        # Get answer from LLM agent
        return self.llm_rl.llm_agent.answer_query(query, llm_state)


# Example usage
if __name__ == "__main__":
    # Set up data directory
    data_dir = "/home/ubuntu/martian_habitat_pathfinder/data"
    
    # Initialize bridge
    bridge = SimulationLLMBridge(
        data_dir=data_dir,
        model_name="llama2"
    )
    
    # Example simulation state
    example_sim_state = {
        "time": [10, 14],
        "environment": {
            "temperature": -60.0,
            "pressure": 650.0,
            "dust_opacity": 0.3,
            "solar_irradiance": 500.0
        },
        "resources": {
            "power": 120.0,
            "water": 450.0,
            "oxygen": 180.0,
            "food": 300.0,
            "spare_parts": 50.0
        },
        "subsystems": {
            "power_system": {"status": "operational", "maintenance_needed": 0.1},
            "life_support": {"status": "operational", "maintenance_needed": 0.2},
            "isru": {"status": "operational", "maintenance_needed": 0.3},
            "thermal_control": {"status": "operational", "maintenance_needed": 0.1}
        }
    }
    
    # Get action with explanation
    result = bridge.get_action(example_sim_state, with_explanation=True)
    print("Action:", result["action"])
    print("\nExplanation:", result["explanation"])
    
    # Generate scenario
    scenario = bridge.generate_scenario(difficulty="hard", scenario_type="dust_storm")
    print("\nScenario:", scenario["description"])
    
    # Answer query
    answer = bridge.answer_query("What should I do during a dust storm?", example_sim_state)
    print("\nAnswer:", answer)
