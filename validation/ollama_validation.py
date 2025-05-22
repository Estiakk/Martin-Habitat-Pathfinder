"""
Validation script for Ollama integration in Martian Habitat Pathfinder.

This script tests the Ollama integration with realistic scenarios and sample data
to ensure all components work as intended.
"""

import os
import sys
import json
import time
import logging
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("ollama_validation.log")
    ]
)
logger = logging.getLogger("ollama_validation")

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import Ollama integration modules
try:
    from models.ollama_integration import OllamaClient, MarsHabitatLLMAgent, MarsHabitatLLMRL
    from simulations.simulation_llm_bridge import SimulationLLMBridge
    logger.info("Successfully imported Ollama integration modules")
except ImportError as e:
    logger.error(f"Failed to import Ollama integration modules: {e}")
    sys.exit(1)

class OllamaValidation:
    """
    Validation class for testing Ollama integration in Martian Habitat Pathfinder.
    """
    
    def __init__(self, data_dir: str):
        """
        Initialize the validation class.
        
        Args:
            data_dir: Directory containing data and configuration files
        """
        self.data_dir = data_dir
        self.results = {
            "tests": [],
            "passed": 0,
            "failed": 0,
            "warnings": 0,
            "total": 0
        }
        
        # Create cache directory if it doesn't exist
        self.cache_dir = os.path.join(data_dir, "llm_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Test data
        self.test_states = self._create_test_states()
        
        logger.info(f"Validation initialized with data directory: {data_dir}")
    
    def _create_test_states(self) -> List[Dict[str, Any]]:
        """Create test states for validation."""
        return [
            {
                "name": "Normal Operations",
                "state": {
                    "time": [5, 12],
                    "environment": {
                        "temperature": -60.0,
                        "pressure": 650.0,
                        "dust_opacity": 0.3,
                        "solar_irradiance": 500.0
                    },
                    "habitat": {
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
                },
                "sim_state": {
                    "time": [5, 12],
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
            },
            {
                "name": "Dust Storm",
                "state": {
                    "time": [10, 14],
                    "environment": {
                        "temperature": -60.0,
                        "pressure": 650.0,
                        "dust_opacity": 0.8,
                        "solar_irradiance": 200.0
                    },
                    "habitat": {
                        "power": 80.0,
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
                },
                "sim_state": {
                    "time": [10, 14],
                    "environment": {
                        "temperature": -60.0,
                        "pressure": 650.0,
                        "dust_opacity": 0.8,
                        "solar_irradiance": 200.0
                    },
                    "resources": {
                        "power": 80.0,
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
            },
            {
                "name": "Equipment Failure",
                "state": {
                    "time": [15, 8],
                    "environment": {
                        "temperature": -70.0,
                        "pressure": 650.0,
                        "dust_opacity": 0.4,
                        "solar_irradiance": 450.0
                    },
                    "habitat": {
                        "power": 100.0,
                        "water": 400.0,
                        "oxygen": 150.0,
                        "food": 250.0,
                        "spare_parts": 30.0
                    },
                    "subsystems": {
                        "power_system": {"status": "operational", "maintenance_needed": 0.2},
                        "life_support": {"status": "degraded", "maintenance_needed": 0.7},
                        "isru": {"status": "operational", "maintenance_needed": 0.3},
                        "thermal_control": {"status": "operational", "maintenance_needed": 0.2}
                    }
                },
                "sim_state": {
                    "time": [15, 8],
                    "environment": {
                        "temperature": -70.0,
                        "pressure": 650.0,
                        "dust_opacity": 0.4,
                        "solar_irradiance": 450.0
                    },
                    "resources": {
                        "power": 100.0,
                        "water": 400.0,
                        "oxygen": 150.0,
                        "food": 250.0,
                        "spare_parts": 30.0
                    },
                    "subsystems": {
                        "power_system": {"status": "operational", "maintenance_needed": 0.2},
                        "life_support": {"status": "degraded", "maintenance_needed": 0.7},
                        "isru": {"status": "operational", "maintenance_needed": 0.3},
                        "thermal_control": {"status": "operational", "maintenance_needed": 0.2}
                    }
                }
            }
        ]
    
    def _record_test_result(self, test_name: str, passed: bool, details: str = "") -> None:
        """Record test result."""
        result = {
            "name": test_name,
            "passed": passed,
            "details": details,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        self.results["tests"].append(result)
        self.results["total"] += 1
        
        if passed:
            self.results["passed"] += 1
            logger.info(f"✅ PASSED: {test_name}")
            if details:
                logger.info(f"  Details: {details}")
        else:
            self.results["failed"] += 1
            logger.error(f"❌ FAILED: {test_name}")
            if details:
                logger.error(f"  Details: {details}")
    
    def _record_warning(self, test_name: str, warning: str) -> None:
        """Record test warning."""
        result = {
            "name": test_name,
            "passed": True,
            "warning": warning,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        self.results["tests"].append(result)
        self.results["total"] += 1
        self.results["passed"] += 1
        self.results["warnings"] += 1
        
        logger.warning(f"⚠️ WARNING: {test_name}")
        logger.warning(f"  Details: {warning}")
    
    def test_ollama_client(self) -> None:
        """Test OllamaClient functionality."""
        logger.info("Testing OllamaClient...")
        
        try:
            # Initialize client
            client = OllamaClient(
                default_model="llama2",
                cache_dir=self.cache_dir
            )
            
            # Test connection
            models = client.list_models()
            
            if models:
                self._record_test_result(
                    "OllamaClient Connection",
                    True,
                    f"Connected to Ollama server. Available models: {', '.join(models)}"
                )
            else:
                self._record_warning(
                    "OllamaClient Connection",
                    "Connected to Ollama server but no models found. You may need to pull models."
                )
            
            # Test text generation
            try:
                result = client.generate(
                    prompt="What is Mars?",
                    max_tokens=100
                )
                
                if "response" in result and result["response"]:
                    self._record_test_result(
                        "OllamaClient Text Generation",
                        True,
                        f"Successfully generated text ({len(result['response'])} chars)"
                    )
                else:
                    self._record_test_result(
                        "OllamaClient Text Generation",
                        False,
                        f"Empty response: {result}"
                    )
            except Exception as e:
                self._record_test_result(
                    "OllamaClient Text Generation",
                    False,
                    f"Error: {str(e)}"
                )
            
            # Test JSON generation
            try:
                schema = {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "description": {"type": "string"}
                    }
                }
                
                result = client.generate_json(
                    prompt="Describe Mars in JSON format with name and description fields",
                    schema=schema,
                    temperature=0.2
                )
                
                if "parsed_json" in result and result["parsed_json"]:
                    self._record_test_result(
                        "OllamaClient JSON Generation",
                        True,
                        f"Successfully generated JSON: {result['parsed_json']}"
                    )
                else:
                    self._record_test_result(
                        "OllamaClient JSON Generation",
                        False,
                        f"Failed to parse JSON: {result}"
                    )
            except Exception as e:
                self._record_test_result(
                    "OllamaClient JSON Generation",
                    False,
                    f"Error: {str(e)}"
                )
            
        except Exception as e:
            self._record_test_result(
                "OllamaClient Initialization",
                False,
                f"Error: {str(e)}"
            )
    
    def test_llm_agent(self) -> None:
        """Test MarsHabitatLLMAgent functionality."""
        logger.info("Testing MarsHabitatLLMAgent...")
        
        try:
            # Initialize agent
            agent = MarsHabitatLLMAgent(
                data_dir=self.data_dir,
                model_name="llama2",
                cache_dir=self.cache_dir
            )
            
            self._record_test_result(
                "MarsHabitatLLMAgent Initialization",
                True,
                "Successfully initialized agent"
            )
            
            # Test action selection
            for test_case in self.test_states:
                try:
                    state = test_case["state"]
                    action = agent.select_action(state)
                    
                    # Validate action format
                    if (
                        isinstance(action, dict) and
                        "power_allocation" in action and
                        "isru_mode" in action and
                        isinstance(action["power_allocation"], dict)
                    ):
                        self._record_test_result(
                            f"MarsHabitatLLMAgent Action Selection - {test_case['name']}",
                            True,
                            f"Action: {action}"
                        )
                    else:
                        self._record_test_result(
                            f"MarsHabitatLLMAgent Action Selection - {test_case['name']}",
                            False,
                            f"Invalid action format: {action}"
                        )
                except Exception as e:
                    self._record_test_result(
                        f"MarsHabitatLLMAgent Action Selection - {test_case['name']}",
                        False,
                        f"Error: {str(e)}"
                    )
            
            # Test explanation generation
            try:
                state = self.test_states[0]["state"]
                action = agent.select_action(state)
                explanation = agent.explain_decision(state, action)
                
                if explanation and len(explanation) > 50:
                    self._record_test_result(
                        "MarsHabitatLLMAgent Explanation Generation",
                        True,
                        f"Generated explanation ({len(explanation)} chars)"
                    )
                else:
                    self._record_test_result(
                        "MarsHabitatLLMAgent Explanation Generation",
                        False,
                        f"Insufficient explanation: {explanation}"
                    )
            except Exception as e:
                self._record_test_result(
                    "MarsHabitatLLMAgent Explanation Generation",
                    False,
                    f"Error: {str(e)}"
                )
            
            # Test scenario generation
            try:
                scenario = agent.generate_scenario(
                    difficulty="normal",
                    scenario_type="dust_storm"
                )
                
                if (
                    isinstance(scenario, dict) and
                    "description" in scenario and
                    "environment" in scenario and
                    "resources" in scenario and
                    "subsystems" in scenario
                ):
                    self._record_test_result(
                        "MarsHabitatLLMAgent Scenario Generation",
                        True,
                        f"Generated scenario: {scenario['description']}"
                    )
                else:
                    self._record_test_result(
                        "MarsHabitatLLMAgent Scenario Generation",
                        False,
                        f"Invalid scenario format: {scenario}"
                    )
            except Exception as e:
                self._record_test_result(
                    "MarsHabitatLLMAgent Scenario Generation",
                    False,
                    f"Error: {str(e)}"
                )
            
        except Exception as e:
            self._record_test_result(
                "MarsHabitatLLMAgent Initialization",
                False,
                f"Error: {str(e)}"
            )
    
    def test_llm_rl(self) -> None:
        """Test MarsHabitatLLMRL functionality."""
        logger.info("Testing MarsHabitatLLMRL...")
        
        try:
            # Initialize LLM-RL integration
            llm_rl = MarsHabitatLLMRL(
                data_dir=self.data_dir,
                model_name="llama2",
                cache_dir=self.cache_dir
            )
            
            self._record_test_result(
                "MarsHabitatLLMRL Initialization",
                True,
                "Successfully initialized LLM-RL integration"
            )
            
            # Test action selection with different modes
            state = self.test_states[0]["state"]
            
            try:
                llm_action = llm_rl.select_action(state, mode="llm")
                
                if (
                    isinstance(llm_action, dict) and
                    "power_allocation" in llm_action and
                    "isru_mode" in llm_action
                ):
                    self._record_test_result(
                        "MarsHabitatLLMRL LLM Action Selection",
                        True,
                        f"Action: {llm_action}"
                    )
                else:
                    self._record_test_result(
                        "MarsHabitatLLMRL LLM Action Selection",
                        False,
                        f"Invalid action format: {llm_action}"
                    )
            except Exception as e:
                self._record_test_result(
                    "MarsHabitatLLMRL LLM Action Selection",
                    False,
                    f"Error: {str(e)}"
                )
            
            try:
                hybrid_action = llm_rl.select_action(state, mode="hybrid", temperature=0.7)
                
                if (
                    isinstance(hybrid_action, dict) and
                    "power_allocation" in hybrid_action and
                    "isru_mode" in hybrid_action
                ):
                    self._record_test_result(
                        "MarsHabitatLLMRL Hybrid Action Selection",
                        True,
                        f"Action: {hybrid_action}"
                    )
                else:
                    self._record_test_result(
                        "MarsHabitatLLMRL Hybrid Action Selection",
                        False,
                        f"Invalid action format: {hybrid_action}"
                    )
            except Exception as e:
                self._record_test_result(
                    "MarsHabitatLLMRL Hybrid Action Selection",
                    False,
                    f"Error: {str(e)}"
                )
            
            # Test training data creation
            try:
                training_data_path = llm_rl.create_training_data(
                    num_examples=2,
                    output_path=os.path.join(self.data_dir, "test_training_data.txt")
                )
                
                if os.path.exists(training_data_path) and os.path.getsize(training_data_path) > 0:
                    self._record_test_result(
                        "MarsHabitatLLMRL Training Data Creation",
                        True,
                        f"Created training data at {training_data_path}"
                    )
                else:
                    self._record_test_result(
                        "MarsHabitatLLMRL Training Data Creation",
                        False,
                        f"Failed to create valid training data"
                    )
            except Exception as e:
                self._record_test_result(
                    "MarsHabitatLLMRL Training Data Creation",
                    False,
                    f"Error: {str(e)}"
                )
            
        except Exception as e:
            self._record_test_result(
                "MarsHabitatLLMRL Initialization",
                False,
                f"Error: {str(e)}"
            )
    
    def test_simulation_bridge(self) -> None:
        """Test SimulationLLMBridge functionality."""
        logger.info("Testing SimulationLLMBridge...")
        
        try:
            # Initialize bridge
            bridge = SimulationLLMBridge(
                data_dir=self.data_dir,
                model_name="llama2",
                cache_dir=self.cache_dir
            )
            
            self._record_test_result(
                "SimulationLLMBridge Initialization",
                True,
                "Successfully initialized simulation bridge"
            )
            
            # Test state conversion
            for test_case in self.test_states:
                try:
                    sim_state = test_case["sim_state"]
                    llm_state = bridge.convert_simulation_state(sim_state)
                    
                    if (
                        isinstance(llm_state, dict) and
                        "time" in llm_state and
                        "environment" in llm_state and
                        "habitat" in llm_state and
                        "subsystems" in llm_state
                    ):
                        self._record_test_result(
                            f"SimulationLLMBridge State Conversion - {test_case['name']}",
                            True,
                            "Successfully converted simulation state to LLM state"
                        )
                    else:
                        self._record_test_result(
                            f"SimulationLLMBridge State Conversion - {test_case['name']}",
                            False,
                            f"Invalid state format: {llm_state}"
                        )
                except Exception as e:
                    self._record_test_result(
                        f"SimulationLLMBridge State Conversion - {test_case['name']}",
                        False,
                        f"Error: {str(e)}"
                    )
            
            # Test action generation
            for test_case in self.test_states:
                try:
                    sim_state = test_case["sim_state"]
                    result = bridge.get_action(sim_state, with_explanation=True)
                    
                    if (
                        isinstance(result, dict) and
                        "action" in result and
                        "explanation" in result and
                        isinstance(result["action"], dict) and
                        "power_allocation" in result["action"] and
                        "isru_mode" in result["action"]
                    ):
                        self._record_test_result(
                            f"SimulationLLMBridge Action Generation - {test_case['name']}",
                            True,
                            f"Action: {result['action']}"
                        )
                    else:
                        self._record_test_result(
                            f"SimulationLLMBridge Action Generation - {test_case['name']}",
                            False,
                            f"Invalid result format: {result}"
                        )
                except Exception as e:
                    self._record_test_result(
                        f"SimulationLLMBridge Action Generation - {test_case['name']}",
                        False,
                        f"Error: {str(e)}"
                    )
            
            # Test scenario generation
            try:
                scenario = bridge.generate_scenario(
                    difficulty="hard",
                    scenario_type="equipment_failure"
                )
                
                if (
                    isinstance(scenario, dict) and
                    "description" in scenario and
                    "environment" in scenario and
                    "resources" in scenario and
                    "subsystems" in scenario
                ):
                    self._record_test_result(
                        "SimulationLLMBridge Scenario Generation",
                        True,
                        f"Generated scenario: {scenario['description']}"
                    )
                else:
                    self._record_test_result(
                        "SimulationLLMBridge Scenario Generation",
                        False,
                        f"Invalid scenario format: {scenario}"
                    )
            except Exception as e:
                self._record_test_result(
                    "SimulationLLMBridge Scenario Generation",
                    False,
                    f"Error: {str(e)}"
                )
            
            # Test query answering
            try:
                sim_state = self.test_states[0]["sim_state"]
                answer = bridge.answer_query(
                    "How should I manage power during a dust storm?",
                    sim_state
                )
                
                if answer and len(answer) > 50:
                    self._record_test_result(
                        "SimulationLLMBridge Query Answering",
                        True,
                        f"Generated answer ({len(answer)} chars)"
                    )
                else:
                    self._record_test_result(
                        "SimulationLLMBridge Query Answering",
                        False,
                        f"Insufficient answer: {answer}"
                    )
            except Exception as e:
                self._record_test_result(
                    "SimulationLLMBridge Query Answering",
                    False,
                    f"Error: {str(e)}"
                )
            
        except Exception as e:
            self._record_test_result(
                "SimulationLLMBridge Initialization",
                False,
                f"Error: {str(e)}"
            )
    
    def test_documentation(self) -> None:
        """Test documentation completeness and accuracy."""
        logger.info("Testing documentation...")
        
        # Check for documentation files
        doc_files = [
            os.path.join(self.data_dir, "..", "docs", "data_formats.md"),
            os.path.join(self.data_dir, "..", "docs", "ollama_usage_guide.md")
        ]
        
        for doc_file in doc_files:
            if os.path.exists(doc_file):
                # Check file size
                file_size = os.path.getsize(doc_file)
                if file_size > 1000:
                    self._record_test_result(
                        f"Documentation - {os.path.basename(doc_file)}",
                        True,
                        f"File exists and has sufficient content ({file_size} bytes)"
                    )
                else:
                    self._record_test_result(
                        f"Documentation - {os.path.basename(doc_file)}",
                        False,
                        f"File exists but has insufficient content ({file_size} bytes)"
                    )
            else:
                self._record_test_result(
                    f"Documentation - {os.path.basename(doc_file)}",
                    False,
                    f"File does not exist: {doc_file}"
                )
        
        # Check for code documentation
        code_files = [
            os.path.join(self.data_dir, "..", "models", "ollama_integration.py"),
            os.path.join(self.data_dir, "..", "simulations", "simulation_llm_bridge.py")
        ]
        
        for code_file in code_files:
            if os.path.exists(code_file):
                # Check for docstrings
                with open(code_file, 'r') as f:
                    content = f.read()
                
                if '"""' in content and "def " in content:
                    # Count docstrings
                    docstring_count = content.count('"""')
                    function_count = content.count("def ")
                    
                    if docstring_count >= function_count / 2:  # At least half of functions documented
                        self._record_test_result(
                            f"Code Documentation - {os.path.basename(code_file)}",
                            True,
                            f"File has sufficient docstrings ({docstring_count} docstrings, {function_count} functions)"
                        )
                    else:
                        self._record_warning(
                            f"Code Documentation - {os.path.basename(code_file)}",
                            f"File has insufficient docstrings ({docstring_count} docstrings, {function_count} functions)"
                        )
                else:
                    self._record_test_result(
                        f"Code Documentation - {os.path.basename(code_file)}",
                        False,
                        "File lacks docstrings"
                    )
            else:
                self._record_test_result(
                    f"Code Documentation - {os.path.basename(code_file)}",
                    False,
                    f"File does not exist: {code_file}"
                )
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all validation tests."""
        logger.info("Starting validation tests...")
        
        # Run tests
        self.test_ollama_client()
        self.test_llm_agent()
        self.test_llm_rl()
        self.test_simulation_bridge()
        self.test_documentation()
        
        # Print summary
        logger.info("\n=== Validation Summary ===")
        logger.info(f"Total tests: {self.results['total']}")
        logger.info(f"Passed: {self.results['passed']}")
        logger.info(f"Failed: {self.results['failed']}")
        logger.info(f"Warnings: {self.results['warnings']}")
        logger.info(f"Pass rate: {self.results['passed'] / self.results['total'] * 100:.2f}%")
        
        # Save results
        results_path = os.path.join(self.data_dir, "validation_results.json")
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Results saved to {results_path}")
        
        return self.results


if __name__ == "__main__":
    # Set data directory
    data_dir = "/home/ubuntu/martian_habitat_pathfinder/data"
    
    # Create validation instance
    validation = OllamaValidation(data_dir)
    
    # Run all tests
    results = validation.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if results["failed"] == 0 else 1)
