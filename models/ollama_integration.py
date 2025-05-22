"""
Ollama Integration Module for Martian Habitat Pathfinder

This module provides a robust integration between Ollama-hosted LLMs and 
the Martian Habitat Pathfinder reinforcement learning system.
"""

import os
import json
import time
import logging
import numpy as np
import requests
from typing import Dict, List, Any, Optional, Union, Tuple
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ollama_integration")

class OllamaClient:
    """
    Client for interacting with Ollama API to run LLMs locally.
    
    This client handles communication with the Ollama server, manages model loading,
    and provides methods for generating text, embeddings, and structured outputs.
    """
    
    def __init__(self, base_url: str = "http://localhost:11434", 
                 default_model: str = "llama2", 
                 timeout: int = 60,
                 cache_dir: Optional[str] = None):
        """
        Initialize the Ollama client.
        
        Args:
            base_url: URL of the Ollama API server
            default_model: Default model to use for generations
            timeout: Request timeout in seconds
            cache_dir: Directory to cache responses (if None, caching is disabled)
        """
        self.base_url = base_url
        self.default_model = default_model
        self.timeout = timeout
        self.cache_dir = cache_dir
        
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            
        # Test connection to Ollama server
        self._test_connection()
    
    def _test_connection(self) -> None:
        """Test connection to Ollama server and log available models."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=self.timeout)
            response.raise_for_status()
            models = response.json().get("models", [])
            model_names = [model.get("name") for model in models]
            logger.info(f"Connected to Ollama server at {self.base_url}")
            logger.info(f"Available models: {', '.join(model_names) if model_names else 'No models found'}")
            
            # Check if default model is available
            if model_names and self.default_model not in model_names:
                logger.warning(f"Default model '{self.default_model}' not found in available models")
                logger.warning(f"You may need to run: ollama pull {self.default_model}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to connect to Ollama server: {e}")
            logger.error("Please ensure Ollama is installed and running")
            logger.error("Installation instructions: https://ollama.com/download")
    
    def _get_cache_path(self, model: str, prompt: str, params: Dict[str, Any]) -> Optional[str]:
        """Get cache file path for a request if caching is enabled."""
        if not self.cache_dir:
            return None
            
        # Create a unique key based on model, prompt and parameters
        cache_key = f"{model}_{hash(prompt)}_{hash(str(params))}"
        return os.path.join(self.cache_dir, f"{cache_key}.json")
    
    def _check_cache(self, cache_path: str) -> Optional[Dict[str, Any]]:
        """Check if a cached response exists and is valid."""
        if not cache_path or not os.path.exists(cache_path):
            return None
            
        try:
            with open(cache_path, 'r') as f:
                cached = json.load(f)
                
            # Check if cache is expired (older than 24 hours)
            cache_time = cached.get('cache_time', 0)
            if time.time() - cache_time > 86400:  # 24 hours
                return None
                
            return cached.get('response')
        except Exception as e:
            logger.warning(f"Failed to read cache: {e}")
            return None
    
    def _save_cache(self, cache_path: str, response: Dict[str, Any]) -> None:
        """Save response to cache."""
        if not cache_path or not self.cache_dir:
            return
            
        try:
            with open(cache_path, 'w') as f:
                json.dump({
                    'response': response,
                    'cache_time': time.time()
                }, f)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def generate(self, prompt: str, model: Optional[str] = None, 
                 system: Optional[str] = None, 
                 format: Optional[str] = None,
                 temperature: float = 0.7,
                 top_p: float = 0.9,
                 top_k: int = 40,
                 max_tokens: int = 2048,
                 stop: Optional[List[str]] = None,
                 use_cache: bool = True) -> Dict[str, Any]:
        """
        Generate text using the specified model.
        
        Args:
            prompt: The prompt to generate text from
            model: Model to use (defaults to self.default_model)
            system: System message for the model
            format: Response format (json, yaml, etc.)
            temperature: Sampling temperature (higher = more creative)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            max_tokens: Maximum tokens to generate
            stop: List of strings to stop generation
            use_cache: Whether to use caching
            
        Returns:
            Dict containing the response and metadata
        """
        model = model or self.default_model
        
        # Prepare parameters
        params = {
            "model": model,
            "prompt": prompt,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "num_predict": max_tokens,
            }
        }
        
        if system:
            params["system"] = system
            
        if format:
            params["format"] = format
            
        if stop:
            params["options"]["stop"] = stop
        
        # Check cache if enabled
        cache_path = self._get_cache_path(model, prompt, params) if use_cache else None
        cached_response = self._check_cache(cache_path) if cache_path else None
        
        if cached_response:
            logger.debug(f"Using cached response for {model}")
            return cached_response
        
        # Make API request
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=params,
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            
            # Save to cache if enabled
            if cache_path:
                self._save_cache(cache_path, result)
                
            return result
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to generate text: {e}")
            return {
                "error": str(e),
                "model": model,
                "response": ""
            }
    
    def generate_json(self, prompt: str, model: Optional[str] = None,
                     system: Optional[str] = None,
                     schema: Optional[Dict[str, Any]] = None,
                     temperature: float = 0.2,  # Lower temperature for structured output
                     max_attempts: int = 3,
                     use_cache: bool = True) -> Dict[str, Any]:
        """
        Generate JSON response using the specified model.
        
        Args:
            prompt: The prompt to generate JSON from
            model: Model to use (defaults to self.default_model)
            system: System message for the model
            schema: JSON schema to validate against (optional)
            temperature: Sampling temperature (lower for more deterministic output)
            max_attempts: Maximum number of attempts to generate valid JSON
            use_cache: Whether to use caching
            
        Returns:
            Dict containing the parsed JSON response and metadata
        """
        model = model or self.default_model
        
        # Add JSON formatting instructions to the prompt
        if schema:
            schema_str = json.dumps(schema, indent=2)
            json_prompt = f"{prompt}\n\nPlease respond with a JSON object that conforms to the following schema:\n{schema_str}\n\nOnly respond with the JSON object, no additional text."
        else:
            json_prompt = f"{prompt}\n\nPlease respond with a JSON object. Only respond with the JSON object, no additional text."
        
        # Try multiple attempts to get valid JSON
        for attempt in range(max_attempts):
            try:
                # Generate with format=json if using a model that supports it
                result = self.generate(
                    prompt=json_prompt,
                    model=model,
                    system=system,
                    format="json" if model.startswith(("llama3", "mistral", "gemma")) else None,
                    temperature=temperature,
                    use_cache=use_cache
                )
                
                response_text = result.get("response", "")
                
                # Try to extract JSON from the response
                try:
                    # First try direct parsing
                    parsed = json.loads(response_text)
                    result["parsed_json"] = parsed
                    return result
                except json.JSONDecodeError:
                    # Try to extract JSON from text
                    start_idx = response_text.find('{')
                    end_idx = response_text.rfind('}') + 1
                    
                    if start_idx >= 0 and end_idx > start_idx:
                        json_str = response_text[start_idx:end_idx]
                        try:
                            parsed = json.loads(json_str)
                            result["parsed_json"] = parsed
                            return result
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse JSON on attempt {attempt+1}/{max_attempts}")
                    
                    # If we're on the last attempt, return the error
                    if attempt == max_attempts - 1:
                        result["error"] = "Failed to parse JSON response"
                        result["parsed_json"] = None
                        return result
            except Exception as e:
                logger.error(f"Error generating JSON: {e}")
                if attempt == max_attempts - 1:
                    return {
                        "error": str(e),
                        "model": model,
                        "response": "",
                        "parsed_json": None
                    }
    
    def get_embeddings(self, text: str, model: Optional[str] = None) -> Dict[str, Any]:
        """
        Get embeddings for the specified text.
        
        Args:
            text: The text to get embeddings for
            model: Model to use (defaults to self.default_model)
            
        Returns:
            Dict containing the embeddings and metadata
        """
        model = model or self.default_model
        
        try:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": model, "prompt": text},
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get embeddings: {e}")
            return {
                "error": str(e),
                "model": model,
                "embedding": []
            }
    
    def list_models(self) -> List[str]:
        """List available models on the Ollama server."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=self.timeout)
            response.raise_for_status()
            models = response.json().get("models", [])
            return [model.get("name") for model in models]
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    def pull_model(self, model: str) -> bool:
        """
        Pull a model from Ollama library if not already available.
        
        Args:
            model: Name of the model to pull
            
        Returns:
            True if successful, False otherwise
        """
        if model in self.list_models():
            logger.info(f"Model {model} is already available")
            return True
            
        logger.info(f"Pulling model {model}...")
        try:
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": model},
                timeout=600  # Longer timeout for model pulling
            )
            response.raise_for_status()
            logger.info(f"Successfully pulled model {model}")
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to pull model {model}: {e}")
            return False


class MarsHabitatLLMAgent:
    """
    LLM-based agent for Mars Habitat management using Ollama.
    
    This agent uses LLMs to:
    1. Generate actions based on environment state
    2. Explain decisions made by RL agents
    3. Generate scenarios for training
    4. Provide natural language responses to queries
    """
    
    def __init__(self, data_dir: str, 
                 model_name: str = "llama2",
                 cache_dir: Optional[str] = None):
        """
        Initialize the Mars Habitat LLM Agent.
        
        Args:
            data_dir: Directory containing data and configuration files
            model_name: Name of the Ollama model to use
            cache_dir: Directory to cache LLM responses
        """
        self.data_dir = data_dir
        self.model_name = model_name
        
        # Initialize Ollama client
        self.ollama = OllamaClient(
            default_model=model_name,
            cache_dir=cache_dir or os.path.join(data_dir, "llm_cache")
        )
        
        # Load configuration
        self.config = self._load_config()
        
        # Define schemas for structured outputs
        self.schemas = self._define_schemas()
        
        logger.info(f"Mars Habitat LLM Agent initialized with model {model_name}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        config_path = os.path.join(self.data_dir, "config.json")
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            logger.warning("Using default configuration")
            return {
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
                },
                "llm": {
                    "system_prompt": "You are an AI assistant specialized in Mars habitat resource management.",
                    "temperature": 0.7,
                    "max_tokens": 2048
                }
            }
    
    def _define_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Define JSON schemas for structured outputs."""
        return {
            "action": {
                "type": "object",
                "properties": {
                    "power_allocation": {
                        "type": "object",
                        "properties": {
                            "life_support": {"type": "number", "minimum": 0, "maximum": 10},
                            "isru": {"type": "number", "minimum": 0, "maximum": 10},
                            "thermal_control": {"type": "number", "minimum": 0, "maximum": 10}
                        },
                        "required": ["life_support", "isru", "thermal_control"]
                    },
                    "isru_mode": {
                        "type": "string",
                        "enum": ["water", "oxygen", "both", "off"]
                    },
                    "maintenance_target": {
                        "type": ["string", "null"],
                        "enum": ["power_system", "life_support", "isru", "thermal_control", None]
                    }
                },
                "required": ["power_allocation", "isru_mode"]
            },
            "scenario": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "environment": {
                        "type": "object",
                        "properties": {
                            "temperature": {"type": "number"},
                            "pressure": {"type": "number"},
                            "dust_opacity": {"type": "number"},
                            "solar_irradiance": {"type": "number"}
                        },
                        "required": ["temperature", "pressure", "dust_opacity", "solar_irradiance"]
                    },
                    "resources": {
                        "type": "object",
                        "properties": {
                            "power": {"type": "number"},
                            "water": {"type": "number"},
                            "oxygen": {"type": "number"},
                            "food": {"type": "number"}
                        },
                        "required": ["power", "water", "oxygen", "food"]
                    },
                    "subsystems": {
                        "type": "object",
                        "properties": {
                            "power_system": {
                                "type": "object",
                                "properties": {
                                    "status": {"type": "string"},
                                    "maintenance_needed": {"type": "number"}
                                },
                                "required": ["status", "maintenance_needed"]
                            },
                            "life_support": {
                                "type": "object",
                                "properties": {
                                    "status": {"type": "string"},
                                    "maintenance_needed": {"type": "number"}
                                },
                                "required": ["status", "maintenance_needed"]
                            },
                            "isru": {
                                "type": "object",
                                "properties": {
                                    "status": {"type": "string"},
                                    "maintenance_needed": {"type": "number"}
                                },
                                "required": ["status", "maintenance_needed"]
                            },
                            "thermal_control": {
                                "type": "object",
                                "properties": {
                                    "status": {"type": "string"},
                                    "maintenance_needed": {"type": "number"}
                                },
                                "required": ["status", "maintenance_needed"]
                            }
                        },
                        "required": ["power_system", "life_support", "isru", "thermal_control"]
                    },
                    "events": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string"},
                                "time": {
                                    "type": "object",
                                    "properties": {
                                        "sol": {"type": "number"},
                                        "hour": {"type": "number"}
                                    },
                                    "required": ["sol", "hour"]
                                },
                                "description": {"type": "string"},
                                "effects": {
                                    "type": "object",
                                    "properties": {
                                        "target": {"type": "string"},
                                        "magnitude": {"type": "number"}
                                    },
                                    "required": ["target", "magnitude"]
                                }
                            },
                            "required": ["type", "time", "description", "effects"]
                        }
                    }
                },
                "required": ["description", "environment", "resources", "subsystems"]
            }
        }
    
    def format_state_prompt(self, state: Dict[str, Any]) -> str:
        """
        Format state as a prompt for the LLM.
        
        Args:
            state: Environment state dictionary
            
        Returns:
            Formatted prompt string
        """
        # Extract state components
        time = state.get('time', [0, 0])
        habitat = state.get('habitat', {})
        environment = state.get('environment', {})
        subsystems = state.get('subsystems', {})
        
        # Format prompt
        prompt = f"""
        You are an AI assistant managing a Mars habitat with the following state:
        
        Time: Sol {time[0]}, Hour {time[1]}
        
        Habitat Resources:
        - Power: {habitat.get('power', 0):.2f} kWh
        - Water: {habitat.get('water', 0):.2f} liters
        - Oxygen: {habitat.get('oxygen', 0):.2f} kg
        - Food: {habitat.get('food', 0):.2f} kg
        - Spare Parts: {habitat.get('spare_parts', 0):.2f} units
        
        Environmental Conditions:
        - Temperature: {environment.get('temperature', 0):.2f} °C
        - Pressure: {environment.get('pressure', 0):.2f} Pa
        - Dust Opacity: {environment.get('dust_opacity', 0):.2f}
        - Solar Irradiance: {environment.get('solar_irradiance', 0):.2f} W/m²
        
        Subsystem Status:
        """
        
        # Add subsystem status
        for subsystem_name, subsystem_data in subsystems.items():
            status = subsystem_data.get('status', 'unknown')
            maintenance = subsystem_data.get('maintenance_needed', 0)
            prompt += f"- {subsystem_name.replace('_', ' ').title()}: {status} (Maintenance: {maintenance:.2f})\n"
        
        return prompt.strip()
    
    def select_action(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select action using LLM based on environment state.
        
        Args:
            state: Environment state dictionary
            
        Returns:
            Action dictionary
        """
        # Format state as prompt
        state_prompt = self.format_state_prompt(state)
        
        # Add action request
        prompt = f"""
        {state_prompt}
        
        Based on this information, determine the optimal resource allocation strategy.
        
        You must provide:
        1. Power allocation to each subsystem (life_support, isru, thermal_control) between 0-10 kW
        2. ISRU mode (water, oxygen, both, or off)
        3. Maintenance target (power_system, life_support, isru, thermal_control, or null if no maintenance needed)
        
        Consider the following factors:
        - Life support requires power to maintain oxygen and water recycling
        - ISRU (In-Situ Resource Utilization) produces water and/or oxygen from Martian resources
        - Thermal control maintains habitat temperature
        - Maintenance improves efficiency and prevents failures
        - Power is limited and must be allocated carefully
        
        Respond with a JSON object containing your decision.
        """
        
        # Get system prompt from config
        system_prompt = self.config.get('llm', {}).get('system_prompt', 
            "You are an AI assistant specialized in Mars habitat resource management.")
        
        # Generate JSON response
        result = self.ollama.generate_json(
            prompt=prompt,
            model=self.model_name,
            system=system_prompt,
            schema=self.schemas["action"],
            temperature=0.2  # Lower temperature for more deterministic decisions
        )
        
        # Extract action from response
        action = result.get("parsed_json")
        
        # Fallback to default action if parsing failed
        if not action:
            logger.warning("Failed to parse LLM response, using default action")
            action = {
                "power_allocation": {
                    "life_support": 4.0,
                    "isru": 3.0,
                    "thermal_control": 3.0
                },
                "isru_mode": "both",
                "maintenance_target": None
            }
        
        return action
    
    def explain_decision(self, state: Dict[str, Any], action: Dict[str, Any]) -> str:
        """
        Generate explanation for a decision.
        
        Args:
            state: Environment state dictionary
            action: Action dictionary
            
        Returns:
            Explanation string
        """
        # Format state as prompt
        state_prompt = self.format_state_prompt(state)
        
        # Format action
        action_str = f"""
        Decision Made:
        - Power to Life Support: {action['power_allocation']['life_support']:.2f} kW
        - Power to ISRU: {action['power_allocation']['isru']:.2f} kW
        - Power to Thermal Control: {action['power_allocation']['thermal_control']:.2f} kW
        - ISRU Mode: {action['isru_mode']}
        - Maintenance Target: {action['maintenance_target'] if action['maintenance_target'] else 'None'}
        """
        
        # Create prompt for explanation
        prompt = f"""
        {state_prompt}
        
        {action_str}
        
        Explain why this decision is optimal given the current state. Consider resource levels, environmental conditions, and subsystem status in your explanation. Keep your explanation concise but informative.
        """
        
        # Get system prompt from config
        system_prompt = self.config.get('llm', {}).get('system_prompt', 
            "You are an AI assistant specialized in Mars habitat resource management.")
        
        # Generate explanation
        result = self.ollama.generate(
            prompt=prompt,
            model=self.model_name,
            system=system_prompt
        )
        
        return result.get("response", "No explanation available.")
    
    def generate_scenario(self, difficulty: str = "normal", 
                         scenario_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a scenario for training or testing.
        
        Args:
            difficulty: Scenario difficulty (easy, normal, hard)
            scenario_type: Type of scenario (dust_storm, solar_flare, etc.)
            
        Returns:
            Scenario dictionary
        """
        # Define scenario types
        scenario_types = [
            "dust_storm", "solar_flare", "equipment_failure", 
            "resource_scarcity", "normal_operations"
        ]
        
        # Select random scenario type if not specified
        if scenario_type is None:
            scenario_type = np.random.choice(scenario_types)
        
        # Create prompt for scenario generation
        prompt = f"""
        Generate a detailed Mars habitat scenario for reinforcement learning training.
        
        Difficulty: {difficulty}
        Scenario Type: {scenario_type}
        
        Provide a description of the scenario including:
        1. Environmental conditions (temperature, pressure, dust opacity, solar irradiance)
        2. Initial resource levels (power, water, oxygen, food)
        3. Subsystem status (power system, life support, ISRU, thermal control)
        4. Any special events or challenges
        
        The scenario should be challenging but solvable, with appropriate difficulty level.
        """
        
        # Get system prompt from config
        system_prompt = self.config.get('llm', {}).get('system_prompt', 
            "You are an AI assistant specialized in Mars habitat resource management.")
        
        # Generate JSON response
        result = self.ollama.generate_json(
            prompt=prompt,
            model=self.model_name,
            system=system_prompt,
            schema=self.schemas["scenario"]
        )
        
        # Extract scenario from response
        scenario = result.get("parsed_json")
        
        # Fallback to default scenario if parsing failed
        if not scenario:
            logger.warning("Failed to parse LLM response, using default scenario")
            scenario = self._default_scenario(difficulty, scenario_type)
        
        return scenario
    
    def _default_scenario(self, difficulty: str, scenario_type: str) -> Dict[str, Any]:
        """Generate a default scenario based on difficulty and type."""
        # Adjust parameters based on difficulty
        if difficulty == "easy":
            resource_multiplier = 1.5
            maintenance_level = 0.1
        elif difficulty == "hard":
            resource_multiplier = 0.7
            maintenance_level = 0.4
        else:  # normal
            resource_multiplier = 1.0
            maintenance_level = 0.2
        
        # Base scenario
        base_scenario = {
            "description": f"{difficulty.title()} {scenario_type.replace('_', ' ')} scenario",
            "environment": {
                "temperature": -60,
                "pressure": 650,
                "dust_opacity": 0.3,
                "solar_irradiance": 500
            },
            "resources": {
                "power": 100 * resource_multiplier,
                "water": 500 * resource_multiplier,
                "oxygen": 200 * resource_multiplier,
                "food": 300 * resource_multiplier
            },
            "subsystems": {
                "power_system": {"status": "operational", "maintenance_needed": maintenance_level},
                "life_support": {"status": "operational", "maintenance_needed": maintenance_level},
                "isru": {"status": "operational", "maintenance_needed": maintenance_level},
                "thermal_control": {"status": "operational", "maintenance_needed": maintenance_level}
            },
            "events": []
        }
        
        # Modify scenario based on type
        if scenario_type == "dust_storm":
            base_scenario["description"] = f"A {difficulty} dust storm is approaching the habitat"
            base_scenario["environment"]["dust_opacity"] = 0.7
            base_scenario["environment"]["solar_irradiance"] = 300
            base_scenario["events"].append({
                "type": "dust_storm_intensifies",
                "time": {"sol": 1, "hour": 12},
                "description": "The dust storm intensifies, reducing solar power generation",
                "effects": {
                    "target": "solar_irradiance",
                    "magnitude": -100
                }
            })
        elif scenario_type == "solar_flare":
            base_scenario["description"] = f"A {difficulty} solar flare is affecting communications"
            base_scenario["environment"]["solar_irradiance"] = 600
            base_scenario["events"].append({
                "type": "solar_flare",
                "time": {"sol": 1, "hour": 8},
                "description": "Solar flare causes electronics interference",
                "effects": {
                    "target": "power_system",
                    "magnitude": 0.2
                }
            })
        elif scenario_type == "equipment_failure":
            base_scenario["description"] = f"A {difficulty} equipment failure in the habitat"
            failed_system = np.random.choice(["power_system", "life_support", "isru", "thermal_control"])
            base_scenario["subsystems"][failed_system]["status"] = "degraded"
            base_scenario["subsystems"][failed_system]["maintenance_needed"] = 0.7
        elif scenario_type == "resource_scarcity":
            base_scenario["description"] = f"A {difficulty} resource scarcity situation"
            base_scenario["resources"]["power"] *= 0.6
            base_scenario["resources"]["water"] *= 0.6
            base_scenario["resources"]["oxygen"] *= 0.6
        
        return base_scenario
    
    def answer_query(self, query: str, state: Optional[Dict[str, Any]] = None) -> str:
        """
        Answer a natural language query about Mars habitat management.
        
        Args:
            query: User query string
            state: Current environment state (optional)
            
        Returns:
            Response string
        """
        # Create prompt based on whether state is provided
        if state:
            state_prompt = self.format_state_prompt(state)
            prompt = f"""
            {state_prompt}
            
            User Query: {query}
            
            Respond to the user's query with helpful information based on the current state of the habitat. If the query is about a specific aspect of habitat management, provide relevant details and recommendations.
            """
        else:
            prompt = f"""
            You are an AI assistant specialized in Mars habitat resource management.
            
            User Query: {query}
            
            Respond to the user's query with helpful information about Mars habitat management. If the query is about a specific aspect of habitat management, provide relevant details and recommendations.
            """
        
        # Get system prompt from config
        system_prompt = self.config.get('llm', {}).get('system_prompt', 
            "You are an AI assistant specialized in Mars habitat resource management.")
        
        # Generate response
        result = self.ollama.generate(
            prompt=prompt,
            model=self.model_name,
            system=system_prompt
        )
        
        return result.get("response", "I'm sorry, I couldn't generate a response.")
    
    def get_state_embedding(self, state: Dict[str, Any]) -> List[float]:
        """
        Get embedding vector for a state.
        
        Args:
            state: Environment state dictionary
            
        Returns:
            Embedding vector
        """
        # Format state as text
        state_text = self.format_state_prompt(state)
        
        # Get embeddings
        result = self.ollama.get_embeddings(state_text, model=self.model_name)
        
        return result.get("embedding", [])
    
    def fine_tune_model(self, training_data_path: str, output_model_name: str) -> bool:
        """
        Fine-tune the LLM model on Mars habitat management data.
        
        Args:
            training_data_path: Path to training data file
            output_model_name: Name for the fine-tuned model
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Fine-tuning model {self.model_name} to create {output_model_name}")
        
        # Check if training data exists
        if not os.path.exists(training_data_path):
            logger.error(f"Training data file not found: {training_data_path}")
            return False
        
        try:
            # Create Modelfile
            modelfile_path = os.path.join(self.data_dir, "Modelfile")
            with open(modelfile_path, 'w') as f:
                f.write(f"FROM {self.model_name}\n")
                f.write("SYSTEM You are an AI assistant specialized in Mars habitat resource management.\n")
                f.write("TEMPLATE <s>[INST] {{ .System }} {{ .Prompt }} [/INST] {{ .Response }}</s>\n")
                f.write("PARAMETER temperature 0.7\n")
                f.write("PARAMETER top_p 0.9\n")
            
            # Create model
            logger.info(f"Creating model {output_model_name}")
            os.system(f"ollama create {output_model_name} -f {modelfile_path}")
            
            # Fine-tune model
            logger.info(f"Fine-tuning model with data from {training_data_path}")
            os.system(f"ollama run {output_model_name} -f {training_data_path}")
            
            logger.info(f"Fine-tuning complete: {output_model_name}")
            return True
        except Exception as e:
            logger.error(f"Error during fine-tuning: {e}")
            return False


class MarsHabitatLLMRL:
    """
    Integration of LLMs with Reinforcement Learning for Mars Habitat management.
    
    This class provides methods to:
    1. Use LLMs as policy networks in RL
    2. Enhance RL with LLM-based explanations
    3. Generate diverse training scenarios
    4. Provide natural language interface to RL systems
    """
    
    def __init__(self, data_dir: str, 
                 model_name: str = "llama2",
                 rl_agent_path: Optional[str] = None,
                 cache_dir: Optional[str] = None):
        """
        Initialize the Mars Habitat LLM-RL integration.
        
        Args:
            data_dir: Directory containing data and configuration files
            model_name: Name of the Ollama model to use
            rl_agent_path: Path to pre-trained RL agent (optional)
            cache_dir: Directory to cache LLM responses
        """
        self.data_dir = data_dir
        self.model_name = model_name
        
        # Initialize LLM agent
        self.llm_agent = MarsHabitatLLMAgent(
            data_dir=data_dir,
            model_name=model_name,
            cache_dir=cache_dir
        )
        
        # Load RL agent if provided
        self.rl_agent = None
        if rl_agent_path:
            self._load_rl_agent(rl_agent_path)
        
        # Initialize metrics
        self.metrics = {
            "llm_actions": 0,
            "rl_actions": 0,
            "hybrid_actions": 0,
            "rewards": []
        }
        
        logger.info(f"Mars Habitat LLM-RL integration initialized with model {model_name}")
    
    def _load_rl_agent(self, agent_path: str) -> None:
        """Load pre-trained RL agent."""
        try:
            # This is a placeholder - actual implementation would depend on RL framework
            logger.info(f"Loading RL agent from {agent_path}")
            # self.rl_agent = torch.load(agent_path)
            logger.info("RL agent loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load RL agent: {e}")
    
    def select_action(self, state: Dict[str, Any], 
                     mode: str = "hybrid",
                     temperature: float = 0.7) -> Dict[str, Any]:
        """
        Select action using the specified mode.
        
        Args:
            state: Environment state dictionary
            mode: Action selection mode (llm, rl, hybrid)
            temperature: Temperature for action selection
            
        Returns:
            Action dictionary
        """
        if mode == "llm" or (mode == "hybrid" and self.rl_agent is None):
            # Use LLM for action selection
            action = self.llm_agent.select_action(state)
            self.metrics["llm_actions"] += 1
            return action
        
        elif mode == "rl" and self.rl_agent is not None:
            # Use RL agent for action selection
            action = self._select_rl_action(state)
            self.metrics["rl_actions"] += 1
            return action
        
        elif mode == "hybrid" and self.rl_agent is not None:
            # Use hybrid approach
            llm_action = self.llm_agent.select_action(state)
            rl_action = self._select_rl_action(state)
            
            # Combine actions based on temperature
            # Higher temperature = more weight to LLM
            action = self._combine_actions(llm_action, rl_action, temperature)
            self.metrics["hybrid_actions"] += 1
            return action
        
        else:
            logger.warning(f"Invalid mode {mode} or missing RL agent, using LLM")
            action = self.llm_agent.select_action(state)
            self.metrics["llm_actions"] += 1
            return action
    
    def _select_rl_action(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Select action using RL agent."""
        # This is a placeholder - actual implementation would depend on RL framework
        # In a real implementation, this would preprocess state and use RL agent
        
        # Fallback to default action
        return {
            "power_allocation": {
                "life_support": 4.0,
                "isru": 3.0,
                "thermal_control": 3.0
            },
            "isru_mode": "both",
            "maintenance_target": None
        }
    
    def _combine_actions(self, llm_action: Dict[str, Any], 
                        rl_action: Dict[str, Any],
                        temperature: float) -> Dict[str, Any]:
        """
        Combine actions from LLM and RL agents.
        
        Args:
            llm_action: Action from LLM agent
            rl_action: Action from RL agent
            temperature: Mixing parameter (0 = all RL, 1 = all LLM)
            
        Returns:
            Combined action
        """
        # Ensure temperature is in [0, 1]
        temperature = max(0, min(1, temperature))
        
        # Combine power allocation
        combined_power = {}
        for subsystem in llm_action["power_allocation"]:
            llm_value = llm_action["power_allocation"].get(subsystem, 0)
            rl_value = rl_action["power_allocation"].get(subsystem, 0)
            combined_power[subsystem] = temperature * llm_value + (1 - temperature) * rl_value
        
        # For discrete choices, use weighted random selection
        if np.random.random() < temperature:
            isru_mode = llm_action["isru_mode"]
        else:
            isru_mode = rl_action["isru_mode"]
        
        if np.random.random() < temperature:
            maintenance_target = llm_action.get("maintenance_target")
        else:
            maintenance_target = rl_action.get("maintenance_target")
        
        return {
            "power_allocation": combined_power,
            "isru_mode": isru_mode,
            "maintenance_target": maintenance_target
        }
    
    def explain_action(self, state: Dict[str, Any], action: Dict[str, Any]) -> str:
        """
        Generate explanation for an action.
        
        Args:
            state: Environment state dictionary
            action: Action dictionary
            
        Returns:
            Explanation string
        """
        return self.llm_agent.explain_decision(state, action)
    
    def generate_training_scenarios(self, num_scenarios: int = 10, 
                                  difficulties: Optional[List[str]] = None,
                                  scenario_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Generate scenarios for training RL agents.
        
        Args:
            num_scenarios: Number of scenarios to generate
            difficulties: List of difficulties to include
            scenario_types: List of scenario types to include
            
        Returns:
            List of scenario dictionaries
        """
        if difficulties is None:
            difficulties = ["easy", "normal", "hard"]
        
        if scenario_types is None:
            scenario_types = [
                "dust_storm", "solar_flare", "equipment_failure", 
                "resource_scarcity", "normal_operations"
            ]
        
        scenarios = []
        for _ in range(num_scenarios):
            difficulty = np.random.choice(difficulties)
            scenario_type = np.random.choice(scenario_types)
            
            scenario = self.llm_agent.generate_scenario(
                difficulty=difficulty,
                scenario_type=scenario_type
            )
            
            scenarios.append(scenario)
        
        return scenarios
    
    def run_episode(self, env, max_steps: int = 500, 
                   mode: str = "hybrid",
                   render: bool = False) -> Tuple[float, int, List[Dict[str, Any]]]:
        """
        Run a full episode using the specified action selection mode.
        
        Args:
            env: Environment to run episode in
            max_steps: Maximum steps per episode
            mode: Action selection mode (llm, rl, hybrid)
            render: Whether to render the environment
            
        Returns:
            Tuple of (total_reward, steps, trajectory)
        """
        state = env.reset()
        total_reward = 0
        trajectory = []
        
        for step in range(max_steps):
            # Select action
            action = self.select_action(state, mode=mode)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Update metrics
            total_reward += reward
            
            # Store transition
            transition = {
                "state": state,
                "action": action,
                "reward": reward,
                "next_state": next_state,
                "done": done,
                "info": info
            }
            trajectory.append(transition)
            
            # Render if requested
            if render:
                env.render()
            
            # Update state
            state = next_state
            
            # Check if episode is done
            if done:
                break
        
        # Update metrics
        self.metrics["rewards"].append(total_reward)
        
        return total_reward, step + 1, trajectory
    
    def evaluate(self, env, num_episodes: int = 10, 
                mode: str = "hybrid") -> Dict[str, Any]:
        """
        Evaluate performance over multiple episodes.
        
        Args:
            env: Environment to evaluate in
            num_episodes: Number of episodes to evaluate
            mode: Action selection mode (llm, rl, hybrid)
            
        Returns:
            Dictionary of evaluation metrics
        """
        rewards = []
        episode_lengths = []
        
        for episode in range(num_episodes):
            reward, length, _ = self.run_episode(env, mode=mode)
            rewards.append(reward)
            episode_lengths.append(length)
            
            logger.info(f"Episode {episode+1}/{num_episodes}: Reward = {reward:.2f}, Length = {length}")
        
        # Calculate metrics
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        mean_length = np.mean(episode_lengths)
        
        return {
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "mean_length": mean_length,
            "rewards": rewards,
            "episode_lengths": episode_lengths
        }
    
    def save_metrics(self, path: Optional[str] = None) -> str:
        """
        Save metrics to file.
        
        Args:
            path: Path to save metrics (optional)
            
        Returns:
            Path where metrics were saved
        """
        if path is None:
            path = os.path.join(self.data_dir, "llm_rl_metrics.json")
        
        with open(path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        logger.info(f"Metrics saved to {path}")
        return path
    
    def create_training_data(self, num_examples: int = 100, 
                           output_path: Optional[str] = None) -> str:
        """
        Create training data for fine-tuning LLMs.
        
        Args:
            num_examples: Number of examples to generate
            output_path: Path to save training data (optional)
            
        Returns:
            Path where training data was saved
        """
        if output_path is None:
            output_path = os.path.join(self.data_dir, "llm_training_data.txt")
        
        # Generate diverse scenarios
        scenarios = self.generate_training_scenarios(num_scenarios=num_examples)
        
        with open(output_path, 'w') as f:
            for i, scenario in enumerate(scenarios):
                # Create state from scenario
                state = {
                    "time": [1, 12],  # Default time
                    "environment": scenario["environment"],
                    "habitat": scenario["resources"],
                    "subsystems": scenario["subsystems"]
                }
                
                # Format state as prompt
                state_prompt = self.llm_agent.format_state_prompt(state)
                
                # Generate optimal action
                action = self.select_action(state, mode="hybrid")
                
                # Generate explanation
                explanation = self.explain_action(state, action)
                
                # Format as training example
                f.write(f"<s>[INST] You are an AI assistant for Mars habitat management. The current state is:\n\n")
                f.write(f"{state_prompt}\n\n")
                f.write(f"What is the optimal resource allocation strategy? [/INST]\n")
                
                # Format action as response
                f.write(f"Based on the current state, I recommend the following allocation:\n\n")
                f.write(f"Power Allocation:\n")
                for subsystem, value in action["power_allocation"].items():
                    f.write(f"- {subsystem.replace('_', ' ').title()}: {value:.1f} kW\n")
                f.write(f"\nISRU Mode: {action['isru_mode'].title()}\n")
                f.write(f"Maintenance Target: {action['maintenance_target'].replace('_', ' ').title() if action['maintenance_target'] else 'None'}\n\n")
                f.write(f"{explanation}</s>\n\n")
        
        logger.info(f"Training data saved to {output_path}")
        return output_path


# Example usage
if __name__ == "__main__":
    # Set up data directory
    data_dir = "/home/ubuntu/martian_habitat_pathfinder/data"
    
    # Initialize LLM-RL integration
    llm_rl = MarsHabitatLLMRL(
        data_dir=data_dir,
        model_name="llama2"
    )
    
    # Example state
    example_state = {
        "time": [10, 14],
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
    }
    
    # Select action using LLM
    action = llm_rl.select_action(example_state, mode="llm")
    print("LLM Action:", action)
    
    # Generate explanation
    explanation = llm_rl.explain_action(example_state, action)
    print("\nExplanation:", explanation)
    
    # Generate scenario
    scenario = llm_rl.llm_agent.generate_scenario(difficulty="hard", scenario_type="dust_storm")
    print("\nScenario:", scenario["description"])
    
    # Create training data
    training_data_path = llm_rl.create_training_data(num_examples=5)
    print(f"\nTraining data saved to: {training_data_path}")
