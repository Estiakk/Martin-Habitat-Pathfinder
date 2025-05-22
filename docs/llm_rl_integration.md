# Integrating LLMs with Reinforcement Learning Using Ollama

This guide explains how to integrate Large Language Models (LLMs) with Reinforcement Learning (RL) using Ollama in the Martian Habitat Pathfinder system.

## Overview

Combining LLMs with RL creates powerful hybrid systems that leverage the reasoning capabilities of language models with the decision-making optimization of reinforcement learning. Ollama provides a convenient way to run LLMs locally, making this integration practical for Mars habitat management.

## Benefits of LLM-RL Integration

1. **Enhanced Decision Making**: LLMs can reason through complex scenarios with nuanced understanding of context
2. **Natural Language Explanations**: LLMs can explain RL decisions in human-readable format
3. **Improved Generalization**: LLMs can generalize to novel situations better than traditional RL networks
4. **Scenario Generation**: LLMs can help create diverse training scenarios for RL agents
5. **Human-AI Collaboration**: Natural language interface improves human operator interaction

## Installation Requirements

1. Install Ollama:
```bash
# For Linux
curl -fsSL https://ollama.com/install.sh | sh

# For macOS
curl -fsSL https://ollama.com/install.sh | sh

# For Windows, download from https://ollama.com/download
```

2. Install Python client:
```bash
pip install ollama
```

3. Pull required models:
```bash
ollama pull llama2
# Or other models as needed
ollama pull mistral
ollama pull gemma:7b
```

## Implementation Approaches

### 1. LLM as Policy Network

Use an LLM to directly map observations to actions:

```python
import ollama
import json
from simulations.rl_environment import MarsHabitatRLEnvironment

class LLMPolicyAgent:
    def __init__(self, model_name="llama2"):
        self.model_name = model_name
        
    def select_action(self, state):
        # Format state as prompt
        prompt = self._format_state_prompt(state)
        
        # Get LLM response
        response = ollama.generate(model=self.model_name, prompt=prompt)
        
        # Parse response to extract action
        action = self._parse_action_from_response(response['response'])
        
        return action
    
    def _format_state_prompt(self, state):
        # Create a detailed prompt describing the current state
        prompt = f"""
        You are an AI assistant managing a Mars habitat with the following state:
        
        Time: Sol {state['time'][0]}, Hour {state['time'][1]}
        
        Habitat Resources:
        - Power: {state['habitat']['power']:.2f} kWh
        - Water: {state['habitat']['water']:.2f} liters
        - Oxygen: {state['habitat']['oxygen']:.2f} kg
        - Food: {state['habitat']['food']:.2f} kg
        - Spare Parts: {state['habitat']['spare_parts']:.2f} units
        
        Environmental Conditions:
        - Temperature: {state['environment']['temperature']:.2f} °C
        - Pressure: {state['environment']['pressure']:.2f} Pa
        - Dust Opacity: {state['environment']['dust_opacity']:.2f}
        - Solar Irradiance: {state['environment']['solar_irradiance']:.2f} W/m²
        
        Subsystem Status:
        - Power System: {state['subsystems']['power_system']['status']} (Maintenance: {state['subsystems']['power_system']['maintenance_needed']:.2f})
        - Life Support: {state['subsystems']['life_support']['status']} (Maintenance: {state['subsystems']['life_support']['maintenance_needed']:.2f})
        - ISRU: {state['subsystems']['isru']['status']} (Maintenance: {state['subsystems']['isru']['maintenance_needed']:.2f})
        - Thermal Control: {state['subsystems']['thermal_control']['status']} (Maintenance: {state['subsystems']['thermal_control']['maintenance_needed']:.2f})
        
        Based on this information, determine the optimal resource allocation strategy.
        
        Provide your response in the following JSON format:
        {{
            "power_allocation": {{
                "life_support": <value between 0-10>,
                "isru": <value between 0-10>,
                "thermal_control": <value between 0-10>
            }},
            "isru_mode": <"water", "oxygen", "both", or "off">,
            "maintenance_target": <"power_system", "life_support", "isru", "thermal_control", or null>
        }}
        
        Only respond with the JSON object, no additional text.
        """
        return prompt
    
    def _parse_action_from_response(self, response_text):
        # Extract JSON from response
        try:
            # Find JSON in the response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                action = json.loads(json_str)
                return action
            else:
                # Fallback to default action if JSON not found
                return {
                    'power_allocation': {
                        'life_support': 4.0,
                        'isru': 3.0,
                        'thermal_control': 3.0
                    },
                    'isru_mode': 'both',
                    'maintenance_target': None
                }
        except json.JSONDecodeError:
            # Fallback to default action if JSON parsing fails
            return {
                'power_allocation': {
                    'life_support': 4.0,
                    'isru': 3.0,
                    'thermal_control': 3.0
                },
                'isru_mode': 'both',
                'maintenance_target': None
            }

# Example usage
if __name__ == "__main__":
    # Create environment
    env = MarsHabitatRLEnvironment()
    state = env.reset()
    
    # Create LLM policy agent
    agent = LLMPolicyAgent(model_name="llama2")
    
    # Run simulation for 10 steps
    for i in range(10):
        # Select action using LLM
        action = agent.select_action(state)
        
        # Take action in environment
        next_state, reward, done, info = env.step(action)
        
        # Print results
        print(f"Step {i+1}:")
        print(f"  Action: {action}")
        print(f"  Reward: {reward:.2f}")
        
        # Update state
        state = next_state
        
        # Check if simulation is done
        if done:
            print("Simulation ended early")
            break
```

### 2. LLM for Decision Explanation

Use an LLM to explain decisions made by traditional RL agents:

```python
import ollama
from models.decision_integration import DecisionIntegrationSystem

class ExplainableDecisionSystem:
    def __init__(self, data_dir, model_name="llama2"):
        self.integration = DecisionIntegrationSystem(data_dir)
        self.model_name = model_name
    
    def make_decision(self, state):
        # Get decision from integration system
        decision = self.integration.make_decision(state)
        
        # Generate explanation using LLM
        explanation = self.explain_decision(state, decision)
        
        return decision, explanation
    
    def explain_decision(self, state, decision):
        # Format prompt for explanation
        prompt = f"""
        You are an AI assistant explaining decisions for Mars habitat resource management.
        
        Current State:
        - Power: {state['habitat']['power']:.2f} kWh
        - Water: {state['habitat']['water']:.2f} liters
        - Oxygen: {state['habitat']['oxygen']:.2f} kg
        - Food: {state['habitat']['food']:.2f} kg
        - Temperature: {state['environment']['temperature']:.2f} °C
        - Dust Opacity: {state['environment']['dust_opacity']:.2f}
        - Solar Irradiance: {state['environment']['solar_irradiance']:.2f} W/m²
        
        Decision Made:
        - Power to Life Support: {decision['power_allocation']['life_support']:.2f} kW
        - Power to ISRU: {decision['power_allocation']['isru']:.2f} kW
        - Power to Thermal Control: {decision['power_allocation']['thermal_control']:.2f} kW
        - ISRU Mode: {decision['isru_mode']}
        - Maintenance Target: {decision['maintenance_target'] if decision['maintenance_target'] else 'None'}
        
        Explain why this decision is optimal given the current state. Consider resource levels, environmental conditions, and subsystem status in your explanation. Keep your explanation concise but informative.
        """
        
        # Get explanation from LLM
        response = ollama.generate(model=self.model_name, prompt=prompt)
        
        return response['response']

# Example usage
if __name__ == "__main__":
    # Create explainable decision system
    system = ExplainableDecisionSystem("/path/to/data")
    
    # Get state from environment
    env = MarsHabitatRLEnvironment()
    state = env.reset()
    
    # Make decision with explanation
    decision, explanation = system.make_decision(state)
    
    # Print decision and explanation
    print("Decision:")
    print(f"  Power to Life Support: {decision['power_allocation']['life_support']:.2f} kW")
    print(f"  Power to ISRU: {decision['power_allocation']['isru']:.2f} kW")
    print(f"  Power to Thermal Control: {decision['power_allocation']['thermal_control']:.2f} kW")
    print(f"  ISRU Mode: {decision['isru_mode']}")
    print(f"  Maintenance Target: {decision['maintenance_target'] if decision['maintenance_target'] else 'None'}")
    print("\nExplanation:")
    print(explanation)
```

### 3. LLM for Scenario Generation

Use an LLM to generate diverse training scenarios:

```python
import ollama
import json
import numpy as np
from simulations.rl_formulation import MarsHabitatRLFormulation

class ScenarioGenerator:
    def __init__(self, model_name="llama2"):
        self.model_name = model_name
        self.formulation = MarsHabitatRLFormulation("/path/to/data")
    
    def generate_scenario(self, difficulty="normal", scenario_type=None):
        # Generate scenario description using LLM
        scenario_desc = self._generate_scenario_description(difficulty, scenario_type)
        
        # Convert description to environment parameters
        env_params = self._convert_description_to_params(scenario_desc)
        
        # Create environment with parameters
        env = self.formulation.create_environment(**env_params)
        
        return env, scenario_desc
    
    def _generate_scenario_description(self, difficulty, scenario_type):
        # Define scenario types
        scenario_types = [
            "dust_storm", "solar_flare", "equipment_failure", 
            "resource_scarcity", "normal_operations"
        ]
        
        # Select random scenario type if not specified
        if scenario_type is None:
            scenario_type = np.random.choice(scenario_types)
        
        # Format prompt for scenario generation
        prompt = f"""
        Generate a detailed Mars habitat scenario for reinforcement learning training.
        
        Difficulty: {difficulty}
        Scenario Type: {scenario_type}
        
        Provide a description of the scenario including:
        1. Environmental conditions (temperature, pressure, dust opacity, solar irradiance)
        2. Initial resource levels (power, water, oxygen, food)
        3. Subsystem status (power system, life support, ISRU, thermal control)
        4. Any special events or challenges
        
        Format your response as a JSON object with the following structure:
        {{
            "description": "Brief description of the scenario",
            "environment": {{
                "temperature": <value>,
                "pressure": <value>,
                "dust_opacity": <value>,
                "solar_irradiance": <value>
            }},
            "resources": {{
                "power": <value>,
                "water": <value>,
                "oxygen": <value>,
                "food": <value>
            }},
            "subsystems": {{
                "power_system": {{"status": "<status>", "maintenance_needed": <value>}},
                "life_support": {{"status": "<status>", "maintenance_needed": <value>}},
                "isru": {{"status": "<status>", "maintenance_needed": <value>}},
                "thermal_control": {{"status": "<status>", "maintenance_needed": <value>}}
            }},
            "events": [
                {{
                    "type": "<event_type>",
                    "time": {{"sol": <value>, "hour": <value>}},
                    "description": "Description of the event",
                    "effects": {{
                        "target": "<target>",
                        "magnitude": <value>
                    }}
                }}
            ]
        }}
        
        Only respond with the JSON object, no additional text.
        """
        
        # Get scenario from LLM
        response = ollama.generate(model=self.model_name, prompt=prompt)
        
        # Extract JSON from response
        try:
            # Find JSON in the response
            start_idx = response['response'].find('{')
            end_idx = response['response'].rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response['response'][start_idx:end_idx]
                scenario = json.loads(json_str)
                return scenario
            else:
                # Return default scenario if JSON not found
                return self._default_scenario(difficulty, scenario_type)
        except json.JSONDecodeError:
            # Return default scenario if JSON parsing fails
            return self._default_scenario(difficulty, scenario_type)
    
    def _default_scenario(self, difficulty, scenario_type):
        # Create default scenario based on difficulty and type
        if scenario_type == "dust_storm":
            return {
                "description": "A major dust storm is approaching the habitat",
                "environment": {
                    "temperature": -80,
                    "pressure": 650,
                    "dust_opacity": 0.8,
                    "solar_irradiance": 200
                },
                "resources": {
                    "power": 100,
                    "water": 500,
                    "oxygen": 200,
                    "food": 300
                },
                "subsystems": {
                    "power_system": {"status": "operational", "maintenance_needed": 0.1},
                    "life_support": {"status": "operational", "maintenance_needed": 0.2},
                    "isru": {"status": "operational", "maintenance_needed": 0.3},
                    "thermal_control": {"status": "operational", "maintenance_needed": 0.1}
                },
                "events": [
                    {
                        "type": "dust_storm_intensifies",
                        "time": {"sol": 1, "hour": 12},
                        "description": "The dust storm intensifies, reducing solar power generation",
                        "effects": {
                            "target": "solar_irradiance",
                            "magnitude": -100
                        }
                    }
                ]
            }
        else:
            # Default normal scenario
            return {
                "description": "Normal operations with standard conditions",
                "environment": {
                    "temperature": -60,
                    "pressure": 650,
                    "dust_opacity": 0.3,
                    "solar_irradiance": 500
                },
                "resources": {
                    "power": 100,
                    "water": 500,
                    "oxygen": 200,
                    "food": 300
                },
                "subsystems": {
                    "power_system": {"status": "operational", "maintenance_needed": 0.1},
                    "life_support": {"status": "operational", "maintenance_needed": 0.1},
                    "isru": {"status": "operational", "maintenance_needed": 0.1},
                    "thermal_control": {"status": "operational", "maintenance_needed": 0.1}
                },
                "events": []
            }
    
    def _convert_description_to_params(self, scenario_desc):
        # Convert scenario description to environment parameters
        params = {
            "initial_temperature": scenario_desc["environment"]["temperature"],
            "initial_pressure": scenario_desc["environment"]["pressure"],
            "initial_dust_opacity": scenario_desc["environment"]["dust_opacity"],
            "initial_solar_irradiance": scenario_desc["environment"]["solar_irradiance"],
            "initial_power": scenario_desc["resources"]["power"],
            "initial_water": scenario_desc["resources"]["water"],
            "initial_oxygen": scenario_desc["resources"]["oxygen"],
            "initial_food": scenario_desc["resources"]["food"]
        }
        
        # Add subsystem status
        for subsystem, status in scenario_desc["subsystems"].items():
            params[f"initial_{subsystem}_status"] = status["status"]
            params[f"initial_{subsystem}_maintenance"] = status["maintenance_needed"]
        
        # Add events
        if "events" in scenario_desc and scenario_desc["events"]:
            params["events"] = scenario_desc["events"]
        
        return params

# Example usage
if __name__ == "__main__":
    # Create scenario generator
    generator = ScenarioGenerator()
    
    # Generate scenarios
    dust_storm_env, dust_storm_desc = generator.generate_scenario(scenario_type="dust_storm")
    equipment_failure_env, equipment_failure_desc = generator.generate_scenario(scenario_type="equipment_failure")
    
    # Print scenario descriptions
    print("Dust Storm Scenario:")
    print(dust_storm_desc["description"])
    print("\nEquipment Failure Scenario:")
    print(equipment_failure_desc["description"])
```

### 4. LLM for Human-AI Interface

Use an LLM to enhance the dashboard with natural language interaction:

```python
import ollama
import dash
from dash import dcc, html, Input, Output, State
import json

class NaturalLanguageInterface:
    def __init__(self, dashboard, model_name="llama2"):
        self.dashboard = dashboard
        self.model_name = model_name
        
        # Add natural language components to dashboard
        self._add_nl_components()
        
        # Add callbacks
        self._add_nl_callbacks()
    
    def _add_nl_components(self):
        # Add natural language input and output to dashboard layout
        nl_components = html.Div([
            html.H2("Natural Language Interface"),
            dcc.Input(
                id="nl-input",
                type="text",
                placeholder="Ask a question or give a command...",
                style={"width": "100%"}
            ),
            html.Button("Submit", id="nl-submit", n_clicks=0),
            html.Div(id="nl-output", style={"marginTop": "10px", "padding": "10px", "backgroundColor": "#f0f0f0", "borderRadius": "5px"})
        ], style={"marginTop": "20px"})
        
        # Add to dashboard layout
        self.dashboard.app.layout.children.append(nl_components)
    
    def _add_nl_callbacks(self):
        # Add callback for natural language input
        @self.dashboard.app.callback(
            Output("nl-output", "children"),
            [Input("nl-submit", "n_clicks")],
            [State("nl-input", "value"),
             State("state-store", "children")]
        )
        def process_nl_input(n_clicks, nl_input, state_json):
            if n_clicks == 0 or not nl_input:
                return "Ask a question or give a command to interact with the system."
            
            if not state_json:
                return "Please start the simulation first."
            
            # Parse state
            state_dict = json.loads(state_json)
            
            # Process natural language input
            response = self._process_nl_input(nl_input, state_dict)
            
            return response
    
    def _process_nl_input(self, nl_input, state_dict):
        # Format prompt for LLM
        prompt = f"""
        You are an AI assistant for a Mars habitat management system. The user has provided the following input:
        
        "{nl_input}"
        
        Current State:
        - Time: Sol {state_dict['time'][0]}, Hour {state_dict['time'][1]}
        - Power: {state_dict['habitat'][0]:.2f} kWh
        - Water: {state_dict['habitat'][1]:.2f} liters
        - Oxygen: {state_dict['habitat'][2]:.2f} kg
        - Food: {state_dict['habitat'][3]:.2f} kg
        
        Environmental Conditions:
        - Temperature: {state_dict['environment'][0]:.2f} °C
        - Pressure: {state_dict['environment'][1]:.2f} Pa
        - Dust Opacity: {state_dict['environment'][2]:.2f}
        - Solar Irradiance: {state_dict['environment'][3]:.2f} W/m²
        
        Respond to the user's input with helpful information or suggestions based on the current state of the habitat. If the user is asking a question, provide an informative answer. If the user is giving a command, explain how that would affect the habitat and what actions would be taken.
        
        Keep your response concise, informative, and focused on Mars habitat management.
        """
        
        # Get response from LLM
        response = ollama.generate(model=self.model_name, prompt=prompt)
        
        return response['response']

# Example usage
if __name__ == "__main__":
    # Create dashboard
    from ui.dashboard import MarsHabitatDashboard
    dashboard = MarsHabitatDashboard("/path/to/data")
    
    # Add natural language interface
    nl_interface = NaturalLanguageInterface(dashboard)
    
    # Run dashboard
    dashboard.run_server(debug=True)
```

## Fine-tuning LLMs for Mars Habitat Management

For optimal performance, fine-tuning LLMs on domain-specific data is recommended:

1. **Create a Dataset**:
   - Generate examples of state-action pairs
   - Include explanations of decisions
   - Cover diverse scenarios and conditions

2. **Fine-tune with Ollama**:
```bash
# Create a fine-tuning dataset
cat > mars_habitat_finetune.txt << EOF
<s>[INST] You are an AI assistant for Mars habitat management. The current state is: Power: 120 kWh, Water: 450 liters, Oxygen: 180 kg, Food: 250 kg, Temperature: -70 °C, Dust Opacity: 0.4, Solar Irradiance: 450 W/m². What is the optimal power allocation? [/INST] Based on the current state, I recommend the following power allocation:
- Life Support: 5.0 kW (priority due to slightly low oxygen levels)
- ISRU: 3.0 kW (focus on water production)
- Thermal Control: 2.0 kW (adequate for current temperature)

This allocation ensures life support systems maintain oxygen levels while continuing water production at a moderate rate. </s>

<s>[INST] You are an AI assistant for Mars habitat management. The current state is: Power: 80 kWh, Water: 600 liters, Oxygen: 150 kg, Food: 300 kg, Temperature: -90 °C, Dust Opacity: 0.8, Solar Irradiance: 200 W/m². What is the optimal power allocation? [/INST] Based on the current state, I recommend the following power allocation:
- Life Support: 6.0 kW (high priority due to low oxygen levels)
- ISRU: 1.0 kW (minimal operation due to power constraints)
- Thermal Control: 3.0 kW (increased due to very low temperature)

This allocation prioritizes life support to address the critical oxygen levels while ensuring thermal control can maintain habitat temperature during the cold conditions. Power to ISRU is minimized since water levels are sufficient. </s>
EOF

# Create a Modelfile
cat > Modelfile << EOF
FROM llama2
SYSTEM You are an AI assistant specialized in Mars habitat resource management.
TEMPLATE <s>[INST] {{ .System }} {{ .Prompt }} [/INST] {{ .Response }}</s>
PARAMETER temperature 0.7
PARAMETER top_p 0.9
EOF

# Create the model
ollama create mars-habitat-assistant -f Modelfile

# Fine-tune the model
ollama run mars-habitat-assistant -f mars_habitat_finetune.txt
```

3. **Use the Fine-tuned Model**:
```python
import ollama

# Use fine-tuned model
response = ollama.generate(model="mars-habitat-assistant", prompt="The current state is: Power: 100 kWh, Water: 500 liters, Oxygen: 200 kg, Food: 300 kg, Temperature: -60 °C, Dust Opacity: 0.3, Solar Irradiance: 500 W/m². What is the optimal power allocation?")

print(response['response'])
```

## Best Practices

1. **Prompt Engineering**:
   - Be specific about the format of expected outputs
   - Include relevant context in prompts
   - Use consistent terminology across prompts

2. **Error Handling**:
   - Always implement robust parsing of LLM outputs
   - Provide fallback options when LLM responses are invalid
   - Log and analyze LLM failures to improve prompts

3. **Performance Optimization**:
   - Cache LLM responses for similar states
   - Use smaller models for real-time decisions
   - Batch process non-time-critical requests

4. **Evaluation**:
   - Compare LLM-based policies with traditional RL policies
   - Measure explanation quality and human understanding
   - Test across diverse scenarios

## Conclusion

Integrating LLMs with RL using Ollama provides a powerful approach for Mars habitat management, combining the reasoning capabilities of language models with the optimization strengths of reinforcement learning. This integration enhances decision-making, improves explainability, and enables more natural human-AI collaboration.

By following the implementation approaches and best practices outlined in this guide, you can create sophisticated hybrid systems that leverage the strengths of both LLMs and RL for more effective Mars habitat resource management.
