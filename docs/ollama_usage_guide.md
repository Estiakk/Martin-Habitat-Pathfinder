# Using LLMs with Ollama in Martian Habitat Pathfinder

This guide provides detailed instructions for using Large Language Models (LLMs) with Ollama in the Martian Habitat Pathfinder project.

## Table of Contents

1. [Installation and Setup](#installation-and-setup)
2. [Basic Usage](#basic-usage)
3. [Advanced Usage](#advanced-usage)
4. [Integration with Simulation](#integration-with-simulation)
5. [Dashboard Integration](#dashboard-integration)
6. [Fine-tuning Models](#fine-tuning-models)
7. [Troubleshooting](#troubleshooting)
8. [Performance Optimization](#performance-optimization)
9. [Examples](#examples)

## Installation and Setup

### Prerequisites

Before using LLMs with Ollama in the Martian Habitat Pathfinder project, ensure you have:

1. Ollama installed on your system
2. Required Python packages
3. Project files in place

### Installing Ollama

```bash
# For Linux
curl -fsSL https://ollama.com/install.sh | sh

# For macOS
curl -fsSL https://ollama.com/install.sh | sh

# For Windows, download from https://ollama.com/download
```

Verify installation:

```bash
ollama --version
```

### Installing Required Python Packages

```bash
pip install requests numpy tqdm
```

### Downloading Required Models

```bash
# Download the default model (llama2)
ollama pull llama2

# For better performance, you can use larger models
ollama pull llama2:13b
ollama pull mistral:7b
```

### Setting Up Project Structure

Ensure your project structure includes:

```
martian_habitat_pathfinder/
├── data/
│   ├── config.json
│   └── llm_cache/
├── models/
│   ├── ollama_integration.py
│   └── ...
├── simulations/
│   ├── simulation_llm_bridge.py
│   └── ...
└── ui/
    └── ...
```

Create the necessary directories:

```bash
mkdir -p martian_habitat_pathfinder/data/llm_cache
```

## Basic Usage

### Initializing the Ollama Client

```python
from models.ollama_integration import OllamaClient

# Initialize client
ollama_client = OllamaClient(
    base_url="http://localhost:11434",  # Default Ollama server URL
    default_model="llama2",             # Default model
    timeout=60,                         # Request timeout in seconds
    cache_dir="/path/to/cache"          # Optional cache directory
)

# Check available models
models = ollama_client.list_models()
print(f"Available models: {models}")

# Pull a model if not available
if "mistral" not in models:
    ollama_client.pull_model("mistral")
```

### Generating Text

```python
# Generate text
result = ollama_client.generate(
    prompt="Explain how to manage power during a dust storm on Mars",
    model="llama2",                 # Optional, defaults to default_model
    system="You are an AI assistant specialized in Mars habitat management",
    temperature=0.7,                # Controls randomness (0.0 to 1.0)
    max_tokens=2048                 # Maximum tokens to generate
)

# Print response
print(result["response"])
```

### Generating Structured JSON

```python
# Define a schema
schema = {
    "type": "object",
    "properties": {
        "power_allocation": {
            "type": "object",
            "properties": {
                "life_support": {"type": "number", "minimum": 0, "maximum": 10},
                "isru": {"type": "number", "minimum": 0, "maximum": 10},
                "thermal_control": {"type": "number", "minimum": 0, "maximum": 10}
            }
        },
        "isru_mode": {
            "type": "string",
            "enum": ["water", "oxygen", "both", "off"]
        }
    }
}

# Generate JSON
result = ollama_client.generate_json(
    prompt="Recommend power allocation during a dust storm on Mars",
    model="llama2",
    system="You are an AI assistant specialized in Mars habitat management",
    schema=schema,
    temperature=0.2  # Lower temperature for more deterministic output
)

# Access parsed JSON
if "parsed_json" in result and result["parsed_json"]:
    allocation = result["parsed_json"]
    print(f"Power to life support: {allocation['power_allocation']['life_support']} kW")
    print(f"ISRU mode: {allocation['isru_mode']}")
else:
    print("Failed to parse JSON response")
```

### Getting Embeddings

```python
# Get embeddings for a text
result = ollama_client.get_embeddings(
    text="Mars habitat power management during dust storm",
    model="llama2"
)

# Use embeddings for similarity comparison, clustering, etc.
embeddings = result.get("embedding", [])
print(f"Embedding dimension: {len(embeddings)}")
```

## Advanced Usage

### Using the Mars Habitat LLM Agent

The `MarsHabitatLLMAgent` class provides specialized functionality for Mars habitat management:

```python
from models.ollama_integration import MarsHabitatLLMAgent

# Initialize agent
agent = MarsHabitatLLMAgent(
    data_dir="/path/to/data",
    model_name="llama2",
    cache_dir="/path/to/cache"  # Optional
)

# Define a state
state = {
    "time": [10, 14],  # Sol 10, Hour 14
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

# Select action
action = agent.select_action(state)
print("Recommended action:", action)

# Get explanation
explanation = agent.explain_decision(state, action)
print("Explanation:", explanation)

# Generate scenario
scenario = agent.generate_scenario(difficulty="hard", scenario_type="dust_storm")
print("Scenario:", scenario["description"])

# Answer query
answer = agent.answer_query("How should I manage resources during a dust storm?", state)
print("Answer:", answer)
```

### Using the LLM-RL Integration

The `MarsHabitatLLMRL` class integrates LLMs with Reinforcement Learning:

```python
from models.ollama_integration import MarsHabitatLLMRL

# Initialize LLM-RL integration
llm_rl = MarsHabitatLLMRL(
    data_dir="/path/to/data",
    model_name="llama2",
    rl_agent_path="/path/to/rl_model.pth",  # Optional
    cache_dir="/path/to/cache"               # Optional
)

# Select action using different modes
llm_action = llm_rl.select_action(state, mode="llm")
rl_action = llm_rl.select_action(state, mode="rl")
hybrid_action = llm_rl.select_action(state, mode="hybrid", temperature=0.7)

print("LLM Action:", llm_action)
print("RL Action:", rl_action)
print("Hybrid Action:", hybrid_action)

# Generate training scenarios
scenarios = llm_rl.generate_training_scenarios(
    num_scenarios=5,
    difficulties=["easy", "normal", "hard"],
    scenario_types=["dust_storm", "equipment_failure"]
)

# Create training data for fine-tuning
training_data_path = llm_rl.create_training_data(
    num_examples=100,
    output_path="/path/to/training_data.txt"
)
```

## Integration with Simulation

The `SimulationLLMBridge` class connects the LLM integration with the simulation environment:

```python
from simulations.simulation_llm_bridge import SimulationLLMBridge

# Initialize bridge
bridge = SimulationLLMBridge(
    data_dir="/path/to/data",
    model_name="llama2",
    cache_dir="/path/to/cache",  # Optional
    rl_agent_path=None           # Optional
)

# Example simulation state
sim_state = {
    "time": [10, 14],
    "environment": {
        "temperature": -60.0,
        "pressure": 650.0,
        "dust_opacity": 0.3,
        "solar_irradiance": 500.0
    },
    "resources": {  # Note: simulation uses "resources" instead of "habitat"
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
result = bridge.get_action(sim_state, mode="llm", with_explanation=True)
print("Action:", result["action"])
print("Explanation:", result["explanation"])

# Generate scenario
scenario = bridge.generate_scenario(difficulty="hard", scenario_type="dust_storm")
print("Scenario:", scenario["description"])

# Save decision history
bridge.save_decision_history("/path/to/decisions.json")
```

### Integration with Simulation Environment

To use the bridge with the actual simulation environment:

```python
from simulations.rl_environment import MarsHabitatRLEnvironment
from simulations.simulation_llm_bridge import SimulationLLMBridge

# Initialize environment
env = MarsHabitatRLEnvironment()

# Initialize bridge
bridge = SimulationLLMBridge(
    data_dir="/path/to/data",
    model_name="llama2"
)

# Run simulation with LLM agent
state = env.reset()
done = False
total_reward = 0

while not done:
    # Get action from LLM
    action_result = bridge.get_action(state, mode="llm", with_explanation=True)
    action = action_result["action"]
    explanation = action_result["explanation"]
    
    # Print explanation (optional)
    print(f"Sol {state['time'][0]}, Hour {state['time'][1]}")
    print(f"Explanation: {explanation}")
    
    # Take action in environment
    next_state, reward, done, info = env.step(action)
    
    # Update total reward
    total_reward += reward
    
    # Update state
    state = next_state

print(f"Simulation complete. Total reward: {total_reward}")
```

## Dashboard Integration

To integrate LLM capabilities with the dashboard:

```python
import dash
from dash import dcc, html, Input, Output, State
import json
from simulations.simulation_llm_bridge import SimulationLLMBridge

# Initialize bridge
bridge = SimulationLLMBridge(
    data_dir="/path/to/data",
    model_name="llama2"
)

# Initialize dashboard app
app = dash.Dash(__name__)

# Define layout with LLM components
app.layout = html.Div([
    # ... existing dashboard components ...
    
    # LLM Integration Section
    html.Div([
        html.H2("LLM Assistant"),
        
        # Natural Language Query
        dcc.Input(
            id="nl-query",
            type="text",
            placeholder="Ask a question about habitat management...",
            style={"width": "100%"}
        ),
        html.Button("Ask", id="nl-submit", n_clicks=0),
        html.Div(id="nl-response", style={"marginTop": "10px"}),
        
        # LLM Action Explanation
        html.H3("Latest Decision Explanation"),
        html.Div(id="llm-explanation"),
        
        # LLM Mode Selection
        html.H3("Decision Mode"),
        dcc.RadioItems(
            id="llm-mode",
            options=[
                {"label": "LLM Only", "value": "llm"},
                {"label": "RL Only", "value": "rl"},
                {"label": "Hybrid", "value": "hybrid"}
            ],
            value="llm"
        ),
        
        # Hybrid Temperature Slider (only visible in hybrid mode)
        html.Div(
            id="hybrid-temp-container",
            children=[
                html.Label("LLM Influence"),
                dcc.Slider(
                    id="hybrid-temperature",
                    min=0,
                    max=1,
                    step=0.1,
                    value=0.5,
                    marks={i/10: str(i/10) for i in range(0, 11)}
                )
            ],
            style={"display": "none"}
        )
    ])
])

# Callback to show/hide hybrid temperature slider
@app.callback(
    Output("hybrid-temp-container", "style"),
    Input("llm-mode", "value")
)
def toggle_hybrid_temp(mode):
    if mode == "hybrid":
        return {"display": "block"}
    return {"display": "none"}

# Callback for natural language query
@app.callback(
    Output("nl-response", "children"),
    Input("nl-submit", "n_clicks"),
    State("nl-query", "value"),
    State("simulation-state", "data")  # Assuming this store contains current sim state
)
def process_query(n_clicks, query, sim_state):
    if n_clicks == 0 or not query:
        return ""
    
    if not sim_state:
        return "Please start the simulation first."
    
    # Parse simulation state
    state_dict = json.loads(sim_state)
    
    # Get answer from bridge
    answer = bridge.answer_query(query, state_dict)
    
    return html.Div([
        html.P(answer)
    ])

# Callback for LLM action and explanation
@app.callback(
    [Output("llm-explanation", "children"),
     Output("action-store", "data")],  # Store to hold the action
    [Input("step-button", "n_clicks")],
    [State("simulation-state", "data"),
     State("llm-mode", "value"),
     State("hybrid-temperature", "value")]
)
def get_llm_action(n_clicks, sim_state, mode, temperature):
    if n_clicks == 0 or not sim_state:
        return "", None
    
    # Parse simulation state
    state_dict = json.loads(sim_state)
    
    # Get action with explanation
    result = bridge.get_action(
        state_dict, 
        mode=mode,
        with_explanation=True
    )
    
    # Store action for simulation step
    action = result["action"]
    explanation = result["explanation"]
    
    return html.Div([
        html.P(explanation)
    ]), json.dumps(action)

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
```

## Fine-tuning Models

To fine-tune an Ollama model for Mars habitat management:

### 1. Create Training Data

```python
from models.ollama_integration import MarsHabitatLLMRL

# Initialize LLM-RL integration
llm_rl = MarsHabitatLLMRL(
    data_dir="/path/to/data",
    model_name="llama2"
)

# Create training data
training_data_path = llm_rl.create_training_data(
    num_examples=100,
    output_path="/path/to/training_data.txt"
)
```

### 2. Create Modelfile

```bash
cat > Modelfile << EOF
FROM llama2
SYSTEM You are an AI assistant specialized in Mars habitat resource management.
TEMPLATE <s>[INST] {{ .System }} {{ .Prompt }} [/INST] {{ .Response }}</s>
PARAMETER temperature 0.7
PARAMETER top_p 0.9
EOF
```

### 3. Create and Fine-tune Model

```bash
# Create the model
ollama create mars-habitat-assistant -f Modelfile

# Fine-tune the model
ollama run mars-habitat-assistant -f /path/to/training_data.txt
```

### 4. Use Fine-tuned Model

```python
from models.ollama_integration import MarsHabitatLLMAgent

# Initialize agent with fine-tuned model
agent = MarsHabitatLLMAgent(
    data_dir="/path/to/data",
    model_name="mars-habitat-assistant"
)

# Use the fine-tuned model
action = agent.select_action(state)
```

## Troubleshooting

### Ollama Server Not Running

**Symptom**: Error connecting to Ollama server

**Solution**:
1. Check if Ollama is running:
   ```bash
   ps aux | grep ollama
   ```
2. Start Ollama if not running:
   ```bash
   ollama serve
   ```

### Model Not Found

**Symptom**: Error about model not being available

**Solution**:
1. List available models:
   ```bash
   ollama list
   ```
2. Pull the required model:
   ```bash
   ollama pull llama2
   ```

### JSON Parsing Errors

**Symptom**: LLM generates invalid JSON that can't be parsed

**Solution**:
1. Lower the temperature (e.g., 0.2 instead of 0.7)
2. Use a more capable model (e.g., llama2:13b instead of llama2)
3. Improve the prompt with clearer instructions
4. Use the `generate_json` method which has built-in error handling

### Slow Response Times

**Symptom**: LLM responses take too long

**Solution**:
1. Enable caching:
   ```python
   ollama_client = OllamaClient(
       cache_dir="/path/to/cache"
   )
   ```
2. Use a smaller model (e.g., llama2:7b instead of llama2:13b)
3. Limit the max_tokens parameter
4. Run on a machine with better GPU support

### Out of Memory Errors

**Symptom**: Ollama crashes with out of memory errors

**Solution**:
1. Use a smaller model
2. Reduce batch size or context length
3. Add more RAM or GPU memory to your system
4. Use Ollama's quantized models (e.g., llama2:7b-q4_0)

## Performance Optimization

### Caching

Enable caching to avoid redundant LLM calls:

```python
ollama_client = OllamaClient(
    cache_dir="/path/to/cache"
)
```

### Batch Processing

Process multiple queries in batches when possible:

```python
# Instead of multiple individual calls
for state in states:
    action = agent.select_action(state)
    
# Process in batches (implementation depends on specific needs)
batch_states = states[:10]
batch_results = process_batch(batch_states)
```

### Model Selection

Choose the right model for your needs:

- **llama2:7b**: Fastest, good for basic tasks
- **llama2:13b**: Better reasoning, slower
- **mistral:7b**: Good balance of performance and speed
- **gemma:7b**: Another good option for balance

### Quantization

Use quantized models for faster inference with minimal quality loss:

```bash
ollama pull llama2:7b-q4_0
```

Then use in your code:

```python
agent = MarsHabitatLLMAgent(
    data_dir="/path/to/data",
    model_name="llama2:7b-q4_0"
)
```

## Examples

### Example 1: Basic Decision Making

```python
from models.ollama_integration import MarsHabitatLLMAgent

# Initialize agent
agent = MarsHabitatLLMAgent(
    data_dir="/home/ubuntu/martian_habitat_pathfinder/data",
    model_name="llama2"
)

# Define a state with dust storm
dust_storm_state = {
    "time": [10, 14],
    "environment": {
        "temperature": -60.0,
        "pressure": 650.0,
        "dust_opacity": 0.8,  # High dust opacity
        "solar_irradiance": 200.0  # Low solar irradiance due to dust
    },
    "habitat": {
        "power": 80.0,  # Lower power due to reduced solar generation
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

# Get action and explanation
action = agent.select_action(dust_storm_state)
explanation = agent.explain_decision(dust_storm_state, action)

print("Dust Storm Scenario")
print("===================")
print(f"Power: {dust_storm_state['habitat']['power']} kWh")
print(f"Dust Opacity: {dust_storm_state['environment']['dust_opacity']}")
print(f"Solar Irradiance: {dust_storm_state['environment']['solar_irradiance']} W/m²")
print("\nRecommended Action:")
print(f"Power to Life Support: {action['power_allocation']['life_support']:.2f} kW")
print(f"Power to ISRU: {action['power_allocation']['isru']:.2f} kW")
print(f"Power to Thermal Control: {action['power_allocation']['thermal_control']:.2f} kW")
print(f"ISRU Mode: {action['isru_mode']}")
print(f"Maintenance Target: {action['maintenance_target']}")
print("\nExplanation:")
print(explanation)
```

### Example 2: Scenario Generation and Simulation

```python
from models.ollama_integration import MarsHabitatLLMRL
from simulations.simulation_llm_bridge import SimulationLLMBridge
from simulations.rl_environment import MarsHabitatRLEnvironment

# Initialize bridge
bridge = SimulationLLMBridge(
    data_dir="/home/ubuntu/martian_habitat_pathfinder/data",
    model_name="llama2"
)

# Generate scenario
scenario = bridge.generate_scenario(difficulty="hard", scenario_type="equipment_failure")

print("Generated Scenario:")
print("==================")
print(f"Description: {scenario['description']}")
print(f"Environment: {scenario['environment']}")
print(f"Resources: {scenario['resources']}")
print(f"Subsystems: {scenario['subsystems']}")
if 'events' in scenario and scenario['events']:
    print("\nEvents:")
    for event in scenario['events']:
        print(f"- {event['description']} at Sol {event['time']['sol']}, Hour {event['time']['hour']}")

# Initialize environment with scenario
env = MarsHabitatRLEnvironment(
    initial_conditions=scenario
)

# Run simulation with LLM agent
state = env.reset()
done = False
total_reward = 0
step = 0

print("\nSimulation Run:")
print("===============")

while not done and step < 10:  # Limit to 10 steps for example
    step += 1
    
    # Get action from LLM
    action_result = bridge.get_action(state, mode="llm", with_explanation=True)
    action = action_result["action"]
    
    print(f"\nStep {step}:")
    print(f"Sol {state['time'][0]}, Hour {state['time'][1]}")
    print(f"Power: {state['resources']['power']:.2f} kWh")
    print(f"Action: {action}")
    
    # Take action in environment
    next_state, reward, done, info = env.step(action)
    
    print(f"Reward: {reward:.2f}")
    
    # Update total reward and state
    total_reward += reward
    state = next_state

print(f"\nSimulation complete. Total reward: {total_reward:.2f}")
```

### Example 3: Fine-tuning and Evaluation

```python
from models.ollama_integration import MarsHabitatLLMRL
import os
import subprocess

# Initialize LLM-RL integration
llm_rl = MarsHabitatLLMRL(
    data_dir="/home/ubuntu/martian_habitat_pathfinder/data",
    model_name="llama2"
)

# Create training data
print("Creating training data...")
training_data_path = llm_rl.create_training_data(
    num_examples=5,  # Small number for example
    output_path="/home/ubuntu/martian_habitat_pathfinder/data/training_data.txt"
)

# Create Modelfile
modelfile_path = "/home/ubuntu/martian_habitat_pathfinder/data/Modelfile"
with open(modelfile_path, 'w') as f:
    f.write("FROM llama2\n")
    f.write("SYSTEM You are an AI assistant specialized in Mars habitat resource management.\n")
    f.write("TEMPLATE <s>[INST] {{ .System }} {{ .Prompt }} [/INST] {{ .Response }}</s>\n")
    f.write("PARAMETER temperature 0.7\n")
    f.write("PARAMETER top_p 0.9\n")

print("Modelfile created at", modelfile_path)

# Note: In a real implementation, you would run these commands
# For this example, we'll just print the commands
print("\nTo create and fine-tune the model, run:")
print(f"ollama create mars-habitat-assistant -f {modelfile_path}")
print(f"ollama run mars-habitat-assistant -f {training_data_path}")

# Simulate evaluation
print("\nEvaluation Simulation:")
print("=====================")

# Define test states
test_states = [
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
        }
    }
]

# Evaluate on test states
for test_case in test_states:
    print(f"\nScenario: {test_case['name']}")
    
    # Get actions using different modes
    llm_action = llm_rl.select_action(test_case['state'], mode="llm")
    
    print(f"LLM Action:")
    print(f"- Power to Life Support: {llm_action['power_allocation']['life_support']:.2f} kW")
    print(f"- Power to ISRU: {llm_action['power_allocation']['isru']:.2f} kW")
    print(f"- Power to Thermal Control: {llm_action['power_allocation']['thermal_control']:.2f} kW")
    print(f"- ISRU Mode: {llm_action['isru_mode']}")
    print(f"- Maintenance Target: {llm_action['maintenance_target']}")
```

These examples demonstrate how to use the Ollama integration in various scenarios within the Martian Habitat Pathfinder project.
