# Data Formats for LLM-RL Integration in Martian Habitat Pathfinder

This document defines the standard data formats used for integrating Large Language Models (LLMs) with Reinforcement Learning (RL) in the Martian Habitat Pathfinder project.

## Overview

The integration between LLMs and RL requires well-defined data formats for:
1. State representations
2. Action representations
3. Prompt templates
4. JSON schemas for structured outputs
5. Training data formats
6. Scenario definitions

These standardized formats ensure consistent communication between components and reliable system behavior.

## 1. State Representation

### 1.1 State Dictionary Format

The environment state is represented as a nested dictionary with the following structure:

```json
{
  "time": [sol, hour],
  "environment": {
    "temperature": float,
    "pressure": float,
    "dust_opacity": float,
    "solar_irradiance": float
  },
  "habitat": {
    "power": float,
    "water": float,
    "oxygen": float,
    "food": float,
    "spare_parts": float
  },
  "subsystems": {
    "power_system": {
      "status": string,
      "maintenance_needed": float
    },
    "life_support": {
      "status": string,
      "maintenance_needed": float
    },
    "isru": {
      "status": string,
      "maintenance_needed": float
    },
    "thermal_control": {
      "status": string,
      "maintenance_needed": float
    }
  }
}
```

### 1.2 State Field Specifications

| Field | Type | Range | Units | Description |
|-------|------|-------|-------|-------------|
| time[0] | int | ≥ 0 | sols | Martian day count |
| time[1] | int | 0-24 | hours | Hour of the Martian day |
| temperature | float | -120 to 20 | °C | External temperature |
| pressure | float | 600 to 700 | Pa | Atmospheric pressure |
| dust_opacity | float | 0.1 to 0.9 | - | Atmospheric dust opacity |
| solar_irradiance | float | 0 to 600 | W/m² | Solar radiation intensity |
| power | float | ≥ 0 | kWh | Available electrical power |
| water | float | ≥ 0 | liters | Available water |
| oxygen | float | ≥ 0 | kg | Available oxygen |
| food | float | ≥ 0 | kg | Available food |
| spare_parts | float | ≥ 0 | units | Available spare parts |
| status | string | "operational", "degraded", "failed" | - | Subsystem operational status |
| maintenance_needed | float | 0.0 to 1.0 | - | Maintenance level (0=none, 1=critical) |

### 1.3 State Embedding Format

When using state embeddings for LLM processing:

```python
{
  "state_text": string,  # Formatted text representation of state
  "embedding": List[float],  # Vector representation, typically 1024 or 4096 dimensions
  "model": string  # Model used to generate embedding
}
```

## 2. Action Representation

### 2.1 Action Dictionary Format

Actions are represented as a dictionary with the following structure:

```json
{
  "power_allocation": {
    "life_support": float,
    "isru": float,
    "thermal_control": float
  },
  "isru_mode": string,
  "maintenance_target": string or null
}
```

### 2.2 Action Field Specifications

| Field | Type | Range | Units | Description |
|-------|------|-------|-------|-------------|
| power_allocation.life_support | float | 0 to 10 | kW | Power allocated to life support |
| power_allocation.isru | float | 0 to 10 | kW | Power allocated to ISRU |
| power_allocation.thermal_control | float | 0 to 10 | kW | Power allocated to thermal control |
| isru_mode | string | "water", "oxygen", "both", "off" | - | ISRU operation mode |
| maintenance_target | string or null | "power_system", "life_support", "isru", "thermal_control", null | - | Subsystem targeted for maintenance |

### 2.3 Action JSON Schema

The JSON schema used for validating LLM-generated actions:

```json
{
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
      "enum": ["power_system", "life_support", "isru", "thermal_control", null]
    }
  },
  "required": ["power_allocation", "isru_mode"]
}
```

## 3. Prompt Templates

### 3.1 State Formatting Template

Template for formatting state as a prompt for LLMs:

```
You are an AI assistant managing a Mars habitat with the following state:

Time: Sol {time[0]}, Hour {time[1]}

Habitat Resources:
- Power: {habitat.power:.2f} kWh
- Water: {habitat.water:.2f} liters
- Oxygen: {habitat.oxygen:.2f} kg
- Food: {habitat.food:.2f} kg
- Spare Parts: {habitat.spare_parts:.2f} units

Environmental Conditions:
- Temperature: {environment.temperature:.2f} °C
- Pressure: {environment.pressure:.2f} Pa
- Dust Opacity: {environment.dust_opacity:.2f}
- Solar Irradiance: {environment.solar_irradiance:.2f} W/m²

Subsystem Status:
- Power System: {subsystems.power_system.status} (Maintenance: {subsystems.power_system.maintenance_needed:.2f})
- Life Support: {subsystems.life_support.status} (Maintenance: {subsystems.life_support.maintenance_needed:.2f})
- ISRU: {subsystems.isru.status} (Maintenance: {subsystems.isru.maintenance_needed:.2f})
- Thermal Control: {subsystems.thermal_control.status} (Maintenance: {subsystems.thermal_control.maintenance_needed:.2f})
```

### 3.2 Action Selection Prompt Template

Template for prompting LLM to select an action:

```
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
```

### 3.3 Explanation Prompt Template

Template for prompting LLM to explain a decision:

```
{state_prompt}

Decision Made:
- Power to Life Support: {action.power_allocation.life_support:.2f} kW
- Power to ISRU: {action.power_allocation.isru:.2f} kW
- Power to Thermal Control: {action.power_allocation.thermal_control:.2f} kW
- ISRU Mode: {action.isru_mode}
- Maintenance Target: {action.maintenance_target if action.maintenance_target else 'None'}

Explain why this decision is optimal given the current state. Consider resource levels, environmental conditions, and subsystem status in your explanation. Keep your explanation concise but informative.
```

### 3.4 Scenario Generation Prompt Template

Template for prompting LLM to generate a scenario:

```
Generate a detailed Mars habitat scenario for reinforcement learning training.

Difficulty: {difficulty}
Scenario Type: {scenario_type}

Provide a description of the scenario including:
1. Environmental conditions (temperature, pressure, dust opacity, solar irradiance)
2. Initial resource levels (power, water, oxygen, food)
3. Subsystem status (power system, life support, ISRU, thermal control)
4. Any special events or challenges

The scenario should be challenging but solvable, with appropriate difficulty level.
```

## 4. Scenario Definition Format

### 4.1 Scenario Dictionary Format

Scenarios are represented as a dictionary with the following structure:

```json
{
  "description": string,
  "environment": {
    "temperature": float,
    "pressure": float,
    "dust_opacity": float,
    "solar_irradiance": float
  },
  "resources": {
    "power": float,
    "water": float,
    "oxygen": float,
    "food": float
  },
  "subsystems": {
    "power_system": {"status": string, "maintenance_needed": float},
    "life_support": {"status": string, "maintenance_needed": float},
    "isru": {"status": string, "maintenance_needed": float},
    "thermal_control": {"status": string, "maintenance_needed": float}
  },
  "events": [
    {
      "type": string,
      "time": {"sol": int, "hour": int},
      "description": string,
      "effects": {
        "target": string,
        "magnitude": float
      }
    }
  ]
}
```

### 4.2 Scenario JSON Schema

The JSON schema used for validating LLM-generated scenarios:

```json
{
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
```

## 5. Training Data Format

### 5.1 LLM Fine-tuning Format

The format for fine-tuning data follows the Ollama convention:

```
<s>[INST] You are an AI assistant for Mars habitat management. The current state is:

{state_prompt}

What is the optimal resource allocation strategy? [/INST]

Based on the current state, I recommend the following allocation:

Power Allocation:
- Life Support: {action.power_allocation.life_support:.1f} kW
- ISRU: {action.power_allocation.isru:.1f} kW
- Thermal Control: {action.power_allocation.thermal_control:.1f} kW

ISRU Mode: {action.isru_mode.title()}
Maintenance Target: {action.maintenance_target.replace('_', ' ').title() if action.maintenance_target else 'None'}

{explanation}</s>
```

### 5.2 RL Experience Format

The format for storing RL experience:

```python
{
  "state": Dict,  # State dictionary as defined in section 1.1
  "action": Dict,  # Action dictionary as defined in section 2.1
  "reward": float,  # Reward received
  "next_state": Dict,  # Next state dictionary
  "done": bool,  # Whether episode is done
  "info": Dict  # Additional information
}
```

## 6. Evaluation Metrics Format

### 6.1 Evaluation Results Format

The format for storing evaluation results:

```json
{
  "mean_reward": float,
  "std_reward": float,
  "mean_length": float,
  "rewards": List[float],
  "episode_lengths": List[int]
}
```

### 6.2 LLM-RL Metrics Format

The format for tracking LLM-RL integration metrics:

```json
{
  "llm_actions": int,
  "rl_actions": int,
  "hybrid_actions": int,
  "rewards": List[float],
  "llm_response_times": List[float],
  "rl_response_times": List[float]
}
```

## 7. API Response Formats

### 7.1 Ollama Generate Response Format

The format of responses from Ollama's generate API:

```json
{
  "model": string,
  "created_at": string,
  "response": string,
  "done": bool,
  "context": List[int],
  "total_duration": int,
  "load_duration": int,
  "prompt_eval_duration": int,
  "eval_duration": int
}
```

### 7.2 Ollama JSON Response Format

The format of JSON responses from the Ollama integration:

```json
{
  "model": string,
  "created_at": string,
  "response": string,
  "parsed_json": object,
  "done": bool,
  "total_duration": int
}
```

## 8. Configuration Format

### 8.1 LLM-RL Configuration Format

The format for configuring the LLM-RL integration:

```json
{
  "environment": {
    "temperature_range": [float, float],
    "pressure_range": [float, float],
    "dust_opacity_range": [float, float],
    "solar_irradiance_range": [float, float]
  },
  "habitat": {
    "initial_power": float,
    "initial_water": float,
    "initial_oxygen": float,
    "initial_food": float,
    "initial_spare_parts": float
  },
  "simulation": {
    "max_steps": int,
    "difficulty": string
  },
  "llm": {
    "system_prompt": string,
    "temperature": float,
    "max_tokens": int,
    "model": string,
    "cache_enabled": bool
  },
  "rl": {
    "algorithm": string,
    "learning_rate": float,
    "discount_factor": float,
    "exploration_rate": float
  },
  "integration": {
    "mode": string,
    "llm_weight": float,
    "hybrid_temperature": float
  }
}
```

## Best Practices

1. **Validation**: Always validate data against the defined schemas before processing
2. **Error Handling**: Provide fallback mechanisms for handling invalid or missing data
3. **Versioning**: Include version information in data formats to support future changes
4. **Logging**: Log data format issues for debugging and improvement
5. **Documentation**: Keep this document updated as formats evolve

## Implementation Examples

See the `ollama_integration.py` module for implementation examples of these data formats.
